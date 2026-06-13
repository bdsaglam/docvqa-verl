# Handoff — teacher-trajectory generation → rejection-sampling SFT (2026-06-13)

Continue the **27B-teacher rejection-sampling SFT** for the ≤8B DocVQA agent. This
doc is self-contained: pipeline, exact commands, environment/infra fixes, and the
session's learnings (VLM throughput, RL, scaffold parity). Work happens on branch
`docvqa-stage0` (this worktree); the RL+SFT scaffold was merged here from `docvqa-rl`.

## Where we are
- **Goal:** beat the in-our-scaffold untrained 4B/27B baseline (see recalibration below)
  by SFT-ing Qwen3.5-4B on **anls==1.0** 27B-teacher CodeAct trajectories.
- **Path:** generate teacher rollouts on a pool → keep the solved ones → 20-epoch LoRA
  SFT → eval on full val. (RL-from-base was tried and parked — single-page-curriculum
  mismatch; see learnings.)
- **Status at handoff:** teacher generation was running under the corrected concat
  scaffold (see Parity) and **stopped** for you to resume here. ~146 parse_first-era
  trajectories are archived (do NOT mix with concat-era data) at
  `~/repos/docvqa-verl-rl/outputs/runs/teacher-gen-pool-parsefirst-archived/`.

## Servers currently UP (reuse them — expensive to restart)
- `:8932` — **27B teacher-AGENT**, DP2 on GPUs 0,1 (tmux `teacher-gen` in the rl worktree).
- `:8927` — **27B VLM** (perception), DP2 on GPUs 2,3.
- `:8928` — **27B VLM**, remote box (3 GPUs). Reached via SSH tunnel on localhost:8928.
- Perception uses a **load-balanced pool** of `:8927`+`:8928` (see VLM learnings).

## Resume generation (the long pole — hours)
Pool file (405 prompts, 38% multi-page incl. 45 docs ≥11pages) lives at
`~/repos/docvqa-verl-rl/data/pool/teacher_gen_pool.json` (rebuild with
`python -m docvqa.scripts.make_teacher_gen_pool` — needs `data/pool/curriculum_rl.parquet`).

```bash
# from this repo, with the RL venv active (see Environment):
POOL='http://localhost:8927@2|http://localhost:8928@3'   # weighted least-loaded pool
python docvqa/scripts/eval.py \
  --questions /home/baris/repos/docvqa-verl-rl/data/pool/teacher_gen_pool.json \
  --base-url http://localhost:8932/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url "$POOL" --vlm-model Qwen/Qwen3.5-27B \
  --concurrency 24 --n 8 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --no-thinking --rollout-timeout 1200 \
  --run-dir outputs/runs/teacher-gen-pool --resume
```
- **`--n 8`**: 8 rollouts/prompt → high chance ≥1 solves (rejection sampling). Yield was
  ~75–88% on easy single-page, dropping on multi-page.
- **`--no-thinking`**: matches the agent design + what the 4B will be trained/deployed with.
- `eval.py` streams `tasks/<doc>/trajectories.jsonl` (full `messages` + `anls`) as questions
  finish; **resumable** (`--resume` skips done questions). Stop anytime; trajectories persist.
- Stop when you have ~400–600 KEPT with multi-page coverage (the ≥11p docs are dripped
  through the pool ordering — don't stop on just the easy head).

## Pipeline after generation
```bash
# 1) Rejection-sample -> verl multi-turn SFT parquet (keeps anls==1.0 + submit, VERBATIM
#    assistant turns incl. multi-fence — the scaffold concatenates fences at inference).
python docvqa/scripts/make_sft_data.py \
  --in outputs/runs/teacher-gen-pool --out data/sft/teacher_pool.parquet \
  --max-per-question 2          # cap near-dup rollouts of the same easy prompt

# 2) SFT (needs 2 GPUs, NO VLM). Free the agent GPUs first. KEY PARAMS:
EPOCHS=20 LR=2e-4 SAVE_FREQ=<~per-2-epochs> MODEL_PATH=Qwen/Qwen3.5-4B \
  bash docvqa/train/run_seqkd.sh data/sft/teacher_pool.parquet seqkd-teacher-pool 2
#   - EPOCHS=20 is LOAD-BEARING: 3 epochs HURT last time; 20 is what beat baseline.
#   - LR=2e-4 (not the 1e-4 default); all-linear LoRA r32 is already the default.

# 3) Merge LoRA ckpt -> HF, then eval. Merge needs the ACTOR subdir:
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/docvqa-verl/seqkd-teacher-pool/global_step_<N>/actor \
  --target_dir <...>/merged_hf
#   then copy preprocessor_config.json + video_preprocessor_config.json from the
#   Qwen3.5-4B HF snapshot into merged_hf (the merge omits the VL image-processor config).

# 4) Eval the SFT model (concat scaffold parity is ON by default in eval.py now):
#    SCREEN on docvqa_mini (29Q), CONFIRM on full questions.json (80Q). --no-thinking.
```

## Scaffold parity (codeact_chat) — DECIDED THIS SESSION
The deployed/leaderboard solver `codeact_chat_solver._extract_code` **concatenates and
runs EVERY complete ```python```` fenced block** in a turn (after stripping paired
`<think>`). Our `agent_loop` historically ran only the FIRST fence (`parse_first_fence`).
For parity we added `concat_fences` (`agent_loop._extract_code_concat`, byte-identical to
the solver) and **`eval.py` sets it True** → generation + SFT-eval now share fence
semantics with deploy. Consequences:
- `make_sft_data` keeps assistant turns **verbatim** (multi-fence included) — they are valid
  actions under concat, not contamination. (No first-fence truncation.)
- A truncated/unclosed block has no closing ``` so it's ignored; inter-block prose (thinking
  off writes rationale between blocks) is ignored — only complete fenced blocks run.

## Environment & infra fixes (REQUIRED — silent failures otherwise)
**Venv:** `~/repos/docvqa-verl-rl/.venv-rl2` (gitignored). The ONLY env with a Qwen3.5-capable
vLLM. Versions: **torch 2.10.0+cu128, vLLM 0.17.0, transformers 5.9.0, numpy 2.4.6,
cupy 14.1.1**. (verl pins vLLM ≤0.12 which predates Qwen3.5 — we install around the pin.)
- **cupy/numpy ABI:** the `nccl` checkpoint_engine imports cupy; cupy must match numpy's major
  (cupy 14.1 ↔ numpy 2.x). A numpy-2-built cupy on numpy-1.x fails silently
  (`numpy.core.multiarray failed to import` → "Checkpoint engine nccl not registered").
- **vLLM patch (required for CUDA graphs + LoRA on Qwen3.5):**
  `patches/vllm-0.17-gdn-lora-cudagraph.patch.md` — applied to `.venv-rl2`. Without it,
  cudagraph capture dies (`IndexError` in dummy-LoRA, vllm#36372). Exclude GDN modules
  (`in_proj_qkvz/in_proj_ba/conv1d/out_proj`) from LoRA wrapping.
- **`rollout.load_format=safetensors`** (NOT dummy) for Qwen3.5 — GDN/linear-attn weights
  otherwise ship split/unwrapped and silently skip → gibberish rollouts, zero reward.
- **vLLM serve flags** that matter for this perception workload (1 image ≈16K-tok prefill,
  ~45-tok output, thinking off):
  `--max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.90
   --async-scheduling --limit-mm-per-prompt '{"image":1}' --dtype bfloat16`. **No
  `--enforce-eager` on a serve-only VLM** (that flag is only for verl-managed engines that
  mutate weights). Agent serve used `--data-parallel-size 2 --max-model-len 40960`.

## Operational gotchas (cost us time)
- **NEVER kill by matching `EngineCore`** — it matches BOTH a verl rollout engine AND the
  served 27B VLMs; you'll take down the VLM. Kill by `--port` (lsof -ti :PORT) or tmux session.
- **`ray stop --force` + kill stale PIDs between RL runs** — stale ray workers get reused and
  silently ignore your fixes.
- **Do NOT set `PYTORCH_ALLOC_CONF=expandable_segments:True`** with the CUDA-IPC weight-transfer
  path — it breaks it (`pidfd_getfd: Operation not permitted` → hang). Lower
  `gpu_memory_utilization` instead to make headroom.
- Bash `pkill`/compound cmds sometimes return exit 144 in this harness but still succeed —
  verify with a follow-up `ps`/`curl` rather than trusting the exit code.

## Session learnings

### VLM throughput optimization (the dominant lever for this agentic task)
- **Client-side image downscale to 16,777,216 px** (= Qwen3.5-27B processor `size.longest_edge`,
  i.e. max AREA): lossless w.r.t. what the model sees (the server resizes to this anyway), but
  done BEFORE base64+HTTP+decode → saves the transfer/CPU cost on the giant pages. Sits AFTER
  the agent's crop (sandbox crops full-res from disk), so survey-coarse/crop-fine is unaffected.
  Also moved image prep OFF the event loop (was blocking all rollouts in the worker).
  (`docvqa/tools.py _prepare_image_b64`.)
- **Weighted least-loaded endpoint pool** (`tools.py EndpointPool`): spec `url@w|url@w`,
  picks `argmin(inflight/weight)`, health-benches a failed endpoint 60s then retries — so a
  remote VLM that comes up mid-run is absorbed without restart. Latency-aware: a tunneled
  remote naturally gets less than its GPU-share (its requests stay in-flight longer).
- **Prefix caching is free throughput**: payload order (system → image → query) means a re-look
  at the same page hits the ~16K-token image prefill in vLLM's cache; only the short query recomputes.
- **Thinking OFF on the VLM** (`enable_thinking:false`, temp 0.3): perception answers are short;
  thinking added tens of seconds/call and dominated rollout wall-clock.
- Serve flags above (prefill-bound → `max-num-batched-tokens`, `async-scheduling`).

### RL (async GRPO) — bottleneck migration via per-phase profiling
We assumed VLM-bound; **per-phase timers in `agent_loop` (t_gen/t_exec/t_vlm) falsified it.**
The binding constraint MIGRATED as each fix landed (profile, don't assume):
1. **Eager policy decode was 80% of rollout wall-clock** @ 15.8 tok/s (Qwen3.5 GDN = many tiny
   kernels CUDA graphs amortize). `enforce_eager=False` → **59.9 tok/s (3.8×)**, but needs two
   capture fixes: the GDN-LoRA patch (vllm#36372) AND `rollout.max_num_seqs=64` (GDN conv-state
   cache has fewer lines than the default 512 capture batch → `assert num_cache_lines>=batch`).
2. Then training became the gate (`update_actor` ~420s on 1 GPU, micro_batch=1). **2 train + 2
   rollout GPUs** (FSDP-2) halved it (~210s). Net step time ~330s.
3. Disaggregated one-step-off (`verl.experimental.one_step_off_policy`): generation hides behind
   training, so the gate is whichever side is slower — every speedup flips it.
- Other verl/Qwen3.5 fixes: `migrate_legacy_reward_impl` (async `main_ppo` dropped it →
  reward NotImplementedError); `bypass_mode=False` (our agent loop doesn't emit rollout
  logprobs); `gpu_memory_utilization=0.25` (fp32 logprob logits ≈16GB on the rollout GPU).
- **Per-step reward ≠ learning** under `shuffle=False` curriculum (prompt-confounded). Judge by
  held-out eval only. Full detail: `project/experiments/RL-async-findings.md`.

### Curriculum / data caveats
- The 8-source pool is **84% single-page** and the RL curriculum was page-ascending, so a short
  RL run trains ONLY single-page — but DocVQA-2026 val is **~45% multi-page (34% ≥10 pages)**.
  RL-from-base on single-page likely doesn't transfer to multi-page navigation. The SFT pool here
  deliberately includes multi-page docs (the 27B teacher CAN navigate them → teaches the 4B).
  Throughput-capped at ≤30 pages (the 31–89p monsters are the ~800s-tail rollouts).
- `make_mixed_curriculum_parquet.py` exists (multi-page-inclusive RL curriculum) if RL is revisited.

### Baseline recalibration (IMPORTANT for "did it beat baseline")
The untrained 4B/27B in the improved `codeact_chat` scaffold is **22.34% ± 3.44 (n=8, full val)**
— NOT the old 15.66%. The scaffold fix itself lifted the untrained model +7pp. So SFT must clear
**~22%**, and the prior SFT's +5.3 (to 20.6% on the OLD scaffold) would NOT clear it — SFT's gain
may overlap the scaffold's. Caveat: our scaffold here vs the docvqa-repo codeact_chat may not have
exact parity, so **our own base-4B eval (same agent_loop) is the valid comparison**, not the
cross-repo 22.34%.

### SFT methodology (from `project/experiments/SFT-synthesis.md`)
- **Screen on docvqa_mini (29Q), CONFIRM on full 80Q val.** The mini set (SE≈5%, median-difficulty
  docs) produced a FALSE NULL that survived FOUR SFT experiments. Any verdict from mini must be
  re-checked on full val before it drives a decision or goes in the report.
- 20 epochs (undertraining hurts), all-linear LoRA, LR ~2e-4, constant/cosine both fine for short runs.
- Per-turn 4K cap clips ~13% of student turns post-SFT — consider raising `max_response_tokens_per_turn`.
