# GRPO pre-launch briefing — base Qwen3.5-4B DocVQA CodeAct agent (2026-06-14)

> ## ⭐ RESOLVED FOR RUN #1 (driver: follow these; they override the "open decisions" in §5)
> Verified + decided 2026-06-15 by the driver-setup pass:
> 1. **TRAIN_FILES = `data/pool/teacher_gen_curriculum_rl.parquet` — ALREADY BUILT** (405 rows,
>    page-sorted easy→hard, num_pages 1–30). Do NOT rebuild. VAL_FILES = same path.
> 2. **NO LEAKAGE — verified.** teacher_gen_pool is the BROAD family pool (mp-docvqa/tatdqa/
>    mmlongbench/chartqa/docvqa-sp/infographicvqa/mapqa/slidevqa), 356 docs / 404 Q. It is
>    **disjoint** from the eval set (questions.json 80Q/25docs, rank13) — **0 overlap at doc AND
>    question level** (checked). So §5.3 concern does NOT apply: this is a clean train-on-broad-pool
>    → eval-on-DocVQA-2026-val generalization run. Eval on rank13 (fast) then questions.json.
> 3. **Efficiency/length reward = OFF (0.0) for run #1** (§5.1). Clean GRPO baseline first; A/B the
>    penalty as a 2nd arm. Do NOT edit reward.py for this run.
> 4. **Page filter = NONE** (§5.2). Keep the full 405 (honors "curriculum by page count"). Accept
>    that the heavy 15–30pg tail may cap (0 reward, zero-variance group) — harmless (no gradient),
>    just slow. Note tail-waste; revisit for run #2 if it dominates.
> 5. **ENFORCE_EAGER=True** (§5.4 / gotcha #10) — safe, VLM-bound. Don't assume the GDN-LoRA
>    cudagraph patch is applied.
> 6. **Plain GRPO** (§5.5). No Dr.GRPO/GSPO/CISPO for run #1.
> 7. **Launch exactly as §1.3** (MAX_STEPS=520, SAVE_FREQ=52, ≥10 epochs). Validation traj dump is
>    whatever `run_grpo.sh` sets (`outputs/async_traj.jsonl` per §3) — use that file for the §3
>    coherence check, not a custom path.
> 8. **Pre-launch cleanup is REQUIRED** (§1.0): kill the n8 eval agent (:8930) + the local 27B VLM
>    DP3 (currently GPUs0,1,2 :8927, tmux docvqarun:0) by PID/port, `ray stop --force`, confirm GPUs
>    clear, THEN bring up the 27B **DP2 on GPUs 2,3** (§1.2) before launching. NEVER kill by EngineCore.
> **ETA ~20–26h for 520 steps (VLM-bound). Validate step 1–3 (§3) and STOP on gibberish/all-0/grad0
> before walking away — do not let a broken run burn 20h.**


Operational checklist for launching async GRPO (`one_step_off_policy`) on the **base** 4B
inside the CodeAct REPL agent, dictated 2+2 GPU layout. Read top to bottom before launching.
Sources: `docvqa/train/run_grpo.sh`, `RL-async-findings.md`, `RL-OPD-design.md`, `CLAUDE.md`
(RL-practices / GPU-layout / Eval-methodology), `.claude/CLAUDE.md` registry.

---

## 0. Decisions baked in (so the launcher doesn't re-derive them)

- **Model:** base `Qwen/Qwen3.5-4B` (MODEL_PATH default). NOT an SFT ckpt.
- **GPU split:** policy on `CUDA_VISIBLE_DEVICES=0,1`, `N_ROLLOUT_GPU=1` (GPU0 gen) +
  `N_TRAIN_GPU=1` (GPU1 train). Local 27B VLM DP2 on GPUs 2,3.
- **Perception pool:** `http://localhost:8927@2|http://144.122.52.7:8927@3` (local DP2 + remote
  DP3 DIRECT — no SSH tunnel; remote reachable directly per the 2026-06-14 registry note).
- **Data:** a NEW curriculum parquet built from the **teacher_gen_pool** 405 prompts (see §1.1).
  This satisfies "train on the same prompts used for teacher rollout generation."
- **Epochs:** ≥10. With 405 rows ÷ batch 8 = **51 steps/epoch** → `MAX_STEPS=520` ≈ 10.2 epochs
  (`SAVE_FREQ=52` ≈ one ckpt/epoch, 10 ckpts).
- **Curriculum:** easy→hard by `num_pages`, `SHUFFLE=False` (builder + flag both required).

---

## 1. Exact launch sequence

### 1.0 Prerequisites (do these first, in order)

1. **Free GPUs / kill stale servers.** A 27B VLM DP3 (prime-rl venv, GPUs 0,1,2, :8927) and an
   SFT-eval agent (:8930) are CURRENTLY RUNNING (live `pgrep` at briefing time). These conflict
   with the dictated layout. Kill the eval agent by `--port 8930`, kill the DP3 27B by its PID /
   tmux window (`docvqarun:vlm`). **NEVER kill by `EngineCore` or `vllm serve` string** — it
   matches every vLLM incl. the one you want to keep. Confirm GPUs ~14 MiB before relaunch.
2. **Stale ray.** `ray stop --force`; then kill any leftover `raylet`/`gcs`/`main_ppo` PIDs.
   A reused stale cluster silently ignores your fixes (registry + findings gotcha).
3. **HF datasets cache** is root-owned globally → the launcher already exports
   `HF_DATASETS_CACHE=outputs/hf_datasets_cache`. Leave it.
4. **venv:** the launcher does `source .venv/bin/activate` itself. `.venv` is the single canonical
   env (vllm 0.17 / torch 2.10 / tf 5.9, has GDN kernels). Do NOT use `.venv-rl2` (promoted into
   `.venv`; may be gone).

### 1.1 Build the curriculum parquet over the teacher_gen_pool (one-time, no GPU)

teacher_gen_pool.json (405 prompts, DocVQA-2026 val, RL schema) exists but has **no `num_pages`
/ `agent_name`** and **no RL parquet yet**. Build it with the existing curriculum script — all 405
docs have `metadata.json` with `num_pages` (verified), so ordering works:

```bash
cd /home/baris/repos/docvqa-verl && source .venv/bin/activate
python docvqa/scripts/make_curriculum_parquet.py \
    --pool data/pool/teacher_gen_pool.json \
    --out  data/pool/teacher_gen_curriculum_rl.parquet
# expect: ~405 rows, num_pages min=1 max=30, sorted easy->hard, datasets interleaved.
```

Page mix of the pool: 250×1pg, 87×2pg, 23×3pg, ~45× 15–30pg (real multi-page tail).

> **Alternative if you'd rather not retrain on val prompts:** use the existing
> `data/pool/curriculum_rl.parquet` (11,464 prompts, 8-source train pool, already page-sorted).
> But then 10 epochs is impossible in any sane budget (51→1,433 steps/epoch). With the broad pool
> you'd run N steps < 1 epoch (e.g. MAX_STEPS=120 as the prior production run did). The user
> asked for ≥10 epochs on the teacher-gen prompts, so **§1.1 is the chosen path**; this is the
> fallback only if training on val prompts is judged a leakage concern. Flag this to the user.

### 1.2 Bring up the local 27B VLM DP2 on GPUs 2,3

Match the serve config used elsewhere in the repo (registry "2-2 layout" + RL-async-findings
serve flags), with prefix caching ON:

```bash
tmux new-session -d -s vllm27b -n serve
tmux send-keys -t vllm27b:serve '
CUDA_VISIBLE_DEVICES=2,3 /home/baris/repos/prime-rl/.venv/bin/vllm serve Qwen/Qwen3.5-27B \
  --port 8927 --data-parallel-size 2 \
  --dtype bfloat16 --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-batched-tokens 32768 \
  --limit-mm-per-prompt "{\"image\":1}" \
  --enable-prefix-caching 2>&1 | tee -a outputs/vllm27b_serve.log
' Enter
# wait until :8927 answers /v1/models before launching RL. Remote :8927 (144.122.52.7) is assumed up.
```

Notes: prefix caching makes same-page re-looks hit the cached ~16K-token image prefill (payload
order is system→image→query). `--enforce-eager` is NOT used on a serve-only box (it's only for
verl-managed engines). The pool is health-aware: if the remote is down at launch it gets benched
and re-probed every 60s, so it can join mid-run without a restart.

### 1.3 Launch GRPO

```bash
tmux new-session -d -s rl -n train
tmux send-keys -t rl:train '
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_PATH=Qwen/Qwen3.5-4B \
TRAIN_FILES=data/pool/teacher_gen_curriculum_rl.parquet \
VAL_FILES=data/pool/teacher_gen_curriculum_rl.parquet \
SHUFFLE=False \
N_ROLLOUT_GPU=1 N_TRAIN_GPU=1 \
VLM_BASE_URL="http://localhost:8927@2|http://144.122.52.7:8927@3" \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH=8 ROLLOUT_N=8 \
MAX_STEPS=520 TOTAL_EPOCHS=11 SAVE_FREQ=52 \
LR=1e-6 KL_COEF=0.001 \
EXP_NAME=docvqa-grpo-4b-base-teacherpool \
PROJECT_NAME=docvqa-verl-rl \
bash docvqa/train/run_grpo.sh 2>&1 | tee outputs/rl_train_teacherpool.log
' Enter
```

**Epoch arithmetic:** 405 rows ÷ TRAIN_BATCH_SIZE 8 = 51 steps/epoch. MAX_STEPS=520 → 10.2 epochs
(`total_training_steps` is the hard stop; TOTAL_EPOCHS=11 is the soft cap, kept above 10.2 so steps
bind). Rollouts/step = batch 8 × n 8 = 64 (matches default `max_num_seqs=64`). Per the prior 4B run
~3 min/step (27B-VLM-bound) → ~26 h for 520 steps; if too long, drop to MAX_STEPS=255 (≈5 ep) or
raise TRAIN_BATCH_SIZE=16 (26 steps/epoch → 260 steps for 10 ep, but 16×8=128 rollouts/step exceeds
`max_num_seqs=64` → also bump `MAX_NUM_SEQS=128`, and re-check VLM saturation).

> **`resume_mode=auto`** is set in the launcher — an OOM/crash resumes from the last
> `global_step_*` for free (cost of a crash = one save interval).

---

## 2. Load-bearing gotchas / silent-failure checklist

Each: symptom → the flag/action that prevents it. Items marked ✅ are ALREADY in `run_grpo.sh`.

| # | Failure | Symptom | Prevention |
|---|---------|---------|------------|
| 1 | **GDN weight-sync drop** | Rollouts = multilingual gibberish + stray special tokens, every reward 0, grad_norm 0 | ✅ `rollout.load_format=safetensors` (NOT `dummy`). Qwen3.5 is hybrid GDN; dummy-init skips ~6/7 linear_attn params/layer. This is THE #1 silent killer here. |
| 2 | **LoRA targets wrap vision tower** | `KeyError 'blocks.0.attn.qkv.base_layer.weight'` at weight-sync | ✅ `TARGET_MODULES` = LM-only `[q/k/v/o_proj,gate/up/down_proj]`, NOT `all-linear`. |
| 3 | **Trainer/generator precision mismatch** | Phantom clipping zeros ~18% of token grads; reward flat, `pg_clipfrac` high early, `mismatch_kl`≈0 | verl default = fp32 master + bf16 autocast forward + bf16 rollout = the recommended recipe (β≈0). DON'T override `model_dtype`/rollout dtype. **Verify** `pg_clipfrac` ~0 at step ~3 (a broken run clips ~13% there). Consider fp32 LM-head logits if asymptote disappoints (ScaleRL lever; not currently set). |
| 4 | **TITO / re-encoding decoded tokens** | Trainer reconstructs different IDs than sampler → corrupt IS ratio, silent | verl's agent-loop machinery is append-only TITO by construction; `multi_turn.enable=True` ✅. Qwen3-family templates can strip non-final `<think>` — but this scaffold runs **native thinking OFF** and only the fenced code is parsed, so the `<think>`-strip prefix-break does NOT apply. Leave `enable_thinking=False` ✅. |
| 5 | **old_log_probs reuse (bypass)** | Our agent loop doesn't emit per-token rollout logprobs → misaligned IS ratio | ✅ `rollout_correction.bypass_mode=False` → trainer recomputes old_log_probs via a forward pass (correct, +1 pass). |
| 6 | **Reward function not migrated** | `NotImplementedError`/`default_compute_score` on our data_source | Fixed in-tree: `one_step_off_policy/main_ppo.py` now calls `migrate_legacy_reward_impl`. ✅ `custom_reward_function.{path,name}` set. After step 1, confirm reward is real ANLS (varies), not all-0. |
| 7 | **Stale ray reuse** | Your config/code fixes "don't take" | `ray stop --force` + kill `raylet`/`gcs` PIDs between runs (§1.0). |
| 8 | **Killing by `EngineCore`/`vllm serve`** | Takes down the 27B VLM (and any other vLLM) | Kill verl by tmux session + `main_ppo`/`raylet`/`gcs` PID; kill served models by `--port`. |
| 9 | **Rollout-GPU OOM on long seq** | Crash at a long sequence (fp32 logits ~16 GB: 16K tok × 248K vocab) | ✅ `gpu_memory_utilization=0.25` (≈20 GB free). Do NOT "fix" with `PYTORCH_ALLOC_CONF=expandable_segments` — it breaks CUDA-IPC weight transfer (`pidfd_getfd: Operation not permitted` → hang). |
| 10 | **CUDA-graph capture (enforce_eager=False)** | dummy-LoRA IndexError (vllm#36372) or GDN conv-state cache assert | Needs the vllm GDN-LoRA capture patch applied to `.venv` (`patches/vllm-0.17-gdn-lora-cudagraph.patch.md`) + `max_num_seqs` ≥ batch×n. If unsure the patch is applied, set `ENFORCE_EAGER=True` (safe; rollouts are VLM-bound, not decode-bound — costs ~3.8× decode but that's hidden behind VLM wait). **Recommend ENFORCE_EAGER=True for this launch** unless you've confirmed the patch. |
| 11 | **cupy/numpy ABI** | "Checkpoint engine nccl not registered" (NCCL weight sync silently off) | Should be fine in `.venv` (cupy 14.x / numpy 2.4). If the nccl engine fails to register, that's the cause. |

---

## 3. Early-validation checklist (first 1–3 steps) — catch bugs cheap

The launcher sets `DOCVQA_TRAJ_DUMP=outputs/async_traj.jsonl` (cleared at start). Use it.

1. **Rollouts coherent (catches GDN sync break #1):**
   ```bash
   head -3 outputs/async_traj.jsonl | python3 -m json.tool | less
   ```
   Look for real perceive→reason→`SUBMIT` English text + `batch_look` calls. Multilingual
   gibberish / stray `<|box_end|>` = GDN sync broke → STOP, recheck `load_format=safetensors`.

2. **Reward has in-group VARIANCE (catches dead groups / no gradient):** in the step-1 console /
   wandb, `critic/score/mean` should be mid-range and `max−min` within a group > 0. All-0 →
   reward not computing or rollouts broken; all-1 → curriculum too easy at cold start (some
   spread expected since easy-first). Grep the log:
   ```bash
   grep -E "critic/score|reward" outputs/rl_train_teacherpool.log | head
   ```

3. **grad_norm > 0:** `grep "grad_norm" outputs/rl_train_teacherpool.log | head`. Expect
   ~0.04–0.1 (prior run). `≈0` = frozen (LR too low / precision mismatch / no signal).

4. **clip fraction sane:** `actor/pg_clipfrac` near 0 in first few steps. `>0.3` early =
   updates too large for staleness OR precision mismatch (#3). `≈0 forever` + flat reward = frozen.

5. **ppo_kl small/stable:** growing or `>0.05` = policy outrunning its (one-step-stale) rollouts.

6. **generation length / num_turns not exploding:** `response_length/mean`, and our
   `num_turns`/`vlm_calls` passthrough — a runaway up = length hacking / wall_cap risk.

7. **No NaN/Inf anywhere** in `actor/loss`, `pg_loss`. Any = immediate death (usually a few steps
   after an ignored grad spike).

8. **Both VLM endpoints active + waiting≈0:** check the local :8927 serve log shows running
   requests; the pool router (least-loaded) should split load local/remote. Deep VLM queue =
   over-saturation → rollout wall_caps (scored 0, censored measurement, not capability).

Write a one-line go/no-go after step 1–3 before walking away. No silent scale-up.

---

## 4. Metrics to watch over the run (and what each warns of)

| Metric | Healthy | Warns of |
|--------|---------|----------|
| `actor/grad_norm` | small, ~stable, gently ↓ | spike ↑ = diverging; ≈0 = frozen |
| `critic/score/mean` (train) | fluctuates by batch difficulty (prompt-confounded under shuffle=False) | **NOT a learning curve** — judge by held-out eval only |
| in-group reward spread | >0 | →0 = dead groups, no GRPO gradient |
| `actor/entropy` | gentle decline | fast crash→0 = mode collapse; spike = diverging |
| `actor/pg_clipfrac` | 0–0.2 | >0.3 = too-large updates / staleness; ≈0 forever = frozen |
| `actor/ppo_kl` | small, stable | growing / >0.05 = outrunning rollouts |
| `response_length` / `num_turns` mean | stable | runaway ↑ = length hacking / wall_cap |
| `actor/pg_loss`, `actor/loss` | smooth | spike / NaN / Inf = divergence/death |
| VLM queue depth (:8927 log) | running, waiting≈0 | deep queue = over-saturation → wall_cap=0 rewards |

**The real verdict is a held-out eval**, not train reward. Per `RL-OPD-design.md` testing
strategy: eval each saved ckpt via the `~/repos/docvqa` harness (binary ANLS@0.9) vs base-4B and
report **overall + submit-only ANLS + wall_cap rate** (Eval-methodology in CLAUDE.md: a capped
rollout = 0 = censored, not wrong; split capability × completion). Baseline-to-beat: base-4B full
val ≈ **22.34% ±3.44 (n=8)** (current impl).

---

## 5. Open decisions / risks — what is NOT yet wired (launcher must decide)

1. **Efficiency / length signal — NOT active by default.** `docvqa/reward.py` HAS a concave
   length penalty (`LENGTH_PENALTY_COEF * C_{k,q}(num_turns)`) and a format penalty, BUT both are
   **module-level constants defaulting to 0.0** (`reward.py:44,49`) and are **NOT env-overridable
   and NOT passed by `run_grpo.sh`.** To enable the "discourage unbounded sequential batch_look"
   signal you must **edit `reward.py`** (e.g. `LENGTH_PENALTY_COEF=0.05`, `LENGTH_PENALTY_Q=1.0`
   for a log-shaped, hard-problems-not-crushed penalty). The signal acts on `num_turns` (passed
   via extra_info ✅). **Decision needed:** ship base GRPO with COEF=0 first (cleaner baseline,
   recommended), then turn on the penalty in a second arm? Or enable from step 1? Honest take:
   start at 0.0 to get a clean GRPO signal, then A/B the penalty — turning it on at cold start
   risks confounding "did GRPO learn" with "did the penalty reshape behavior."

2. **Timeout filtering / penalty — partial.** Non-submission (wall_cap / iter_cap / no SUBMIT)
   already scores **0.0** in reward.py (so chronic timeouts are penalized as wrong). But there is
   **no per-question filter** that drops prompts which *consistently* time out, and **no
   DAPO-style zero-variance-group filter** (an all-capped group has 0 advantage variance = wasted
   step). `RL-OPD-design.md` lists DAPO dynamic sampling as **deferred/not built**. **Decision:**
   accept wasted all-capped groups for now (they're harmless—zero gradient—just slow), OR
   pre-filter the curriculum to drop known-hard multi-page docs (the 45× 15–30pg tail does 100+
   sequential `batch_look` → highest cap risk). Simplest lever: `--max-pages` on the curriculum
   builder (e.g. `--max-pages 3` keeps 360/405, drops the runaway tail). Flag to user.

3. **Training on val prompts (leakage).** The teacher_gen_pool IS DocVQA-2026 **val**. Training
   GRPO on it then evaluating on DocVQA-2026 val = train/test overlap. This is fine IF the eval
   set is disjoint (e.g. the 80Q `questions.json` minus these 405, or a held-out split), but
   **must be checked.** If you want a clean held-out number, eval on prompts NOT in the 405. The
   §1.1-alternative (broad train pool) avoids this but breaks the ≥10-epoch + same-prompts asks.
   **Surface this trade-off to the user.**

4. **CUDA graphs vs enforce_eager.** `run_grpo.sh` defaults `enforce_eager=False` (needs the
   vllm GDN-LoRA patch in `.venv`). If you can't confirm the patch is applied, set
   `ENFORCE_EAGER=True` (safe, VLM-bound anyway). **Recommend confirming the patch OR forcing
   eager for this launch.**

5. **GRPO variant.** Plain GRPO (`adv_estimator=grpo`, std-normalized). Dr.GRPO (mean-only,
   removes difficulty bias) = `algorithm.norm_adv_by_std_in_grpo=False`; GSPO/CISPO = a
   `policy_loss.loss_mode` change (CISPO was ScaleRL's winner; GSPO more precision-tolerant).
   Not needed for the first run — note as a second-arm lever.

6. **Batch size for RL gradient noise.** Best-practice says RL wants large batches (256–1024
   prompts); we run batch 8 (LoRA + tiny 405-prompt pool + VLM-bound throughput force this).
   Expect noisy per-step signal; this is a known tension (small pool + ≥10 epochs vs large-batch
   ideal). Judge by held-out eval, not per-step reward.
