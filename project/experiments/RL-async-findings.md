# RL findings — async GRPO on Qwen3.5-4B DocVQA CodeAct agent

_Date: 2026-06-12. Branch: `docvqa-rl` (isolated worktree; main `docvqa-verl` kept pristine for SFT).
Companion to `RL-OPD-design.md` (the plan) and `SFT-synthesis.md` (the SFT verdict)._

## TL;DR

- **First successful RL run of the project.** Async GRPO (verl `one_step_off_policy`, disaggregated) on
  **base Qwen3.5-4B**, 30 steps in ~3h, clean reward signal (continuous ANLS, variance 0.28–0.92, no
  collapse), `grad_norm` >0 throughout, 3 checkpoints. Final model merged to HF:
  `checkpoints/docvqa-verl-rl/docvqa-grpo-4b-async-curriculum/global_step_30/merged_hf/`.
- **wandb:** https://wandb.ai/bdsaglam/docvqa-verl-rl/runs/g9j8nxtr
- **The hard part was infra, not RL.** Getting verl to run agentic GRPO on Qwen3.5 (a hybrid-attention
  VL model) on the experimental async trainer took **6 distinct fixes** (below) — all reusable knowledge.
- **Held-out eval (RL-4B vs base-4B on docvqa-2026 val, 80Q n4) is running** — that's the real
  "did RL help" number; per-step training reward is prompt-confounded and is NOT a learning curve.

## Setup

- **Method:** GRPO via `verl.experimental.one_step_off_policy` — DISAGGREGATED async-RL (AReaL/async-RLHF
  style): dedicated rollout GPU(s) + train GPU(s), generation overlaps training (one step off), NCCL
  weight sync, no colocated sleep/wake. Chosen over `main_ppo` (sync+colocated) because agentic RL is
  rollout-bound and `main_ppo` hit a sleep/wake OOM; also the right base for OPD/Pedagogical-RL later.
- **GPU layout (4× H100/A100, single box):** GPU0 = agent-LM rollout, GPU1 = train, **GPUs 2,3 = LOCAL 27B
  VLM** (`vllm serve` DP2, :8927). batch_look → localhost (no network latency — the actual throughput gate).
- **Policy:** base Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`, hybrid GDN/linear-attn), LoRA r32
  (LM-only target modules — NOT the VL vision tower).
- **Env:** isolated `.venv-rl2` — torch 2.10 / vLLM 0.17 / transformers 5.9 / cupy 14.1 (numpy 2.4). verl's
  packaged `vllm<=0.12.0` predates Qwen3.5; we install around the pin.
- **Reward:** continuous ANLS (`docvqa/reward.py`), data-source-agnostic → works across the whole pool.
- **Data — CURRICULUM:** `data/pool/curriculum_rl.parquet` (11,464 prompts from the 8-source DocVQA-family
  pool: chartqa/docvqa-sp/infographicvqa/mapqa/mp-docvqa/slidevqa/tatdqa/mmlb), **sorted easy→hard by
  num_pages** (datasets interleaved within each page-level), trained with `data.shuffle=False`. Easy-first
  also mitigates GRPO cold-start (more reward variance early). 30 steps × batch4 × n8 = 32 rollouts/step →
  trained on ~120 diverse 1-page prompts.
- **Stability config (load-bearing):** `load_format=safetensors` (GDN fix), `bypass_mode=False`
  (recompute old_log_probs), `gpu_memory_utilization=0.25` (logits headroom), `resume_mode=auto`.

## Result

- **Trained 4B (merged HF):** `.../global_step_30/merged_hf/` (model.safetensors 8.5G + lora_adapter/).
  Intermediate ckpts at steps 10, 20. ~3.05h wall (10,974s), ~3 min/step (27B-VLM-bound generation).
- **Reward (critic/score/mean per step — PROMPT-CONFOUNDED, not a learning curve):** 0.75 0.73 0.65 0.79
  0.74 0.82 0.77 0.49 0.51 0.33 0.71 0.39 0.66 0.63 0.92 0.68 0.42 0.64 0.28 0.66 … fluctuates 0.28–0.92
  by batch difficulty (each step = different prompts under shuffle=False), healthy variance, no collapse.
- **grad_norm:** 0.039–0.099 throughout (>0 = learning), gently decreasing (convergence), 0 NaN/inf.
- **Eval (running):** RL-4B vs base-4B on **DocVQA-2026 val (80Q, n4)**, faithful CodeAct harness
  (`eval.py`, same agent loop), shared 27B VLM. Runs `outputs/runs/{rl-4b-curriculum-val-n4,base-4b-val-n4}`.
  Held-out generalization (trained on pool, eval on docvqa-2026). Compare vs ≤8B leaderboard 0.1875 and the
  SFT baseline ~0.19 (docvqa_mini). **[ANLS numbers appended on completion.]**

## The 6 verl/Qwen3.5/async gaps fixed (reusable engineering findings)

1. **GDN garbage rollouts → `rollout.load_format=safetensors`.** Qwen3.5 is hybrid (GDN/linear-attn).
   verl's default `load_format=dummy` inits vLLM random + pushes base weights, but the FSDP→vLLM transfer
   ships the GDN `linear_attn` projections split/unwrapped while vLLM wants them fused + LoRA-`base_layer`-
   wrapped → ~6/7 GDN params/layer silently skip → random → **pure-gibberish rollouts, zero reward**.
   `safetensors` loads the real checkpoint at init (`engine_workers.py:614` sets `base_sync_done=True`),
   so verl skips the broken base push and the GDN weights stay correct. (Diagnosed by a weight-sync
   name-dump; standalone vLLM loaded the same model coherently, isolating it to verl's send path.)
2. **vLLM 0.17 + transformers 5.x needed; verl pins ≤0.12.0.** Qwen3.5 didn't exist in vLLM 0.12. Built
   `.venv-rl2` around the pin. verl's *code* already branches for 0.13/0.14; 0.17 needed ~3 small patches
   (AgentLoopConfig `docvqa` field; bucketed_weight_transfer `rebuild_ipc` guard for torch-2.10
   expandable-segments IPC; LM-only LoRA targets to avoid wrapping the VL vision tower).
3. **`one_step_off_policy/main_ppo.py` dropped `migrate_legacy_reward_impl(config)`** (a verl bug — the
   colocated `main_ppo` and fully_async both call it). Without it `custom_reward_function` never migrates
   to where the reward_loop reads it → `default_compute_score` → `NotImplementedError` on our data_source.
   Fix: add the call in the async `main()`.
4. **`bypass_mode=True` needs rollout_log_probs our agent loop doesn't return → `bypass_mode=False`.**
   one_step_off's bypass reuses the rollout's per-token logprobs as old_log_probs; our custom multi-turn
   `DocVQAReplAgentLoop` doesn't emit them. Chose the decoupled mode (trainer recomputes old_log_probs via
   a forward pass) over hand-aligning per-token logprobs through the loop — avoids the silent rollout/
   trainer logprob-mismatch class. (Proper fix for OPD later: make the agent loop return response_logprobs
   aligned with response_ids.)
5. **OOM on the rollout GPU = fp32 log-prob logits (16K tok × 248K vocab × 4B ≈ 16 GB) → `gpu_mem=0.25`.**
   Crashed at step 14 on a long sequence at 0.35. NOTE: `PYTORCH_ALLOC_CONF=expandable_segments:True`
   (the error's own suggestion) **broke the CUDA-IPC weight transfer** (`pidfd_getfd: Operation not
   permitted` → hung) — do NOT use it with the IPC path. Lowering `gpu_memory_utilization` to 0.25 (≈20 GB
   free > worst-case 16 GB logits) fixed the OOM without it.
6. **Resume after crash → `trainer.resume_mode=auto`.** The step-14 OOM cost nothing because save_freq=10
   had checkpointed step 10; resumed from `global_step_10` and continued.

## Operational gotchas (cost us time)

- **Never kill processes by matching `EngineCore`** — it matches BOTH verl's rollout vLLM AND the local
  27B VLM serve; killing it takes down the VLM. Kill verl by tmux session + `main_ppo`/`raylet`/`gcs` PIDs;
  kill served models by their `--port`.
- **Stale ray clusters get reused by the next run** with stale workers (fixes don't take). Always
  `ray stop --force` + kill PIDs between runs.
- **cupy/numpy ABI:** the `nccl` checkpoint_engine imports cupy; a numpy-2-built cupy on numpy-1.26 fails
  silently (`numpy.core.multiarray failed to import`) → "Checkpoint engine nccl not registered".
- Async long-tail slowdowns are EXPECTED (rollout-bound); only treat as a hang after 2+ monitor cycles
  with zero step progress.

## Lessons / what's next

- **Async-RL is the right architecture for this rollout-bound agentic task** and now works end-to-end on
  our verl+Qwen3.5 scaffold. The local VLM (vs remote tunnel) is the dominant throughput lever.
- **Per-step training reward ≠ improvement** under a shuffle=False curriculum (prompt-confounded). Always
  judge by a held-out eval (in progress).
- **Next levers:** (a) the held-out eval verdict gates whether base-4B RL helps; (b) SFT-init RL arm
  (warm-start from `seqkd-mmlb-long`); (c) OPD / Pedagogical-RL on this same async stack (needs the proper
  rollout_log_probs return from the agent loop, fix #4 above); (d) DAPO dynamic-sampling (skip all-0/all-1
  reward groups) — deferred, low-prio; (e) image downscaling for VLM throughput.

_Full driver timeline + every cron-fire state: `outputs/ASYNC-RL-DRIVER-PLAN.md`. Result card:
`outputs/RESULT-4b-async-curriculum.md`._

---

# Addendum 2026-06-12/13 — the speed sprint: profile-driven bottleneck migration

_Context: 120-step production run (`docvqa-grpo-4b-curric-lr1e4-cg`, lr 1e-4, batch8 n8,
thinking OFF per A/B result) had to fit a ~12h window incl. eval. We assumed VLM-bound;
per-phase profiling falsified that and the bottleneck then MIGRATED twice as each fix
landed. Method: agent_loop.py now has permanent per-phase timers (t_gen/t_exec/t_vlm/
t_template + gen_tokens) flowing into the traj dump — profile any config with a 2-step
shuffled run (batch8 n4) and read the split._

## Bottleneck migration (measured, 64-rollout profiles)

| stage | config | split | per-stream tok/s | rollout p50/p90/max |
|---|---|---|---|---|
| assumed | — | "VLM-bound" | — | — |
| measured | eager policy engine | **gen 80%** / vlm 19% / exec 0.4% | 15.8 | 91/276/509s |
| graphs on | + CUDA graphs (2 fixes below) | gen 60% / **vlm 39%** | **59.9 (3.8x)** | 54/117/212s |
| production | 3 rollout + 1 train GPUs | **update_actor ~420s = 70% of step** | gen idle (5s/step) | step ~580s |
| rebalanced | 2 rollout + 2 train | (expected ~310s/step) | — | — |

Moral: in disaggregated one-step-off, generation hides behind training — so the binding
constraint is whichever side is SLOWER, and every speedup flips it. Profile, don't assume.

## CUDA graphs on Qwen3.5-GDN + LoRA: the two capture blockers

`enforce_eager=True` (verl default) was costing 3.8x on decode — GDN hybrid layers are
exactly the many-small-kernels workload CUDA graphs amortize. `enforce_eager=False` needs:

1. **vllm#36372 (dummy-LoRA capture IndexError):** warmup builds dummy LoRAs for ALL
   linears incl. fused GDN `in_proj_qkvz` whose packed mapping declares 2 subloras vs 4
   output slices → `set_lora` IndexError. Fix: blocklist GDN modules
   (`in_proj_qkvz/in_proj_ba/conv1d/out_proj`) from `supported_lora_modules`
   (patch in `patches/vllm-0.17-gdn-lora-cudagraph.patch.md`, applied to .venv-rl2).
   Sound because our adapter is LM-standard-modules only → GDN never carries LoRA;
   capture state == runtime state. Validated: coherent rollouts, reward band unchanged.
2. **GDN conv-state cache assert (`num_cache_lines >= batch`):** at gpu_mem_util=0.25 the
   GDN state cache has fewer slots than the default max capture batch (512). Fix:
   `rollout.max_num_seqs=64` (= actual concurrency batch*n; capture sizes track it).

## VLM-side levers (landed before profiling showed they weren't the gate — still real)

- **Client-side downscale to 16,777,216 px** (= the 27B processor's `size.longest_edge`
  cap, i.e. max AREA): lossless w.r.t. what the VLM sees (server resizes anyway, but
  only after paying b64+HTTP+decode at full res). Sits AFTER agent cropping (sandbox
  crops full-res pages from disk) → survey-coarse/crop-fine unaffected. PNG kept.
  Only ~1% of training-pool pages exceed the cap (mostly protects eval maps/posters).
  Also moved image prep off the event loop (was blocking all rollouts in the worker).
- **Weighted least-loaded endpoint pool** in tools.py (`url@w|url@w` spec, health-bench
  +60s-retry, transport failover): worked (remote took ~42% with tunnel latency —
  latency-aware self-correction, by design less than the 3/5 GPU share). Retired to
  single-endpoint remote-only when 3 local GPUs went to rollout engines; pool code now
  runs as pool-of-1 (retry/bench still cushions tunnel blips).
- **Serve flags for the perception workload** (1 image ~16K tok prefill, median 45-tok
  output, thinking off): `--max-model-len 32768 --max-num-batched-tokens 32768
  --limit-mm-per-prompt '{"image":1}' --gpu-memory-utilization 0.90 --async-scheduling`,
  and NO `--enforce-eager` on a serve-only box (that flag is for verl-managed engines).
  vLLM v1 prefix caching + our payload order (system -> image -> query) makes same-page
  re-looks hit the 16K-token image prefill in cache.

## Train-side numbers to beat (2026-06-13, 1 train GPU, micro=1)

update_actor ~420s + old_log_prob ~71s + ref ~60s + sync ~32s ≈ 580s serial / step
(64 trajs, ~213K tokens global, ppo_micro_batch_size_per_gpu=1 = 64 sequential fwd+bwd).
Unexplored safe-ish levers if needed: actor micro-batch 2 (logits-memory risk on long-seq
batches — fp32/bf16 [B,T,vocab] materialization), use_dynamic_bsz (compat with
use_remove_padding=False unverified), fused-CE kernels. We took FSDP-x2 instead (zero risk).
