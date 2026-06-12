# RL / OPD training — design spec (async substrate)

_Originally drafted 2026-06-06 around verl's sync `main_ppo`. **Rewritten 2026-06-12** after the
project switched to verl's **async `one_step_off_policy`** (disaggregated) trainer — the GRPO
rung now runs end-to-end on it. The reward design and the OPD / future-RL plan below carry over;
the env, architecture, and build sequence are updated to the async stack. GRPO bring-up is DONE;
its outcome + the 6 infra fixes that got it running live in `RL-async-findings.md` (read that for
"what happened"). This doc is "the design / forward plan", chiefly the path to **OPD**._

## Goal

Reinforcement-learning and on-policy-distillation fine-tuning of the ≤9B DocVQA CodeAct agent on
top of the SFT warm-start, **GRPO first, then OPD from the 27B teacher**. Push past the SFT
ceiling (the SFT investigation concluded trajectory-SFT only ties the untrained baseline — see
`SFT-synthesis.md`; on-policy signal is the lever).

Non-goals: changing the agent scaffold (lives in `~/repos/docvqa`), VLM fine-tuning (frozen HTTP
endpoint), eval-harness changes.

## Env (resolved — async stack)

**`.venv-rl2`** (isolated; the working `.venv` for SFT/eval is untouched): **torch 2.10 /
vLLM 0.17 / transformers 5.9 / cupy 14.1 (numpy 2.4)**. verl's packaged `vllm<=0.12.0` predates
Qwen3.5 (a `Qwen3_5ForConditionalGeneration` VL model that only exists in transformers 5.x +
vLLM ≥0.17), so we install around the pin; verl's code already branches for 0.13/0.14 and needed
~3 small patches (`RL-async-findings.md` fix #2). `cupy` is required by the `nccl`
checkpoint_engine (NCCL weight sync) and must be ABI-matched to numpy 2.x.

> The original 2026-06-06 `.venv-rl` (vllm 0.12) is **dead** — it cannot load Qwen3.5. All RL/OPD
> runs use `.venv-rl2`, run from the `docvqa-verl-rl` worktree (branch `docvqa-rl`).

## Key technical decisions

1. **GRPO first (done), then OPD.** verl's OPD trainer is GRPO + `distillation.*` flags on the
   same rollout/env/reward/precision substrate. GRPO is validated end-to-end on the async stack,
   so OPD is now mostly additive. Honors the SFT-warmstart → RL sequencing.

2. **Async disaggregated trainer (`one_step_off_policy`), not sync colocated `main_ppo`.** This
   agentic task is **rollout-bound** — the 27B VLM perception (`batch_look`) is the throughput
   gate. The async trainer puts generation on dedicated rollout GPU(s) and training on dedicated
   train GPU(s), overlapping the two (one step stale) with NCCL weight sync and **no** colocated
   sleep/wake (which OOM'd the colocated path). It is also the cleaner substrate for OPD /
   Pedagogical-RL. Canonical launcher: `docvqa/train/run_grpo.sh`.

3. **Fresh isolated `.venv-rl2`.** Never touch the working `.venv` (SFT/eval). See Env above.

4. **Continuous ANLS reward, single module (`docvqa/reward.py`).** GRPO needs in-group reward
   *variance*; binary@0.9 is mostly-0 on hard DocVQA → dead groups / cold-start. Reward is
   continuous ANLS (`get_anls`, 0..1) plus the numeric `extra_info` passthrough (num_turns,
   vlm_calls, wall_clock_s) for reward-vs-turns/length diagnostics. Data-source-agnostic so it
   works across the whole DocVQA-family pool. Evaluation stays binary@0.9 — reward ≠ metric on
   purpose. Non-submission (wall_cap / iter_cap / no SUBMIT) → 0.0. (The old duplicate
   `rl_reward.py` was reconciled into this one.)

5. **Agent LM is text-only.** `batch_look` outputs are injected as text observations; the agent
   never sees images. Consequence for OPD: the teacher is the **27B as a text policy** over the
   agent's reasoning tokens → use the **text** OPD recipe
   (`examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh`), NOT the VL one. The 27B
   teacher loads as an in-process vLLM logprob cluster on its own GPUs.

6. **Read the RL-training-practices guide (CLAUDE.md) in full before scaling.** GRPO is a
   clipped-surrogate + rollout run; precision matching, TITO, and the silent-bug checklist all
   bite here (SFT was immune). The async path recomputes `old_log_probs` on the trainer side (see
   decision 8), which sidesteps the rollout/trainer logprob-mismatch class for GRPO.

7. **Format reward components (trajectory-scalar).** Reward = ANLS minus two optional,
   independently-tunable penalties, each a trajectory-level scalar (default 0.0):
   - **length:** `LENGTH_PENALTY_COEF * C_{k,q}(num_turns)` — a **concave-down, increasing**
     penalty (Cursor "Composer 2" report, 2026): `C_{k,q}(x) = ((1+kx)^(1-q) - 1) / (k(1-q))`,
     marginal cost `(1+kx)^(-q)` *decays* with length. Easy problems pushed to be short; hard
     problems needing many turns aren't crushed. `q=0`→linear, `q=1`→log, `q>1`→saturates to
     `1/(k(q-1))`. Attacks the wall_cap / token_cap runaway. Acts on `num_turns`. GRPO synergy:
     advantages are group-relative per question, so it differentiates within-question; concavity
     compresses differences among a hard question's uniformly-long rollouts → difficulty-adaptive
     twice over.
   - **format:** `FORMAT_PENALTY_PER_VIOLATION * (multi_block_turns + empty_output_turns)` — a
     turn that emits **>1** ```` ```python ```` block, or whose code **printed nothing** ("forgot
     to print"). Counts are computed in the agent loop and passed via `extra_fields`. 0-block
     turns are NOT counted here (already handled by the loop's parse_error path → no double
     penalty). Score clamped ≥ 0.

8. **OPD prerequisite — the agent loop must return per-token `response_logprobs`.** The async
   trainer's `bypass_mode` reuses the rollout's per-token logprobs as `old_log_probs`; our custom
   `DocVQAReplAgentLoop` does **not** emit them, so GRPO runs with `bypass_mode=False` (trainer
   recomputes `old_log_probs` via a forward pass — correct, just an extra pass). **OPD needs the
   teacher/student token alignment, so before the OPD rung the agent loop must return
   `response_logprobs` aligned with `response_ids`** (TITO; never re-encode decoded tokens). This
   is the one concrete piece of plumbing OPD adds beyond flipping `distillation.*` flags.

9. **RL-only first-fence parsing.** The agent loop has a `parse_first_fence` knob
   (`_DEFAULTS=False` → last-fence for eval/collection); the GRPO recipe sets it `True` (RL
   rollouts select the **first** python fence per turn — more predictable). The format penalty
   makes multi-block turns rare; once in-flight eval/collection finishes, flipping eval to
   first-fence too restores exact train==deploy parsing.

## Architecture / components (async)

```
.venv-rl2  ──  verl one_step_off_policy (DISAGGREGATED)
  torch2.10/vllm0.17/tf5.9        ┌─ ROLLOUT GPU(s): vllm AsyncLLM (4B LoRA)
                                  │     └─ DocVQAReplAgentLoop (docvqa_repl)
                                  │            └─ batch_look() ──HTTP──► 27B VLM :8927 (frozen, separate GPUs)
   gen overlaps train  ◄──NCCL──► ├─ TRAIN GPU(s): FSDP2 actor (LoRA r32, LM-only targets)
   (one step stale)               │     └─ recompute old_log_probs (bypass_mode=False) + ref KL
                                  └─ reward: docvqa/reward.py:compute_score (continuous ANLS)
                                         └─ OPD phase: + distillation teacher (27B, in-proc vLLM logprobs)
```

Units and their boundaries:
- **`.venv-rl2`** — isolated interpreter; the only env that loads Qwen3.5 under vllm. Interface:
  `source .venv-rl2/bin/activate`, run from the `docvqa-verl-rl` worktree.
- **`docvqa/reward.py:compute_score(data_source, solution_str, ground_truth, extra_info)`** —
  pure function, verl reward contract. Returns scalar ∈[0,1] (minus optional penalties) + numeric
  `extra_info`. No I/O, unit-testable. Data-source-agnostic (whole pool).
- **`docvqa/train/run_grpo.sh`** — the async GRPO entry (`one_step_off_policy.main_ppo`).
  Parameterized so init ckpt, group size, penalties, curriculum order (`SHUFFLE`), and the
  rollout/train GPU split (`N_ROLLOUT_GPU`/`N_TRAIN_GPU`) are env-overridable. Carries the
  load-bearing knobs: `load_format=safetensors` (GDN fix), LM-only LoRA targets, low
  `gpu_memory_utilization` (logits headroom), `bypass_mode=False`, `resume_mode=auto`.
- **OPD recipe (phase 2)** — the same launcher + `distillation.*`; teacher context built in
  `verl/experimental/teacher_loop/teacher_manager.py` (the PI seam).

## Data — curriculum

`data/pool/curriculum_rl.parquet` (the 8-source DocVQA-family pool, sorted **easy→hard by
num_pages**, datasets interleaved within each page-level), built by
`docvqa/scripts/make_curriculum_parquet.py`, trained with `data.shuffle=False`. Easy-first also
mitigates GRPO cold-start (more in-group reward variance early). The pool reward is
data-source-agnostic so all sources score uniformly.

## Build sequence

**Phase 0–3 — Environment / reward / GRPO dry-run / GRPO real run: DONE.** See
`RL-async-findings.md` for the outcome (first successful RL run: async GRPO on base Qwen3.5-4B,
30 steps, clean reward signal) and the 6 verl/Qwen3.5/async fixes. Held-out eval (RL-4B vs base-4B
on DocVQA-2026 val) gates whether base-4B RL helps.

**Phase 4 — OPD from 27B teacher (NEXT).**
1. **Plumb `response_logprobs`** out of `DocVQAReplAgentLoop`, aligned with `response_ids` (TITO).
   This is the gating prerequisite (decision 8).
2. Text OPD recipe; 27B teacher as in-process vLLM logprob cluster on its own GPUs. Reuse the
   async GRPO substrate (`distillation.enabled=True`, `distillation_loss.loss_mode=k1/k3`,
   `use_policy_gradient` toggles OPD-only vs OPD+RL).
3. GPU layout is the tightest case (policy train + rollout + 27B teacher + frozen 27B perception
   VLM all live) — document the split before launching; schedule when GPUs free.

## Designed-for extensions (NOT built now; seams only)

- **RL variants — CISPO, GSPO, DAPO controls.** verl's `actor.loss_mode` built-ins are
  `vanilla`/`clip-cov`/`kl-cov`/`gpg` (`verl/workers/config/actor.py`). GSPO (sequence-level IS
  ratio) and CISPO (clipped IS-weight) are **not** built-in named modes in our verl version →
  each needs a recipe-level policy-loss variant (port from upstream or implement). The launcher
  parameterizes the policy loss so these slot in without a fork. DAPO dynamic sampling (skip
  all-0 / all-1 reward groups) is a deferred, low-prio control.
- **Privileged-information (PI) teacher for OPD.** Stock OPD feeds the teacher the student's
  prompt+response verbatim (`verl/workers/config/distillation.py`); teacher input ids are
  assembled in `verl/experimental/teacher_loop/teacher_manager.py`. PI = inject extra context
  into the **teacher's** context only — reference solution (already available as
  `reward_model.ground_truth`), a solution summary, or a hint (latter two need generation, e.g.
  27B-produced). This is the "pedagogical / privileged-teacher distillation" direction. Keep
  teacher-context construction as the override point; do not hardcode "teacher sees exactly
  student context" anywhere in our recipe.

## Resource / sequencing

- 27B perception VLM is served **separately** (e.g. GPUs 2,3 DP2 `:8927`) — rollout `batch_look`
  and (later) the OPD teacher's perception both need it. The policy trainer drives the remaining
  GPUs, disaggregated into rollout + train (`N_ROLLOUT_GPU` + `N_TRAIN_GPU`).
- OPD adds the 27B teacher *into* the training job (own GPUs) → schedule only when GPUs free.
- Long runs: dedicated tmux + heartbeat cron per the repo's async-work convention; register
  running work in `.claude/CLAUDE.md`.

## Testing strategy

- Reward: unit tests (exact→1.0, close→partial, wrong→0, no-submit→0, extra_info numeric-only),
  no GPU.
- GRPO/OPD: the dry-run IS the integration test — explicit validation checklist (env imports →
  vllm rollout → agent hits 27B `batch_look` → reward computes with sane spread → no precision/
  clip blowup) + metric capture (clip-frac, advantage mean/std, KL, generation length,
  prefix-break / token-|δ|) + written go/no-go before scaling. No silent scale-up.
- Each trained checkpoint: eval via the `~/repos/docvqa` harness (binary@0.9) against the SFT and
  untrained baselines; report overall + submit-only + wall_cap/token_cap rate.

## Open risks

- **OPD logprob alignment (decision 8):** if the agent loop's returned `response_logprobs` don't
  exactly align with `response_ids`, OPD silently distills against misaligned tokens. Verify
  against a trainer recompute before trusting OPD loss.
- vLLM weight sync after actor update is the historically fragile step on this hybrid model
  (GDN fix #1, the IPC/expandable-segments trap fix #5) — watch on the first OPD step too.
- Small/medium curriculum early signal is noisy (dev-set ±0.04–0.08 floor); judge by held-out
  eval, never by per-step training reward (prompt-confounded under `shuffle=False`).
