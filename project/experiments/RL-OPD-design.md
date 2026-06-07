# RL / OPD training bring-up — design spec

_Date: 2026-06-06. Author: pairing session. Supersedes the RL groundwork notes in
`AUTONOMOUS-DRIVER-PLAN.md` (PHASE 5) — that doc described scaffold that was written but
never executed; this spec is the agreed plan to actually run it._

## Goal

Bring up reinforcement-learning and on-policy-distillation fine-tuning of the ≤9B DocVQA
CodeAct agent on top of the existing SFT warm-start, starting with **GRPO** and then
**OPD from the 27B teacher**. Push past the SFT ceiling (the autonomous driver concluded
SFT transfer is at/near ceiling for this rung, dragged by the ~37% wall_cap runaway).

Non-goals: changing the agent scaffold (lives in `~/repos/docvqa`), VLM fine-tuning (frozen
HTTP endpoint), eval-harness changes.

## Env (resolved)

`.venv-rl` built 2026-06-06 via `uv pip install -e '.[vllm]'`: **torch 2.9.0+cu128, vllm
0.12.0** (top of verl's [0.8.5,0.12.0] range), verl 0.8.0.dev, CUDA 12.8, `cuda.is_available()`
True. `import ray; from docvqa.agent_loop import DocVQAReplAgentLoop` works under `.venv-rl`.
The working `.venv` (torch 2.12/cu130, SFT/eval) is untouched. Install log:
`outputs/venv_rl_install.log`.

## Context: what already exists (and its single blocker)

The 2026-06-06 autonomous driver wrote ~90% of the GRPO scaffold; **none of it has run
once**. It is hard-blocked on environment:

- `docvqa/train/run_grpo.sh` — GRPO + LoRA, inits actor from v1 SFT (`checkpoints/
  docvqa-verl/seqkd-transfer-mp/merged_hf`), `adv_estimator=grpo`, agent rollout via the
  registered `docvqa_repl` loop, reward = `docvqa/reward.py:compute_score`. Written, untested.
- Reward — **two files disagree**: `docvqa/rl_reward.py` (continuous ANLS, no `extra_info`)
  vs `docvqa/reward.py` (binary + numeric `extra_info` plumbing). The recipes point at
  different ones.
- RL data: `data/docvqa-2026/val/train_rl.parquet` (56 Q, in-dist, no val leak) +
  `heldout_rl.parquet` (24 Q), each carrying `agent_name` + question/doc_dir/category/
  question_id + prompt/data_source/reward_model/extra_info.
- **Blocker (gates GRPO *and* OPD):** `.venv` has `torch 2.12.0+cu130` + verl but **no
  vllm**. verl pins `vllm>=0.8.5,<=0.12.0` (`setup.py:52`). The only vllm on the box is
  prime-rl's 0.17.0 (too new, no verl). Installing a compatible vllm into `.venv` would
  likely drag torch 2.12 → older and risk breaking the working SFT/eval pipeline.

## Key technical decisions

1. **GRPO first, then OPD.** verl's OPD trainer is literally GRPO + `distillation.*` flags
   on the same rollout/env/reward/precision substrate. Validating GRPO end-to-end first
   de-risks OPD for almost no wasted work, and honors the SFT-warmstart → RL sequencing.

2. **Fresh isolated `.venv-rl`.** Never touch the working `.venv` (torch 2.12, SFT/eval).
   All RL/OPD runs use `.venv-rl`; SFT/eval stay on `.venv`.

3. **Continuous ANLS reward, single module.** GRPO needs in-group reward *variance*;
   binary@0.9 is mostly-0 on hard DocVQA → dead groups / cold-start. Keep continuous ANLS
   (`get_anls`, 0..1) but merge in the numeric `extra_info` passthrough so we get
   reward-vs-turns/length diagnostics. Evaluation stays binary@0.9 — reward ≠ metric on
   purpose. Non-submission (wall_cap / iter_cap / no FINAL) → 0.0.

4. **Agent LM is text-only.** `batch_look` outputs are injected as text observations; the
   agent never sees images. Consequence for OPD: the teacher is the **27B as a text policy**
   over the agent's reasoning tokens → use the **text** OPD recipe
   (`examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh`), NOT the VL one. The
   27B teacher loads as an in-process vLLM logprob cluster on its own GPUs.

5. **Read the RL-training-practices guide (CLAUDE.md) in full before the first run.** This
   is the first clipped-surrogate + rollout run; precision matching, TITO, and the
   silent-bug checklist all bite here (SFT was immune).

6. **Format reward components (trajectory-scalar).** Reward = ANLS minus two optional,
   independently-tunable penalties, each a trajectory-level scalar (default 0.0):
   - **length:** `LENGTH_PENALTY_COEF * C_{k,q}(num_turns)` — a **concave-down, increasing**
     penalty (Cursor "Composer 2" report, 2026): `C_{k,q}(x) = ((1+kx)^(1-q) - 1) / (k(1-q))`,
     marginal cost `(1+kx)^(-q)` *decays* with length. Easy problems pushed to be short; hard
     problems needing many turns aren't crushed. `q=0`→linear, `q=1`→log, `q>1`→saturates to
     `1/(k(q-1))`. Attacks the wall_cap runaway. Acts on `num_turns` (chosen over a token/tool
     weighted combo to keep one tunable knob on the noisy 56-Q set). GRPO synergy: advantages
     are group-relative per question, so it differentiates within-question; concavity compresses
     differences among a hard question's uniformly-long rollouts → difficulty-adaptive twice
     over. (Cursor also drops GRPO's length-standardization term — deferred for now, one change
     at a time.)
   - **format:** `FORMAT_PENALTY_PER_VIOLATION * (multi_block_turns + empty_output_turns)`
     — a turn that emits **>1** ```` ```python ```` block, or whose code **printed nothing**
     ("forgot to print"). Counts are computed in the agent loop and passed via `extra_fields`.
     0-block turns are NOT counted here (already handled by the loop's parse_error path → no
     double penalty). Score clamped ≥ 0. (Per-turn *dense* format reward is a later upgrade;
     for now every component is one scalar at the trajectory level.)

7. **RL-only first-fence parsing.** The agent loop gets a `parse_first_fence` knob
   (`_DEFAULTS=False` → last-fence, preserving eval/collection). The GRPO recipe sets it
   `True`: RL rollouts select the **first** python fence per turn (more predictable than
   last). **Parity nuance:** eval.py shares this agent loop and stays last-fence, so RL
   (first) vs eval/deploy (last) diverge on multi-block turns — the format penalty makes
   such turns rare, but once in-flight eval/collection finishes, flip eval to first-fence
   too to restore train==deploy parsing. NB: the docvqa reference repo has *no* first/last
   policy at all — its CodeAct uses a dspy structured `code` field (no fence regex), so
   first-fence is a verl-side raw-text artifact, not a divergence from the reference.

## Architecture / components

```
.venv-rl (uv)  ──  verl main_ppo  ──┬─ actor (4B LoRA, FSDP, init = v1 SFT merged_hf)
   verl + vllm[0.8.5,0.12.0]        ├─ rollout (vllm) → DocVQAReplAgentLoop (docvqa_repl)
   + the torch it pins              │      └─ batch_look() ──HTTP──► 27B VLM :8927 (frozen)
                                    ├─ ref policy (KL)
                                    └─ reward: docvqa/reward.py:compute_score (cont. ANLS)
                                            └─ OPD phase: + distillation teacher (27B, in-proc vLLM logprobs)
```

Units and their boundaries:
- **`.venv-rl`** — isolated interpreter; the only thing that knows about vllm. Interface:
  `source .venv-rl/bin/activate`. Depends on: verl repo (`-e '.[vllm]'`), CUDA driver.
- **`docvqa/reward.py:compute_score(data_source, solution_str, ground_truth, extra_info)`**
  — pure function, verl reward contract. Returns scalar ∈[0,1] (minus optional length
  penalty) + numeric `extra_info` (num_turns, vlm_calls, wall_clock_s). No I/O, unit-testable.
- **`docvqa/train/run_grpo.sh`** — the GRPO entry. Parameterized so policy-loss mode,
  reward, init ckpt, group size, penalty, and GPU layout are env-overridable.
- **OPD recipe (phase 2)** — GRPO recipe + `distillation.*`; teacher context built in
  `verl/experimental/teacher_loop/teacher_manager.py` (the PI seam).

## Build sequence

**Phase 0 — Environment**
```
uv venv .venv-rl --python 3.12
uv pip install --python .venv-rl -e '.[vllm]'
```
Verify: `import vllm, verl, torch`; 1-GPU vllm smoke-load of Qwen3.5-4B. Record the resolved
torch/vllm/cuda versions in the experiments log.

**Phase 1 — Reward reconciliation**
Merge `reward.py`'s numeric `extra_info` passthrough into `rl_reward.py`; keep continuous
ANLS. Point both recipes at `rl_reward.py:compute_score`. Delete the redundant `reward.py`
(or make it a thin re-export). Unit-test: exact→1.0, close→partial, wrong→0, no-submit→0,
extra_info numeric-only.

**Phase 2 — GRPO dry-run (mandatory; never run yet)**
Read RL-practices guide first. On the free GPU, tiny config: 1–2 prompts, `n=4`, ~2 steps,
`val_before_train=True` (step-0 baseline). Validate in order: env imports → vllm rollout
wakes → agent hits 27B `batch_look` → reward computes (continuous, sane spread) → no
precision/clip blowup. Expect 2–3 config shake-out iterations. Capture metrics: clip-frac,
advantage mean/std, KL, generation length, prefix-break/token-|δ|. Write go/no-go verdict.

**Phase 3 — GRPO real run**
`train_rl.parquet` (56 Q), `n≥4`, length/step penalty ON (attacks the wall_cap runaway).
Dedicated tmux + heartbeat cron. Eval merged checkpoint via the `~/repos/docvqa` harness
(binary@0.9) vs SFT v1 + baseline. Keep best.

**Phase 4 — OPD from 27B teacher**
Text OPD recipe; 27B teacher as in-process vLLM logprob cluster. Reuse the GRPO substrate
(`distillation.enabled=True`, `distillation_loss.loss_mode=k1/k3`, `use_policy_gradient`
toggles OPD-only vs OPD+RL). Heavier GPU → schedule when GPUs free.

## Designed-for extensions (NOT built now; seams only)

- **RL variants — CISPO, GSPO.** verl's `actor.loss_mode` built-ins are
  `vanilla`/`clip-cov`/`kl-cov`/`gpg` (`verl/workers/config/actor.py:84`). GSPO (sequence-level
  IS ratio) and CISPO (clipped IS-weight) are **not** built-in named modes in our verl
  version → each needs a recipe-level policy-loss variant (port from upstream recipe or
  implement). The GRPO recipe is written to parameterize the policy loss so these slot in
  without a fork. Dr.GRPO/DAPO-style controls similarly.
- **Privileged-information (PI) teacher for OPD.** Stock OPD feeds the teacher the student's
  prompt+response verbatim (`verl/workers/config/distillation.py:161`); teacher input ids are
  assembled in `verl/experimental/teacher_loop/teacher_manager.py:125`. PI = inject extra
  context into the **teacher's** context only — reference solution (already available as
  `reward_model.ground_truth`), a solution summary, or a hint (latter two need generation,
  e.g. 27B-produced). This is the "pedagogical / privileged-teacher distillation" direction.
  Keep teacher-context construction as the override point; do not hardcode "teacher sees
  exactly student context" anywhere in our recipe.

## Resource / sequencing

- 27B perception VLM stays up on GPUs 0–2 (`:8927`) — rollout `batch_look` and (later) the
  OPD teacher's perception both need it.
- GRPO dry-run needs the free GPU (currently GPU 3), which **collides with the in-flight
  `seqkd-v3` SFT** (Arm B queued). Do not preempt SFT — wait for it to free GPU 3, or the
  user explicitly hands it over.
- OPD adds the 27B teacher into the training job (own GPUs) → schedule only when GPUs free.
- Long runs: dedicated tmux + heartbeat cron per the repo's async-work convention; stay
  responsive. Register running work in `.claude/CLAUDE.md`.

## Testing strategy

- Reward: unit tests (the 5 cases above), no GPU needed.
- Env: import + 1-GPU vllm smoke load.
- GRPO: the dry-run IS the integration test — explicit validation checklist + metric
  capture + written go/no-go before scaling. No silent scale-up.
- Each trained checkpoint: eval via `~/repos/docvqa` harness (binary@0.9) against SFT v1 +
  untrained baseline; report overall + submit-only + wall_cap rate.

## Open risks

- `.[vllm]` may resolve a torch/cuda that needs a specific wheel index; if it fails, fall
  back to pinning `vllm==0.12.0` explicitly and letting it choose torch.
- vLLM wake-up after actor update has crashed before (`multi_stage_wake_up=True` /
  `multi_stage_wake_up` knobs already in the smoke recipe) — watch on first real step.
- GRPO on 56 Q is small; in-group variance + the dev set's ±0.04–0.08 noise floor mean
  early signal may be noisy. May need to grow the RL prompt set (mmlb) before trusting deltas.
