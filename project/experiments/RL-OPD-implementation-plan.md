# RL / OPD Training Bring-up — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring up GRPO RL training for the DocVQA CodeAct agent end-to-end (env → reward → validated dry-run → real run), then layer on-policy distillation (OPD) from the 27B teacher.

**Architecture:** verl `main_ppo` with `adv_estimator=grpo`; the 4B LoRA actor rolls out multi-turn agent trajectories through the existing `DocVQAReplAgentLoop` (registered `docvqa_repl`), whose `batch_look` hits the frozen 27B VLM over HTTP; reward is continuous ANLS. Runs in an isolated `.venv-rl` so the working `.venv` (SFT/eval) is untouched. OPD = the same substrate + a `distillation.*` teacher (27B as in-process vLLM logprob cluster).

**Tech Stack:** verl 0.8.0.dev, vLLM (∈[0.8.5,0.12.0]), PyTorch (vllm-pinned), uv, Ray, FSDP, LoRA, pytest.

**Spec:** `project/experiments/RL-OPD-design.md`

---

## File Structure

- `.venv-rl/` — **Create.** Isolated uv venv: verl + vllm + vllm-pinned torch. The only interpreter that knows about vllm. RL/OPD run here; SFT/eval stay on `.venv`.
- `docvqa/reward.py` — **Modify.** The single canonical reward module. Switch from binary `evaluate_prediction` to continuous `get_anls`; keep the clean `submitted_answer` extraction, dict-return, and numeric `extra_info` passthrough; add a length-penalty knob.
- `docvqa/rl_reward.py` — **Delete.** Redundant once `reward.py` is continuous. (It regex-parses `solution_str`; `reward.py`'s `extra_info["submitted_answer"]` path is cleaner.)
- `docvqa/tests/test_reward.py` — **Create.** Unit tests for `compute_score`.
- `recipe/docvqa/run_grpo.sh` — **Modify.** Point reward at `docvqa/reward.py`; fix the venv note in the header.
- `recipe/docvqa/run_opd.sh` — **Create (Phase 4).** GRPO recipe + `distillation.*` teacher flags (text OPD).
- `.claude/CLAUDE.md` — **Modify.** Register running RL work (session registry convention).

---

## Task 1: Isolated RL environment (`.venv-rl`)

**Files:**
- Create: `.venv-rl/` (uv venv)

- [ ] **Step 1: Create the venv**

```bash
cd /home/baris/repos/docvqa-verl
uv venv .venv-rl --python 3.12
```

- [ ] **Step 2: Install verl + vllm extra (resolves a self-consistent vllm+torch+tensordict)**

```bash
uv pip install --python .venv-rl -e '.[vllm]'
```

Expected: resolves and installs vllm in [0.8.5, 0.12.0] plus the torch it pins (expect ~2.8/cu124, independent of `.venv`'s 2.12/cu130). If it fails on torch/cuda resolution, fall back to an explicit pin and retry: `uv pip install --python .venv-rl -e . && uv pip install --python .venv-rl 'vllm==0.12.0'`.

- [ ] **Step 3: Verify imports and record versions**

```bash
.venv-rl/bin/python -c "import torch, verl, vllm; print('torch', torch.__version__); print('verl', verl.__version__); print('vllm', vllm.__version__); print('cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
```

Expected: all three import; `vllm` ∈ [0.8.5, 0.12.0]; `torch.cuda.is_available()` True. **Record the printed versions in `project/experiments/RL-OPD-design.md` under a new `## Env (resolved)` line** (the spec said to record them).

- [ ] **Step 4: Verify the agent_loop imports under `.venv-rl` (ray + verl + our code)**

```bash
.venv-rl/bin/python -c "import ray; from docvqa.agent_loop import DocVQAReplAgentLoop; print('agent_loop OK', DocVQAReplAgentLoop.__name__)"
```

Expected: prints `agent_loop OK DocVQAReplAgentLoop` (no `ModuleNotFoundError`). This is the import that crashed under a bare `python` before — confirms `.venv-rl` has ray+verl+our package.

- [ ] **Step 5: Commit (env is not a file, so commit the recorded versions only)**

```bash
git add project/experiments/RL-OPD-design.md
git commit -m "docvqa(rl): record resolved .venv-rl torch/vllm versions"
```

---

## Task 2: Reconcile reward into one continuous-ANLS module

**Files:**
- Modify: `docvqa/reward.py`
- Test: `docvqa/tests/test_reward.py`

- [ ] **Step 1: Write the failing tests**

Create `docvqa/tests/test_reward.py`:

```python
"""Unit tests for the continuous-ANLS GRPO reward (docvqa/reward.py)."""
import pytest

from docvqa import reward as R


def _score(submitted, gt, **extra):
    extra_info = {"submitted_answer": submitted, **extra}
    return R.compute_score(data_source="docvqa", solution_str="", ground_truth=gt, extra_info=extra_info)


def test_exact_match_scores_one():
    out = _score("2048.88", "2048.88")
    assert out["score"] == 1.0
    assert out["anls"] == 1.0


def test_close_answer_gets_partial_credit():
    # "2038.94" vs "2048.88": Levenshtein distance 3 over len 7 -> anls = 1 - 3/7 = 0.5714
    out = _score("2038.94", "2048.88")
    assert out["anls"] == pytest.approx(1 - 3 / 7, abs=1e-6)
    assert 0.0 < out["score"] < 1.0


def test_wrong_answer_scores_zero():
    out = _score("completely different", "2048.88")
    assert out["score"] == 0.0


def test_no_submission_scores_zero():
    out = _score(None, "2048.88")
    assert out["score"] == 0.0
    assert out["anls"] == 0.0


def test_empty_submission_scores_zero():
    out = _score("", "2048.88")
    assert out["score"] == 0.0


def test_extra_info_passthrough_is_numeric():
    out = _score("x", "y", num_turns=5, vlm_calls=3, wall_clock_s=12.5)
    assert out["num_turns"] == 5
    assert out["vlm_calls"] == 3
    assert out["wall_clock_s"] == 12.5
    # No non-numeric keys leak into the reward dict (verl aggregates every key with np.mean).
    for k, v in out.items():
        assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"


def test_length_penalty_reduces_score_but_not_anls(monkeypatch):
    monkeypatch.setattr(R, "LENGTH_PENALTY_PER_TURN", 0.1)
    out = _score("2048.88", "2048.88", num_turns=3)
    assert out["anls"] == 1.0           # raw anls unaffected
    assert out["score"] == pytest.approx(1.0 - 0.1 * 3)  # penalized reward


def test_length_penalty_floors_at_zero(monkeypatch):
    monkeypatch.setattr(R, "LENGTH_PENALTY_PER_TURN", 1.0)
    out = _score("2048.88", "2048.88", num_turns=10)
    assert out["score"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest docvqa/tests/test_reward.py -v
```

Expected: failures — current `reward.py` is binary (`test_close_answer_gets_partial_credit` fails, score is 0.0 not partial) and has no `LENGTH_PENALTY_PER_TURN` (the penalty tests error on `monkeypatch.setattr`).

- [ ] **Step 3: Rewrite `docvqa/reward.py` to continuous ANLS + length penalty**

Replace the entire contents of `docvqa/reward.py` with:

```python
# docvqa/reward.py
"""Continuous-ANLS reward for the DocVQA CodeAct agent. verl custom_reward_function hook.

GRPO needs in-group reward *variance*. The eval metric (binary ANLS @ 0.9) is mostly-0 on
hard DocVQA -> dead groups / cold-start. So the *reward* is continuous ANLS (get_anls, 0..1):
dense partial credit, still maximized by exact answers. Evaluation stays binary@0.9 in the
~/repos/docvqa harness -- reward != metric on purpose.

The agent loop populates extra_fields (docvqa/agent_loop.py:307) which verl merges into
extra_info: submitted_answer, num_turns, vlm_calls, wall_clock_s. We read submitted_answer
directly (no regex on solution_str). Non-submission (None / "") -> 0.0, which -- with the
optional per-turn length penalty -- discourages the ~37% wall_cap runaway the 4B exhibits.
"""
from __future__ import annotations

from typing import Any

from docvqa.metrics import get_anls

# Subtract this * num_turns from the score (0 disables). Off for the first dry-run (validate
# the bare ANLS signal first); turn on for the real run to attack the wall_cap runaway.
LENGTH_PENALTY_PER_TURN = 0.0

# Numeric keys propagated into reward_extra_info. ONLY numerics: verl's
# process_validation_metrics runs np.mean/std/max/min on every key (metric_utils.py),
# and protocol.py asserts every rollout emits the same key set -- so always emit all,
# with defaults. Non-numeric metadata travels via the agent-loop JSONL dump instead.
_NUMERIC_PASSTHROUGH: dict[str, float | int] = {
    "num_turns": 0,
    "vlm_calls": 0,
    "wall_clock_s": 0.0,
}


def compute_score(
    data_source: str = "",   # noqa: ARG001 -- verl signature
    solution_str: str = "",  # noqa: ARG001 -- answer comes from extra_info, not the text
    ground_truth: str = "",
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return {'score': reward, 'anls': raw_anls, **numeric passthrough}.

    verl unpacks result['score'] as the scalar reward and feeds every other key into
    reward_extra_info. score == penalized reward; anls == raw ANLS (diagnostic, unpenalized).
    """
    extra = extra_info or {}
    submitted = extra.get("submitted_answer")
    if submitted is None or submitted == "":
        raw_anls = 0.0
    else:
        raw_anls = float(get_anls(str(submitted), str(ground_truth)))

    score = raw_anls
    if submitted and LENGTH_PENALTY_PER_TURN:
        num_turns = float(extra.get("num_turns") or 0)
        score = max(0.0, raw_anls - LENGTH_PENALTY_PER_TURN * num_turns)

    out: dict[str, Any] = {k: extra.get(k, d) for k, d in _NUMERIC_PASSTHROUGH.items()}
    out["score"] = score
    out["anls"] = raw_anls
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest docvqa/tests/test_reward.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Delete the redundant module and point the recipe at `reward.py`**

```bash
git rm docvqa/rl_reward.py 2>/dev/null || rm -f docvqa/rl_reward.py
sed -i 's#custom_reward_function.path=docvqa/rl_reward.py#custom_reward_function.path=docvqa/reward.py#' recipe/docvqa/run_grpo.sh
grep -n "custom_reward_function.path" recipe/docvqa/run_grpo.sh
```

Expected: the grep shows `custom_reward_function.path=docvqa/reward.py`.

- [ ] **Step 6: Fix the venv note in the recipe header**

In `recipe/docvqa/run_grpo.sh`, change the header comment line `# Run with `.venv` active.` (~line 13) to `# Run with `.venv-rl` active (the only env with a verl-compatible vllm).`

```bash
sed -i 's/Run with `.venv` active/Run with `.venv-rl` active (the only env with a verl-compatible vllm)/' recipe/docvqa/run_grpo.sh
```

- [ ] **Step 7: Commit**

```bash
git add docvqa/reward.py docvqa/tests/test_reward.py recipe/docvqa/run_grpo.sh
git add -u docvqa/rl_reward.py
git commit -m "docvqa(rl): single continuous-ANLS reward module + length-penalty knob + tests"
```

---

## Task 3: GRPO dry-run (integration smoke — never run before; MANDATORY before scaling)

**Files:** none created; this is an execution + validation task. Read the RL-training-practices guide first (CLAUDE.md mandates it before the first clipped-surrogate + rollout run).

- [ ] **Step 1: Read the RL-training-practices guide**

Read the RL-training-practices section in `CLAUDE.md` in full (precision matching, TITO, which metrics to watch, the silent-bug checklist). Note: SeqKD was immune; this GRPO run is the first time the precision-mismatch and clip-fraction issues can bite.

- [ ] **Step 2: Confirm preconditions (GPU free + 27B VLM up)**

```bash
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
curl -s -m 3 http://localhost:8927/health -o /dev/null -w "27B health: %{http_code}\n"
tmux ls | grep -E "seqkd-v3" && echo "WARNING: seqkd-v3 SFT may still hold GPU 3 -- do NOT preempt; wait or get user's OK"
```

Expected: a GPU with ~free memory (GPU 3), `27B health: 200`. **If `seqkd-v3` is still running on GPU 3, STOP — do not preempt the user's SFT.** Wait for it to finish or get explicit handover.

- [ ] **Step 3: Launch the tiny dry-run in a dedicated tmux window**

```bash
tmux new-session -d -s grpo -n train
tmux send-keys -t grpo:train 'cd /home/baris/repos/docvqa-verl && source .venv-rl/bin/activate && CUDA_VISIBLE_DEVICES=3 NGPUS=1 bash recipe/docvqa/run_grpo.sh 2>&1 | tee outputs/train_grpo_dryrun.log' Enter
```

(Recipe defaults are already a tiny dry-run: `TRAIN_BATCH_SIZE=4`, `ROLLOUT_N=4`, `MAX_STEPS=2`, `NGPUS=1`, `save_freq=-1`, `test_freq=-1`.) Expect to iterate on 2–3 config errors; fix in the recipe and relaunch.

- [ ] **Step 4: Watch for the failure modes the guide names, in order**

```bash
# stream the log
tail -f outputs/train_grpo_dryrun.log
```

Validate, in order (each is a real failure point):
1. **Env / import** — no `ModuleNotFoundError`, vllm initializes.
2. **vLLM rollout wakes** — actor weights load into vllm; watch for the wake-up crash (`multi_stage_wake_up` is the known mitigation; the smoke recipe sets it — add to `run_grpo.sh` if the dry-run hits it).
3. **Agent hits the 27B** — the 27B server log / a rollout with `num_turns > 0` and `vlm_calls > 0` (the agent actually called `batch_look`).
4. **Reward computes** — log shows non-degenerate reward; some in-group variance (not all-0, not all-1).
5. **No precision/clip blowup** — `clip_frac`, `kl`, `advantage` mean/std, generation length finite and sane.

- [ ] **Step 5: Capture the go/no-go metrics**

```bash
grep -iE "clip|kl|advantage|reward|anls|response_length|num_turns" outputs/train_grpo_dryrun.log | tail -40
```

Record clip-fraction, advantage mean/std, KL, generation length, reward/anls spread.

- [ ] **Step 6: Write the go/no-go verdict into the design doc**

Append a `## GRPO dry-run verdict (YYYY-MM-DD)` section to `project/experiments/RL-OPD-design.md`: what ran, the captured metrics, any config fixes made, and an explicit **GO** (scale up) or **NO-GO** (what's broken). Do NOT scale up on a NO-GO.

- [ ] **Step 7: Commit the verdict + any recipe fixes**

```bash
git add project/experiments/RL-OPD-design.md recipe/docvqa/run_grpo.sh
git commit -m "docvqa(rl): GRPO dry-run verdict + config fixes"
```

---

## Task 4: GRPO real run + eval

**Files:**
- Modify: `.claude/CLAUDE.md` (register the running job)

Gated on Task 3 = GO.

- [ ] **Step 1: Turn the length penalty on (attacks the wall_cap runaway)**

In `docvqa/reward.py`, set `LENGTH_PENALTY_PER_TURN = 0.02` (start small). Re-run the reward tests to confirm nothing breaks:

```bash
.venv/bin/python -m pytest docvqa/tests/test_reward.py -v
```

Expected: PASS (the penalty tests monkeypatch their own value; the default-change tests still hold).

- [ ] **Step 2: Launch the scaled run in tmux with a heartbeat cron**

```bash
tmux new-session -d -s grpo-real -n train
tmux send-keys -t grpo-real:train 'cd /home/baris/repos/docvqa-verl && source .venv-rl/bin/activate && CUDA_VISIBLE_DEVICES=3 NGPUS=1 TRAIN_BATCH_SIZE=32 PPO_MINI_BATCH=8 ROLLOUT_N=8 MAX_STEPS=100 EXPERIMENT_NAME=docvqa-grpo-v1 bash recipe/docvqa/run_grpo.sh trainer.save_freq=20 trainer.test_freq=20 trainer.rollout_data_dir=outputs/docvqa-grpo-v1/rollouts 2>&1 | tee outputs/train_grpo_v1.log' Enter
```

(Batch/group/steps are starting points; tune to GPU memory. `save_freq`/`test_freq` re-enabled for the real run.)

- [ ] **Step 3: Register the job in the session registry**

Add a row to the "Running work" table in `.claude/CLAUDE.md`: what (`docvqa-grpo-v1`), how (tmux `grpo-real:train`, `.venv-rl`), GPU (3), status (RUNNING), log path. Per the repo's async-work convention, also schedule a heartbeat cron to check liveness/health every ~15 min (use `tmux capture-pane`/log growth/GPU mem for liveness, **not** `pgrep -f` self-matching tokens).

- [ ] **Step 4: Merge the trained LoRA and eval via the docvqa harness**

When a checkpoint lands (`checkpoints/docvqa-verl/docvqa-grpo-v1/global_step_*`):

```bash
source .venv/bin/activate
python -m verl.model_merger merge --backend fsdp \
  --local_dir  checkpoints/docvqa-verl/docvqa-grpo-v1/global_step_<N> \
  --target_dir checkpoints/docvqa-verl/docvqa-grpo-v1/merged_hf
BASE=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
cp -L "$BASE"/preprocessor_config.json "$BASE"/video_preprocessor_config.json \
   checkpoints/docvqa-verl/docvqa-grpo-v1/merged_hf/
```

Then serve the merged 4B on a free GPU and run the eval harness (binary@0.9) per the PHASE 3 commands in `AUTONOMOUS-DRIVER-PLAN.md`, comparing GRPO-v1 vs SFT-v1 (`seqkd-transfer-mp/merged_hf`) vs untrained baseline. Report overall + submit-only + wall_cap rate.

- [ ] **Step 5: Commit eval results**

```bash
git add project/experiments/ .claude/CLAUDE.md
git commit -m "docvqa(rl): GRPO v1 run config + eval vs SFT/baseline"
```

---

## Task 5: OPD from the 27B teacher (phase 2 — gated on GRPO GO + GPUs free)

**Files:**
- Create: `recipe/docvqa/run_opd.sh`

The OPD trainer is GRPO + `distillation.*`. Our agent LM is **text-only** (batch_look outputs are text observations; the agent never sees images), so the teacher is the 27B as a **text** policy over the agent's reasoning tokens → base on the **text** example `examples/on_policy_distillation_trainer/run_qwen3_8b_fsdp.sh`, NOT the VL one.

- [ ] **Step 1: Create `recipe/docvqa/run_opd.sh` from `run_grpo.sh` + distillation flags**

Copy `recipe/docvqa/run_grpo.sh` to `recipe/docvqa/run_opd.sh` and add the distillation block (teacher = 27B served as an in-process vLLM logprob cluster on its own GPUs):

```bash
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE:-2} \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL:-Qwen/Qwen3.5-27B}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP:-2} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM:-0.4} \
    distillation.distillation_loss.loss_mode=${DISTILL_LOSS_MODE:-k3} \
    distillation.distillation_loss.topk=${DISTILL_TOPK:-64} \
    distillation.distillation_loss.use_task_rewards=${USE_TASK_REWARDS:-False} \
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT:-False} \
```

(`use_policy_gradient=False` = pure OPD; `True` = OPD + RL advantages summed — the "combinations" method. `use_task_rewards` wires our ANLS reward into the distill objective.) Keep the same `docvqa_repl` rollout + `+...agent.docvqa.vlm_*` perception flags from `run_grpo.sh`. Verify bash syntax: `bash -n recipe/docvqa/run_opd.sh`.

- [ ] **Step 2: Plan the GPU layout (teacher + student + rollout + perception VLM)**

This needs: 27B teacher (≈2 GPUs in-job) + 4B student FSDP + 4B vllm rollout + the frozen 27B perception VLM (separate, :8927). On a 4×80GB box this is tight — document the layout (e.g. teacher on 2 GPUs, student colocated on 1–2, perception VLM stays external) before launching. Schedule only when GPUs are free; pause collection.

- [ ] **Step 3: Tiny OPD dry-run, same validation discipline as Task 3**

```bash
tmux new-session -d -s opd -n train
tmux send-keys -t opd:train 'cd /home/baris/repos/docvqa-verl && source .venv-rl/bin/activate && bash recipe/docvqa/run_opd.sh 2>&1 | tee outputs/train_opd_dryrun.log' Enter
```

Validate: teacher vLLM cluster loads + produces logprobs over the student's rollout tokens; distill loss is finite (the recipe sets `loss_max_clamp`/`log_prob_min_clamp` in the example — port them if missing); no precision blowup. Write a go/no-go verdict into the design doc.

- [ ] **Step 4: Commit the OPD recipe + verdict**

```bash
git add recipe/docvqa/run_opd.sh project/experiments/RL-OPD-design.md
git commit -m "docvqa(opd): on-policy distillation recipe (27B teacher) + dry-run verdict"
```

---

## Future extensions (seams only — NOT in this plan's scope)

Documented so the recipes above don't foreclose them; build under their own spec/plan later.

- **RL variants (CISPO, GSPO, Dr.GRPO).** verl's built-in `actor.loss_mode` is
  `vanilla`/`clip-cov`/`kl-cov`/`gpg` (`verl/workers/config/actor.py:84`) — GSPO (sequence-level
  IS ratio) and CISPO (clipped IS-weight) are **not** built-in named modes here, so each is a
  recipe-level policy-loss variant (port from an upstream recipe or implement). `run_grpo.sh`
  keeps the policy loss parameterizable so they slot in without forking the recipe.
- **Privileged-information (PI) teacher for OPD.** Stock OPD feeds the teacher the student's
  prompt+response verbatim (`verl/workers/config/distillation.py:161`); teacher input ids are
  assembled in `verl/experimental/teacher_loop/teacher_manager.py:125`. PI = inject extra
  context into the **teacher's** context only — reference solution (already on the row as
  `reward_model.ground_truth`), a solution summary, or a hint (latter two need generation).
  The seam is teacher-context construction; do not hardcode "teacher sees exactly student
  context" in `run_opd.sh`.

---

## Self-Review

- **Spec coverage:** Env → Task 1. Reward reconciliation (continuous ANLS + extra_info) → Task 2. GRPO dry-run w/ RL-practices read + validation checklist + go/no-go → Task 3. GRPO real run + eval → Task 4. OPD text recipe + teacher-as-vLLM → Task 5. CISPO/GSPO + PI-teacher extensions → Future-extensions section (spec marked them not-built-now). Resource/sequencing (don't preempt SFT, 27B stays up) → Task 3 Step 2 + Task 5 Step 2. All spec sections mapped.
- **Placeholder scan:** No TBD/TODO; reward code, tests, recipe edits, and commands are complete. `<N>` in the merge command is a runtime checkpoint number, not a plan placeholder.
- **Type consistency:** `compute_score(data_source, solution_str, ground_truth, extra_info) -> dict` with keys `score`/`anls`/`num_turns`/`vlm_calls`/`wall_clock_s` is used identically in the tests and the recipe (`custom_reward_function.name=compute_score`). `LENGTH_PENALTY_PER_TURN` name matches between module and tests.
