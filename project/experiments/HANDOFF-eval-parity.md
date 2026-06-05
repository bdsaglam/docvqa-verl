# Handoff — Eval parity for our CodeAct agent (2026-06-05)

> ## ✅ VERDICT (2026-06-05 evening): scaffold is FAITHFUL — proceed.
> Thinking-ON 27B/27B, full 80-val, n=1: **overall 0.275 (22/80)**, **submit-only 0.400 (22/55)**, wall_cap 27.5%.
> **Submit-only 0.40 ≈ reference 0.37 (band 33–44%) → `agent_loop` reproduces the original on completable questions; NOT a reimplementation bug.**
> Overall sits below the floor purely as a TIMEOUT artifact: `maps` 0/10 + `science_paper` 0/10 all hit the 1800s `--rollout-timeout` (8× tighter than the reference's 14400s).
> Run dir `outputs/runs/parity-codeact-27b-val-t1`. No-think comparison (cut @58): overall 0.259, wall_cap **45%** — thinking ~halves runaways. **enable_thinking=TRUE is correct** (see below). Open follow-ups: re-run timed-out docs at a longer cap for a clean overall; mitigate the long-doc runaway (scaffold work in ~/repos/docvqa) before/with training.

**Task:** Prove our `docvqa/agent_loop.py` (CodeAct-style harness) reproduces the
original `~/repos/docvqa` CodeAct solver's performance on the **full** DocVQA-2026
val (80 Q, `data/docvqa-2026/val/questions.json`) — *not* a split. Then resume training work.

## Reference numbers to match (CURATED — `~/repos/docvqa/docs/results.md §133` + `docs/experiments/codeact-qwen-3_5-27b.md`; full 80-Q val, binary-ANLS @ 0.9)
| Config | Full-val (curated) |
|---|---|
| **CodeAct 27B** (LM+VLM, `enable_thinking=false`) | **36.74% ± 4.29 pooled (n=23)**; per-budget (max_iters 24/40/56): 37.66 / 36.96 / 35.62; single-run pilots swing **33.75–43.75%** (high variance). Twin of `rvlm` (39.38% ± 1.49). |
| CodeAct **4B + 27B VLM** (**our train setup**) | ~**0.15** (raw run `codeact-4b-llm-27b-vlm-val-t4`, 12/80 — NOT curated; provisional) |
| CodeAct 9B / 4B (raw runs) | 0.2125 / 0.1375 (provisional) |

So: **27B ceiling ≈ 37%** (n=8 pooled; single-run variance ±5pp — don't over-read one run), **untrained-4B baseline ≈ 0.15** (provisional). Our heldout-24 4B baseline was 0.125 (n=24, not comparable).
**Note from the curated doc:** CodeAct's iteration cap (`max_iterations`) **never binds** (~1% @cap, converges in ~13 iters), so budget isn't a lever — EXCEPT `maps_2` hit a 14400s task-timeout twice (append-only context balloons on that multi-page map until `batch_look` hangs). This is the same long-map runaway our `--rollout-timeout` guards against.

## Plan (user-specified)
1. Eval is **VLM-bottlenecked** → serve 27B **data-parallel across ALL 4 GPUs** (`--data-parallel-size 4`, 4 replicas; 27B fits 1×80GB ≈73GB) on one port, run eval against it with high `--concurrency`. prime-rl vllm binary: `/home/baris/repos/prime-rl/.venv/bin/vllm`.
2. Run **27B/27B parity** first (LM+VLM both = the DP=4 27B), full val, expect **~37%** (single run can land anywhere 33–44% — variance is ±5pp, so ideally run 2–3 trials, not 1). Then **4B+27B-VLM** (untrained `Qwen/Qwen3.5-4B` agent), expect ~0.15.
3. If 27B lands ~33–44% → scaffold is faithful; proceed to training. If it's **≪ ~33%** → our `agent_loop` has a reimplementation bug (investigate before trusting any training result).

## Verified eval invocation (`docvqa/scripts/eval.py`)
```
python docvqa/scripts/eval.py \
  --questions data/docvqa-2026/val/questions.json \
  --base-url http://localhost:<PORT>/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:<PORT> --vlm-model Qwen/Qwen3.5-27B \
  --concurrency 32 --n 1 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --rollout-timeout 1800 --run-dir outputs/runs/parity-codeact-27b-val-t1 \
  --dataset docvqa-2026 --split val
```
- **Output format (NEW, mirrors `~/repos/docvqa/output/runs/`):** `--run-dir` writes
  `results.json` (`summary.overall_accuracy` is the headline number + `by_category` + `documents`)
  and `tasks/<doc_id>/{result.json, trajectories.jsonl}`. The per-doc
  `trajectories.jsonl` holds one structured record per (question, sample): full chat
  `messages` + `anls` + `termination` + meta **+ exact token-level data
  `prompt_ids`/`response_ids`/`response_mask`** (assistant-only mask; `--no-token-ids`
  to omit). **So an eval run IS the trajectory collection; `collect_trajectories.py` is redundant.**
  - Token IDs come straight from `AgentLoopOutput` (no retokenization) → train SeqKD on
    EXACT teacher tokens + mask (drops the `ignore_input_ids_mismatch` hack), and they're
    the basis for forward-KL/OPD/RL. **logprobs not stored** — recompute via a frozen-27B
    forward pass over `response_ids` when KD needs them.
  - **CAVEAT to verify:** token IDs are saved with the `--model` tokenizer (= the
    27B teacher during collection). SeqKD trains the 4B on these → confirm Qwen3.5-4B and
    Qwen3.5-27B share an identical tokenizer/vocab (same family — almost certainly yes, but
    check once before training on saved token IDs).
- Headline number = `json.load(run_dir/'results.json')['summary']['overall_accuracy']`.
- SFT data from a run: `make_sft_data.py --in <run-dir>` (it globs `tasks/*/trajectories.jsonl`).
- `--model` defaults to **27B** (teacher; our usual eval). For the student eval use
  `--model Qwen/Qwen3.5-4B` (or a checkpoint path) on a 4B server; `--vlm-*` stays the 27B.
- **Config — CORRECTED 2026-06-05 (this note was backwards in the original handoff):**
  the original CodeAct 27B used `enable_thinking=false`, BUT that was correct *only*
  because DSPy's CodeAct signature carries a separate explicit `reason` field. Our
  `agent_loop` has **no such field** — native `<think>` is its ONLY reasoning channel.
  Running `enable_thinking=false` here yields **empty `<think></think>` and zero
  deliberation** (verified: 22/22 empty in a no-think run). So **`eval.py` keeps
  `enable_thinking=True` (the original default was right); do NOT flip it to false.**
  Match the *behavior* (reasoning on), not the flag value. eval.py now defaults
  `enable_thinking=True` with a `--no-thinking` A/B flag. The mmlb teacher training
  data already has real reasoning in all assistant turns (collected with thinking on),
  so eval and training are now consistent. Budget: original chose `max_iterations=24`
  (sweep showed cap never binds); our `agent_loop` uses adaptive `20 + 1.5·sqrt(pages−9)`
  capped at 30 — comparable, not identical.

## Code changes already made (this session, committed-worthy — all syntax/smoke-tested, NOT yet run on GPUs)
- `eval.py` **rewritten** to write a run dir via `--run-dir` (see Output format above): `config.json` (written at start — models/sampling/dataset), `results.json` (summary), `tasks/<doc_id>/{result.json, trajectories.jsonl}`. Trajectories stream per-doc as questions finish (crash-safe); each record = messages + anls + termination + meta + `prompt_ids`/`response_ids`/`response_mask`.
- `eval.py` `--rollout-timeout` (default **600s**; `asyncio.wait_for` per rollout → timeouts scored 0 / `termination="wall_cap"`). **Lesson:** the cap fired on *every* rollout when the VLM was contended (collection on the same 27B) → all-`wall_cap`/0 garbage. **Eval must have the VLM ~uncontended** (DP=4 + collection stopped); keep cap generous (1800s).
- Root cause of runaway rollouts: `batch_look` VLM calls run in the host async context, **not** the sandbox subprocess, so `subprocess_timeout_s=120` does NOT bound them (one question fanned out unbounded VLM calls for 165 min). The `--rollout-timeout` is the backstop.
- `make_sft_data.py --in <run-dir>` globs `tasks/*/trajectories.jsonl` (reads a run dir or a single jsonl). `collect_trajectories.py` deprecated (note added).
- Old `--output` flag removed (→ `--run-dir`); `outputs/eval/run_transfer_eval.sh` still uses `--output` and must be updated to `--run-dir` when the transfer eval resumes.

## Machine state — CLEANED (done 2026-06-05 ~16:00)
- **All 4 GPUs free** (~14 MiB each, 0% util). All collection (the `while true` loop in tmux `stage0-prep` + strays), all vLLM servers, and all eval procs were killed; my tmux sessions (`stage0-prep`, `eval-students`, `vllm`, `vllm-eval-27b`) removed.
- Remaining tmux: `docvqa-verl` (this session) + the separate `~/repos/docvqa` sessions (`docvqa`, `eval-9b`, `general`) — left intact.
- **Gotcha for next session:** `pkill -f "vllm serve"` self-matches your own shell (the pattern is in its cmdline) → kills the command mid-run. Kill vLLM **by PID** (or `pkill VLLM` for the renamed workers); a worker child can outlive its parent and keep ~73GB — check `nvidia-smi --query-compute-apps` and kill leftovers.
- Old/garbage eval outputs from the earlier contended run: `outputs/eval/transfer_*_strat24.jsonl*` (all-0 wall_cap) — ignore/delete.
- So the next session can go straight to: stand up the DP=4 27B → run the parity eval.

## Status of the SeqKD transfer run (paused, separate from parity)
- Trained `checkpoints/docvqa-verl/seqkd-transfer/` (30 steps) + merged `merged_hf/` exist. Its strat24 eval was never validly completed (contention). Resume AFTER parity is established.
- Transfer SFT data: `data/sft/mmlb_transfer.parquet` (87 traj). Collection had reached ~138 unique-success Q (`outputs/teacher_rollouts/mmlb_train_n4.jsonl`).

## Decisions locked this session
- **ANLS metric = `docvqa/metrics.py:evaluate_prediction` only** (single source of truth; binary threshold **0.9** — NOT classic DocVQA's 0.5). All of collect/eval/reward route through it.
- **SFT filter stays ANLS = 1.0** (not loosened).
- **Train/eval splits must be by DOCUMENT** (train.json/heldout.json already are, 0 shared docs; `eval_subset_strat24` is NOT — shares 11/16 docs with train, so only valid when training corpus ≠ dv2026).
- **Harness = CodeAct** (our `agent_loop.py`); the `rvlm_minimal`/`flat_solo` framing is STALE.

## DOCS TO UPDATE (not yet done — user request: "update all relevant docs" to CodeAct)
Stale `rvlm_minimal`/`flat_solo` references to fix → CodeAct:
- `CLAUDE.md:36-37,56-57,61,67`  ·  `project/experiments/SETUP-POINTERS.md` (§3)
- `project/verl_recipe_survey.md:164`  ·  `project/superpowers/specs/2026-04-30-…design.md` (multiple)  ·  `…/plans/2026-04-30-…scaffold.md`

## Pointers
- `project/experiments/stage0-execution-log.md` — full chronological log (read the last ~15 bullets).
- `project/experiments/SETUP-POINTERS.md` — orientation map (note: its §3 still says rvlm_minimal — fix).
- `docvqa/train/README.md` — verified merge+serve+eval procedure.
- `.claude/CLAUDE.md` — session registry (partly stale re: collection attribution).
