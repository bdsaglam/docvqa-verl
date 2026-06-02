# DocVQA Stage 0 — setup, data prep, teacher trajectories, eval harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up everything needed to train and evaluate the Qwen3.5-4B CodeAct agent — corrected model defaults, an n=8 / pass@8 / SC-8 eval harness, MMLongBench-Doc training data, and ANLS-filtered 27B teacher trajectories projected to verl SFT parquet — without using a training GPU.

**Architecture:** All work sits under `docvqa/` and `tests/docvqa/`. The agent loop (`docvqa/agent_loop.py`, CodeAct, append-only) is unchanged in behaviour; we re-point its model defaults, extend the eval/collection scripts around it, add an MMLongBench-Doc data adapter, and add a trajectory→SFT-parquet projection. The 27B teacher runs through this same loop against the already-serving vLLM endpoints (8927/8928).

**Tech Stack:** Python 3.12, verl (`verl.trainer.sft_trainer` for Plan 2), HuggingFace `datasets` + `huggingface_hub`, `pypdfium2` (PDF render), `pandas`/`pyarrow` (parquet), `httpx` (vLLM HTTP), `pytest`, `Levenshtein` (ANLS via `docvqa/metrics.py`).

**Scope:** This is Plan 1 of 3. Plan 2 = SeqKD training (`verl.trainer.sft_trainer`, LoRA, needs GPUs). Plan 3 = forward-KL top-k soft-target KD. This plan produces: materialized MMLongBench-Doc train data, teacher trajectories + SFT parquet, the extended eval harness, and the measured before-baseline (4B zero-shot) + teacher-ceiling (27B) on DocVQA-2026 val.

**Spec:** `project/superpowers/specs/2026-06-02-docvqa-offpolicy-distillation-design.md` (Stage 0 = §7; eval = §9; data = §5).

---

## File Structure

| Path | Responsibility | This plan |
|---|---|---|
| `docvqa/agent_loop.py` | CodeAct loop; model defaults + docstring | Modify (T1) |
| `docvqa/scripts/collect_trajectories.py` | Teacher/student rollout collection | Modify defaults (T1), run (T7) |
| `docvqa/eval_metrics.py` | Pure metric helpers: mean±std, pass@k, SC-k vote | **Create** (T2) |
| `docvqa/scripts/eval.py` | DocVQA-2026 val eval harness (§9 protocol) | Modify (T1 defaults, T3 n-rollout + top_k + metrics) |
| `docvqa/scripts/make_sft_data.py` | Filter ANLS-pass trajectories → verl SFT parquet | **Create** (T4) |
| `docvqa/scripts/prepare_data.py` | Dataset → per-doc dirs + questions.json | Modify: add MMLongBench-Doc adapter (T5) |
| `docvqa/scripts/check_leakage.py` | Best-effort DocVQA-2026 ↔ training-set overlap report | **Create** (T6) |
| `tests/docvqa/test_eval_metrics.py` | Tests for T2 | **Create** (T2) |
| `tests/docvqa/test_make_sft_data.py` | Tests for T4 | **Create** (T4) |
| `tests/docvqa/test_prepare_data_mmlb.py` | Tests for T5 pure logic | **Create** (T5) |

---

## Task 1: Re-point model defaults to Qwen3.5 and fix the CodeAct docstring

**Files:**
- Modify: `docvqa/agent_loop.py:1-16` (docstring), `docvqa/agent_loop.py:90-94` (`_vlm_model_id` default)
- Modify: `docvqa/scripts/collect_trajectories.py:32-37,195,200` (defaults)
- Modify: `docvqa/scripts/eval.py:12-13,208-210` (defaults)

- [ ] **Step 1: Fix the agent_loop docstring (CodeAct, not rvlm)**

In `docvqa/agent_loop.py`, replace the first docstring paragraph:

```python
"""DocVQA CodeAct REPL agent loop for verl.

Implements the **CodeAct** scaffold: a strictly append-only transcript in
which the model emits ``<think>...</think>`` + a single ``` ```python ... ``` ``
fence per turn, the code runs in a persistent CPython subprocess with
``batch_look`` (27B VLM perception) + ``SUBMIT``, and the captured stdout is
appended verbatim as the next ``user`` turn. The full message list grows
monotonically (an MDP, fully observable) — this is the property RL / SFT /
distillation losses assume. Contrast the deployed ``rvlm`` solver, which uses
LeanRLM's hidden REPL namespace + ``RESET_HISTORY`` (a POMDP) and is *not* a
fine-tuning target. ANLS reward end-of-trajectory.
"""
```

- [ ] **Step 2: Re-point the VLM model id default**

In `docvqa/agent_loop.py:90-94`, change the fallback:

```python
        self._vlm_model_id: str = (
            agent_cfg.get("vlm_model_id")
            or os.environ.get("DOCVQA_VLM_MODEL_ID")
            or "Qwen/Qwen3.5-27B"
        )
```

- [ ] **Step 3: Re-point collect_trajectories defaults**

In `docvqa/scripts/collect_trajectories.py`, change the argparse defaults (around lines 195, 200) and the docstring example (lines 32-37):

```python
    ap.add_argument("--lm-model", default="Qwen/Qwen3.5-27B",
                    help="Model id served at --lm-base-url "
                         "(27B for teacher rollouts, 4B for student RS).")
    ...
    ap.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B")
```

Update the docstring example block to use `Qwen/Qwen3.5-27B` and base URL `http://localhost:8927/v1`.

- [ ] **Step 4: Re-point eval.py defaults**

In `docvqa/scripts/eval.py:207-210`:

```python
    ap.add_argument("--student-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--student-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--vlm-base-url", default="http://localhost:8927")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B")
```

- [ ] **Step 5: Verify imports still work and no stale ids remain**

Run:
```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
python -c "import docvqa.agent_loop, docvqa.scripts.eval, docvqa.scripts.collect_trajectories; print('ok')"
grep -rn "Qwen3-8B\|qwen3.6-27b\|Qwen3.6-27B\|willcb" docvqa/ || echo "no stale ids"
```
Expected: `ok`, then `no stale ids` (or only intentional references). 

- [ ] **Step 6: Commit**

```bash
git add docvqa/agent_loop.py docvqa/scripts/collect_trajectories.py docvqa/scripts/eval.py
git commit -m "docvqa: re-point defaults to Qwen3.5-4B/27B; correct CodeAct docstring"
```

---

## Task 2: Pure metric helpers — mean±std, pass@k, SC-k vote

**Files:**
- Create: `docvqa/eval_metrics.py`
- Test: `tests/docvqa/test_eval_metrics.py`

The §9 protocol needs three aggregations over a question's N rollouts. These
are pure functions over `(submitted_answer, gold_answer)` lists, scored with
the existing `docvqa.metrics.evaluate_prediction` (returns `(is_correct,
extracted)`). Build TDD-first.

- [ ] **Step 1: Write the failing test**

`tests/docvqa/test_eval_metrics.py`:

```python
import math
from docvqa.eval_metrics import score_rollouts, aggregate_question, majority_vote


def test_score_rollouts_marks_correct():
    # gold "Paris"; one exact, one wrong
    scores = score_rollouts(["Paris", "London"], "Paris")
    assert scores == [1.0, 0.0]


def test_aggregate_question_mean_passk_sc():
    # 3 rollouts, 2 say "Paris" (correct), 1 says "London"
    answers = ["Paris", "Paris", "London"]
    gold = "Paris"
    agg = aggregate_question(answers, gold)
    assert math.isclose(agg["mean"], 2 / 3)
    assert agg["passk"] == 1.0          # at least one correct
    assert agg["sc"] == 1.0             # majority vote "Paris" is correct
    assert agg["n"] == 3


def test_majority_vote_normalizes_and_breaks_ties_by_first_seen():
    # normalization: punctuation/case/articles stripped via metrics._clean_text
    assert majority_vote(["The Paris.", "paris", "London"]) == "The Paris."


def test_aggregate_all_wrong():
    agg = aggregate_question(["London", "Berlin"], "Paris")
    assert agg["mean"] == 0.0 and agg["passk"] == 0.0 and agg["sc"] == 0.0


def test_none_submission_scores_zero():
    assert score_rollouts([None, "Paris"], "Paris") == [0.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docvqa/test_eval_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: docvqa.eval_metrics`.

- [ ] **Step 3: Write the implementation**

`docvqa/eval_metrics.py`:

```python
"""Pure aggregation helpers for the n=8 / pass@k / SC-k eval protocol (spec §9).

Scoring delegates to docvqa.metrics.evaluate_prediction (the official
DocVQA-2026 metric: strict numeric/date match + relaxed ANLS).
"""
from __future__ import annotations

from collections import Counter

from docvqa.metrics import _clean_text, evaluate_prediction


def score_rollouts(submitted: list[str | None], gold: str | None) -> list[float]:
    """1.0 / 0.0 per rollout under the official metric. None or no-gold -> 0.0."""
    out: list[float] = []
    for ans in submitted:
        if ans is None or gold is None:
            out.append(0.0)
            continue
        is_correct, _ = evaluate_prediction(ans, gold)
        out.append(1.0 if is_correct else 0.0)
    return out


def majority_vote(submitted: list[str | None]) -> str | None:
    """Most common answer by normalized form; returns the first-seen raw
    surface of the winning normalized class. Ties broken by first appearance."""
    norm_to_raw: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for ans in submitted:
        if ans is None:
            continue
        key = _clean_text(str(ans))
        if key == "":
            continue
        if key not in norm_to_raw:
            norm_to_raw[key] = ans
        counts[key] += 1
    if not counts:
        return None
    # Counter.most_common is stable on insertion order for ties (py3.7+ dict order).
    best_key = counts.most_common(1)[0][0]
    return norm_to_raw[best_key]


def aggregate_question(submitted: list[str | None], gold: str | None) -> dict:
    """Per-question aggregation across rollouts."""
    scores = score_rollouts(submitted, gold)
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    passk = 1.0 if any(s == 1.0 for s in scores) else 0.0
    voted = majority_vote(submitted)
    sc = score_rollouts([voted], gold)[0] if voted is not None else 0.0
    return {"n": n, "mean": mean, "passk": passk, "sc": sc, "scores": scores,
            "voted_answer": voted}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/docvqa/test_eval_metrics.py -v`
Expected: PASS (5 tests). If `test_majority_vote_normalizes...` fails, confirm `_clean_text` lowercases + strips punctuation/articles (it does, per `docvqa/metrics.py:88-94`).

- [ ] **Step 5: Commit**

```bash
git add docvqa/eval_metrics.py tests/docvqa/test_eval_metrics.py
git commit -m "docvqa: add mean/pass@k/SC-k eval-metric helpers (spec §9)"
```

---

## Task 3: Extend eval.py to the §9 protocol (n rollouts, top_k, new metrics)

**Files:**
- Modify: `docvqa/scripts/eval.py:54-81` (`generate` — add `top_k`), `:126-201` (`_solve` → n rollouts + aggregation), `:204-216` (argparse)

- [ ] **Step 1: Thread `top_k` and matched sampling through the server manager**

In `_OpenAIClientServerManager.generate` (`eval.py:54-81`), add `top_k` to the request JSON (vLLM `/completions` accepts it):

```python
            json={
                "model": self._model,
                "prompt": prompt_text,
                "max_tokens": sampling_params.get("max_tokens", 4096),
                "temperature": sampling_params.get("temperature", 0.6),
                "top_p": sampling_params.get("top_p", 0.95),
                "top_k": sampling_params.get("top_k", 20),
                "stop": sampling_params.get("stop", ["<|im_end|>"]),
            },
```

- [ ] **Step 2: Replace `_solve` with an n-rollout collector that returns raw answers**

Replace `_solve` (`eval.py:126-155`) with:

```python
async def _solve_n(loop: DocVQAReplAgentLoop, q: dict, n: int,
                   sampling: dict) -> dict[str, Any]:
    """Run n rollouts for one question; return raw submitted answers + meta."""
    submitted: list[str | None] = []
    terminations: list[str | None] = []
    turns: list[int] = []
    for _ in range(n):
        try:
            out = await loop.run(
                sampling_params=sampling,
                question_id=q["question_id"], question=q["question"],
                doc_dir=q["doc_dir"], gold_answer=q.get("answer"),
                category=q.get("category", "unknown"),
            )
            submitted.append(out.extra_fields.get("submitted_answer"))
            terminations.append(out.extra_fields.get("termination"))
            turns.append(out.extra_fields.get("num_turns") or 0)
        except Exception as e:  # one failed rollout shouldn't kill the question
            submitted.append(None)
            terminations.append(f"error:{e!r}")
            turns.append(0)
    return {
        "question_id": q["question_id"], "doc_id": q.get("doc_id"),
        "category": q.get("category", "unknown"),
        "gold_answer": q.get("answer"),
        "submitted_answers": submitted,
        "terminations": terminations, "num_turns": turns,
    }
```

- [ ] **Step 3: Rewrite `_main_async` to aggregate with eval_metrics and report mean±std / pass@8 / SC-8**

Replace the body of `_main_async` (`eval.py:158-201`) from the `sem`/`_bound` block onward:

```python
    import statistics
    from docvqa.eval_metrics import aggregate_question

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}
    sem = asyncio.Semaphore(args.concurrency)

    async def _bound(q: dict) -> dict[str, Any]:
        async with sem:
            raw = await _solve_n(loop_obj, q, args.n, sampling)
        agg = aggregate_question(raw["submitted_answers"], raw["gold_answer"])
        return {**raw, **{k: agg[k] for k in ("mean", "passk", "sc", "scores",
                                              "voted_answer")}}

    results = await asyncio.gather(*(_bound(q) for q in questions))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    def _overall(key: str) -> float:
        return statistics.mean(r[key] for r in results) if results else 0.0

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"=== DocVQA-2026 eval (n={args.n}, model={args.student_model}) ===")
    means = [r["mean"] for r in results]
    print(f"  mean ANLS : {statistics.mean(means):.4f} "
          f"± {statistics.pstdev(means):.4f}  (n_q={len(results)})")
    print(f"  pass@{args.n}  : {_overall('passk'):.4f}")
    print(f"  SC-{args.n}    : {_overall('sc'):.4f}")
    for cat, rows in sorted(by_cat.items()):
        m = statistics.mean(r["mean"] for r in rows)
        print(f"    {cat:20s} mean={m:.4f} pass={statistics.mean(r['passk'] for r in rows):.4f} "
              f"sc={statistics.mean(r['sc'] for r in rows):.4f} (n_q={len(rows)})")
    print(f"  -> {out_path}")
```

- [ ] **Step 4: Add the new argparse flags**

In `main()` (`eval.py:204-216`) add:

```python
    ap.add_argument("--n", type=int, default=8, help="rollouts per question")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
```

- [ ] **Step 5: Smoke-check the harness wiring without a GPU**

The full eval needs a served 4B (T8). Verify the script parses and imports:
```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
python docvqa/scripts/eval.py --help | grep -E "\-\-n|\-\-top-k"
python -c "from docvqa.scripts.eval import _solve_n, _build_loop; print('ok')"
```
Expected: the `--n` / `--top-k` flags listed; `ok`.

- [ ] **Step 6: Commit**

```bash
git add docvqa/scripts/eval.py
git commit -m "docvqa: eval.py -> n-rollout mean/pass@k/SC-k protocol (spec §9)"
```

---

## Task 4: make_sft_data.py — filter ANLS-pass trajectories → verl SFT parquet

**Files:**
- Create: `docvqa/scripts/make_sft_data.py`
- Test: `tests/docvqa/test_make_sft_data.py`

Input: the JSONL from `collect_trajectories.py` (one rollout/line with
`messages`, `anls`, ...). Output: a parquet with a single `messages` column
(verl `MultiTurnSFTDataset` format) containing only ANLS==1.0 trajectories,
de-duplicated and optionally capped per question.

- [ ] **Step 1: Write the failing test**

`tests/docvqa/test_make_sft_data.py`:

```python
import json
import pandas as pd
from docvqa.scripts.make_sft_data import filter_and_project


def _rollout(anls, qid, answer, n_msgs=4):
    return {
        "record_id": f"ds:val:{qid}:0", "question_id": qid,
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "q"},
                     {"role": "assistant", "content": "<think>..</think>"},
                     {"role": "user", "content": "obs"}][:n_msgs],
        "submitted_answer": answer, "anls": anls,
        "termination": "submit", "num_turns": 1,
    }


def test_keeps_only_anls_pass():
    rows = [_rollout(1.0, "q1", "a"), _rollout(0.0, "q2", "b")]
    kept = filter_and_project(rows, max_per_question=None)
    assert len(kept) == 1
    assert kept[0]["messages"][0]["role"] == "system"


def test_caps_per_question():
    rows = [_rollout(1.0, "q1", "a") for _ in range(5)]
    kept = filter_and_project(rows, max_per_question=2)
    assert len(kept) == 2


def test_drops_empty_or_non_submit():
    rows = [{"question_id": "q3", "messages": [], "anls": 1.0,
             "termination": "submit", "submitted_answer": "x"}]
    assert filter_and_project(rows, max_per_question=None) == []


def test_output_is_messages_only(tmp_path):
    from docvqa.scripts.make_sft_data import write_parquet
    kept = filter_and_project([_rollout(1.0, "q1", "a")], max_per_question=None)
    out = tmp_path / "train.parquet"
    write_parquet(kept, out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["messages"]
    assert df.iloc[0]["messages"][0]["role"] == "system"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docvqa/test_make_sft_data.py -v`
Expected: FAIL — `ModuleNotFoundError: docvqa.scripts.make_sft_data`.

- [ ] **Step 3: Write the implementation**

`docvqa/scripts/make_sft_data.py`:

```python
#!/usr/bin/env python
"""Filter ANLS-passing trajectories and project to verl multi-turn SFT parquet.

verl's MultiTurnSFTDataset expects parquet with a single `messages` column
(list of {role, content}); assistant turns are trained, others masked. We keep
only `anls == 1.0` rollouts that terminated via SUBMIT with a non-empty
message list, optionally capping the number kept per question for balance.

Usage:
    python docvqa/scripts/make_sft_data.py \\
        --in outputs/teacher_rollouts/train_n8.jsonl \\
        --out data/sft/teacher_seqkd_train.parquet \\
        --max-per-question 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


def filter_and_project(rows: list[dict], max_per_question: int | None) -> list[dict]:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("anls") != 1.0:
            continue
        if r.get("termination") != "submit":
            continue
        msgs = r.get("messages") or []
        if not msgs or not any(m.get("role") == "assistant" for m in msgs):
            continue
        by_q[r.get("question_id", r.get("record_id", ""))].append(r)

    kept: list[dict] = []
    for qid, rs in by_q.items():
        chosen = rs if max_per_question is None else rs[:max_per_question]
        for r in chosen:
            kept.append({"messages": r["messages"]})
    return kept


def write_parquet(kept: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"messages": [k["messages"] for k in kept]}).to_parquet(out_path)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-question", type=int, default=None)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.inp))
    kept = filter_and_project(rows, args.max_per_question)
    write_parquet(kept, Path(args.out))
    n_q = len({tuple(sorted(k.items(), key=str)) for k in kept})  # rough
    print(f"kept {len(kept)} trajectories from {len(rows)} rollouts -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/docvqa/test_make_sft_data.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add docvqa/scripts/make_sft_data.py tests/docvqa/test_make_sft_data.py
git commit -m "docvqa: add trajectory->verl SFT parquet projection (ANLS filter)"
```

---

## Task 5: MMLongBench-Doc adapter in prepare_data.py

**Files:**
- Modify: `docvqa/scripts/prepare_data.py` (add adapter + register; add deps `pypdfium2`, `huggingface_hub`)
- Test: `tests/docvqa/test_prepare_data_mmlb.py`

Materialize from HF (`yubo2333/MMLongBench-Doc`, single `train` split, 1091 Q /
135 docs). doc_id = source `*.pdf`; category = `doc_type`; `answer="Not
answerable"` is real gold. Render PDFs with pypdfium2 (150 DPI, max 80 pages)
to `docs/<doc_id>/pages/page_*.png`. PDFs fetched via `hf_hub_download(...,
f"documents/{doc_id}", repo_type="dataset")`. Carve our own train split (HF has
no train/val/test boundary) and EXCLUDE doc_ids listed in an optional
`--exclude-doc-ids` file (the leakage-flagged set from T6).

- [ ] **Step 1: Write the failing test for the pure helpers**

`tests/docvqa/test_prepare_data_mmlb.py` (tests the row/id logic without HF or PDF rendering):

```python
from docvqa.scripts.prepare_data import _mmlb_question_id, _mmlb_rows_for_doc


def test_question_id_is_stable_and_indexed():
    assert _mmlb_question_id("COSTCO_2021_10K.pdf", 3) == "COSTCO_2021_10K.pdf::q3"


def test_rows_for_doc_builds_canonical_rows(tmp_path):
    hf_rows = [
        {"doc_id": "A.pdf", "question": "Q1", "answer": "1", "doc_type": "Financial report"},
        {"doc_id": "A.pdf", "question": "Q2", "answer": "Not answerable", "doc_type": "Financial report"},
    ]
    doc_dir = tmp_path / "docs" / "A.pdf"
    rows = _mmlb_rows_for_doc("train", hf_rows, doc_dir)
    assert len(rows) == 2
    assert rows[0]["question_id"] == "A.pdf::q0"
    assert rows[0]["category"] == "Financial report"
    assert rows[0]["dataset"] == "mmlongbench-doc"
    assert rows[1]["answer"] == "Not answerable"   # real gold, not nulled
    assert rows[0]["doc_dir"] == str(doc_dir)
    assert rows[0]["record_id"] == "mmlongbench-doc:train:A.pdf:A.pdf::q0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docvqa/test_prepare_data_mmlb.py -v`
Expected: FAIL — `ImportError: cannot import name '_mmlb_question_id'`.

- [ ] **Step 3: Add the pure helpers + adapter to prepare_data.py**

Add near the other adapters in `docvqa/scripts/prepare_data.py`:

```python
def _mmlb_question_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}::q{idx}"


def _mmlb_rows_for_doc(split: str, hf_rows: list[dict], doc_dir: Path) -> list[dict]:
    """Build canonical question rows for one MMLongBench doc (no I/O)."""
    rows = []
    for i, r in enumerate(hf_rows):
        rows.append(_build_row(
            dataset="mmlongbench-doc", split=split, doc_id=r["doc_id"],
            question_id=_mmlb_question_id(r["doc_id"], i),
            question=r["question"], answer=r["answer"],
            category=r["doc_type"], doc_dir_abs=doc_dir,
        ))
    return rows


_MMLB_REPO = "yubo2333/MMLongBench-Doc"
_MMLB_DPI = 150
_MMLB_MAX_PAGES = 80


def _mmlb_render_pdf(pdf_path: Path, out_dir: Path) -> int:
    """Render PDF pages to out_dir/page_<i>.png (idempotent). Returns page count."""
    import pypdfium2 as pdfium
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_total = len(pdf)
    n_render = min(n_total, _MMLB_MAX_PAGES)
    scale = _MMLB_DPI / 72.0
    for i in range(n_render):
        png = out_dir / f"page_{i}.png"
        if png.exists():
            continue
        img = pdf[i].render(scale=scale).to_pil()
        img.save(png, format="PNG", optimize=True)
    pdf.close()
    return n_render


def adapter_mmlongbench_doc(split: str, split_dir: Path) -> list[dict]:
    from collections import defaultdict
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    docs_dir = split_dir / "docs"

    exclude: set[str] = set()
    exclude_file = split_dir.parent / "exclude_doc_ids.txt"   # written by check_leakage
    if exclude_file.exists():
        exclude = {ln.strip() for ln in exclude_file.read_text().splitlines() if ln.strip()}

    ds = load_dataset(_MMLB_REPO, split="train")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in ds:
        if r["doc_id"] in exclude:
            continue
        by_doc[r["doc_id"]].append(r)

    all_rows: list[dict] = []
    for doc_id, hf_rows in by_doc.items():
        doc_dir = docs_dir / doc_id
        pdf_path = Path(hf_hub_download(_MMLB_REPO, f"documents/{doc_id}",
                                        repo_type="dataset"))
        num_pages = _mmlb_render_pdf(pdf_path, doc_dir / "pages")
        (doc_dir / "metadata.json").write_text(json.dumps({
            "doc_id": doc_id, "num_pages": num_pages,
            "doc_category": hf_rows[0]["doc_type"],
            "dataset": "mmlongbench-doc", "split": split,
        }))
        all_rows.extend(_mmlb_rows_for_doc(split, hf_rows, doc_dir))
    return all_rows
```

Register it in `ADAPTERS`:

```python
ADAPTERS: dict[str, Callable[[str, Path], list[dict]]] = {
    "docvqa-2026": adapter_docvqa_2026,
    "mmlongbench-doc": adapter_mmlongbench_doc,
}
```

- [ ] **Step 4: Run the pure-logic test to verify it passes**

Run: `pytest tests/docvqa/test_prepare_data_mmlb.py -v`
Expected: PASS. (If `_build_row` requires a kwarg not passed, align the call to its real signature in `prepare_data.py`.)

- [ ] **Step 5: Add deps and materialize a 3-doc smoke slice (real HF + render)**

```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
uv pip install pypdfium2 huggingface_hub datasets
# tiny smoke: cap docs by editing a temp question set is overkill; instead run
# the adapter directly on a 3-doc subset via a one-off python snippet:
python - <<'PY'
from pathlib import Path
from docvqa.scripts import prepare_data as P
from datasets import load_dataset
ds = load_dataset(P._MMLB_REPO, split="train")
doc_ids = list(dict.fromkeys(r["doc_id"] for r in ds))[:3]
print("smoke doc_ids:", doc_ids)
PY
```
Expected: prints 3 doc_ids (confirms HF access). Full materialization is Step 6.

- [ ] **Step 6: Materialize the MMLongBench-Doc train split**

```bash
python docvqa/scripts/prepare_data.py --dataset mmlongbench-doc --splits train \
    --out-root data
ls data/mmlongbench-doc/train/ && \
python -c "import json; d=json.load(open('data/mmlongbench-doc/train/questions.json')); \
print(len(d), 'questions over', len({r['doc_id'] for r in d}), 'docs')"
```
Expected: `questions.json` + `docs/` present; counts printed (≈1091 Q / ≤135 docs, fewer if T6 excludes any). **Record these counts** — this is the spec §5.2 volume gate. If question count is far below ~1k after exclusions, flag for adding another non-overlapping source (a follow-up, not in this plan).

- [ ] **Step 7: Commit**

```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_mmlb.py
git commit -m "docvqa: add MMLongBench-Doc data adapter (HF render + canonical rows)"
```

---

## Task 6: Leakage check — DocVQA-2026 ↔ MMLongBench-Doc

**Files:**
- Create: `docvqa/scripts/check_leakage.py`

Metadata gives no shared id (DocVQA-2026 doc_ids are synthetic `<category>_<N>`;
MMLongBench are source `*.pdf`), so the only robust test is **content-based**:
perceptual-hash the rendered first pages and compare. Emit a report and write
`data/mmlongbench-doc/exclude_doc_ids.txt` listing any MMLongBench doc whose
page-hash collides with a DocVQA-2026 page (the adapter reads this file).

- [ ] **Step 1: Write the script**

`docvqa/scripts/check_leakage.py`:

```python
#!/usr/bin/env python
"""Best-effort content overlap check between DocVQA-2026 and a training corpus.

Compares perceptual hashes (average-hash) of rendered pages. Any MMLongBench
doc with a page within HAMMING<=THRESH of a DocVQA-2026 page is flagged and
written to <train_root>/exclude_doc_ids.txt (the prepare_data adapter excludes
these). Conservative: false positives only cost us a few training docs.

Usage:
    python docvqa/scripts/check_leakage.py \\
        --ref-docs ~/repos/docvqa/data/docvqa-2026/val/ocr \\
        --ref-pages-glob '*/page_*.png' \\
        --train-root data/mmlongbench-doc/train \\
        --thresh 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


def ahash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size, size))
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= (1 << i)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-pages-root", required=True,
                    help="dir whose **/page_*.png are the DocVQA-2026 pages")
    ap.add_argument("--train-root", required=True,
                    help="data/<dataset>/<split> with docs/<doc_id>/pages/*.png")
    ap.add_argument("--thresh", type=int, default=4)
    args = ap.parse_args()

    ref_hashes = [ahash(p) for p in Path(args.ref_pages_root).rglob("page_*.png")]
    print(f"[ref] {len(ref_hashes)} reference pages")

    flagged: set[str] = set()
    docs_root = Path(args.train_root) / "docs"
    for doc_dir in sorted(docs_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        for page in (doc_dir / "pages").glob("page_*.png"):
            h = ahash(page)
            if any(hamming(h, r) <= args.thresh for r in ref_hashes):
                flagged.add(doc_dir.name)
                break

    out = Path(args.train_root) / "exclude_doc_ids.txt"
    out.write_text("\n".join(sorted(flagged)) + ("\n" if flagged else ""))
    print(f"[result] flagged {len(flagged)} docs as possible overlap -> {out}")
    for d in sorted(flagged):
        print("   ", d)
```

- [ ] **Step 2: Run it after materialization (T5 step 6)**

```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
python docvqa/scripts/check_leakage.py \
    --ref-pages-root ~/repos/docvqa/data/docvqa-2026/val/ocr \
    --train-root data/mmlongbench-doc/train --thresh 4
```
Expected: prints reference page count and a (likely empty) flagged list, writes `exclude_doc_ids.txt`. If non-empty, **re-run T5 step 6** so the adapter drops the flagged docs, then re-record counts.

Note: `val/ocr/<doc_id>/` holds `page_*.md`, not images — if there are no
`page_*.png` under the OCR dir, point `--ref-pages-root` at the DocVQA-2026
`pages/` materialized by `prepare_data.py --dataset docvqa-2026 --splits val`
instead (materialize it first if absent).

- [ ] **Step 3: Commit**

```bash
git add docvqa/scripts/check_leakage.py
git commit -m "docvqa: add best-effort content-overlap leakage check"
```

---

## Task 7: Collect 27B teacher trajectories (long run, no training GPU)

**Files:** none (uses `collect_trajectories.py` + `make_sft_data.py`).

This runs the CodeAct loop with the 27B as agent + VLM against the live
servers. Long-running → tmux + monitoring per the project's async conventions.

- [ ] **Step 1: Launch collection in a dedicated tmux session**

```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
tmux new-session -d -s teacher-collect
tmux send-keys -t teacher-collect "cd ~/repos/docvqa-verl && source .venv/bin/activate && \
python docvqa/scripts/collect_trajectories.py \
  --questions data/mmlongbench-doc/train/train.json \
  --lm-base-url http://localhost:8927/v1 --lm-model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8928 --vlm-model Qwen/Qwen3.5-27B \
  --num-samples-per-q 8 --temperature 1.0 --concurrency 8 --resume \
  --output outputs/teacher_rollouts/mmlb_train_n8.jsonl \
  2>&1 | tee outputs/teacher_rollouts/collect.log" Enter
```
Note: `train.json` exists only if the train split had gold (it does — MMLongBench answers are gold). If `_split_train_heldout` produced `train.json`/`heldout.json`, use `train.json`; otherwise use `questions.json`.

- [ ] **Step 2: Schedule a heartbeat to monitor completion**

Use a background monitor (Monitor tool or a `tail`/`wc -l` poll on
`outputs/teacher_rollouts/mmlb_train_n8.jsonl`) and a periodic check that the
27B endpoints are alive (`curl -s localhost:8927/v1/models`). Expected: line
count grows toward `num_questions * 8`.

- [ ] **Step 3: Inspect collection quality (the CodeAct-27B teacher ceiling)**

When done (or partway):
```bash
python - <<'PY'
import json, statistics
from collections import Counter
rows = [json.loads(l) for l in open("outputs/teacher_rollouts/mmlb_train_n8.jsonl")]
print("rollouts:", len(rows), "mean ANLS:", statistics.mean(r["anls"] for r in rows))
print("termination:", Counter(r["termination"] for r in rows))
solved_q = {r["question_id"] for r in rows if r["anls"] == 1.0}
allq = {r["question_id"] for r in rows}
print(f"questions with >=1 success (pass@8): {len(solved_q)}/{len(allq)}")
PY
```
Expected: a mean ANLS and a pass@8 coverage. **This is the teacher-quality
gate** — if very few questions ever get a correct teacher rollout, surface it
(weak teacher → re-evaluate before Plan 2).

- [ ] **Step 4: Project to SFT parquet**

```bash
python docvqa/scripts/make_sft_data.py \
  --in outputs/teacher_rollouts/mmlb_train_n8.jsonl \
  --out data/sft/teacher_seqkd_train.parquet --max-per-question 2
python -c "import pandas as pd; df=pd.read_parquet('data/sft/teacher_seqkd_train.parquet'); \
print(len(df),'SFT trajectories; first has', len(df.iloc[0]['messages']),'messages')"
```
Expected: a positive count; first row's `messages` is a list of role/content dicts.

- [ ] **Step 5: Commit the data manifest (not the heavy artifacts)**

```bash
echo "data/" ; grep -q "^outputs/" .gitignore || echo "outputs/" >> .gitignore
grep -q "^data/sft/" .gitignore || echo "data/sft/" >> .gitignore
git add .gitignore
git commit -m "docvqa: ignore teacher rollout + SFT data artifacts"
```
(Record the counts from Steps 3–4 in the run log / a short note; do not commit multi-GB page/rollout data.)

---

## Task 8: Measure the before-baseline (4B) and teacher-ceiling (27B) on val

**Files:** none (uses `eval.py`). Needs a GPU to serve the 4B; the 27B is already served.

- [ ] **Step 1: Materialize DocVQA-2026 val (if not already present)**

```bash
cd ~/repos/docvqa-verl && source .venv/bin/activate
python docvqa/scripts/prepare_data.py --dataset docvqa-2026 --splits val --out-root data
ls data/docvqa-2026/val/questions.json
```
Expected: `questions.json` (80 questions / 25 docs).

- [ ] **Step 2: Serve Qwen3.5-4B in vLLM (when a GPU is free)**

```bash
tmux new-session -d -s serve-4b
tmux send-keys -t serve-4b "CUDA_VISIBLE_DEVICES=<free_gpu> vllm serve Qwen/Qwen3.5-4B \
  --port 8000 --max-model-len 32768 --enable-prefix-caching" Enter
# wait until ready:
until curl -s localhost:8000/v1/models | grep -q Qwen3.5-4B; do sleep 5; done; echo READY
```

- [ ] **Step 3: Run the §9 before-eval (zero-shot 4B)**

```bash
python docvqa/scripts/eval.py \
  --questions data/docvqa-2026/val/questions.json \
  --student-base-url http://localhost:8000/v1 --student-model Qwen/Qwen3.5-4B \
  --vlm-base-url http://localhost:8928 --vlm-model Qwen/Qwen3.5-27B \
  --n 8 --temperature 0.6 --top-p 0.95 --top-k 20 --concurrency 8 \
  --output outputs/eval/val_4b_zeroshot_n8.json
```
Expected: a report with `mean ANLS ± std`, `pass@8`, `SC-8`, and per-category lines. **This is the floor to beat in Plan 2.** Record it.

- [ ] **Step 4: Run the teacher-ceiling eval (27B CodeAct on val)**

```bash
python docvqa/scripts/eval.py \
  --questions data/docvqa-2026/val/questions.json \
  --student-base-url http://localhost:8927/v1 --student-model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8928 --vlm-model Qwen/Qwen3.5-27B \
  --n 8 --temperature 0.6 --top-p 0.95 --top-k 20 --concurrency 4 \
  --output outputs/eval/val_27b_codeact_n8.json
```
Expected: the CodeAct-27B ceiling (compare to rvlm-27B ~0.39 / SC-8 0.51 — a large gap would confirm the spec §3.2 teacher-quality risk).

- [ ] **Step 5: Record both numbers in the run log**

Append `mean±std / pass@8 / SC-8` for both to `outputs/eval/RESULTS.md` (a tracked note is fine; the per-question JSON stays untracked). Commit only the note:

```bash
git add outputs/eval/RESULTS.md 2>/dev/null && \
git commit -m "docvqa: record 4B floor + 27B CodeAct ceiling on DocVQA-2026 val" || \
echo "no RESULTS.md to commit yet"
```

---

## Self-Review

**Spec coverage (Stage 0 = §7, eval = §9, data = §5):**
- §7.1 re-point defaults + docstring → T1. Chat-template `<think>` gate → see note below. Data prep → T5. Teacher trajectories → T7. Before-eval/ceiling → T8. ✓
- §9 eval protocol (n=8, mean±std, pass@8, SC-8, matched sampling, frozen VLM) → T2 + T3 + T8. ✓
- §5 data (MMLongBench primary, no-leakage rule, volume gate) → T5 (materialize + count) + T6 (leakage) . ✓
- §5.4 SFT projection (parquet `messages`) → T4 + T7 step 4. ✓

**Gap found & resolved:** the spec §7.1 step-2 "chat-template `<think>` gate" is not a standalone task here. Rationale: the load-bearing check (per the verl investigation) is `MultiTurnSFTDataset`'s per-turn-vs-whole-conversation tokenization mismatch, which is mitigated by `data.ignore_input_ids_mismatch=True` and validated against the actual SFT loss mask — that belongs in **Plan 2 (SeqKD training)**, where the dataset is instantiated. Added as an explicit note here so it is not lost: **Plan 2 must set `data.ignore_input_ids_mismatch=True` and validate the assistant-only loss mask on a few samples before a long run.** For Stage 0, the trajectories are stored as raw `messages`; no tokenization gate is required to produce them.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every run step has an exact command + expected output. ✓

**Type/name consistency:** `filter_and_project`/`write_parquet` (T4) match their test and T7 usage; `aggregate_question`/`score_rollouts`/`majority_vote` (T2) match T3 usage; `_mmlb_question_id`/`_mmlb_rows_for_doc`/`adapter_mmlongbench_doc` (T5) match the test and `ADAPTERS` registration; `_build_row` is called with the kwargs the investigation reported (verify against the real signature in T5 step 4). ✓

**Residual risk to verify at execution time (not placeholders — real unknowns):**
- `_build_row`'s exact kwargs (T5) — confirm against `prepare_data.py` when implementing; the test will catch a mismatch.
- vLLM `/completions` accepting `top_k` directly vs needing `extra_body` (T3) — if rejected, move `top_k` into an `extra_body` dict.
- DocVQA-2026 `val/ocr/` may hold `.md` not `.png` (T6) — fallback documented (use materialized `pages/`).
