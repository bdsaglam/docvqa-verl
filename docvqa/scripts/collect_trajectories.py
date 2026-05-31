#!/usr/bin/env python
"""Collect agent-loop trajectories for SFT with rejection sampling.

Runs ``DocVQAReplAgentLoop`` against an OpenAI-compatible LM endpoint over
a question split, sampling N diverse rollouts per question (high temperature
for diversity), and writes one JSONL line per rollout with the full chat-
format trajectory + ANLS score.

Output format (one JSON object per line):
    {
      "record_id": "<dataset>:<split>:<doc_id>:<question_id>",
      "question_id": ..., "doc_id": ..., "category": ...,
      "messages": [{"role": "system", "content": ...}, ...],
      "submitted_answer": ..., "gold_answer": ...,
      "anls": 0.0 | 1.0,
      "termination": "submit" | "iter_cap" | "parse_error" | "token_cap" | ...,
      "num_turns": int, "vlm_calls": int,
      "wall_clock_s": float,
      "sample_idx": int, "lm_model": str, "temperature": float
    }

Use cases:
  1. Teacher trajectories for SFT-RS (point ``--lm-model`` at the 27B
     and filter the output by ``anls == 1.0``).
  2. Student-side rejection sampling (point ``--lm-model`` at the student
     to bootstrap from rare successes when teacher rollouts are too costly).

Example (teacher rollouts, 4 samples per question):
    python docvqa/scripts/collect_trajectories.py \\
        --questions data/docvqa-2026/val/train.json \\
        --lm-base-url http://localhost:8928/v1 \\
        --lm-model Qwen/Qwen3.6-27B \\
        --vlm-base-url http://localhost:8928 \\
        --vlm-model Qwen/Qwen3.6-27B \\
        --num-samples-per-q 4 \\
        --temperature 1.0 \\
        --concurrency 4 \\
        --output outputs/teacher_rollouts/train_n4.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Project root on path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docvqa.metrics import evaluate_prediction  # noqa: E402
from docvqa.scripts.eval import _build_loop  # noqa: E402


async def _one_rollout(loop_obj, q: dict, sample_idx: int,
                       temperature: float, lm_model: str) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        out = await loop_obj.run(
            sampling_params={"temperature": temperature, "top_p": 0.95},
            question_id=q["question_id"],
            question=q["question"],
            doc_dir=q["doc_dir"],
            gold_answer=q.get("answer"),
            category=q.get("category", "unknown"),
        )
    except Exception as e:
        return {
            "record_id": q.get("record_id"),
            "question_id": q["question_id"],
            "doc_id": q.get("doc_id"),
            "category": q.get("category", "unknown"),
            "messages": [],
            "submitted_answer": None,
            "gold_answer": q.get("answer"),
            "anls": 0.0,
            "termination": "error",
            "error": repr(e),
            "num_turns": 0,
            "vlm_calls": 0,
            "wall_clock_s": time.monotonic() - t0,
            "sample_idx": sample_idx,
            "lm_model": lm_model,
            "temperature": temperature,
        }

    submitted = out.extra_fields.get("submitted_answer")
    gold = q.get("answer")
    if submitted is None or gold is None:
        anls = 0.0
    else:
        is_correct, _ = evaluate_prediction(submitted, gold)
        anls = 1.0 if is_correct else 0.0

    return {
        "record_id": q.get("record_id"),
        "question_id": q["question_id"],
        "doc_id": q.get("doc_id"),
        "category": q.get("category", "unknown"),
        "messages": out.extra_fields.get("messages", []),
        "submitted_answer": submitted,
        "gold_answer": gold,
        "anls": anls,
        "termination": out.extra_fields.get("termination"),
        "num_turns": out.extra_fields.get("num_turns"),
        "vlm_calls": out.extra_fields.get("vlm_calls"),
        "wall_clock_s": out.extra_fields.get("wall_clock_s",
                                              time.monotonic() - t0),
        "sample_idx": sample_idx,
        "lm_model": lm_model,
        "temperature": temperature,
    }


async def _main_async(args) -> None:
    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    loop_obj = _build_loop(
        args.lm_base_url, args.lm_model,
        args.vlm_base_url, args.vlm_model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip (record_id, sample_idx) pairs already in the file.
    done: set[tuple[str, int]] = set()
    if out_path.exists() and args.resume:
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r.get("record_id") or r["question_id"],
                              r["sample_idx"]))
                except json.JSONDecodeError:
                    continue
        print(f"[resume] {len(done)} rollouts already in {out_path}",
              file=sys.stderr)

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def _bound(q: dict, sample_idx: int) -> dict[str, Any] | None:
        key = (q.get("record_id") or q["question_id"], sample_idx)
        if key in done:
            return None
        async with sem:
            r = await _one_rollout(loop_obj, q, sample_idx,
                                    args.temperature, args.lm_model)
        async with write_lock:
            with out_path.open("a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return r

    tasks = [
        _bound(q, s)
        for q in questions
        for s in range(args.num_samples_per_q)
    ]
    raw_results = await asyncio.gather(*tasks)
    results = [r for r in raw_results if r is not None]

    if not results:
        print("[no new rollouts collected]", file=sys.stderr)
        return

    # Aggregate report.
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "unknown"), []).append(r["anls"])

    correct = sum(1 for r in results if r["anls"] == 1.0)
    print(f"=== Collection report (this run) ===")
    print(f"  total rollouts: {len(results)}  "
          f"(correct: {correct}, mean ANLS: "
          f"{statistics.mean(r['anls'] for r in results):.4f})")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat:22s}: mean={statistics.mean(scores):.4f}  "
              f"(n={len(scores)}, correct={int(sum(scores))})")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True,
                    help="Path to questions.json (or train.json/heldout.json)")
    ap.add_argument("--lm-base-url", default="http://localhost:8928/v1",
                    help="OpenAI-compatible base URL for the agent's LM "
                         "(use the 27B for teacher rollouts, the 8B for student).")
    ap.add_argument("--lm-model", default="Qwen/Qwen3.6-27B",
                    help="Model id served at --lm-base-url.")
    ap.add_argument("--vlm-base-url", default="http://localhost:8928",
                    help="Base URL for the VLM tool (batch_look). "
                         "Same endpoint as the teacher's LM works.")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--num-samples-per-q", type=int, default=4,
                    help="How many rollouts to sample per question.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap number of questions (for smoke runs).")
    ap.add_argument("--output", required=True,
                    help="Output JSONL path. Resumable (--resume).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (record_id, sample_idx) already in --output.")
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
