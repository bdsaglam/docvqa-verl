#!/usr/bin/env python
"""Filter ANLS-passing trajectories and project to verl multi-turn SFT parquet.

verl's MultiTurnSFTDataset expects parquet with a single `messages` column
(list of {role, content}); assistant turns are trained, others masked. We keep
only `anls == 1.0` rollouts that terminated via SUBMIT with a non-empty
message list (containing at least one assistant turn), optionally capping the
number kept per question for balance.

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
    print(f"kept {len(kept)} trajectories from {len(rows)} rollouts -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
