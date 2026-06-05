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


def _approx_tokens(messages: list[dict], tokenizer) -> int:
    """Approximate the trained sequence length: sum of per-message content tokens
    plus a small per-turn template overhead. Close enough to drop trajectories
    that would OOM under SDPA (which can't pack long seqs cheaply)."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        total += len(tokenizer.encode(content, add_special_tokens=False)) + 8
    return total


def filter_and_project(rows: list[dict], max_per_question: int | None,
                       max_tokens: int | None = None, tokenizer=None) -> list[dict]:
    by_q: dict[str, list[dict]] = defaultdict(list)
    dropped_long = 0
    for r in rows:
        if r.get("anls") != 1.0:
            continue
        if r.get("termination") != "submit":
            continue
        msgs = r.get("messages") or []
        if not msgs or not any(m.get("role") == "assistant" for m in msgs):
            continue
        if max_tokens is not None and tokenizer is not None:
            if _approx_tokens(msgs, tokenizer) > max_tokens:
                dropped_long += 1
                continue
        by_q[r.get("question_id", r.get("record_id", ""))].append(r)

    kept: list[dict] = []
    for qid, rs in by_q.items():
        chosen = rs if max_per_question is None else rs[:max_per_question]
        for r in chosen:
            kept.append({"messages": r["messages"]})
    if max_tokens is not None:
        print(f"[max-tokens={max_tokens}] dropped {dropped_long} over-length "
              f"trajectories", file=sys.stderr)
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
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="Drop trajectories whose approx token length exceeds this "
                         "(SDPA can OOM on very long seqs). Loads the tokenizer.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()

    tokenizer = None
    if args.max_tokens is not None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    rows = _read_jsonl(Path(args.inp))
    kept = filter_and_project(rows, args.max_per_question,
                              max_tokens=args.max_tokens, tokenizer=tokenizer)
    write_parquet(kept, Path(args.out))
    print(f"kept {len(kept)} trajectories from {len(rows)} rollouts -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
