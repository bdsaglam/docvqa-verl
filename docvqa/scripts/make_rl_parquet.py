#!/usr/bin/env python
"""Convert a prepared DocVQA-family questions JSON into a verl RL prompt parquet.

The RL trainer (verl main_ppo / GRPO) ingests one parquet of *prompts*: each row
carries the verl-required `prompt` / `data_source` / `reward_model` / `extra_info`
fields plus the human-readable metadata, and `agent_name` selecting the rollout
agent loop. This mirrors the schema of `data/docvqa-2026/val/train_rl.parquet`
(built by prepare_data.py) but starts from an already-materialized questions JSON
(e.g. the MMLB `solvable141.json` / `collect_v4_set.json` subsets), so we can
turn any curated question subset into an RL prompt set without re-materializing
docs.

Usage:
    python docvqa/scripts/make_rl_parquet.py \
        --questions data/mmlongbench-doc/train/solvable141.json \
        --docs-dir  data/mmlongbench-doc/train/docs \
        --out       data/mmlongbench-doc/train/solvable141_rl.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Reuse the canonical verl row builder so the schema stays identical to
# prepare_data.py's output (record_id / prompt / reward_model / extra_info ...).
from prepare_data import _build_row  # type: ignore


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--questions", required=True, type=Path, help="prepared questions JSON (list of rows)")
    ap.add_argument("--docs-dir", required=True, type=Path, help="dir holding <doc_id>/ materialized docs")
    ap.add_argument("--out", required=True, type=Path, help="output parquet path")
    ap.add_argument("--agent-name", default="docvqa_repl", help="rollout agent loop name")
    ap.add_argument("--limit", type=int, default=0, help="keep only first N rows (0 = all)")
    ap.add_argument("--require-answer", action="store_true", help="drop rows with empty gold answer")
    args = ap.parse_args()

    rows_in = json.loads(args.questions.read_text())
    docs_dir = args.docs_dir.resolve()

    out_rows: list[dict] = []
    skipped_no_doc, skipped_no_ans = 0, 0
    for r in rows_in:
        doc_id = r["doc_id"]
        doc_dir_abs = docs_dir / doc_id
        if not doc_dir_abs.exists():
            skipped_no_doc += 1
            continue
        answer = r.get("answer")
        if args.require_answer and (answer is None or answer == ""):
            skipped_no_ans += 1
            continue
        row = _build_row(
            dataset=r["dataset"],
            split=r["split"],
            doc_id=doc_id,
            question_id=r["question_id"],
            question=r["question"],
            answer=answer,
            category=r.get("category", ""),
            doc_dir_abs=doc_dir_abs,
        )
        row["agent_name"] = args.agent_name
        out_rows.append(row)
        if args.limit and len(out_rows) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out_rows)
    df.to_parquet(args.out, index=False)
    n_docs = df["doc_id"].nunique() if len(df) else 0
    print(
        f"wrote {len(df)} rows / {n_docs} docs -> {args.out}"
        f"  (skipped: {skipped_no_doc} no-doc, {skipped_no_ans} no-answer)"
    )


if __name__ == "__main__":
    main()
