#!/usr/bin/env python
"""Build a CURRICULUM RL parquet from the DocVQA-family pool, sorted easy->hard by num_pages.

The pool (`data/pool/prompts.json`, built by sample_pool.py) is already in the verl RL schema
(prompt / data_source / reward_model / extra_info / doc_dir). This script:
  - reads each doc's metadata.json to get `num_pages`,
  - drops rows with no gold answer or missing doc,
  - SORTS ascending by (num_pages, record_id) so a shuffle=False run sees a difficulty
    curriculum (1-page chartqa/infographic/mapqa/docvqa-sp first -> multi-page mmlb/slidevqa last),
  - adds `agent_name` and `num_pages`,
  - writes a parquet.

Use with `data.shuffle=False` so the curriculum order is preserved. Easy-first also gives the
cold-starting policy more reward variance early (GRPO needs in-group spread to learn).

Usage:
    python docvqa/scripts/make_curriculum_parquet.py \
        --pool data/pool/prompts.json \
        --out  data/pool/curriculum_rl.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _num_pages(doc_dir: str) -> int:
    try:
        m = json.loads((Path(doc_dir) / "metadata.json").read_text())
        return int(m.get("num_pages", 1))
    except Exception:
        return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, type=Path, help="pool prompts.json (verl RL schema)")
    ap.add_argument("--out", required=True, type=Path, help="output curriculum parquet")
    ap.add_argument("--agent-name", default="docvqa_repl")
    ap.add_argument("--max-pages", type=int, default=0, help="drop docs with more pages than this (0=keep all)")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N after sorting (0=all)")
    args = ap.parse_args()

    rows = json.loads(args.pool.read_text())
    out = []
    skipped_doc = skipped_ans = skipped_pages = 0
    for r in rows:
        dd = r.get("doc_dir")
        if not dd or not Path(dd).is_dir():
            skipped_doc += 1
            continue
        ans = (r.get("reward_model") or {}).get("ground_truth") or r.get("answer")
        if ans is None or ans == "":
            skipped_ans += 1
            continue
        np_ = _num_pages(dd)
        if args.max_pages and np_ > args.max_pages:
            skipped_pages += 1
            continue
        row = dict(r)
        row["num_pages"] = np_
        row["agent_name"] = args.agent_name
        out.append(row)

    # Curriculum order: easy (few pages) -> hard. Within each page-level, INTERLEAVE datasets
    # (round-robin) so the easy tier is diverse (not 9k chartqa then 1.5k mapqa). Per row, rank =
    # its index among same-(num_pages,dataset) rows; sort by (num_pages, rank, dataset) so we get
    # ds0[0],ds1[0],...,ds0[1],ds1[1],... at each difficulty level.
    from collections import defaultdict

    seen: dict[tuple, int] = defaultdict(int)
    for r in sorted(out, key=lambda r: (r["num_pages"], r["dataset"], r["record_id"])):
        key = (r["num_pages"], r["dataset"])
        r["_rank"] = seen[key]
        seen[key] += 1
    out.sort(key=lambda r: (r["num_pages"], r["_rank"], r["dataset"]))
    for r in out:
        r.pop("_rank", None)
    if args.limit:
        out = out[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_parquet(args.out, index=False)
    pages = df["num_pages"]
    print(f"wrote {len(df)} rows -> {args.out}")
    print(f"  skipped: {skipped_doc} no-doc, {skipped_ans} no-answer, {skipped_pages} >max_pages")
    print(f"  num_pages: min={pages.min()} max={pages.max()} mean={pages.mean():.2f}")
    print("  page buckets:", df.groupby("num_pages").size().head(12).to_dict())
    print("  datasets (first 500 = easiest):", df.head(500)["dataset"].value_counts().to_dict())


if __name__ == "__main__":
    main()
