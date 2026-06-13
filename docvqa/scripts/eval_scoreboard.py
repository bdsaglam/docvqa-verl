#!/usr/bin/env python3
"""Build a per-page-bucket scoreboard across eval run-dirs.

Reads each eval run's per-question ANLS (tasks/<doc>/result.json) and the doc
page counts (metadata.json under each doc_dir), then reports binary-ANLS@0.9
accuracy overall and split by page bucket (1p / 2-9p / >=10p). This is the lens
that disambiguates the single-page-vs-multi-page transfer question (see
RL-async-findings addendum): RL trained on single-page only, eval is ~45%
multi-page, so the bucketed delta vs base is the real verdict.

Usage:
    python -m docvqa.scripts.eval_scoreboard \
        --questions /home/baris/repos/docvqa-verl/data/docvqa-2026/val/docvqa_mini.json \
        --runs base=outputs/runs/base-4b-mini \
               s20=outputs/runs/ckpt20-mini ... \
               s100=outputs/runs/ckpt100-mini
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pages_by_doc(questions_path: str) -> dict[str, int]:
    qs = json.loads(Path(questions_path).read_text())
    pages: dict[str, int] = {}
    for q in qs:
        doc = q.get("doc_id")
        if doc in pages:
            continue
        m = Path(q["doc_dir"]) / "metadata.json"
        if m.exists():
            pages[doc] = json.loads(m.read_text()).get("num_pages", 1)
    return pages


def _bucket(p: int) -> str:
    if p <= 1:
        return "1p"
    if p <= 9:
        return "2-9p"
    return ">=10p"


def _load_run(run_dir: str) -> dict[tuple, float]:
    """Return {(doc_id, question_id): mean_anls} for a run dir."""
    out: dict[tuple, float] = {}
    for rj in Path(run_dir).glob("tasks/*/result.json"):
        d = json.loads(rj.read_text())
        doc = d["doc_id"]
        for q in d["questions"]:
            out[(doc, q["question_id"])] = q["mean"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="tag=run_dir pairs, in display order")
    args = ap.parse_args()

    pages = _pages_by_doc(args.questions)
    runs = {}
    for spec in args.runs:
        tag, _, rd = spec.partition("=")
        runs[tag] = _load_run(rd)

    # Common question set actually scored across all runs (so columns compare
    # apples-to-apples even if some run is mid-flight).
    keysets = [set(r) for r in runs.values() if r]
    common = set.intersection(*keysets) if keysets else set()
    buckets = ["1p", "2-9p", ">=10p", "ALL"]

    def acc(tag: str, bucket: str) -> tuple[float, int]:
        r = runs[tag]
        ks = [k for k in common if bucket == "ALL" or _bucket(pages.get(k[0], 1)) == bucket]
        vals = [r[k] for k in ks if k in r]
        return (sum(vals) / len(vals) if vals else float("nan"), len(vals))

    n_by_bucket = {b: acc(next(iter(runs)), b)[1] for b in buckets}
    print(f"\n=== Bucketed binary-ANLS@0.9 (common scored Qs: {len(common)}) ===")
    print(f"{'bucket':>8} (n) | " + " | ".join(f"{t:>7}" for t in runs))
    for b in buckets:
        cells = []
        base_tag = next(iter(runs))
        base_v, _ = acc(base_tag, b)
        for t in runs:
            v, _ = acc(t, b)
            mark = ""
            if t != base_tag and v == v and base_v == base_v:
                d = v - base_v
                mark = f"({d:+.2f})" if abs(d) >= 0.005 else "(=)"
            cells.append(f"{v:6.3f}{mark:>7}" if v == v else f"{'--':>13}")
        print(f"{b:>8} ({n_by_bucket[b]:>2}) | " + " | ".join(cells))
    print("\n(delta vs first column = base; per-question mean over n samples)")


if __name__ == "__main__":
    main()
