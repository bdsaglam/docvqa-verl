#!/usr/bin/env python3
"""Build a *mixed-page* RL curriculum from the existing page-ascending one.

Motivation (2026-06-13): the default `curriculum_rl.parquet` is sorted strictly
easy->hard by num_pages, and the pool is 84% single-page (9,632/11,464) with the
first multi-page prompt at row 9,632 (= step 1,205 at batch 8). So any short run
(our 120-step run trains on rows 0-959) sees **only single-page docs** — yet
DocVQA-2026 val is ~45% multi-page (34% >= 10 pages). The policy never practices
the survey->locate page-navigation half of the CodeAct loop it's evaluated on,
and a 1-page-only prior ("look at pages[0], submit fast") can transfer badly to a
36-page doc.

This builder keeps the easy-first **cold-start** property (early steps mostly
single-page, for in-group reward variance) but **ramps multi-page in** across the
training window so the model learns the num_pages>1 behavior: when there's more
than one page, survey and locate before extracting.

Throughput cap: the cheap multi-page tier in this pool is only 2-3 page docs
(mp-docvqa/tatdqa); beyond that the pool jumps to 11-89 pages, which are the
~800s-tail rollouts that gate step time (a single 36-page rollout stalls the
whole 64-rollout step under the disaggregated trainer). So we cap at
``--max-pages 8`` by default (= 1/2/3p here) — real navigation signal at ~1.5x
single-page rollout cost, not 10x. Raise the cap only if you accept the tail.

Ordering within a draw stays dataset round-robin (preserve source diversity), and
prompts beyond the ramp window are appended page-ascending so a longer run / resume
stays valid. Pair with ``data.shuffle=False`` (preserves this order), same as the
original curriculum.

Usage:
    python -m docvqa.scripts.make_mixed_curriculum_parquet \
        --in data/pool/curriculum_rl.parquet \
        --out data/pool/curriculum_mixed_rl.parquet \
        --max-pages 8 --window 960 --seed 0
"""
from __future__ import annotations

import argparse
import itertools
from collections import deque

import pandas as pd


def _round_robin(frame: pd.DataFrame) -> list[int]:
    """Indices of `frame` interleaved round-robin across its `dataset` values."""
    queues = {
        ds: deque(sub.index.tolist())
        for ds, sub in frame.groupby("dataset", sort=True)
    }
    order: list[int] = []
    for ds in itertools.cycle(sorted(queues)):
        if not queues:
            break
        q = queues.get(ds)
        if q is None:
            continue
        order.append(q.popleft())
        if not q:
            del queues[ds]
    return order


def build(
    df: pd.DataFrame,
    max_pages: int,
    window: int,
    seed: int,  # reserved; ordering is deterministic without RNG
) -> pd.DataFrame:
    sub = df[df["num_pages"] <= max_pages].copy()
    single = sub[sub["num_pages"] == 1]
    multi = sub[sub["num_pages"] > 1].sort_values("num_pages", kind="stable")

    # Round-robin orderings (diversity), easy-first within multi (2p before 3p).
    single_q = deque(_round_robin(single))
    multi_q = deque(_round_robin(multi))  # already page-ascending blocks

    # Ramp the multi-page fraction across the training window in 3 blocks:
    # mostly-single early (cold start) -> ~half multi late (match eval's 45%).
    block = max(1, window // 3)
    schedule = [(block, 0.15), (block, 0.35), (window - 2 * block, 0.50)]

    ordered: list[int] = []
    for n, frac in schedule:
        want_multi = round(n * frac)
        want_single = n - want_multi
        # Interleave evenly: step a fractional accumulator so multi prompts are
        # spread through the block rather than clumped at its end.
        m = s = 0
        acc = 0.0
        for _ in range(n):
            take_multi = (
                multi_q
                and (acc + frac >= 1.0 or (want_single - s) <= 0)
                and (want_multi - m) > 0
            )
            if take_multi:
                ordered.append(multi_q.popleft())
                m += 1
                acc = acc + frac - 1.0
            elif single_q:
                ordered.append(single_q.popleft())
                s += 1
                acc += frac
            elif multi_q:
                ordered.append(multi_q.popleft())
                m += 1

    # Append everything left, page-ascending (for resumes / longer runs).
    tail = list(single_q) + list(multi_q)
    ordered.extend(tail)
    # And finally the docs we capped out (>max_pages), hardest last.
    capped = df[df["num_pages"] > max_pages].sort_values("num_pages", kind="stable")
    ordered.extend(capped.index.tolist())

    out = df.loc[ordered].reset_index(drop=True)
    assert len(out) == len(df), f"row count drift: {len(out)} != {len(df)}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pool/curriculum_rl.parquet")
    ap.add_argument("--out", dest="out", default="data/pool/curriculum_mixed_rl.parquet")
    ap.add_argument("--max-pages", type=int, default=8)
    ap.add_argument("--window", type=int, default=960, help="ramp window in prompts (steps*batch)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    out = build(df, args.max_pages, args.window, args.seed)
    out.to_parquet(args.out)

    w = out.iloc[: args.window]
    mp = (w["num_pages"] > 1).mean()
    print(f"wrote {args.out}: {len(out)} prompts")
    print(f"first {args.window} (training window): {mp:.0%} multi-page, "
          f"max {w['num_pages'].max()}p")
    print("  page mix:", w["num_pages"].value_counts().sort_index().to_dict())
    for lo, hi, lbl in [(0, 320, "steps 1-40"), (320, 640, "41-80"), (640, 960, "81-120")]:
        b = out.iloc[lo:hi]
        print(f"  {lbl}: {(b['num_pages']>1).mean():.0%} multi")


if __name__ == "__main__":
    main()
