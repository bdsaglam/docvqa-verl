#!/usr/bin/env python3
"""Build a teacher-generation pool (JSON) for rejection-sampling SFT.

Selects a stratified, multi-page-inclusive slice of the 8-source pool for the 27B
teacher to roll out on (via eval.py with the 27B as the agent LM). Rejection
sampling (make_sft_data, anls==1.0) then keeps the solved trajectories for SFT.

Why multi-page-inclusive: SFT's measured benefit (SFT-synthesis.md) is robustness
on harder docs, i.e. the multi-page navigation the single-page RL curriculum
couldn't teach. The 27B teacher CAN navigate multi-page docs, so its trajectories
over them teach the 4B that skill. The pool is bimodal (abundant 1-3p, then a jump
to 11+; near-empty 4-10p), so we draw 1p (fast/high-yield) + 2-3p ("more than one
page" signal) + a modest 11-30p slice (genuine navigation, slow to generate).

Output schema matches eval.py --questions (drops num_pages/agent_name).
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd

KEEP = ["record_id", "dataset", "split", "doc_id", "question_id", "question",
        "answer", "category", "doc_dir", "prompt", "data_source",
        "reward_model", "extra_info"]


def stratified(frame: pd.DataFrame, k: int) -> list[dict]:
    by = dict(tuple(frame.groupby("dataset")))
    srcs = sorted(by)
    pools = {s: by[s].sample(frac=1, random_state=0).to_dict("records") for s in srcs}
    out: list[dict] = []
    i = 0
    while len(out) < k and any(pools.values()):
        s = srcs[i % len(srcs)]
        i += 1
        if pools[s]:
            out.append(pools[s].pop())
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, np.ndarray)):
        return [_clean(x) for x in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pool/curriculum_rl.parquet")
    ap.add_argument("--out", default="data/pool/teacher_gen_pool.json")
    ap.add_argument("--n-single", type=int, default=950)
    ap.add_argument("--n-multi-small", type=int, default=400)  # 2-3p
    ap.add_argument("--n-multi-real", type=int, default=150)   # 11-30p
    args = ap.parse_args()

    df = pd.read_parquet(args.inp)
    print("pool buckets:", {f"{lo}-{hi}p": int(((df.num_pages >= lo) & (df.num_pages <= hi)).sum())
                            for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 999)]})

    sng = stratified(df[df.num_pages == 1], args.n_single)
    ms = stratified(df[(df.num_pages >= 2) & (df.num_pages <= 3)], args.n_multi_small)
    mr = stratified(df[(df.num_pages >= 11) & (df.num_pages <= 30)], args.n_multi_real)
    print(f"drawn: single={len(sng)} multi2-3p={len(ms)} multi11-30p={len(mr)}")

    rng = random.Random(0)
    for x in (sng, ms, mr):
        rng.shuffle(x)

    # Weave: 2 single : 1 small-multi for fast early yield; drip slow nav docs every ~9.
    # Guarantee forward progress every iteration (else once single+small drain while
    # nav docs remain and the modulo misses, the loop would spin forever).
    comb: list[dict] = []
    si = mi = ri = 0
    while si < len(sng) or mi < len(ms) or ri < len(mr):
        progressed = False
        for _ in range(2):
            if si < len(sng):
                comb.append(sng[si]); si += 1; progressed = True
        if mi < len(ms):
            comb.append(ms[mi]); mi += 1; progressed = True
        drained = si >= len(sng) and mi >= len(ms)
        if ri < len(mr) and (len(comb) % 9 == 0 or drained or not progressed):
            comb.append(mr[ri]); ri += 1; progressed = True
        if not progressed:
            break

    out = [_clean({k: r.get(k) for k in KEEP}) for r in comb]
    json.dump(out, open(args.out, "w"))

    idmap = dict(zip(df.record_id, df.num_pages))
    pg = [idmap.get(r["record_id"]) for r in out]
    print(f"wrote {args.out}: {len(out)} prompts | "
          f"multi-frac {sum(1 for p in pg if p and p > 1) / len(pg):.2f} | "
          f">=11p {sum(1 for p in pg if p and p >= 11)}")


if __name__ == "__main__":
    main()
