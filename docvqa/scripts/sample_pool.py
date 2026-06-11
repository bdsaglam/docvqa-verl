#!/usr/bin/env python
"""Draw a weighted, balanced prompt set from the pool manifest.

Reads ``docvqa/train/pool.yaml``; for each dataset whose ``questions.json`` is
materialized, draws a seeded random sample of ``n_sample`` rows (capped at
availability). Writes per-source samples to ``data/pool/sampled/<name>.json``
and a shuffled combined set to ``data/pool/prompts.json``. Sources not yet
prepared are skipped with a warning (so this is safe to run after Phase 1
before Phase 2 exists).

The per-source ``n_sample`` in the manifest is the weight: equal values give a
balanced mix; tune individual values to reshape it. This de-dominates large
sources (e.g. MapQA's ~483k questions -> n_sample).

Usage:
    python docvqa/scripts/sample_pool.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def _sample_rows(rows: list, n: int | None, seed: int) -> list:
    """Deterministic sample of up to ``n`` rows. n=None or n>=len -> all rows (order kept)."""
    if n is None or n >= len(rows):
        return list(rows)
    return random.Random(seed).sample(rows, n)


def _source_seed(base: int, name: str) -> int:
    """Per-source seed so sources are independent yet reproducible."""
    return base + sum(ord(c) for c in name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="docvqa/train/pool.yaml")
    ap.add_argument("--out-dir", type=Path, default=Path("data/pool"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    man = yaml.safe_load(Path(args.manifest).read_text())
    sampled_dir = args.out_dir / "sampled"
    sampled_dir.mkdir(parents=True, exist_ok=True)

    combined: list = []
    for d in man["datasets"]:
        name = d["name"]
        qp = Path(d["questions"])
        n = d.get("n_sample")
        if not qp.exists():
            print(f"SKIP {name}: {qp} not prepared yet")
            continue
        rows = json.loads(qp.read_text())
        sample = _sample_rows(rows, n, _source_seed(args.seed, name))
        (sampled_dir / f"{name}.json").write_text(
            json.dumps(sample, indent=2, ensure_ascii=False))
        combined.extend(sample)
        print(f"{name}: sampled {len(sample)} / {len(rows)}")

    random.Random(args.seed).shuffle(combined)
    out = args.out_dir / "prompts.json"
    out.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"combined: {len(combined)} prompts -> {out}")


if __name__ == "__main__":
    main()
