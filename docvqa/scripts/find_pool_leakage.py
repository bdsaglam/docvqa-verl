#!/usr/bin/env python
"""Detect pool images that are (near-)duplicates of DocVQA-2026 val/test pages.

DocVQA-2026 doc_ids carry no provenance, so we match by image *content* via a
perceptual hash (dHash) — robust to PNG re-encoding, unlike a raw byte hash.
Cheap: hash the small reference set (DocVQA-2026 val+test pages) once, then scan
each pool dataset's pages. Reports exact-hash matches (distance 0) and near-dups
(Hamming <= --threshold), with the colliding doc/page on each side.

Usage:
    python docvqa/scripts/find_pool_leakage.py \
        --pool-datasets docvqa-sp/validation infographicvqa/validation \
                        mp-docvqa/val slidevqa/train \
        --threshold 4 --out outputs/pool_leakage.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000
DATA = Path("data")


def dhash(path: Path, size: int = 8) -> int | None:
    try:
        img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        base = r * (size + 1)
        for c in range(size):
            bits = (bits << 1) | (1 if px[base + c] > px[base + c + 1] else 0)
    return bits


def _pages(split_dir: Path):
    for png in sorted(split_dir.glob("docs/*/pages/*.png")):
        yield png.parent.parent.name, png.name, png  # doc_id, page_name, path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", nargs="+",
                    default=["docvqa-2026/val", "docvqa-2026/test"],
                    help="<dataset>/<split> dirs whose pages are the leakage target")
    ap.add_argument("--pool-datasets", nargs="+",
                    default=["docvqa-sp/validation", "infographicvqa/validation",
                             "mp-docvqa/val", "slidevqa/train"],
                    help="<dataset>/<split> pool dirs to check (default: CVC-lineage)")
    ap.add_argument("--threshold", type=int, default=4, help="max Hamming for a near-dup")
    ap.add_argument("--out", type=Path, default=Path("outputs/pool_leakage.json"))
    args = ap.parse_args()

    # Reference hashes (small side).
    ref: dict[int, list[str]] = {}
    ref_list: list[tuple[int, str]] = []
    for rel in args.reference:
        d = DATA / rel
        if not d.exists():
            print(f"  (skip reference {rel}: not materialized)")
            continue
        n = 0
        for doc_id, page, p in _pages(d):
            h = dhash(p)
            if h is None:
                continue
            tag = f"{rel}:{doc_id}:{page}"
            ref.setdefault(h, []).append(tag)
            ref_list.append((h, tag))
            n += 1
        print(f"  reference {rel}: {n} pages hashed")
    print(f"reference total: {len(ref_list)} pages, {len(ref)} distinct hashes")

    report = {"reference_pages": len(ref_list), "threshold": args.threshold, "datasets": {}}
    for rel in args.pool_datasets:
        d = DATA / rel
        if not d.exists():
            print(f"{rel}: not materialized, skip")
            continue
        exact, near = [], []
        n_imgs = 0
        matched_docs: set[str] = set()
        for doc_id, page, p in _pages(d):
            h = dhash(p)
            if h is None:
                continue
            n_imgs += 1
            if h in ref:  # exact dHash match (re-encode-robust)
                exact.append({"pool": f"{doc_id}:{page}", "ref": ref[h], "dist": 0})
                matched_docs.add(doc_id)
                continue
            if args.threshold > 0:
                best = min(((bin(h ^ rh).count("1"), tag) for rh, tag in ref_list),
                           default=(99, None))
                if best[0] <= args.threshold:
                    near.append({"pool": f"{doc_id}:{page}", "ref": best[1], "dist": best[0]})
                    matched_docs.add(doc_id)
        report["datasets"][rel] = {
            "pool_images": n_imgs, "exact_matches": len(exact),
            "near_matches": len(near), "matched_docs": sorted(matched_docs),
            "exact": exact[:50], "near": near[:50],
        }
        print(f"{rel}: {n_imgs} imgs | exact={len(exact)} near(<= {args.threshold})={len(near)} "
              f"| {len(matched_docs)} pool docs implicated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nfull report -> {args.out}")


if __name__ == "__main__":
    main()
