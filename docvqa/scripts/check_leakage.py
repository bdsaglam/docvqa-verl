#!/usr/bin/env python
"""Best-effort content-overlap check between DocVQA-2026 and a training corpus.

Why dhash + multi-page agreement: an 8x8 average-hash on text-heavy / mostly-
white document pages collides for almost everything (an earlier version flagged
111/115 docs). We instead use a 256-bit difference hash (dhash, resolution-
invariant) with a STRICT per-page threshold, and flag a training doc only when
SEVERAL of its pages each closely match a reference page — a genuine duplicate
document shares many pages, a coincidental blank page does not.

Writes the flagged ids to <train-root>/exclude_doc_ids.txt (the prepare_data
adapter drops them on the next re-emit). Always prints the per-doc match-count
distribution so the thresholds can be sanity-checked.

Usage:
    python docvqa/scripts/check_leakage.py \\
        --ref-pages-root data/docvqa-2026/val/docs \\
        --train-root data/mmlongbench-doc/train \\
        --page-thresh 16 --min-pages 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


def dhash(path: Path, size: int = 16) -> int:
    """256-bit difference hash (compares adjacent pixels left->right)."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.getdata())  # row-major, width = size+1
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _min_hamming(h: int, refs: list[int]) -> int:
    best = 10_000
    for r in refs:
        d = hamming(h, r)
        if d < best:
            best = d
            if best == 0:
                break
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-pages-root", required=True,
                    help="dir whose **/page_*.png are the DocVQA-2026 pages")
    ap.add_argument("--train-root", required=True,
                    help="data/<dataset>/<split> with docs/<doc_id>/pages/*.png")
    ap.add_argument("--page-thresh", type=int, default=16,
                    help="max dhash hamming (of 256) for a page to count as a match")
    ap.add_argument("--min-pages", type=int, default=3,
                    help="min matching pages for a doc to be flagged as overlap")
    args = ap.parse_args()

    refs = [dhash(p) for p in sorted(Path(args.ref_pages_root).rglob("page_*.png"))]
    print(f"[ref] {len(refs)} reference pages")

    docs_root = Path(args.train_root) / "docs"
    dist: dict[int, int] = {}
    flagged: list[tuple[str, int, int]] = []
    for doc_dir in sorted(d for d in docs_root.iterdir() if d.is_dir()):
        pages = sorted((doc_dir / "pages").glob("page_*.png"))
        matches = sum(1 for p in pages if _min_hamming(dhash(p), refs) <= args.page_thresh)
        dist[matches] = dist.get(matches, 0) + 1
        if matches >= args.min_pages:
            flagged.append((doc_dir.name, matches, len(pages)))

    print("[match-count distribution] {matching_pages: n_docs} =",
          dict(sorted(dist.items())))
    out = Path(args.train_root) / "exclude_doc_ids.txt"
    out.write_text("\n".join(name for name, _, _ in flagged)
                   + ("\n" if flagged else ""))
    print(f"[result] flagged {len(flagged)} docs (>= {args.min_pages} matching "
          f"pages @ hamming<={args.page_thresh}) -> {out}")
    for name, m, total in flagged:
        print(f"    {name}: {m}/{total} pages match")


if __name__ == "__main__":
    main()
