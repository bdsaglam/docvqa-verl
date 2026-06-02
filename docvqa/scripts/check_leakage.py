#!/usr/bin/env python
"""Best-effort content overlap check between DocVQA-2026 and a training corpus.

Compares perceptual hashes (average-hash) of rendered pages. Any MMLongBench
doc with a page within HAMMING<=THRESH of a DocVQA-2026 page is flagged and
written to <train_root>/exclude_doc_ids.txt (the prepare_data adapter excludes
these). Conservative: false positives only cost us a few training docs.

Usage:
    python docvqa/scripts/check_leakage.py \\
        --ref-pages-root ~/repos/docvqa/data/docvqa-2026/val/pages \\
        --train-root data/mmlongbench-doc/train \\
        --thresh 4
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000


def ahash(path: Path, size: int = 8) -> int:
    img = Image.open(path).convert("L").resize((size, size))
    px = list(img.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= avg:
            bits |= (1 << i)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-pages-root", required=True,
                    help="dir whose **/page_*.png are the DocVQA-2026 pages")
    ap.add_argument("--train-root", required=True,
                    help="data/<dataset>/<split> with docs/<doc_id>/pages/*.png")
    ap.add_argument("--thresh", type=int, default=4)
    args = ap.parse_args()

    ref_hashes = [ahash(p) for p in Path(args.ref_pages_root).rglob("page_*.png")]
    print(f"[ref] {len(ref_hashes)} reference pages")

    flagged: set[str] = set()
    docs_root = Path(args.train_root) / "docs"
    for doc_dir in sorted(docs_root.iterdir()):
        if not doc_dir.is_dir():
            continue
        for page in (doc_dir / "pages").glob("page_*.png"):
            h = ahash(page)
            if any(hamming(h, r) <= args.thresh for r in ref_hashes):
                flagged.add(doc_dir.name)
                break

    out = Path(args.train_root) / "exclude_doc_ids.txt"
    out.write_text("\n".join(sorted(flagged)) + ("\n" if flagged else ""))
    print(f"[result] flagged {len(flagged)} docs as possible overlap -> {out}")
    for d in sorted(flagged):
        print("   ", d)


if __name__ == "__main__":
    main()
