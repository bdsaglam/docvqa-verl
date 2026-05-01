#!/usr/bin/env python
"""Materialize per-document working directories for DocVQA-2026.

Source 1: ~/repos/docvqa/data/{val,test}/{ocr,bm25}/<doc_id>/...
          (pre-computed OCR markdown + BM25 indexes; reused as-is).
Source 2: HuggingFace dataset `VLR-CVC/DocVQA-2026` (one row per document,
          with a `document` field that is a list of page images).

Output:
  data/{split}/docs/{doc_id}/
    metadata.json           (extended copy of source: adds doc_category)
    pages/page_*.png        (rendered from HF row's `document` images)
    ocr/page_*.md           (copied from source 1)
    bm25/...                (copied from source 1)
  data/{split}/questions.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image
from datasets import load_dataset

# DocVQA test scans can exceed PIL's default ~178MP cap (some hit 246MP).
# Match the convention in ~/repos/docvqa/ (solvers + scripts use 500_000_000).
Image.MAX_IMAGE_PIXELS = 500_000_000


SRC_DOCVQA_DATA = Path.home() / "repos" / "docvqa" / "data"


def _materialize_doc(row: Any, split: str, docs_dir: Path) -> dict:
    """Build per-doc directory; idempotent. Returns extended metadata dict."""
    doc_id = row["doc_id"]
    category = row.get("doc_category", "unknown")
    doc_out = docs_dir / doc_id
    pages_dir = doc_out / "pages"
    ocr_dir = doc_out / "ocr"
    bm25_dir = doc_out / "bm25"
    pages_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(exist_ok=True)
    bm25_dir.mkdir(exist_ok=True)

    src_split = SRC_DOCVQA_DATA / split
    src_ocr = src_split / "ocr" / doc_id
    src_bm25 = src_split / "bm25" / doc_id

    if src_ocr.exists():
        for p in src_ocr.iterdir():
            if p.is_file() and p.name != "metadata.json":
                dst = ocr_dir / p.name
                if not dst.exists():
                    shutil.copy(p, dst)
    if src_bm25.exists():
        for p in src_bm25.iterdir():
            if p.is_file():
                dst = bm25_dir / p.name
                if not dst.exists():
                    shutil.copy(p, dst)

    images = row.get("document") or []
    for i, img in enumerate(images):
        out_path = pages_dir / f"page_{i}.png"
        if out_path.exists():
            continue
        if isinstance(img, Image.Image):
            img.save(out_path, format="PNG")

    src_meta_path = src_ocr / "metadata.json"
    if src_meta_path.exists():
        meta = json.loads(src_meta_path.read_text())
    else:
        meta = {"doc_id": doc_id, "num_pages": len(images)}
    meta["doc_category"] = category
    meta["source_dataset"] = f"docvqa-2026-{split}"
    (doc_out / "metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def _materialize_split(split: str, out_root: Path) -> None:
    docs_dir = out_root / split / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("VLR-CVC/DocVQA-2026", split=split)
    questions: list[dict] = []
    seen_docs = 0

    for row in ds:
        row_d: Any = row
        _materialize_doc(row_d, split, docs_dir)
        seen_docs += 1
        doc_id = row_d["doc_id"]
        category = row_d.get("doc_category", "unknown")
        doc_dir_abs = (docs_dir / doc_id).resolve()

        qs = row_d.get("questions") or {}
        ans = row_d.get("answers") or {}
        q_ids = qs.get("question_id") or []
        q_texts = qs.get("question") or []
        a_ids = ans.get("question_id") or []
        a_texts = ans.get("answer") or []
        ans_lookup = {qid: a for qid, a in zip(a_ids, a_texts, strict=False)}

        for qid, qtext in zip(q_ids, q_texts, strict=False):
            gold = ans_lookup.get(qid)
            # Test split has answers stubbed as the literal string "NULL".
            if gold == "NULL":
                gold = None
            data_source = f"docvqa-2026-{split}"
            # Schema is shaped for both human inspection AND verl's RLHFDataset
            # consumption (which reads JSON directly via datasets.load_dataset).
            # `prompt` is a placeholder for the prompt-length filter — the agent
            # loop builds the real prompt from `question`/`category`/`doc_dir`.
            questions.append({
                "question_id": qid,
                "doc_id": doc_id,
                "question": qtext,
                "answer": gold,
                "category": category,
                "source_dataset": data_source,
                "doc_dir": str(doc_dir_abs),
                # --- verl-required fields -------------------------------------
                "prompt": [{"role": "user", "content": qtext}],
                "data_source": data_source,
                "reward_model": {"style": "rule", "ground_truth": gold or ""},
                "extra_info": {
                    "split": split,
                    "category": category,
                    "doc_id": doc_id,
                    "question_id": qid,
                },
            })

    out_questions = out_root / split / "questions.json"
    out_questions.write_text(json.dumps(questions, indent=2, ensure_ascii=False))
    print(f"[{split}] wrote {len(questions)} questions across {seen_docs} docs -> {out_questions}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("data"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _materialize_split(split, args.out_root)


if __name__ == "__main__":
    main()
