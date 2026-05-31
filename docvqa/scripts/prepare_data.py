#!/usr/bin/env python
"""Materialize per-document working directories for DocVQA-family datasets.

Layout:
    data/<dataset>/<split>/
        questions.json            # canonical full split (every question)
        train.json                # if split has gold; complement of heldout
        heldout.json              # if split has gold; 1 doc/category by smallest doc_id
        docs/<doc_id>/
            metadata.json
            pages/page_*.png

The ``rvlm_minimal_solver`` scaffold we mirror only consumes page images
(``pages``); the older OCR/BM25 sidecar dirs were removed when the scaffold
dropped the ``search`` tool.

Each row in *.json carries a globally unique
    record_id = "<dataset>:<original_split>:<doc_id>:<question_id>"

so trajectories dumped during training can be traced back to their source
unambiguously even when datasets/splits are concatenated for verl ingest.

Adapters live below in `ADAPTERS`. Each one knows how to populate
docs/ from its source corpus and emit raw question rows; the common
post-processing pass (verl schema, train/heldout split) is shared.

Usage:
    python docvqa/scripts/prepare_data.py --dataset docvqa-2026 --splits val test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image
from datasets import load_dataset

# DocVQA test scans can exceed PIL's default ~178MP cap (some hit 246MP).
# Match the convention in ~/repos/docvqa/ (solvers + scripts use 500_000_000).
Image.MAX_IMAGE_PIXELS = 500_000_000


# ---------------------------------------------------------------------------
# Common helpers (dataset-agnostic)
# ---------------------------------------------------------------------------

def _make_record_id(dataset: str, split: str, doc_id: str, question_id: str) -> str:
    return f"{dataset}:{split}:{doc_id}:{question_id}"


def _build_row(
    *,
    dataset: str,
    split: str,
    doc_id: str,
    question_id: str,
    question: str,
    answer: str | None,
    category: str,
    doc_dir_abs: Path,
) -> dict:
    """Final row schema: human-readable + verl-ready."""
    record_id = _make_record_id(dataset, split, doc_id, question_id)
    data_source = f"{dataset}-{split}"
    return {
        "record_id": record_id,
        "dataset": dataset,
        "split": split,
        "doc_id": doc_id,
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "category": category,
        "doc_dir": str(doc_dir_abs),
        # --- verl-required fields ------------------------------------------
        # `prompt` is a placeholder for verl's prompt-length filter — the
        # agent loop builds the real prompt from `question`/`category`/`doc_dir`.
        "prompt": [{"role": "user", "content": question}],
        "data_source": data_source,
        "reward_model": {"style": "rule", "ground_truth": answer or ""},
        "extra_info": {
            "record_id": record_id,
            "dataset": dataset,
            "split": split,
            "doc_id": doc_id,
            "question_id": question_id,
            "category": category,
        },
    }


def _split_train_heldout(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Deterministic stratified split: smallest doc_id per category to heldout.

    Returns (train_rows, heldout_rows). If no rows have a gold answer
    (e.g. test split), returns ([], []) — caller decides what to write.
    """
    if not rows or all(r.get("answer") in (None, "") for r in rows):
        return [], []
    by_cat: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["doc_id"] not in by_cat[r["category"]]:
            by_cat[r["category"]].append(r["doc_id"])
    heldout_doc_ids: set[str] = set()
    for doc_ids in by_cat.values():
        heldout_doc_ids.add(sorted(doc_ids)[0])
    train = [r for r in rows if r["doc_id"] not in heldout_doc_ids]
    heldout = [r for r in rows if r["doc_id"] in heldout_doc_ids]
    return train, heldout


def _write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


def _emit_split(split_dir: Path, rows: list[dict]) -> None:
    """Write questions.json plus optional train.json / heldout.json."""
    _write_json(split_dir / "questions.json", rows)
    n_docs = len({r["doc_id"] for r in rows})
    print(f"  questions.json: {len(rows)} Qs / {n_docs} docs")

    train, heldout = _split_train_heldout(rows)
    if train or heldout:
        _write_json(split_dir / "train.json", train)
        _write_json(split_dir / "heldout.json", heldout)
        print(f"  train.json:   {len(train)} Qs / {len({r['doc_id'] for r in train})} docs")
        print(f"  heldout.json: {len(heldout)} Qs / {len({r['doc_id'] for r in heldout})} docs")


# ---------------------------------------------------------------------------
# Adapter: docvqa-2026 (VLR-CVC/DocVQA-2026)
# ---------------------------------------------------------------------------

def _materialize_docvqa_2026_doc(row: Any, split: str, docs_dir: Path) -> None:
    """Idempotently materialize a single doc directory for DocVQA-2026."""
    doc_id = row["doc_id"]
    category = row.get("doc_category", "unknown")
    doc_out = docs_dir / doc_id
    pages_dir = doc_out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    images = row.get("document") or []
    for i, img in enumerate(images):
        out_path = pages_dir / f"page_{i}.png"
        if out_path.exists():
            continue
        if isinstance(img, Image.Image):
            img.save(out_path, format="PNG")

    meta = {
        "doc_id": doc_id,
        "num_pages": len(images),
        "doc_category": category,
        "dataset": "docvqa-2026",
        "split": split,
    }
    (doc_out / "metadata.json").write_text(json.dumps(meta, indent=2))


def adapter_docvqa_2026(split: str, split_dir: Path) -> list[dict]:
    """Build docs/ and return raw rows for VLR-CVC/DocVQA-2026."""
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("VLR-CVC/DocVQA-2026", split=split)
    rows: list[dict] = []
    for row in ds:
        row_d: Any = row
        _materialize_docvqa_2026_doc(row_d, split, docs_dir)

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
            if gold == "NULL":  # test split sentinel
                gold = None
            rows.append(_build_row(
                dataset="docvqa-2026",
                split=split,
                doc_id=doc_id,
                question_id=qid,
                question=qtext,
                answer=gold,
                category=category,
                doc_dir_abs=doc_dir_abs,
            ))
    return rows


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Callable[[str, Path], list[dict]]] = {
    "docvqa-2026": adapter_docvqa_2026,
    # Future:
    #   "docvqa-1.0": adapter_docvqa_1_0,
    #   "mp-docvqa": adapter_mp_docvqa,
    #   "infographic-vqa": adapter_infographic_vqa,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="docvqa-2026", choices=sorted(ADAPTERS))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out-root", type=Path, default=Path("data"))
    args = ap.parse_args()

    adapter = ADAPTERS[args.dataset]
    for split in args.splits:
        split_dir = args.out_root / args.dataset / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{args.dataset}/{split}] materializing -> {split_dir}")
        rows = adapter(split, split_dir)
        _emit_split(split_dir, rows)


if __name__ == "__main__":
    main()
