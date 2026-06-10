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
import ast
import json
import shutil
import sys
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


def _materialize_single_image_doc(
    *, doc_id: str, image: Image.Image, category: str,
    dataset: str, split: str, docs_dir: Path,
) -> Path:
    """Save a one-page doc dir for a single-image dataset. Returns abs doc_dir."""
    doc_out = docs_dir / doc_id
    pages_dir = doc_out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = pages_dir / "page_0.png"
    if not out_path.exists():
        image.convert("RGB").save(out_path, format="PNG")
    (doc_out / "metadata.json").write_text(json.dumps({
        "doc_id": doc_id, "num_pages": 1, "doc_category": category,
        "dataset": dataset, "split": split,
    }, indent=2))
    return doc_out.resolve()


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
# Adapter: mmlongbench-doc (yubo2333/MMLongBench-Doc)
# ---------------------------------------------------------------------------

_MMLB_REPO = "yubo2333/MMLongBench-Doc"
_MMLB_DPI = 150
_MMLB_MAX_PAGES = 80


def _mmlb_question_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}::q{idx}"


def _mmlb_rows_for_doc(split: str, hf_rows: list[dict], doc_dir: Path) -> list[dict]:
    """Build canonical question rows for one MMLongBench doc (no I/O).

    Note: ``answer == "Not answerable"`` is REAL gold for this dataset, not a
    sentinel, so it is passed through unchanged (unlike DocVQA-2026's "NULL").
    """
    rows = []
    for i, r in enumerate(hf_rows):
        rows.append(_build_row(
            dataset="mmlongbench-doc",
            split=split,
            doc_id=r["doc_id"],
            question_id=_mmlb_question_id(r["doc_id"], i),
            question=r["question"],
            answer=r["answer"],
            category=r["doc_type"],
            doc_dir_abs=doc_dir,
        ))
    return rows


def _mmlb_render_pdf(pdf_path: Path, out_dir: Path) -> int:
    """Render PDF pages to out_dir/page_<i>.png (idempotent). Returns page count."""
    import pypdfium2 as pdfium

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_total = len(pdf)
    n_render = min(n_total, _MMLB_MAX_PAGES)
    scale = _MMLB_DPI / 72.0
    for i in range(n_render):
        png = out_dir / f"page_{i}.png"
        if png.exists():
            continue
        img = pdf[i].render(scale=scale).to_pil()
        img.save(png, format="PNG", optimize=True)
    pdf.close()
    return n_render


def adapter_mmlongbench_doc(split: str, split_dir: Path) -> list[dict]:
    """Build docs/ and return raw rows for yubo2333/MMLongBench-Doc.

    The dataset has a single ``train`` split (~1091 Q / 135 docs). PDFs are
    fetched from the HF dataset repo's ``documents/`` dir and rendered locally.
    """
    from huggingface_hub import hf_hub_download

    docs_dir = split_dir / "docs"

    exclude: set[str] = set()
    exclude_file = split_dir.parent / "exclude_doc_ids.txt"
    if exclude_file.exists():
        exclude = {ln.strip() for ln in exclude_file.read_text().splitlines() if ln.strip()}

    ds = load_dataset(_MMLB_REPO, split="train")
    by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in ds:
        if r["doc_id"] in exclude:
            continue
        by_doc[r["doc_id"]].append(r)

    all_rows: list[dict] = []
    failed: list[str] = []
    for doc_id, hf_rows in by_doc.items():
        # Absolute path, consistent with adapter_docvqa_2026 — a relative
        # doc_dir would break the agent loop when train/rollout runs from
        # another CWD.
        doc_dir = (docs_dir / doc_id).resolve()
        try:
            pdf_path = Path(hf_hub_download(_MMLB_REPO, f"documents/{doc_id}",
                                            repo_type="dataset"))
            num_pages = _mmlb_render_pdf(pdf_path, doc_dir / "pages")
        except Exception as e:
            # One bad remote PDF (corrupt download, render error) must not abort
            # the whole corpus. Drop the partial dir, skip the doc, keep going.
            print(f"[mmlb] SKIP {doc_id}: {type(e).__name__}: {e}", file=sys.stderr)
            shutil.rmtree(doc_dir, ignore_errors=True)
            failed.append(doc_id)
            continue
        (doc_dir / "metadata.json").write_text(json.dumps({
            "doc_id": doc_id,
            "num_pages": num_pages,
            "doc_category": hf_rows[0]["doc_type"],
            "dataset": "mmlongbench-doc",
            "split": split,
        }))
        all_rows.extend(_mmlb_rows_for_doc(split, hf_rows, doc_dir))
    if failed:
        print(f"[mmlb] skipped {len(failed)} docs that failed to "
              f"download/render: {failed}", file=sys.stderr)
    return all_rows


# ---------------------------------------------------------------------------
# Adapter: docvqa-sp (lmms-lab/DocVQA, single-page)
# ---------------------------------------------------------------------------

def _gold_answer_str(answers) -> str | None:
    """Single alias -> bare str; multiple -> repr(list) for the ast.literal_eval scorer."""
    if isinstance(answers, str):
        return answers
    if not answers:
        return None
    answers = [str(a) for a in answers]
    return answers[0] if len(answers) == 1 else repr(answers)


def adapter_docvqa_sp(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        doc_id = str(r["docId"])
        if doc_id not in seen_docs:
            _materialize_single_image_doc(
                doc_id=doc_id, image=r["image"], category="business_report",
                dataset="docvqa-sp", split=split, docs_dir=docs_dir)
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="docvqa-sp", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="business_report",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows


def adapter_infographicvqa(split: str, split_dir: Path) -> list[dict]:
    """Build docs/ and return raw rows for lmms-lab/DocVQA 'InfographicVQA' config.

    Schema NOTE: unlike the 'DocVQA' (single-page) config, this config has NO
    ``docId`` field — its keys are ``questionId, question, answers, answer_type,
    image, image_url, operation/reasoning, ocr, data_split``. Questions on the
    same infographic share the same ``image_url``, so we derive a stable,
    filesystem-safe ``doc_id`` from it (short blake2b hash) to group questions
    per document and keep the single-image-doc materializer semantics.
    """
    import hashlib

    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        doc_key = r.get("image_url") or f"q{r['questionId']}"
        doc_id = "infvqa_" + hashlib.blake2b(
            doc_key.encode("utf-8"), digest_size=8).hexdigest()
        if doc_id not in seen_docs:
            _materialize_single_image_doc(
                doc_id=doc_id, image=r["image"], category="infographics",
                dataset="infographicvqa", split=split, docs_dir=docs_dir)
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="infographicvqa", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="infographics",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows


def adapter_chartqa(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
    rows: list[dict] = []
    for idx, r in enumerate(ds):
        doc_id = f"chartqa_{split}_{idx}"
        label = r.get("label")
        answer = _gold_answer_str(label if isinstance(label, list) else [label])
        _materialize_single_image_doc(
            doc_id=doc_id, image=r["image"], category="science_poster",
            dataset="chartqa", split=split, docs_dir=docs_dir)
        rows.append(_build_row(
            dataset="chartqa", split=split, doc_id=doc_id, question_id=str(idx),
            question=r["query"], answer=answer, category="science_poster",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows


# ---------------------------------------------------------------------------
# Adapter: mapqa (nimapourjafar/mm_mapqa)
# ---------------------------------------------------------------------------

def _mapqa_qa_from_data(data) -> list[tuple[str, str]]:
    """Extract all (question, answer) pairs from mm_mapqa's `data` field.

    `data` is a list of turn dicts with keys {data, modality, role}: a leading
    `modality == 'image'` turn (the image placeholder), then interleaved text
    turns alternating role 'user' (question) / 'assistant' (answer). A single
    row carries MANY Q/A pairs over one image, so we return every pair. Each
    user text turn is paired with the next assistant text turn.
    """
    pairs: list[tuple[str, str]] = []
    pending_q: str | None = None
    for turn in data:
        if turn.get("modality") != "text":
            continue
        role = turn.get("role")
        text = str(turn.get("data", "")).strip()
        if role == "user":
            pending_q = text
        elif role == "assistant" and pending_q is not None:
            pairs.append((pending_q, text))
            pending_q = None
    return pairs


def adapter_mapqa(split: str, split_dir: Path) -> list[dict]:
    """Build docs/ and return raw rows for nimapourjafar/mm_mapqa.

    Single ``train`` split. Each HF row is one map image carrying multiple Q/A
    pairs; we materialize the image once and emit one row per Q/A pair.
    """
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nimapourjafar/mm_mapqa", split=split)
    rows: list[dict] = []
    for idx, r in enumerate(ds):
        doc_id = f"mapqa_{split}_{idx}"
        pairs = _mapqa_qa_from_data(r["data"])
        if not pairs:
            continue
        _materialize_single_image_doc(
            doc_id=doc_id, image=r["images"][0], category="maps",
            dataset="mapqa", split=split, docs_dir=docs_dir)
        doc_dir_abs = (docs_dir / doc_id).resolve()
        for qi, (question, answer) in enumerate(pairs):
            rows.append(_build_row(
                dataset="mapqa", split=split, doc_id=doc_id,
                question_id=f"{idx}_{qi}", question=question,
                answer=answer or None, category="maps",
                doc_dir_abs=doc_dir_abs))
    return rows


# ---------------------------------------------------------------------------
# Adapter: mp-docvqa (lmms-lab/MP-DocVQA), <= max_pages filter
# ---------------------------------------------------------------------------

def _page_count(page_ids) -> int:
    """Page count from MP-DocVQA's ``page_ids`` without decoding images.

    ``page_ids`` arrives as a stringified Python list (e.g. "['p0', 'p1']");
    handle a real list too. Returns 0 for None / unparseable.
    """
    if page_ids is None:
        return 0
    if isinstance(page_ids, str):
        try:
            page_ids = ast.literal_eval(page_ids)
        except (ValueError, SyntaxError):
            return 0
    return len(page_ids)


def _mp_doc_images(row) -> list:
    """Collect image_1..image_20 up to the first None (doc's page images)."""
    imgs = []
    for i in range(1, 21):
        v = row.get(f"image_{i}")
        if v is None:
            break
        imgs.append(v)
    return imgs


def adapter_mp_docvqa(split: str, split_dir: Path, max_pages: int = 3) -> list[dict]:
    """Build docs/ and return raw rows for lmms-lab/MP-DocVQA.

    One HF row per (doc, question). Keep only docs with ``<= max_pages`` pages,
    decided from ``page_ids`` WITHOUT decoding images. Materialize each doc's
    pages once; emit one row per question.
    """
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/MP-DocVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        if _page_count(r.get("page_ids")) > max_pages:
            continue
        doc_id = str(r["doc_id"])
        if doc_id not in seen_docs:
            doc_out = docs_dir / doc_id / "pages"
            doc_out.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(_mp_doc_images(r)):
                p = doc_out / f"page_{i}.png"
                if not p.exists() and isinstance(img, Image.Image):
                    img.convert("RGB").save(p, format="PNG")
            (docs_dir / doc_id / "metadata.json").write_text(json.dumps({
                "doc_id": doc_id, "num_pages": _page_count(r.get("page_ids")),
                "doc_category": "business_report", "dataset": "mp-docvqa", "split": split,
            }, indent=2))
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="mp-docvqa", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="business_report",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTERS: dict[str, Callable[[str, Path], list[dict]]] = {
    "docvqa-2026": adapter_docvqa_2026,
    "mmlongbench-doc": adapter_mmlongbench_doc,
    "docvqa-sp": adapter_docvqa_sp,
    "infographicvqa": adapter_infographicvqa,
    "chartqa": adapter_chartqa,
    "mapqa": adapter_mapqa,
    "mp-docvqa": adapter_mp_docvqa,
    # Future:
    #   "docvqa-1.0": adapter_docvqa_1_0,
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
