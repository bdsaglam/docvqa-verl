"""Host-side tool handlers: batch_look (HTTP→VLM) and search (BM25)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

# Process-local cache: doc_dir -> (BM25 retriever, list of chunk dicts)
_BM25_CACHE: dict[str, tuple[Any, list[dict]]] = {}


def clear_bm25_cache() -> None:
    """Drop the BM25 cache (used by tests)."""
    _BM25_CACHE.clear()


def _load_bm25(doc_dir: str) -> tuple[Any, list[dict]]:
    if doc_dir in _BM25_CACHE:
        return _BM25_CACHE[doc_dir]
    bm25_dir = Path(doc_dir) / "bm25"
    retriever = bm25s.BM25.load(str(bm25_dir), load_corpus=False)
    chunks = json.loads((bm25_dir / "chunks.json").read_text())
    _BM25_CACHE[doc_dir] = (retriever, chunks)
    return retriever, chunks


def search(doc_dir: str, query: str, k: int = 5) -> list[dict]:
    """BM25 search over the document's per-page OCR.

    Returns: list of {"page": int, "score": float, "text": str}, sorted by
    decreasing score, with score > 0.
    """
    retriever, chunks = _load_bm25(doc_dir)
    tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
    n = min(k, len(chunks))
    indices, scores = retriever.retrieve(tokens, k=n)
    out = []
    for idx, score in zip(indices[0], scores[0]):
        if score <= 0:
            continue
        c = chunks[idx]
        out.append({"page": c["page"], "score": round(float(score), 2), "text": c["text"]})
    return out
