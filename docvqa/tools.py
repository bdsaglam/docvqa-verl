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


# ---------------------------------------------------------------------------
# batch_look — parallel VLM image-query handler
# ---------------------------------------------------------------------------
import asyncio
import base64

import httpx


async def _one_look(
    client: httpx.AsyncClient, base_url: str, model_id: str,
    path: str, query: str,
) -> str:
    img_b64 = base64.b64encode(Path(path).read_bytes()).decode()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": query},
        ]}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[VLM error: {type(e).__name__}: {e}]"


async def batch_look(
    requests: list[dict],
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
) -> list[str]:
    """Send (path, query) pairs to the VLM in parallel. Returns answers in order."""
    coros = [_one_look(client, base_url, model_id, r["path"], r["query"]) for r in requests]
    return await asyncio.gather(*coros)
