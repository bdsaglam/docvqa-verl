"""Host-side tool handlers.

Mirrors the ``rvlm_minimal_solver`` tool surface in ``~/repos/docvqa``: just
``batch_look``. Earlier revisions also shipped a BM25 ``search`` over OCR;
the deployment-time scaffold dropped it (recursive VLM perception is the
load-bearing mechanism) and we follow suit.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path

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
