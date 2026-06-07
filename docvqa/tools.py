"""Host-side tool handlers.

Mirrors the ``rvlm_minimal_solver`` tool surface in ``~/repos/docvqa``: just
``batch_look``. Earlier revisions also shipped a BM25 ``search`` over OCR;
the deployment-time scaffold dropped it (recursive VLM perception is the
load-bearing mechanism) and we follow suit.
"""
from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path

import httpx

# Match the deployed scaffold's VLM instruction verbatim (docvqa repo
# codeact_solver.py:256-266 `vlm_predict` signature). The original wraps the call
# in a dspy.Predict whose signature instruction steers the VLM toward a concise,
# exact answer; we send a raw chat completion, so we replicate that instruction as
# a system message. Without it we were forwarding the bare query, which let the VLM
# ramble / editorialize instead of returning just the answer.
# Based on the deployed scaffold's VLM instruction (docvqa repo
# codeact_solver.py:256-266) but deliberately relaxed from "output ONLY the concise
# answer": a bare answer gives the main agent nothing to cross-check, so we ask for
# the concise answer FIRST (so it's still up front) followed by a one-sentence
# rationale citing the supporting evidence. This gives the agent grounding to verify
# / reason without ballooning the observation.
_VLM_INSTRUCTION = (
    "Analyze the image content strictly to answer the query. "
    "Transcribe numbers and characters exactly. "
    "When a label is separated from the item it identifies, trace any visual "
    "connector (leader line, arrow, callout, alignment) to determine which item "
    "it refers to. "
    "Give the concise final answer first, then a brief one-sentence rationale "
    "citing the supporting evidence in the image. "
    "If the information is missing, answer 'Unknown'."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S)


def _strip_thinking(text: str) -> str:
    """Remove any reasoning content the VLM leaks despite enable_thinking=False.

    Handles a complete ``<think>...</think>`` block and the Qwen variant where the
    opening tag is template-supplied (so only a trailing ``</think>`` appears)."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


async def _one_look(
    client: httpx.AsyncClient, base_url: str, model_id: str,
    path: str, query: str,
) -> str:
    img_b64 = base64.b64encode(Path(path).read_bytes()).decode()
    # Mirror the deployed scaffold's VLM call (docvqa repo
    # configs/vlm/qwen-3_5-27b-vllm-local.yaml): thinking DISABLED, temp 0.3,
    # top_k 20, plus the concise-answer system instruction above. Without
    # enable_thinking=false the Qwen3.5 VLM defaults to thinking, which both
    # mismatches deployment and adds tens of seconds of reasoning to every
    # perception call (it dominated rollout wall-clock). max_tokens capped at
    # 2048: thinking-off perception answers are short, so this is ample headroom
    # while bounding pathological long VLM replies.
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _VLM_INSTRUCTION},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": query},
            ]},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return _strip_thinking(resp.json()["choices"][0]["message"]["content"])
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
