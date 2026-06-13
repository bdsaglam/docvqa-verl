"""Host-side tool handlers.

Mirrors the ``rvlm_minimal_solver`` tool surface in ``~/repos/docvqa``: just
``batch_look``. Earlier revisions also shipped a BM25 ``search`` over OCR;
the deployment-time scaffold dropped it (recursive VLM perception is the
load-bearing mechanism) and we follow suit.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time
from pathlib import Path

import httpx
from PIL import Image

Image.MAX_IMAGE_PIXELS = 500_000_000  # match sandbox; competition pages reach 240MP

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

# Qwen3.5-27B's image processor caps every image at 16,777,216 total pixels
# (preprocessor_config.json `size.longest_edge` — Qwen-VL semantics: max pixel AREA,
# = 4096x4096). Downscaling to that cap client-side is lossless w.r.t. what the VLM
# sees: the server performs the identical resize anyway, but only AFTER we have paid
# base64-encode + HTTP transfer + server-side decode at full resolution. Crops made
# by the agent happen upstream (in the sandbox, on the full-res pages from disk) and
# are small, so they pass through untouched — the survey-coarse/crop-fine loop keeps
# its full effective resolution.
_VLM_MAX_PIXELS = 16_777_216


def _prepare_image_b64(path: str) -> str:
    """Read an image and base64 it, downscaling to the VLM's pixel cap if larger.

    Runs in a worker thread (PIL decode/resize of a 240MP map takes seconds and
    must not block the AgentLoopWorker event loop). PNG is kept (no lossy step)."""
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        w, h = im.size
        if w * h > _VLM_MAX_PIXELS:
            scale = (_VLM_MAX_PIXELS / (w * h)) ** 0.5
            im = im.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.Resampling.LANCZOS,
            )
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            raw = buf.getvalue()
    return base64.b64encode(raw).decode()


# ---------------------------------------------------------------------------
# VLM endpoint pool — least-loaded, health-aware load balancing
# ---------------------------------------------------------------------------

_ENDPOINT_COOLDOWN_S = 60.0


class _Endpoint:
    __slots__ = ("base_url", "weight", "inflight", "down_until")

    def __init__(self, base_url: str, weight: float):
        self.base_url = base_url
        self.weight = weight
        self.inflight = 0
        self.down_until = 0.0


class EndpointPool:
    """Least-loaded pool of VLM base URLs (one shared instance per process).

    Spec: ``http://host:8927@2|http://host:8928@3`` — '|'-separated (hydra would
    parse commas as a list), ``@weight`` ~ #GPUs behind the URL (default 1). Each
    request goes to the live endpoint with the lowest inflight/weight, which
    self-corrects for latency asymmetry (a slower endpoint accumulates in-flight
    and organically receives less traffic). An endpoint that fails at transport
    level is benched for _ENDPOINT_COOLDOWN_S then retried automatically — so a
    server that comes up mid-run starts absorbing traffic without any restart.
    """

    def __init__(self, spec: str):
        self.endpoints: list[_Endpoint] = []
        for part in spec.split("|"):
            part = part.strip()
            if not part:
                continue
            url, _, w = part.partition("@")
            self.endpoints.append(_Endpoint(url.rstrip("/"), float(w) if w else 1.0))
        if not self.endpoints:
            raise ValueError(f"empty VLM endpoint spec: {spec!r}")

    def pick(self, exclude: tuple = ()) -> _Endpoint:
        now = time.monotonic()
        pool = [e for e in self.endpoints if e not in exclude] or self.endpoints
        live = [e for e in pool if e.down_until <= now]
        return min(live or pool, key=lambda e: e.inflight / e.weight)

    def bench(self, ep: _Endpoint) -> None:
        ep.down_until = time.monotonic() + _ENDPOINT_COOLDOWN_S


_pools: dict[str, EndpointPool] = {}


def _get_pool(spec: str) -> EndpointPool:
    pool = _pools.get(spec)
    if pool is None:
        pool = _pools[spec] = EndpointPool(spec)
    return pool


def _strip_thinking(text: str) -> str:
    """Remove any reasoning content the VLM leaks despite enable_thinking=False.

    Handles a complete ``<think>...</think>`` block and the Qwen variant where the
    opening tag is template-supplied (so only a trailing ``</think>`` appears)."""
    text = _THINK_BLOCK_RE.sub("", text)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.strip()


async def _one_look(
    client: httpx.AsyncClient, pool: EndpointPool, model_id: str,
    path: str, query: str,
) -> str:
    img_b64 = await asyncio.to_thread(_prepare_image_b64, path)
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
    # Transport failures / 5xx bench the endpoint and fail over to the next one;
    # anything else (4xx = our request's fault, JSON shape) returns the error string
    # to the agent as before — failing over would not change the outcome.
    last_err: Exception | None = None
    tried: list[_Endpoint] = []
    for _ in range(len(pool.endpoints)):
        ep = pool.pick(exclude=tuple(tried))
        ep.inflight += 1
        try:
            resp = await client.post(f"{ep.base_url}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            return _strip_thinking(resp.json()["choices"][0]["message"]["content"])
        except httpx.TransportError as e:
            pool.bench(ep)
            tried.append(ep)
            last_err = e
        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                pool.bench(ep)
                tried.append(ep)
                last_err = e
            else:
                return f"[VLM error: {type(e).__name__}: {e}]"
        except Exception as e:
            return f"[VLM error: {type(e).__name__}: {e}]"
        finally:
            ep.inflight -= 1
    return f"[VLM error: {type(last_err).__name__}: {last_err}]"


async def batch_look(
    requests: list[dict],
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
) -> list[str]:
    """Send (path, query) pairs to the VLM in parallel. Returns answers in order.

    ``base_url`` may be a single URL or a '|'-separated weighted endpoint spec
    (see EndpointPool); pool state (in-flight counters, health) is shared across
    all rollouts in this process."""
    pool = _get_pool(base_url)
    coros = [_one_look(client, pool, model_id, r["path"], r["query"]) for r in requests]
    return await asyncio.gather(*coros)
