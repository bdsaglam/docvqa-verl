#!/usr/bin/env python
"""Layer-3 ANLS reproduction.

Runs DocVQAReplAgentLoop over a question split with a running vLLM (student)
and the running 27B VLM, computes per-question ANLS, and writes a JSON report.

Usage:
    python docvqa/scripts/eval.py \\
        --questions data/docvqa-2026/val/questions.json \\
        --student-base-url http://localhost:8000/v1 \\
        --student-model Qwen/Qwen3.5-4B \\
        --vlm-base-url http://localhost:8927 \\
        --vlm-model Qwen/Qwen3.5-27B \\
        --concurrency 4 \\
        --output outputs/eval/val_qwen3_8b_layer3.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx

# Make sure the project root is on sys.path when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docvqa.agent_loop import DocVQAReplAgentLoop  # noqa: E402


# --- vLLM-backed server_manager ---------------------------------------------


class _OpenAIClientServerManager:
    """Implements .generate(prompt_ids, sampling_params, ...) by calling a
    vLLM-served student model at an OpenAI-compatible /v1/completions endpoint.

    We decode prompt_ids -> text and use the prompt-string completions endpoint
    so the wire format stays simple. vLLM tokenizes server-side; we re-encode
    the response text on our side to recover token ids for response_mask.
    """

    def __init__(self, base_url: str, model: str, tokenizer):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._tokenizer = tokenizer

    async def generate(self, request_id: str, prompt_ids, sampling_params, **_):
        del request_id
        prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(
                f"{self._base_url}/completions",
                json={
                    "model": self._model,
                    "prompt": prompt_text,
                    "max_tokens": sampling_params.get("max_tokens", 4096),
                    "temperature": sampling_params.get("temperature", 0.6),
                    "top_p": sampling_params.get("top_p", 0.95),
                    # vLLM's OpenAI /completions accepts top_k as a top-level
                    # sampling-param extension (not in the OpenAI spec proper).
                    "top_k": sampling_params.get("top_k", 20),
                    "stop": sampling_params.get("stop", ["<|im_end|>"]),
                },
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)

        class _Out:
            def __init__(self, ids):
                self.token_ids = ids
                self.log_probs = None
                self.num_preempted = 0
                self.extra_fields: dict = {}
                self.routed_experts = None

        return _Out(token_ids)


# --- Agent loop construction -------------------------------------------------


def _build_loop(student_base_url: str, student_model: str,
                vlm_base_url: str, vlm_model: str,
                response_length: int = 32768) -> DocVQAReplAgentLoop:
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    cfg = OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 16384,
                "response_length": response_length,
                "agent": {
                    "agent_loop_config_path": None,
                    "docvqa": {
                        "vlm_base_url": vlm_base_url,
                        "vlm_model_id": vlm_model,
                    },
                },
                "multi_turn": {},
                "trace": {"project_name": "eval", "experiment_name": "eval"},
            },
            "model": {},
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
    })
    return DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_OpenAIClientServerManager(
            student_base_url, student_model, tokenizer,
        ),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=type("_StubDataset", (), {}),
        data_config=DictConfigWrap(cfg.data),
    )


async def _solve_n(loop: DocVQAReplAgentLoop, q: dict, n: int,
                   sampling: dict) -> dict[str, Any]:
    """Run n rollouts for one question; return raw submitted answers + meta."""
    submitted: list[str | None] = []
    terminations: list[str | None] = []
    turns: list[int] = []
    for _ in range(n):
        try:
            out = await loop.run(
                sampling_params=sampling,
                question_id=q["question_id"], question=q["question"],
                doc_dir=q["doc_dir"], gold_answer=q.get("answer"),
                category=q.get("category", "unknown"),
            )
            submitted.append(out.extra_fields.get("submitted_answer"))
            terminations.append(out.extra_fields.get("termination"))
            turns.append(out.extra_fields.get("num_turns") or 0)
        except Exception as e:
            submitted.append(None)
            terminations.append(f"error:{e!r}")
            turns.append(0)
    return {
        "question_id": q["question_id"], "doc_id": q.get("doc_id"),
        "category": q.get("category", "unknown"),
        "gold_answer": q.get("answer"),
        "submitted_answers": submitted,
        "terminations": terminations, "num_turns": turns,
    }


async def _main_async(args) -> None:
    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    loop_obj = _build_loop(
        args.student_base_url, args.student_model,
        args.vlm_base_url, args.vlm_model,
    )

    import statistics
    from docvqa.eval_metrics import aggregate_question

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}
    sem = asyncio.Semaphore(args.concurrency)

    async def _bound(q: dict) -> dict[str, Any]:
        async with sem:
            raw = await _solve_n(loop_obj, q, args.n, sampling)
        agg = aggregate_question(raw["submitted_answers"], raw["gold_answer"])
        return {**raw, **{k: agg[k] for k in ("mean", "passk", "sc", "scores",
                                              "voted_answer")}}

    results = await asyncio.gather(*(_bound(q) for q in questions))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    def _overall(key: str) -> float:
        return statistics.mean(r[key] for r in results) if results else 0.0

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"=== DocVQA-2026 eval (n={args.n}, model={args.student_model}) ===")
    means = [r["mean"] for r in results]
    print(f"  mean ANLS : {statistics.mean(means):.4f} "
          f"± {statistics.pstdev(means):.4f}  (n_q={len(results)})")
    print(f"  pass@{args.n}  : {_overall('passk'):.4f}")
    print(f"  SC-{args.n}    : {_overall('sc'):.4f}")
    for cat, rows in sorted(by_cat.items()):
        m = statistics.mean(r["mean"] for r in rows)
        print(f"    {cat:20s} mean={m:.4f} pass={statistics.mean(r['passk'] for r in rows):.4f} "
              f"sc={statistics.mean(r['sc'] for r in rows):.4f} (n_q={len(rows)})")
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--student-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--student-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--vlm-base-url", default="http://localhost:8927")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8, help="rollouts per question")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
