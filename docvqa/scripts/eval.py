#!/usr/bin/env python
"""Layer-3 ANLS reproduction.

Runs DocVQAReplAgentLoop over a question split with a running vLLM (student)
and the running 27B VLM, computes per-question ANLS, and writes a JSON report.

Usage:
    python docvqa/scripts/eval.py \\
        --questions data/docvqa-2026/val/questions.json \\
        --base-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3.5-4B \\
        --vlm-base-url http://localhost:8927 \\
        --vlm-model Qwen/Qwen3.5-27B \\
        --concurrency 4 \\
        --run-dir outputs/runs/val-qwen3_5-4b-t1

The agent LM (`--model`/`--base-url`) may be the teacher (27B) or the student
(4B); the run dir doubles as a trajectory collection (see _write_summary).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
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


def _build_loop(base_url: str, model: str,
                vlm_base_url: str, vlm_model: str,
                response_length: int = 32768,
                enable_thinking: bool = True) -> DocVQAReplAgentLoop:
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    tokenizer = AutoTokenizer.from_pretrained(model)
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
                        # Deploy-parity with codeact_chat: concat all complete fenced
                        # blocks per turn (matches the leaderboard scaffold). Applies to
                        # both teacher generation and SFT-model eval run through eval.py.
                        "concat_fences": True,
                    },
                },
                "multi_turn": {},
                "trace": {"project_name": "eval", "experiment_name": "eval"},
            },
            "model": {},
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": enable_thinking}},
    })
    return DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_OpenAIClientServerManager(
            base_url, model, tokenizer,
        ),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=type("_StubDataset", (), {}),
        data_config=DictConfigWrap(cfg.data),
    )


async def _solve_n(loop: DocVQAReplAgentLoop, q: dict, n: int,
                   sampling: dict, rollout_timeout: float | None = None,
                   save_token_ids: bool = True,
                   max_keep: int | None = None) -> dict[str, Any]:
    """Run up to n rollouts for one question; return full per-sample records
    (incl. the chat ``messages`` trajectory) + scoring.

    ``n`` is the rollout BUDGET. With ``max_keep`` set (rejection-sampling
    generation), rollouts run sequentially and stop early as soon as the
    question has ``max_keep`` keepers (anls==1.0) — easy prompts finish in 2
    rollouts and free their concurrency slot, so the in-flight pool stays
    saturated while hard prompts get the full budget. ``max_keep=None`` (eval)
    always runs all n (pass@n / SC metrics need every sample).

    Each rollout is bounded by ``rollout_timeout`` seconds (wall clock). The
    agent loop's per-turn caps do NOT bound ``batch_look``'s VLM HTTP calls
    (they run in the host async context, not the sandbox subprocess), so a
    single pathological question can fan out unbounded VLM calls and never
    terminate. A wall-clock cap makes eval time predictable; a timed-out
    rollout is recorded as termination="wall_cap" and scored as a non-answer.

    The per-sample records carry the full trajectory + ANLS, so an eval run is
    also a teacher/student trajectory collection (no separate collect script).
    """
    from docvqa.metrics import evaluate_prediction

    gold = q.get("answer")
    samples: list[dict[str, Any]] = []
    for sample_idx in range(n):
        t0 = time.monotonic()
        rec: dict[str, Any] = {
            "sample_idx": sample_idx, "submitted_answer": None,
            "extracted_answer": None, "is_correct": False, "anls": 0.0,
            "termination": None, "num_turns": 0, "vlm_calls": 0,
            "turns_truncated": 0, "max_turn_tokens": 0,
            "wall_clock_s": 0.0, "messages": [],
            # token-level (exact teacher tokens + assistant-only loss mask), for
            # SFT-on-exact-tokens and as the basis for forward-KL/OPD/RL. Empty
            # if --no-token-ids. logprobs are NOT stored (recompute via a frozen-
            # teacher forward pass over response_ids when KD needs them).
            "prompt_ids": [], "response_ids": [], "response_mask": [],
        }
        try:
            coro = loop.run(
                sampling_params=sampling,
                question_id=q["question_id"], question=q["question"],
                doc_dir=q["doc_dir"], gold_answer=gold,
                category=q.get("category", "unknown"),
            )
            if rollout_timeout is not None:
                out = await asyncio.wait_for(coro, timeout=rollout_timeout)
            else:
                out = await coro
            ef = out.extra_fields
            submitted = ef.get("submitted_answer")
            rec.update(
                submitted_answer=submitted,
                termination=ef.get("termination"),
                num_turns=ef.get("num_turns") or 0,
                vlm_calls=ef.get("vlm_calls") or 0,
                turns_truncated=ef.get("turns_truncated") or 0,
                max_turn_tokens=ef.get("max_turn_tokens") or 0,
                wall_clock_s=ef.get("wall_clock_s", time.monotonic() - t0),
                messages=ef.get("messages", []),
            )
            if save_token_ids:
                rec.update(
                    prompt_ids=list(out.prompt_ids),
                    response_ids=list(out.response_ids),
                    response_mask=list(out.response_mask),
                )
            if submitted is not None and gold is not None:
                is_correct, extracted = evaluate_prediction(submitted, gold)
                rec.update(is_correct=bool(is_correct),
                           extracted_answer=extracted,
                           anls=1.0 if is_correct else 0.0)
        except asyncio.TimeoutError:
            rec.update(termination="wall_cap",
                       wall_clock_s=time.monotonic() - t0)
        except Exception as e:
            rec.update(termination=f"error:{e!r}",
                       wall_clock_s=time.monotonic() - t0)
        samples.append(rec)
        # Rejection-sampling early stop: once we have enough keepers, stop
        # spending rollouts on this (easy) prompt and free the slot.
        if max_keep is not None and sum(s["anls"] == 1.0 for s in samples) >= max_keep:
            break
    return {
        "question_id": q["question_id"], "doc_id": q.get("doc_id"),
        "category": q.get("category", "unknown"),
        "question": q.get("question"),
        "gold_answer": gold,
        "samples": samples,
    }


def _write_summary(run_dir: Path, results: list[dict], meta: dict) -> dict:
    """Write per-doc result.json + top-level results.json (metrics, no messages).
    Trajectories were already streamed to tasks/<doc>/trajectories.jsonl."""
    import statistics
    by_doc: dict[str, list[dict]] = {}
    for r in results:
        by_doc.setdefault(str(r["doc_id"] or r["question_id"]), []).append(r)

    documents = []
    for doc_id, qs in sorted(by_doc.items()):
        q_entries = [{
            "question_id": r["question_id"], "question": r.get("question"),
            "ground_truth": r["gold_answer"], "prediction": r.get("voted_answer"),
            "is_correct": bool(r["sc"] >= 1.0), "mean": r["mean"],
            "passk": r["passk"], "sc": r["sc"], "n": r["n"],
        } for r in qs]
        acc = statistics.mean(r["mean"] for r in qs)
        (run_dir / "tasks" / doc_id / "result.json").write_text(json.dumps({
            "doc_id": doc_id, "doc_category": qs[0]["category"],
            "accuracy": acc, "questions": q_entries,
        }, indent=2, ensure_ascii=False))
        documents.append({"doc_id": doc_id, "doc_category": qs[0]["category"],
                          "accuracy": acc, "num_questions": len(qs)})

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    means = [r["mean"] for r in results]
    summary = {
        "overall_accuracy": statistics.mean(means) if means else 0.0,
        "std": statistics.pstdev(means) if len(means) > 1 else 0.0,
        "total_questions": len(results),
        "correct": sum(1 for m in means if m >= 1.0),
        "n_samples": meta["n"],
        f"pass@{meta['n']}": statistics.mean(r["passk"] for r in results) if results else 0.0,
        f"sc-{meta['n']}": statistics.mean(r["sc"] for r in results) if results else 0.0,
        "by_category": {c: {"accuracy": statistics.mean(r["mean"] for r in rs),
                            "total": len(rs)} for c, rs in sorted(by_cat.items())},
    }
    (run_dir / "results.json").write_text(json.dumps(
        {"summary": summary, "config": meta, "documents": documents},
        indent=2, ensure_ascii=False))
    return summary


def _load_done_results(run_dir: Path, n: int, aggregate_question,
                       max_keep: int | None = None) -> tuple[set, list[dict]]:
    """For ``--resume``: scan already-streamed trajectories and rebuild per-question
    result entries so the final summary covers resumed + newly-run questions.

    Without ``max_keep`` a question counts as done iff it has >= n streamed
    samples (eval; samples are written atomically per-question). With
    ``max_keep`` (rejection-sampling generation, variable rollout count) a
    question is done iff it reached ``max_keep`` keepers (anls==1.0) OR exhausted
    the budget (>= n samples) — so solved-early and proven-unsolvable prompts are
    both skipped, and only genuinely-partial ones re-run."""
    by_qid: dict[str, list[dict]] = {}
    for fp in run_dir.glob("tasks/*/trajectories.jsonl"):
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    by_qid.setdefault(rec.get("question_id"), []).append(rec)
    done_ids: set = set()
    done_results: list[dict] = []
    for qid, recs in by_qid.items():
        if max_keep is not None:
            keepers = sum(1 for r in recs if r.get("anls") == 1.0)
            if keepers < max_keep and len(recs) < n:
                continue
        elif len(recs) < n:
            continue
        recs = sorted(recs, key=lambda r: r.get("sample_idx", 0))[:n]
        agg = aggregate_question([r.get("submitted_answer") for r in recs],
                                 recs[0].get("gold_answer"))
        done_ids.add(qid)
        done_results.append({
            "question_id": qid, "doc_id": recs[0].get("doc_id"),
            "category": recs[0].get("category", "unknown"),
            "question": recs[0].get("question"),
            "gold_answer": recs[0].get("gold_answer"),
            "samples": recs, "n": len(recs),
            **{k: agg[k] for k in ("mean", "passk", "sc", "scores", "voted_answer")},
        })
    return done_ids, done_results


async def _main_async(args) -> None:
    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    loop_obj = _build_loop(
        args.base_url, args.model,
        args.vlm_base_url, args.vlm_model,
        enable_thinking=args.enable_thinking,
    )

    from docvqa.eval_metrics import aggregate_question

    sampling = {"temperature": args.temperature, "top_p": args.top_p,
                "top_k": args.top_k}
    sem = asyncio.Semaphore(args.concurrency)
    rollout_timeout = args.rollout_timeout if args.rollout_timeout > 0 else None

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    total_questions = len(questions)
    done_results: list[dict] = []
    if args.resume:
        done_ids, done_results = _load_done_results(run_dir, args.n, aggregate_question,
                                                     max_keep=args.max_keep)
        if done_ids:
            questions = [q for q in questions if q.get("question_id") not in done_ids]
            print(f"[resume] {len(done_ids)} questions already complete in {run_dir}; "
                  f"running {len(questions)} remaining (of {total_questions}).",
                  file=sys.stderr)
        else:
            print(f"[resume] no completed questions found in {run_dir}; "
                  f"running all {total_questions}.", file=sys.stderr)

    meta = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.dataset, "split": args.split,
        "questions": args.questions, "num_questions": total_questions,
        "n": args.n, "max_keep": args.max_keep,
        "model": args.model, "base_url": args.base_url,
        "vlm_model": args.vlm_model, "vlm_base_url": args.vlm_base_url,
        "sampling": dict(sampling), "rollout_timeout": args.rollout_timeout,
        "save_token_ids": args.save_token_ids,
        "enable_thinking": args.enable_thinking,
    }
    # Persist config up front, so a crashed/partial run is still identifiable
    # (which models / sampling produced these trajectories).
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_lock = asyncio.Lock()

    async def _bound(q: dict) -> dict[str, Any]:
        async with sem:
            raw = await _solve_n(loop_obj, q, args.n, sampling, rollout_timeout,
                                 save_token_ids=args.save_token_ids,
                                 max_keep=args.max_keep)
        samples = raw["samples"]
        agg = aggregate_question([s["submitted_answer"] for s in samples],
                                 raw["gold_answer"])
        doc_id = str(raw["doc_id"] or raw["question_id"])
        tdir = run_dir / "tasks" / doc_id
        # Stream the structured trajectory immediately (crash-safe; this is the
        # collection output — full messages + anls + termination per sample).
        async with write_lock:
            tdir.mkdir(parents=True, exist_ok=True)
            with (tdir / "trajectories.jsonl").open("a") as f:
                for s in samples:
                    f.write(json.dumps({
                        "record_id": f"{args.dataset}:{args.split}:{doc_id}:{raw['question_id']}:{s['sample_idx']}",
                        "question_id": raw["question_id"], "doc_id": doc_id,
                        "category": raw["category"], "question": raw["question"],
                        "gold_answer": raw["gold_answer"],
                        "model": args.model, "vlm_model": args.vlm_model,
                        "sampling": sampling,
                        **s,
                    }, ensure_ascii=False) + "\n")
        return {**{k: raw[k] for k in ("question_id", "doc_id", "category",
                                       "question", "gold_answer", "samples")},
                "n": len(samples),
                **{k: agg[k] for k in ("mean", "passk", "sc", "scores",
                                       "voted_answer")}}

    new_results = await asyncio.gather(*(_bound(q) for q in questions))
    # Combine resumed (reconstructed-from-disk) + newly-run so the summary covers
    # the full set, not just this invocation's questions.
    results = done_results + list(new_results)
    summary = _write_summary(run_dir, results, meta)

    print(f"=== DocVQA eval (n={args.n}, model={args.model}) -> {run_dir} ===")
    print(f"  overall   : {summary['overall_accuracy']:.4f} "
          f"± {summary['std']:.4f}  ({summary['correct']}/{summary['total_questions']})")
    print(f"  pass@{args.n}  : {summary[f'pass@{args.n}']:.4f}   "
          f"SC-{args.n}: {summary[f'sc-{args.n}']:.4f}")
    for cat, s in summary["by_category"].items():
        print(f"    {cat:20s} acc={s['accuracy']:.4f} (n={s['total']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1",
                    help="Agent LM endpoint (OpenAI-compatible). The agent LM may "
                         "be the teacher (e.g. 27B) or the student (e.g. 4B).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B",
                    help="Agent LM model id served at --base-url (default 27B — "
                         "we mostly evaluate the teacher; pass the 4B/checkpoint "
                         "path for student evals).")
    ap.add_argument("--vlm-base-url", default="http://localhost:8927")
    ap.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=8, help="rollouts per question "
                    "(the BUDGET / max rollouts; see --max-keep)")
    ap.add_argument("--max-keep", type=int, default=None,
                    help="Rejection-sampling early stop: stop a question once it "
                         "has this many keepers (anls==1.0), capped at --n. Easy "
                         "prompts finish fast and free their slot (pool stays "
                         "saturated); hard prompts get the full --n budget. "
                         "Default None = run all --n (required for eval metrics).")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--rollout-timeout", type=float, default=600.0,
                    help="Per-rollout wall-clock cap in seconds (0 disables). "
                         "Timed-out rollouts are scored as non-answers "
                         "(termination='wall_cap').")
    ap.add_argument("--run-dir", required=True,
                    help="Output run directory (output/runs/<run_id> style): "
                         "writes results.json + tasks/<doc_id>/{result.json, "
                         "trajectories.jsonl}. The per-doc trajectories.jsonl "
                         "carries full chat messages + anls + termination per "
                         "sample, so an eval run doubles as trajectory collection.")
    ap.add_argument("--dataset", default="docvqa-2026",
                    help="Dataset slug for trajectory record_ids.")
    ap.add_argument("--split", default="val",
                    help="Split name for trajectory record_ids.")
    ap.add_argument("--no-token-ids", dest="save_token_ids", action="store_false",
                    help="Omit prompt_ids/response_ids/response_mask from "
                         "trajectories (smaller files; messages text still saved). "
                         "Default: save them (exact tokens + assistant-only mask "
                         "for SFT-on-exact-tokens / KD / RL).")
    ap.set_defaults(save_token_ids=True)
    ap.add_argument("--no-thinking", dest="enable_thinking",
                    action="store_false",
                    help="Disable the agent LM's native thinking. Default ON. "
                         "Our agent_loop has NO explicit reasoning field, so "
                         "native <think> is the ONLY reasoning channel — turning "
                         "it off yields empty <think></think> and no deliberation. "
                         "(The original DSPy CodeAct used enable_thinking=false "
                         "ONLY because DSPy's signature carries a separate `reason` "
                         "field; that does NOT transfer to our setup. Do not "
                         "'match' the flag value — match the behavior: reasoning ON.)")
    ap.set_defaults(enable_thinking=True)
    ap.add_argument("--resume", action="store_true",
                    help="Resume into an existing --run-dir: skip questions that "
                         "already have >= n streamed samples and only run the rest. "
                         "The final results.json summary still covers all questions "
                         "(reconstructed-from-disk + newly-run).")
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
