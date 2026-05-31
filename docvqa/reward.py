# docvqa/reward.py
"""ANLS reward for DocVQA. Plugs into verl's custom_reward_function hook.

The metric is identical to what evaluation uses — no train/eval drift.
Score is 1.0 if the predicted answer is correct under
`evaluate_prediction`, else 0.0.
"""
from __future__ import annotations

from typing import Any

from docvqa.metrics import evaluate_prediction


_NUMERIC_PASSTHROUGH: dict[str, float | int] = {
    # agent-loop summary stats. ONLY numerics here: verl runs
    # `process_validation_metrics` which does np.mean/std/max/min on
    # every reward_extra_info value (`metric_utils.py:645`), so any
    # string column would crash validation aggregation.
    "num_turns": 0,
    "vlm_calls": 0,
    "wall_clock_s": 0.0,
}
"""Numeric keys we propagate into reward_extra_info.

Verl's contract: every reward_extra_info key gets aggregated as a
numpy mean across the validation batch. Non-numeric metadata
(record_id, doc_id, submitted_answer, etc.) lives instead in the
agent-loop output's extra_fields and travels to the rollout JSONL
dump via `trainer.rollout_data_dir`.

Verl also asserts every rollout has the same set of keys
(`protocol.py:concat`), so we always emit the full set with defaults.
"""


def compute_score(
    data_source: str,  # noqa: ARG001 — verl signature
    solution_str: str,  # noqa: ARG001 — we use extra_info from agent loop
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """verl reward function. Returns a dict with `score` plus extras.

    Verl's reward manager unpacks `result["score"]` as the scalar reward
    and feeds every other key into `reward_extra_info` (rollout JSONL
    dump). All values must be numpy-array-compatible scalars/strings;
    variable-shape values are filtered out via `_PASSTHROUGH_KEYS`.

    `extra_info` arrives from the dataset row plus
    `AgentLoopOutput.extra_fields` merged in by the agent-loop worker.
    Correctness is computed against the model's `submitted_answer`
    (None ⇒ 0 score) using the same `evaluate_prediction` as eval.
    """
    extra = extra_info or {}
    submitted = extra.get("submitted_answer")
    if submitted is None:
        score = 0.0
    else:
        is_correct, _extracted = evaluate_prediction(submitted, ground_truth)
        score = 1.0 if is_correct else 0.0
    out: dict[str, Any] = {k: extra.get(k, default) for k, default in _NUMERIC_PASSTHROUGH.items()}
    out["score"] = score
    out["anls"] = score
    return out
