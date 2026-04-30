# docvqa/reward.py
"""ANLS reward for DocVQA. Plugs into verl's custom_reward_function hook.

The metric is identical to what evaluation uses — no train/eval drift.
Score is 1.0 if the predicted answer is correct under
`evaluate_prediction`, else 0.0.
"""
from __future__ import annotations

from typing import Any

from docvqa.metrics import evaluate_prediction


def compute_score(
    data_source: str,
    solution_str: str,  # noqa: ARG001 — verl signature; we use extra_info
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """verl reward function: returns (score, extra_info_to_dump).

    `extra_info` arrives from `AgentLoopOutput.extra_fields`. We compute
    correctness against the model's `submitted_answer` (None ⇒ 0 score)
    and pass every other field through to the rollout JSONL dump.
    """
    extra = dict(extra_info or {})
    submitted = extra.get("submitted_answer")
    if submitted is None:
        score = 0.0
    else:
        is_correct, _extracted = evaluate_prediction(submitted, ground_truth)
        score = 1.0 if is_correct else 0.0
    extra["anls"] = score
    return score, extra
