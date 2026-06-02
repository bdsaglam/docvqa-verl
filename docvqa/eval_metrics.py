"""Pure aggregation helpers for the n=8 / pass@k / SC-k eval protocol (spec §9).

Scoring delegates to docvqa.metrics.evaluate_prediction (the official
DocVQA-2026 metric: strict numeric/date match + relaxed ANLS).
"""
from __future__ import annotations

from collections import Counter

from docvqa.metrics import _clean_text, evaluate_prediction


def score_rollouts(submitted: list[str | None], gold: str | None) -> list[float]:
    """1.0 / 0.0 per rollout under the official metric. None or no-gold -> 0.0."""
    out: list[float] = []
    for ans in submitted:
        if ans is None or gold is None:
            out.append(0.0)
            continue
        is_correct, _ = evaluate_prediction(ans, gold)
        out.append(1.0 if is_correct else 0.0)
    return out


def majority_vote(submitted: list[str | None]) -> str | None:
    """Most common answer by normalized form; returns the first-seen raw
    surface of the winning normalized class. Ties broken by first appearance."""
    norm_to_raw: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for ans in submitted:
        if ans is None:
            continue
        key = _clean_text(str(ans))
        if key == "":
            continue
        if key not in norm_to_raw:
            norm_to_raw[key] = ans
        counts[key] += 1
    if not counts:
        return None
    best_key = counts.most_common(1)[0][0]
    return norm_to_raw[best_key]


def aggregate_question(submitted: list[str | None], gold: str | None) -> dict:
    """Per-question aggregation across rollouts."""
    scores = score_rollouts(submitted, gold)
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    passk = 1.0 if any(s == 1.0 for s in scores) else 0.0
    voted = majority_vote(submitted)
    sc = score_rollouts([voted], gold)[0] if voted is not None else 0.0
    return {"n": n, "mean": mean, "passk": passk, "sc": sc, "scores": scores,
            "voted_answer": voted}
