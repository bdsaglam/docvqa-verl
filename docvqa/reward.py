# docvqa/reward.py
"""Continuous-ANLS reward for the DocVQA CodeAct agent. verl custom_reward_function hook.

GRPO needs in-group reward *variance*. The eval metric (binary ANLS @ 0.9) is mostly-0 on
hard DocVQA -> dead groups / cold-start. So the *reward* is continuous ANLS (get_anls, 0..1):
dense partial credit, still maximized by exact answers. Evaluation stays binary@0.9 in the
~/repos/docvqa harness -- reward != metric on purpose.

The agent loop populates extra_fields (docvqa/agent_loop.py:307) which verl merges into
extra_info: submitted_answer, num_turns, vlm_calls, wall_clock_s, multi_block_turns,
empty_output_turns. We read submitted_answer directly (no regex on solution_str).
Non-submission (None / "") -> 0.0.

Reward = ANLS minus two optional, independently-tunable trajectory-scalar penalties:
- length: LENGTH_PENALTY_COEF * C_{k,q}(num_turns) -- a Cursor Composer-2-style *concave*
  length penalty (see _concave_length_cost). Concave-down + increasing: the marginal cost of
  an extra turn decays with length, so easy problems are pushed to be short while hard problems
  that genuinely need many turns aren't crushed. Attacks the ~37% wall_cap runaway without a
  flat per-turn tax. (q=0 recovers the plain linear LENGTH_PENALTY_COEF*num_turns.)
- FORMAT_PENALTY_PER_VIOLATION * (multi_block_turns + empty_output_turns) -- format shaping:
  a turn that emits >1 ```python``` block, or whose executed code printed nothing ("forgot to
  print"). Both counts come from the agent loop. (0-block turns are already handled by the
  loop's parse_error path, so they are NOT counted here -- no double penalty.)
Both penalties default to 0.0 (validate the bare ANLS signal on the first dry-run, then turn
on). The final score is clamped to >= 0. Per-turn (dense) format reward is a later upgrade;
for now every component is a trajectory-level scalar.

GRPO synergy: advantages are group-relative per question, so the length penalty mostly
differentiates shorter-vs-longer rollouts of the SAME problem; the concavity compresses
penalty differences among a hard question's uniformly-long rollouts -> difficulty-adaptive
twice over (group-relative x concave). Source: Cursor "Composer 2" technical report (2026).
"""
from __future__ import annotations

import ast
import math
from typing import Any

from docvqa.metrics import get_anls

# Concave length penalty: subtract LENGTH_PENALTY_COEF * C_{k,q}(num_turns) (COEF=0 disables).
# C_{k,q}(x) = ((1+kx)^(1-q) - 1) / (k(1-q)); marginal cost (1+kx)^(-q) decays with x for q>0.
#   q=0 -> linear (x);  q=1 -> log(1+kx)/k;  q>1 -> saturates to the cap 1/(k(q-1)).
LENGTH_PENALTY_COEF = 0.0
LENGTH_PENALTY_K = 1.0
LENGTH_PENALTY_Q = 1.0

# Subtract this * (multi_block_turns + empty_output_turns) from the score (0 disables).
FORMAT_PENALTY_PER_VIOLATION = 0.0


def _gt_candidates(ground_truth) -> list[str]:
    """Multi-alias gold may be stored as repr([...]); return the candidate list."""
    s = str(ground_truth)
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)) and v:
            return [str(c) for c in v]
    except (ValueError, SyntaxError):
        pass
    return [s]


def _concave_length_cost(x: float, k: float, q: float) -> float:
    """Cursor Composer-2 nonlinear length cost: concave-down, increasing for q>0.

    C_{k,q}(x) = ((1+kx)^(1-q) - 1) / (k(1-q)). Returns 0 for x<=0 or k<=0. Handles the
    q=1 singularity via the log limit. q=0 -> linear; q>1 -> bounded by 1/(k(q-1))."""
    if x <= 0 or k <= 0:
        return 0.0
    if abs(q - 1.0) < 1e-9:
        return math.log1p(k * x) / k
    return ((1.0 + k * x) ** (1.0 - q) - 1.0) / (k * (1.0 - q))

# Numeric keys propagated into reward_extra_info. ONLY numerics: verl's
# process_validation_metrics runs np.mean/std/max/min on every key (metric_utils.py),
# and protocol.py asserts every rollout emits the same key set -- so always emit all,
# with defaults. Non-numeric metadata travels via the agent-loop JSONL dump instead.
_NUMERIC_PASSTHROUGH: dict[str, float | int] = {
    "num_turns": 0,
    "vlm_calls": 0,
    "wall_clock_s": 0.0,
    "multi_block_turns": 0,
    "empty_output_turns": 0,
}


def compute_score(
    data_source: str = "",   # noqa: ARG001 -- verl signature
    solution_str: str = "",  # noqa: ARG001 -- answer comes from extra_info, not the text
    ground_truth: str = "",
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return {'score', 'anls', 'length_penalty', 'format_penalty', **numeric passthrough}.

    verl unpacks result['score'] as the scalar reward and feeds every other key into
    reward_extra_info. score == penalized reward; anls == raw ANLS (diagnostic, unpenalized).
    """
    extra = extra_info or {}
    submitted = extra.get("submitted_answer")
    if submitted is None or submitted == "":
        raw_anls = 0.0
    else:
        raw_anls = max(float(get_anls(str(submitted), c)) for c in _gt_candidates(ground_truth))

    num_turns = float(extra.get("num_turns") or 0)
    length_penalty = LENGTH_PENALTY_COEF * _concave_length_cost(
        num_turns, LENGTH_PENALTY_K, LENGTH_PENALTY_Q
    )

    format_violations = float(extra.get("multi_block_turns") or 0) + float(
        extra.get("empty_output_turns") or 0
    )
    format_penalty = FORMAT_PENALTY_PER_VIOLATION * format_violations

    score = max(0.0, raw_anls - length_penalty - format_penalty)

    out: dict[str, Any] = {k: extra.get(k, d) for k, d in _NUMERIC_PASSTHROUGH.items()}
    out["score"] = score
    out["anls"] = raw_anls
    out["length_penalty"] = length_penalty
    out["format_penalty"] = format_penalty
    return out
