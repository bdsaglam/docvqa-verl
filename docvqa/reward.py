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
- LENGTH_PENALTY_PER_TURN * num_turns -- attacks the ~37% wall_cap runaway (4B doesn't stop).
- FORMAT_PENALTY_PER_VIOLATION * (multi_block_turns + empty_output_turns) -- format shaping:
  a turn that emits >1 ```python``` block, or whose executed code printed nothing ("forgot to
  print"). Both counts come from the agent loop. (0-block turns are already handled by the
  loop's parse_error path, so they are NOT counted here -- no double penalty.)
Both penalties default to 0.0 (validate the bare ANLS signal on the first dry-run, then turn
on). The final score is clamped to >= 0. Per-turn (dense) format reward is a later upgrade;
for now every component is a trajectory-level scalar.
"""
from __future__ import annotations

from typing import Any

from docvqa.metrics import get_anls

# Subtract this * num_turns from the score (0 disables).
LENGTH_PENALTY_PER_TURN = 0.0

# Subtract this * (multi_block_turns + empty_output_turns) from the score (0 disables).
FORMAT_PENALTY_PER_VIOLATION = 0.0

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
    """Return {'score': reward, 'anls': raw_anls, 'format_penalty': ..., **numeric passthrough}.

    verl unpacks result['score'] as the scalar reward and feeds every other key into
    reward_extra_info. score == penalized reward; anls == raw ANLS (diagnostic, unpenalized).
    """
    extra = extra_info or {}
    submitted = extra.get("submitted_answer")
    if submitted is None or submitted == "":
        raw_anls = 0.0
    else:
        raw_anls = float(get_anls(str(submitted), str(ground_truth)))

    length_penalty = LENGTH_PENALTY_PER_TURN * float(extra.get("num_turns") or 0)

    format_violations = float(extra.get("multi_block_turns") or 0) + float(
        extra.get("empty_output_turns") or 0
    )
    format_penalty = FORMAT_PENALTY_PER_VIOLATION * format_violations

    score = max(0.0, raw_anls - length_penalty - format_penalty)

    out: dict[str, Any] = {k: extra.get(k, d) for k, d in _NUMERIC_PASSTHROUGH.items()}
    out["score"] = score
    out["anls"] = raw_anls
    out["format_penalty"] = format_penalty
    return out
