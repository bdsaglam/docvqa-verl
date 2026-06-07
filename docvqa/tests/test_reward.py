"""Unit tests for the continuous-ANLS GRPO reward (docvqa/reward.py)."""
import math

import pytest

from docvqa import reward as R


def _score(submitted, gt, **extra):
    extra_info = {"submitted_answer": submitted, **extra}
    return R.compute_score(data_source="docvqa", solution_str="", ground_truth=gt, extra_info=extra_info)


def test_exact_match_scores_one():
    out = _score("2048.88", "2048.88")
    assert out["score"] == 1.0
    assert out["anls"] == 1.0


def test_close_answer_gets_partial_credit():
    # "2038.94" vs "2048.88": Levenshtein distance 3 over len 7 -> anls = 1 - 3/7 = 0.5714
    out = _score("2038.94", "2048.88")
    assert out["anls"] == pytest.approx(1 - 3 / 7, abs=1e-6)
    assert 0.0 < out["score"] < 1.0


def test_wrong_answer_scores_zero():
    out = _score("completely different", "2048.88")
    assert out["score"] == 0.0


def test_no_submission_scores_zero():
    out = _score(None, "2048.88")
    assert out["score"] == 0.0
    assert out["anls"] == 0.0


def test_empty_submission_scores_zero():
    out = _score("", "2048.88")
    assert out["score"] == 0.0


def test_extra_info_passthrough_is_numeric():
    out = _score("x", "y", num_turns=5, vlm_calls=3, wall_clock_s=12.5)
    assert out["num_turns"] == 5
    assert out["vlm_calls"] == 3
    assert out["wall_clock_s"] == 12.5
    # No non-numeric keys leak into the reward dict (verl aggregates every key with np.mean).
    for k, v in out.items():
        assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"


def test_linear_length_penalty_q0_reduces_score_but_not_anls(monkeypatch):
    # q=0 recovers the plain linear penalty: COEF * num_turns
    monkeypatch.setattr(R, "LENGTH_PENALTY_COEF", 0.1)
    monkeypatch.setattr(R, "LENGTH_PENALTY_Q", 0.0)
    out = _score("2048.88", "2048.88", num_turns=3)
    assert out["anls"] == 1.0                          # raw anls unaffected
    assert out["length_penalty"] == pytest.approx(0.3)
    assert out["score"] == pytest.approx(0.7)


def test_length_penalty_floors_at_zero(monkeypatch):
    monkeypatch.setattr(R, "LENGTH_PENALTY_COEF", 1.0)
    monkeypatch.setattr(R, "LENGTH_PENALTY_Q", 0.0)
    out = _score("2048.88", "2048.88", num_turns=10)
    assert out["score"] == 0.0


def test_length_penalty_default_off_is_inert():
    out = _score("2048.88", "2048.88", num_turns=50)
    assert out["length_penalty"] == 0.0
    assert out["score"] == 1.0


# --- concave length cost (Cursor Composer-2 form) ---


def test_concave_cost_q0_is_linear():
    assert R._concave_length_cost(5, 1.0, 0.0) == pytest.approx(5.0)


def test_concave_cost_q1_is_log():
    assert R._concave_length_cost(5, 2.0, 1.0) == pytest.approx(math.log1p(2 * 5) / 2)


def test_concave_cost_zero_at_zero():
    assert R._concave_length_cost(0, 1.0, 0.5) == 0.0


def test_concave_marginal_penalty_decays():
    # marginal cost of an extra turn shrinks as the rollout grows (q>0)
    k, q = 1.0, 0.5
    early = R._concave_length_cost(2, k, q) - R._concave_length_cost(1, k, q)
    late = R._concave_length_cost(11, k, q) - R._concave_length_cost(10, k, q)
    assert early > late > 0


def test_concave_q_gt_1_saturates_to_cap():
    # q>1 -> bounded by 1/(k(q-1)); large x approaches but never exceeds the cap
    k, q = 1.0, 2.0
    cap = 1.0 / (k * (q - 1))  # = 1.0
    big = R._concave_length_cost(10_000, k, q)
    assert big < cap
    assert big == pytest.approx(cap, abs=1e-3)


# --- format penalty (trajectory-scalar: sum of per-turn violations) ---


def test_format_penalty_subtracts_per_violation(monkeypatch):
    # violations = multi_block_turns + empty_output_turns = 2 + 1 = 3
    monkeypatch.setattr(R, "FORMAT_PENALTY_PER_VIOLATION", 0.1)
    out = _score("2048.88", "2048.88", multi_block_turns=2, empty_output_turns=1)
    assert out["anls"] == 1.0                       # raw anls unaffected
    assert out["format_penalty"] == pytest.approx(0.3)
    assert out["score"] == pytest.approx(0.7)


def test_format_penalty_default_off_is_inert():
    out = _score("2048.88", "2048.88", multi_block_turns=2, empty_output_turns=1)
    assert out["format_penalty"] == 0.0
    assert out["score"] == 1.0


def test_format_penalty_floors_at_zero(monkeypatch):
    monkeypatch.setattr(R, "FORMAT_PENALTY_PER_VIOLATION", 1.0)
    out = _score("2048.88", "2048.88", multi_block_turns=5)
    assert out["score"] == 0.0


def test_format_violation_counts_passthrough_numeric():
    out = _score("x", "y", multi_block_turns=3, empty_output_turns=2)
    assert out["multi_block_turns"] == 3
    assert out["empty_output_turns"] == 2
    for k, v in out.items():
        assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"


def test_length_and_format_penalties_stack(monkeypatch):
    # 1.0 - 0.05*2 (linear length, q=0) - 0.1*1 (format) = 0.8
    monkeypatch.setattr(R, "LENGTH_PENALTY_COEF", 0.05)
    monkeypatch.setattr(R, "LENGTH_PENALTY_Q", 0.0)
    monkeypatch.setattr(R, "FORMAT_PENALTY_PER_VIOLATION", 0.1)
    out = _score("2048.88", "2048.88", num_turns=2, multi_block_turns=1)
    assert out["score"] == pytest.approx(0.8)
