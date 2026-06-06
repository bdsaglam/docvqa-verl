"""Unit tests for the continuous-ANLS GRPO reward (docvqa/reward.py)."""
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


def test_length_penalty_reduces_score_but_not_anls(monkeypatch):
    monkeypatch.setattr(R, "LENGTH_PENALTY_PER_TURN", 0.1)
    out = _score("2048.88", "2048.88", num_turns=3)
    assert out["anls"] == 1.0           # raw anls unaffected
    assert out["score"] == pytest.approx(1.0 - 0.1 * 3)  # penalized reward


def test_length_penalty_floors_at_zero(monkeypatch):
    monkeypatch.setattr(R, "LENGTH_PENALTY_PER_TURN", 1.0)
    out = _score("2048.88", "2048.88", num_turns=10)
    assert out["score"] == 0.0
