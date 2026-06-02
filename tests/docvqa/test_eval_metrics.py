import math
from docvqa.eval_metrics import score_rollouts, aggregate_question, majority_vote


def test_score_rollouts_marks_correct():
    scores = score_rollouts(["Paris", "London"], "Paris")
    assert scores == [1.0, 0.0]


def test_aggregate_question_mean_passk_sc():
    answers = ["Paris", "Paris", "London"]
    gold = "Paris"
    agg = aggregate_question(answers, gold)
    assert math.isclose(agg["mean"], 2 / 3)
    assert agg["passk"] == 1.0
    assert agg["sc"] == 1.0
    assert agg["n"] == 3


def test_majority_vote_normalizes_and_breaks_ties_by_first_seen():
    assert majority_vote(["The Paris.", "paris", "London"]) == "The Paris."


def test_aggregate_all_wrong():
    agg = aggregate_question(["London", "Berlin"], "Paris")
    assert agg["mean"] == 0.0 and agg["passk"] == 0.0 and agg["sc"] == 0.0


def test_none_submission_scores_zero():
    assert score_rollouts([None, "Paris"], "Paris") == [0.0, 1.0]
