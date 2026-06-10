"""Tests for the ANLS reward function and the vendored metric."""
from docvqa.metrics import evaluate_prediction, get_anls
from docvqa.reward import compute_score


def test_get_anls_exact_match():
    assert get_anls("42", "42") == 1.0


def test_get_anls_close_match_high_score():
    score = get_anls("$1.2 billion", "$1.2 billion.")
    assert score > 0.9


def test_evaluate_exact_numeric_with_unit():
    is_correct, extracted = evaluate_prediction("50 kg", "50 kg")
    assert is_correct is True


def test_evaluate_unit_alias_match():
    # parse_magnitude_unit normalizes 'kilograms' -> 'kg'
    is_correct, _ = evaluate_prediction("50 kilograms", "50 kg")
    assert is_correct is True


def test_evaluate_unknown_typo_variants():
    is_correct, _ = evaluate_prediction("Unknown", "Unkown")  # GT typo
    assert is_correct is True


def test_evaluate_wrong_answer():
    is_correct, _ = evaluate_prediction("apple", "battlefield")
    assert is_correct is False


def test_compute_score_no_submission():
    info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={"submitted_answer": None, "termination": "iter_cap"},
    )
    assert info["score"] == 0.0
    assert info["anls"] == 0.0
    # Strings (termination, submitted_answer, doc_id, ...) are NOT in
    # reward_extra_info — they break verl's numeric metric aggregation.
    assert "termination" not in info


def test_compute_score_correct_submission():
    info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={"submitted_answer": "42", "termination": "submit"},
    )
    assert info["score"] == 1.0
    assert info["anls"] == 1.0


def test_compute_score_wrong_submission():
    info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={"submitted_answer": "totally different", "termination": "submit"},
    )
    assert info["score"] == 0.0


def test_compute_score_handles_multi_alias_ground_truth():
    from docvqa.reward import compute_score
    # multi-alias gold stored as repr(list); a correct answer must score ~1.0
    gt = repr(["paris", "Paris"])
    out = compute_score(ground_truth=gt, extra_info={"submitted_answer": "Paris", "num_turns": 1})
    assert out["anls"] >= 0.99, out
    # single bare string still works
    out2 = compute_score(ground_truth="Paris", extra_info={"submitted_answer": "Paris", "num_turns": 1})
    assert out2["anls"] >= 0.99, out2
    # a clearly wrong answer stays low
    out3 = compute_score(ground_truth=gt, extra_info={"submitted_answer": "London", "num_turns": 1})
    assert out3["anls"] < 0.5, out3


def test_compute_score_passes_through_metadata():
    info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={
            "submitted_answer": "42", "termination": "submit",
            "num_turns": 5, "vlm_calls": 2,
            "messages": [{"role": "system", "content": "..."}],  # filtered out
            "doc_id": "doc1", "question_id": "q1",
        },
    )
    assert info["num_turns"] == 5
    assert info["vlm_calls"] == 2
    # Strings and variable-shape fields are filtered out; only numerics.
    assert "doc_id" not in info
    assert "submitted_answer" not in info
    assert "messages" not in info
    # search/search_calls no longer flow through anywhere.
    assert "search_calls" not in info
