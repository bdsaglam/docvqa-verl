"""Tests for prompt-template construction."""
from docvqa.prompts import (
    ANSWER_FORMATTING_RULES,
    build_first_user_message,
    build_observation_message,
    build_system_prompt,
)


def test_system_prompt_mentions_tools_and_format():
    sp = build_system_prompt(category="business_report")
    for required in ("batch_look", "SUBMIT", "<think>", "```python"):
        assert required in sp, f"missing: {required}"


def test_system_prompt_does_not_mention_dropped_tools():
    """rvlm_minimal scaffold dropped BM25 search and the `look` helper."""
    sp = build_system_prompt(category="business_report")
    assert "search(query" not in sp
    assert "BM25" not in sp
    assert "page_texts" not in sp
    assert "look(image" not in sp  # `look(image, query)` is gone — only batch_look


def test_system_prompt_is_category_agnostic():
    """rvlm_minimal intentionally strips per-category tips so the body is
    byte-identical regardless of category."""
    a = build_system_prompt(category="business_report")
    b = build_system_prompt(category="comics")
    assert a == b


def test_system_prompt_includes_answer_format_rules():
    sp = build_system_prompt(category="business_report")
    first_line = next(line for line in ANSWER_FORMATTING_RULES.strip().splitlines() if line.strip())
    assert first_line.strip() in sp


def test_first_user_message_includes_question_and_doc_meta():
    msg = build_first_user_message(
        question="What was Q3 revenue?",
        category="business_report",
        num_pages=3,
    )
    assert "What was Q3 revenue?" in msg
    assert "business_report" in msg
    assert "3" in msg


def test_first_user_message_does_not_advertise_page_texts():
    msg = build_first_user_message(question="?", category="c", num_pages=1)
    assert "page_texts" not in msg


def test_observation_message_includes_turn_counter():
    msg = build_observation_message(turn=3, max_iter=20, output="hello")
    assert "Turn 3/20" in msg
    assert "hello" in msg
