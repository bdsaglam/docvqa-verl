"""Tests for prompt-template construction."""
from docvqa.prompts import (
    build_system_prompt, build_first_user_message, build_observation_message,
    ANSWER_FORMATTING_RULES, get_category_tips, CATEGORY_TIPS,
)


def test_system_prompt_mentions_tools():
    sp = build_system_prompt(category="business_report")
    for required in ("batch_look", "search", "SUBMIT", "<think>", "```python"):
        assert required in sp, f"missing: {required}"


def test_system_prompt_includes_answer_format_rules():
    sp = build_system_prompt(category="business_report")
    # First non-empty line of the rules should appear.
    first_line = next(line for line in ANSWER_FORMATTING_RULES.strip().splitlines() if line.strip())
    assert first_line.strip() in sp


def test_system_prompt_includes_category_tips_when_present():
    tips = get_category_tips("business_report")
    assert tips  # business_report has tips
    sp = build_system_prompt(category="business_report")
    # `tips` is the formatted block returned by get_category_tips, including its header.
    # We just check that one distinctive line of the tips body shows up.
    body = CATEGORY_TIPS["business_report"]
    distinctive = next(line for line in body.strip().splitlines() if line.strip())
    assert distinctive.strip() in sp


def test_system_prompt_no_category_tips_when_unknown():
    sp = build_system_prompt(category="non_existent_category_xyz")
    assert sp  # still produces a prompt
    assert "CATEGORY" not in sp  # no category tips heading


def test_first_user_message_includes_question_and_doc_meta():
    page_texts = ["First page content here.", "Second page.", "Third."]
    msg = build_first_user_message(
        question="What was Q3 revenue?",
        category="business_report",
        num_pages=3,
        page_texts=page_texts,
    )
    assert "What was Q3 revenue?" in msg
    assert "business_report" in msg
    assert "3" in msg
    assert "First page content here." in msg


def test_first_user_message_truncates_long_first_page():
    long_text = "x" * 1000
    msg = build_first_user_message(
        question="?", category="c", num_pages=1, page_texts=[long_text],
    )
    assert "…" in msg
    assert "x" * 500 not in msg  # not the full thing


def test_observation_message_includes_turn_counter():
    msg = build_observation_message(turn=3, max_iter=20, output="hello")
    assert "Turn 3/20" in msg
    assert "hello" in msg
