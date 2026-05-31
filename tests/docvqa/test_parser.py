"""Tests for code-fence extraction."""
from docvqa.parser import parse_last_python_fence


def test_single_python_fence():
    text = "Some prose.\n```python\nprint(1)\n```"
    assert parse_last_python_fence(text) == "print(1)"


def test_no_lang_fence_accepted():
    text = "```\nprint(2)\n```"
    assert parse_last_python_fence(text) == "print(2)"


def test_py_alias_accepted():
    text = "```py\nprint(3)\n```"
    assert parse_last_python_fence(text) == "print(3)"


def test_multiple_fences_returns_last():
    text = "```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```"
    assert parse_last_python_fence(text) == "print(2)"


def test_no_fence_returns_none():
    assert parse_last_python_fence("just prose, no code") is None


def test_fences_inside_think_are_ignored():
    text = (
        "<think>\nLet me draft:\n```python\nprint('draft')\n```\nMaybe rethink.\n</think>\n"
        "```python\nprint('final')\n```"
    )
    assert parse_last_python_fence(text) == "print('final')"


def test_only_think_fence_returns_none():
    text = "<think>\n```python\nprint('inner')\n```\n</think>\n\nNo final code."
    assert parse_last_python_fence(text) is None


def test_unclosed_think_falls_through():
    # Defensive: if the model emits <think> without </think>, treat the rest as outside-think.
    text = "<think>\nplanning... ```python\nprint(0)\n```"
    # Policy: an unclosed <think> means no fence counts.
    assert parse_last_python_fence(text) is None


def test_strips_outer_whitespace():
    text = "```python\n   x = 1\n   y = 2\n```"
    assert parse_last_python_fence(text) == "   x = 1\n   y = 2"
