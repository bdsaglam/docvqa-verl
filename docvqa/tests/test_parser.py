"""Tests for python-fence extraction (docvqa/parser.py)."""

from docvqa.parser import (
    parse_first_python_fence,
    parse_last_python_fence,
    parse_python_fences,
)

_TWO_BLOCKS = """<think>planning</think>
Here is the first:
```python
print("first")
```
and a second:
```python
print("second")
```
"""

_FENCE_IN_THINK_ONLY = """<think>
```python
print("scratch inside think — must be ignored")
```
</think>
no real fence here
"""


def test_parse_python_fences_returns_all_in_order():
    fences = parse_python_fences(_TWO_BLOCKS)
    assert len(fences) == 2
    assert 'print("first")' in fences[0]
    assert 'print("second")' in fences[1]


def test_parse_python_fences_ignores_fences_inside_think():
    assert parse_python_fences(_FENCE_IN_THINK_ONLY) == []


def test_parse_python_fences_empty_text():
    assert parse_python_fences("") == []
    assert parse_python_fences("no code at all") == []


def test_first_vs_last_fence_selection():
    assert 'print("first")' in parse_first_python_fence(_TWO_BLOCKS)
    assert 'print("second")' in parse_last_python_fence(_TWO_BLOCKS)


def test_single_fence_first_equals_last():
    one = "intro\n```python\nx = 1\n```\n"
    assert parse_first_python_fence(one) == parse_last_python_fence(one)
    assert len(parse_python_fences(one)) == 1


def test_no_fence_returns_none():
    assert parse_first_python_fence("nothing") is None
    assert parse_last_python_fence("nothing") is None
