"""Extract the last ```python ... ``` fence from a model response.

Fences nested inside <think>...</think> blocks are ignored — only fences
emitted *after* the last </think> (or in plain text if no <think>) count.
"""
from __future__ import annotations

import re

# Match a code fence with optional `python` or `py` language tag.
_FENCE_RE = re.compile(
    r"```(?:python|py)?[ \t]*\n(.*?)\n```",
    re.DOTALL,
)


def _strip_think_blocks(text: str) -> str:
    """Remove all closed <think>...</think> blocks. Unclosed <think> => empty after the tag."""
    # Drop closed think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # If an unclosed <think> remains, drop it and everything after it.
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    return text


def parse_last_python_fence(text: str) -> str | None:
    """Return the contents of the last python fence outside <think>, or None."""
    if not text:
        return None
    outside = _strip_think_blocks(text)
    matches = _FENCE_RE.findall(outside)
    if not matches:
        return None
    return matches[-1]
