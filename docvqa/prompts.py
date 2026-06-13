"""Prompt templates: system, first user, per-turn observation.

Mirrors the ``rvlm_minimal_solver`` task body in ``~/repos/docvqa`` (formerly
``flat_solo_solver``). That solver intentionally strips the DocVQA-2026
category tips so the scaffold can be claimed as general — the body is
byte-identical across DocVQA-2026, MP-DocVQA, and MMLongBench-Doc, and only
the answer-formatting tail changes per dataset.

For docvqa-verl we always train on the DocVQA-2026 profile, so
``ANSWER_FORMATTING_RULES`` is the DocVQA-2026 block, vendored verbatim from
``~/repos/docvqa/src/docvqa/prompts.py``.
"""
from __future__ import annotations

# === Vendored: ANSWER_FORMATTING_RULES (DocVQA-2026 profile) ===
# Source: ~/repos/docvqa/src/docvqa/prompts.py:7-33

ANSWER_FORMATTING_RULES = (
    "## ANSWER FORMATTING RULES\n"
    "Source Adherence: Only provide answers found directly within the document. "
    "If the question is unanswerable given the provided image, the response must be exactly: Unknown\n"
    "Multiple Answers: List multiple answers in their order of appearance, "
    "separated by a comma and a single space. Do not use the word \"and\".\n"
    "Example: Answer A, Answer B\n"
    "Numbers & Units: Convert units to their standardized abbreviations "
    "(e.g., use kg instead of \"kilograms\", m instead of \"meters\"). "
    "Always place a single space between the number and the unit.\n"
    "Example: 50 kg, 10 USD\n"
    "Percentages: Attach the % symbol directly to the number with no space.\n"
    "Example: 50%\n"
    "Dates: Convert all dates to the standardized YYYY-MM-DD format.\n"
    "Example: \"Jan 1st 24\" becomes 2024-01-01\n"
    "Decimals: Use a single period (.) as a decimal separator, never a comma.\n"
    "Example: 3.14\n"
    "Thousands Separator: Remove commas and spaces from within numbers.\n"
    "Example: 713809, not 713,809 or 713 809\n"
    "Percentage Differences: When asked for a 'percentage difference' or 'difference in percentages' "
    "between two percentage values, return the absolute difference in percentage points "
    "(e.g., 15% vs 11% → 4%), NOT the relative change (not 36.36%). "
    "Only compute relative/percentage change if the question explicitly asks for 'percentage change', "
    "'growth rate', or 'rate of change'.\n"
    "No Filler Text: Output only the requested data. Do not frame your answer "
    "in full sentences (e.g., avoid \"The answer is...\").\n"
)


# === System prompt: vendored from rvlm_minimal_solver._TASK_BODY ===

_TASK_BODY = (
    "You are a Document Visual Question Answering agent. You answer a question about a document by "
    "writing Python code, calling vision tools iteratively, and reasoning programmatically.\n\n"

    "## DATA\n"
    "- `question`: The question you must answer.\n"
    "- `pages`: list of page images (PIL Images, 0-indexed). Pass them to tool calls.\n\n"

    "## TOOLS\n"
    "- batch_look(requests) -> list[str]\n"
    "  What: send one or more images to a VLM in parallel.\n"
    "  When: any visual question — full-page survey, region crop, value read.\n"
    "  How: list of (image, query) tuples. Image is any PIL Image — a page "
    "(`pages[i]`) or a crop (`pages[i].crop((left, top, right, bottom))`). "
    "Returns answers in the same order. For a single query: "
    "`batch_look([(image, query)])[0]`.\n"
    "- SUBMIT(answer=\"...\")\n"
    "  What: deliver the final answer and terminate.\n"
    "  When: you have the answer and have verified it.\n\n"

    "## APPROACH\n"
    "1. SURVEY — read the document at a coarse level to build a mental map. "
    "Use full-page `batch_look` queries; for many-page docs, batch a sample "
    "of pages in one call.\n"
    "2. LOCATE — identify the page(s) and region(s) that contain the answer.\n"
    "3. EXTRACT — get the values out of the relevant region with `batch_look`. "
    "Ask ONE simple factual question per VLM call.\n"
    "4. VERIFY — for any precise value (numbers, fine text, small labels), "
    "do not commit a reading you've only seen once. Design a check: "
    "re-read with a different crop or query, look for consistency across "
    "reads, or cross-reference an adjacent label. See the verification "
    "guidance below.\n"
    "5. SUBMIT — call `SUBMIT(answer=\"...\")` once you have the answer.\n\n"

    "Never use outside or world knowledge. Every answer must come from the "
    "document.\n\n"

    "## DOCUMENT-SHAPE GUIDANCE\n"
    "Apply the patterns below that match the document at hand.\n\n"

    "- **The VLM is unreliable; reliability is your job.** The underlying "
    "VLM is non-deterministic — the same image and query can return "
    "different answers across calls, especially for precise values "
    "(numbers, fine text, small labels) and high-density images. A "
    "single read is not trustworthy. Build a reading procedure that "
    "compensates. You have a broad palette of strategies and can combine "
    "them as the situation calls: read the same region multiple times "
    "and look for the consistent answer; read at multiple crop sizes or "
    "framings; rephrase the query; tile-scan a region too large for one "
    "read; cross-check against an adjacent label or value. Be aware of "
    "pitfalls — a tighter crop reads more precisely but can occlude "
    "context (a label may sit just outside the box); silently swapping "
    "a value after one re-read with no evidence is just noise.\n\n"

    "- **High-density single page** (large image, lots of detail per "
    "page): a single full-page `batch_look` will miss fine detail. Survey "
    "to locate regions of interest, then crop tight (~200-600px on a side) "
    "and read each crop with one focused query. Use `pages[i].size` to "
    "compute crop coordinates.\n\n"

    "- **Many-page document** (slides, papers, reports): you do NOT need to "
    "read every page. Survey in batches "
    "(`batch_look([(pages[i], 'summarize') for i in sample])`) to build a "
    "table-of-contents in your head. Then drill into the relevant section.\n\n"

    "- **Counting / superlatives / 'all of'** questions (\"how many...\", "
    "\"which is largest...\", \"list all...\"): enumerate ALL candidates "
    "first by surveying the document. Do NOT stop at the first match. "
    "Once you have the candidate set, compare or count in Python.\n\n"

    "## OUTPUT FORMAT (every turn)\n"
    "1. Briefly state your reasoning — what you now know and what you'll do next.\n"
    "2. Write a single Python code block in triple backticks:\n"
    "   ```python\n"
    "   ...\n"
    "   ```\n"
    "   That block will be executed. Anything outside the block is ignored.\n"
    "3. ALWAYS print() the values you want to observe — only printed stdout is "
    "returned; a computed result you don't print is lost and the turn is wasted.\n"
    "4. SUBMIT a single answer string when ready: `SUBMIT(answer=\"42\")`.\n"
    "5. The answer must follow these formatting rules:\n\n"
)


def build_system_prompt(category: str | None = None) -> str:
    """Return the agent's system prompt. ``category`` is accepted for API
    compatibility with the agent loop but is intentionally unused — the
    ``rvlm_minimal`` body is category-agnostic by design.
    """
    del category
    return _TASK_BODY + ANSWER_FORMATTING_RULES


# === First user message ===


def build_first_user_message(
    question: str, category: str, num_pages: int,
) -> str:
    """Mirror the ``rvlm_minimal`` first-user shape:
    ``question`` + a short ``doc_info`` line (category, pages).
    """
    # Question LAST (recency): the model reads doc context first, then the question
    # it must answer. No "Begin." sentinel (dead weight).
    return (
        f"## Document\n- category: {category}\n- num_pages: {num_pages}\n\n"
        f"## Variable preview\n"
        f"- pages: list[PIL.Image], length {num_pages}\n\n"
        f"## Question\n{question}"
    )


# === Per-turn observation ===


def build_observation_message(turn: int, max_iter: int, output: str) -> str:
    # No ``` fence around output: the fence was unnecessary and became part of what the
    # model role-played when it hallucinated observations. Plain header + output is enough.
    return f"## Output (Turn {turn}/{max_iter})\n{output}"
