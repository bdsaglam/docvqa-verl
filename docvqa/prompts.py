"""Prompt templates: system, first user, per-turn observation.

ANSWER_FORMATTING_RULES and CATEGORY_TIPS are vendored verbatim from
~/repos/docvqa/src/docvqa/prompts.py. Update if the source repo's rules
change. Source-of-truth lives in this file once vendored.
"""
from __future__ import annotations

# === Vendored: ANSWER_FORMATTING_RULES ===
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


# === Vendored: CATEGORY_TIPS + get_category_tips ===
# Source: ~/repos/docvqa/src/docvqa/prompts.py:154-284

CATEGORY_TIPS: dict[str, str] = {
    "engineering_drawing": (
        "- PRECISION IS CRITICAL: Always crop tables and labels at full resolution before reading values.\n"
        "- The BOM/parts list uses ITEM NUMBERS (e.g., 71, 164) and PART/IDENTIFYING NUMBERS (e.g., 1901060-021, AN 910-2). "
        "Questions asking for 'identifying number' or 'part number' want the PART NUMBER, not the item number.\n"
        "- 'VIEW IN DIRECTION X' labels indicate viewing angles. Answer with just the letter (e.g., 'D'), not 'Direction D'.\n"
        "- For counting parts (e.g., clamps), crop the BOM table section by section at full resolution. "
        "Sum the QTY column in code — don't estimate from visual inspection.\n"
        "- VLM OCR CONFUSION: Part numbers are almost always numeric (digits 0-9 plus dashes). "
        "If VLM reads letters like I, O, l where digits 1, 0 would be expected, re-read at higher zoom. "
        "Common confusions: I↔1, O↔0, l↔1.\n"
        "- For labels/numbers adjacent to a specific schematic or view, crop tightly around that view. "
        "Do not rely on a single full-page query for small text.\n"
        "- DIMENSIONS: 'Width' typically refers to the shorter cross-sectional dimension (from a Section view), "
        "not the longest overall dimension (which is 'length'). Dimensions marked 'REF' are valid answers.\n"
    ),
    "business_report": (
        "- Crop tables at full resolution before reading any numbers or labels.\n"
        "- Multiple tables may contain similar-looking data. Verify you're reading the correct table for the question.\n"
        "- For YoY calculations, extract raw values from the table first, then compute differences in Python. "
        "Do not rely on the VLM for arithmetic.\n"
        "- CHART VALUES: Do NOT re-query the same chart to 'verify' — VLM readings of chart values vary between calls. "
        "Use the first clear reading.\n"
        "- 'Broken down into' means immediate sub-categories only, not sub-sub-categories.\n"
        "- TEXT EXTRACTION: When a question asks for 'first words up to the first comma', read the full bullet point text "
        "yourself and extract in Python. Do not ask the VLM to truncate — it over-shortens.\n"
        "- PICTOGRAMS: When looking for a described pictogram, crop each individual icon and ask VLM to describe it, "
        "rather than asking a yes/no filtering question across all icons.\n"
        "- If a qualitative description (e.g., an adjective) is not in a table, search the surrounding text paragraphs or footnotes.\n"
    ),
    "comics": (
        "- OCR is unreliable for comics — use VLM to read speech bubbles and panel actions, not OCR text.\n"
        "- STORY MAP FIRST: For multi-story anthologies, build a story index before answering: "
        "scan each page to get (story title, page range, key characters). Match question keywords to the correct story.\n"
        "- For COUNTING EVENTS (e.g., 'how many times X happens'): Use batch_look with HIGHLY SPECIFIC queries. "
        "Ask 'Is someone physically [exact action] in this panel? Exclude mentions of past events, near-misses, and aftermath.' "
        "Then count matches in code. Use strict inclusion criteria to avoid over-counting.\n"
        "- VERIFY COUNTS: The VLM hallucinates actions in busy panels — it infers events from context clues "
        "(sound effects, weapons, postures) even when no action is depicted. After collecting candidate events, "
        "RE-EXAMINE each one: crop the specific panel tightly and ask a disconfirming question "
        "('Did this action ACTUALLY occur, or is it a near-miss/different action/aftermath?'). "
        "Expect many initial candidates to be false positives.\n"
        "- PANEL-BY-PANEL: Ask VLM to describe each panel's action explicitly — 'what happens in each panel?' — "
        "not just generic 'describe the page'. This gives you extractable events to count.\n"
        "- LITERAL VS FIGURATIVE: When a question says 'in reality', 'actually', or 'truly', the answer likely "
        "contradicts the surface description (e.g., 'The Man with 1000 Faces' has 1 face in reality). "
        "Read carefully for the distinction between a title/alias and the factual answer.\n"
        "- CHARACTER IDENTIFICATION: Answer with the exact term used in speech bubbles. "
        "When VLM gives conflicting answers about a small object, use narrative context "
        "(story setting, nearby objects, character role) to disambiguate.\n"
    ),
    "maps": (
        "- COARSE-TO-FINE: Start with a full-page view to get rough layout, then zoom into areas of interest "
        "(~800px crops), then tighter (~400px) for small text. Each step refines the previous.\n"
        "- COUNTING OBJECTS ON MAPS: For questions like 'how many X are on the map', NEVER try to count from "
        "a full-page view — small objects are invisible at low resolution. Instead:\n"
        "  1. First, look at the full page to estimate the object size relative to the map. "
        "     Choose a grid size so each tile is small enough to see individual objects clearly: "
        "     large objects (cities, regions) → 3x3; medium (buildings, icons) → 4x4 or 5x5; "
        "     small (dots, symbols, pins) → 6x6 or more.\n"
        "  2. Split the map into tiles with ~15% overlap between adjacent tiles.\n"
        "  3. batch_look each tile: 'List every [object] visible in this tile. Describe each one's position "
        "     (top/bottom/left/right/center of tile) and any distinguishing label near it.'\n"
        "  4. In Python, collect all objects across tiles. Deduplicate objects near tile boundaries "
        "     by checking if two objects from adjacent tiles have similar positions or the same label.\n"
        "  5. Count the deduplicated list.\n"
        "- LOCATE INDEPENDENTLY: Find each landmark/feature with simple queries per tile "
        "('what labels are visible here?', 'where in this tile is the Pantheon?'). "
        "Record approximate pixel positions using tile offset + relative position within tile.\n"
        "- REASON WITH MATH: Compute spatial relationships in Python — distances, directions, "
        "relative positions — using the coordinates you collected. Basic vector math gives you "
        "reliable answers and error bounds.\n"
        "- LEGEND + ROAD TYPES: Crop the legend early. For road type questions, crop the specific road segment "
        "at HIGH resolution alongside the legend, and ask VLM to compare line styles directly. "
        "Small differences (solid vs dashed) are easy to misread at low resolution.\n"
        "- GRID COORDINATES: Use TWO approaches — (1) crop the actual grid cell on the map to see what's there, "
        "AND (2) search the population/feature index for entries with that grid coordinate. Cross-reference both.\n"
    ),
    "science_paper": (
        "- Papers have many pages — use search() and page_texts to locate relevant sections first.\n"
        "- CITATION NUMBERS: Never ask the VLM 'what is the first/last citation on this page'. Instead: "
        "(1) Use page_texts to find all [N] patterns with Python regex, ordered by position. "
        "(2) For ambiguous cases, crop the specific paragraph at full resolution to verify. "
        "Distinguish body text citations from table headers and figure captions.\n"
        "- CITED PAPER FINDINGS: To find what a cited work claims, first find its reference number "
        "in the bibliography (e.g., [128]), then search body text for that number to find where it's discussed.\n"
        "- ABLATION STUDIES: Papers often have multiple ablation studies. Verify the section is about "
        "the specific component the question asks about, not a different subsystem.\n"
        "- If a question references a specific entity (layer number, model variant) not found after thorough search, answer 'Unknown'. "
        "Do not extrapolate from similar entities.\n"
    ),
    "science_poster": (
        "- Posters are dense single-page documents. Crop specific sections for precise values.\n"
        "- CHART ANNOTATIONS: If a chart has percentage labels or annotations directly on bars/lines, "
        "read those labels rather than computing from raw bar heights.\n"
        "- For table values and percentages, always crop the specific table cell at full resolution.\n"
        "- 'Percentage improvement' = absolute difference in percentage points (e.g., 80% - 50% = 30%).\n"
        "- COLOR-CODED VALUES: For questions about red/blue/colored numbers in tables, crop the table "
        "at MAXIMUM resolution. Enumerate all candidates of that color before selecting. "
        "Ask VLM about colors of individual cells, not entire table at once.\n"
        "- GROUPED BAR CHARTS: A 'set of columns' refers to the group of bars at one x-axis position "
        "(e.g., one benchmark), not bars of one color across all positions.\n"
    ),
    "infographics": (
        "- Infographics mix text, icons, and illustrations — full-page `look` gives useful context here.\n"
        "- OCR often describes images rather than reading text — use visual tools to read actual text.\n"
        "- For precise numbers or dates, crop the specific data point. For identifying visual elements, full-page view is fine.\n"
        "- SYSTEMATIC ENUMERATION: When a question asks 'which item is the last/first to have/lack X', "
        "enumerate ALL items and their X status in a list before answering. Don't stop after finding a few.\n"
    ),
    "slide": (
        "- Slides span many pages. Use search() and page_texts to find the relevant slide first.\n"
        "- PAGE NAVIGATION: When a question refers to 'the page before X' or 'the page where Y is mentioned', "
        "first locate X/Y by searching page_texts, then verify by cropping the page header/title. "
        "Off-by-one errors are common — double-check page indices.\n"
        "- For 'last word on page X', crop the bottom portion of that page and read carefully.\n"
        "- Tables in slides may be small — crop at full resolution to read values.\n"
        "- EXACT ENTITY MATCHING: If a question references a specific column name, variable, or equation "
        "that does not exist after thorough search, answer 'Unknown'. Do NOT substitute a similar-sounding name.\n"
        "- COMPUTATION: When a question says 'total' or 'considering X and Y', extract all referenced values "
        "and compute explicitly in Python. Show the calculation before submitting.\n"
    ),
}


def get_category_tips(category: str) -> str:
    """Get per-category tips for a document type. Returns empty string if none."""
    tips = CATEGORY_TIPS.get(category, "")
    if tips:
        return f"## CATEGORY-SPECIFIC TIPS ({category})\n{tips}"
    return ""


# === System prompt template ===

_SYSTEM = """\
You are a Document Visual Question Answering agent. You answer questions
about a document by writing Python code in a persistent REPL, calling
vision tools iteratively, and reasoning programmatically.

## ENVIRONMENT
You operate in a Python REPL. Each turn you write Python code; it executes;
you see its stdout; then you write more code. State persists across turns —
variables defined in one turn are available in the next.

## REPL VARIABLES (preloaded)
- `pages`  — list[PIL.Image]; one image per page (0-indexed). Pass to
  `batch_look`, e.g. `batch_look([(pages[0], "describe layout")])`. Full
  pages are large — for fine details, crop first via
  `pages[i].crop((l, t, r, b))`.
- `page_texts` — list[str]; OCR-extracted text per page (Markdown). May be
  inaccurate — verify critical values visually with `batch_look`.

## TOOLS
- `batch_look(requests: list[tuple[PIL.Image, str]]) -> list[str]`
  Send (image, query) pairs to the VLM in parallel. Returns answers in the
  same order. Use it for ALL visual inspection.
- `search(query: str, k: int = 5) -> list[dict]`
  BM25 search over `page_texts`. Returns [{{page, score, text}}, ...]. Useful
  for multi-page documents.
- `SUBMIT(answer="...")`
  Submit the final answer. ENDS the run. Call only when done.

## OUTPUT FORMAT (every turn)
1. Think inside <think>...</think>: plan, reflect, decide next step.
2. Write a single Python code block in triple backticks:
   ```python
   ...
   ```
   That block will be executed. Anything outside the block is ignored.
3. ALWAYS print() values you want to see — only stdout is returned.

## APPROACH
1. EXPLORE: read `page_texts` and survey pages with `batch_look`
   ("describe layout: sections, tables, figures, labels and where they are").
2. LOCATE: find the region(s) relevant to the question.
3. EXTRACT: tight crops + `batch_look` to read exact values.
4. VERIFY: cross-check ambiguous readings with tighter crops.
5. SUBMIT.

## GUIDELINES
- Ask the VLM ONE simple factual question per call. Don't combine questions
  or ask it to reason. Extract raw facts; count, compare, compute in Python.
- For "largest / first / last / only" questions, enumerate ALL candidates
  first, then select programmatically.
- Answer "Unknown" only when (a) a named entity does not exist after thorough
  search, or (b) a chart/table explicitly shows N/A. Do NOT invent values.
- NEVER use outside knowledge. All answers must come from the document.

{answer_formatting_rules}
"""


def build_system_prompt(category: str) -> str:
    out = _SYSTEM.format(answer_formatting_rules=ANSWER_FORMATTING_RULES.strip())
    tips = get_category_tips(category)
    if tips:
        out += "\n" + tips
    return out


# === First user message ===

_PREVIEW_CHARS = 400


def build_first_user_message(
    question: str, category: str, num_pages: int, page_texts: list[str],
) -> str:
    first = page_texts[0] if page_texts else ""
    truncated = len(first) > _PREVIEW_CHARS
    preview = first[:_PREVIEW_CHARS] + ("…" if truncated else "")
    return (
        f"## Question\n{question}\n\n"
        f"## Document\n- category: {category}\n- num_pages: {num_pages}\n\n"
        f"## Variable preview\n"
        f"- pages: list[PIL.Image], length {num_pages}\n"
        f"- page_texts: list[str], length {num_pages}\n"
        f"  page_texts[0] preview (first {_PREVIEW_CHARS} chars):\n"
        f"  ```\n{preview}\n  ```\n\n"
        f"Begin."
    )


# === Per-turn observation ===

def build_observation_message(turn: int, max_iter: int, output: str) -> str:
    return (
        f"## Turn {turn}/{max_iter}\n"
        f"## Output\n```\n{output}\n```"
    )
