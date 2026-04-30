# DocVQA-verl Scaffold + Phase-1 GRPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a verl-native `AgentLoopBase` subclass that replicates the deployed `flat_solo` REPL agent on DocVQA-2026 (Layer-3 ANLS within ±5pp of the reference) and a Phase-1 GRPO training launcher.

**Architecture:** Custom `DocVQAReplAgentLoop` runs a persistent CPython subprocess via JSON-line IPC; the model emits `<think>...</think>` + a single ```` ```python ```` ` ``` ` fence per turn; tools `batch_look` (HTTP to a frozen 27B VLM) and `search` (BM25 per-doc) live host-side; trajectories return as standard `AgentLoopOutput` with `extra_fields` carrying structured rollout metadata for the JSONL dump.

**Tech Stack:** verl (mainline fork), PyTorch + FSDP-2 + LoRA, vLLM, asyncio + httpx, bm25s, PIL, pytest.

**Spec:** `docs/superpowers/specs/2026-04-30-docvqa-verl-training-design.md`

---

## File Structure

```
recipe/docvqa/
  __init__.py                  # Empty package marker.
  agent_loop.py                # DocVQAReplAgentLoop(AgentLoopBase). Orchestrator.
  subprocess_interp.py         # Vendored CPython REPL with JSON-line IPC.
  sandbox.py                   # Inline Python: pages, page_texts, SUBMIT, IPC proxies.
  tools.py                     # Host-side: batch_look (HTTP), search (BM25).
  prompts.py                   # System prompt + first-user-message + observation templates.
  parser.py                    # Fenced-code-block extraction.
  reward.py                    # ANLS reward function for verl.
  dataset.py                   # Optional verl RLHFDataset adapter (or just JSON->row mapping).
  agent.yaml                   # Hydra registration of docvqa_repl agent.
  scripts/
    prepare_data.py            # Materialize data/{val,test,train}/docs/.
    eval.py                    # Layer-3 ANLS reproduction runner.
    run_smoke_grpo.sh          # Layer-4 sanity training launcher.
    run_phase1_grpo.sh         # Phase-1 production launcher.

tests/recipe/docvqa/
  __init__.py
  conftest.py                  # Shared pytest fixtures.
  test_parser.py               # Code-fence extraction.
  test_subprocess_interp.py    # IPC + state persistence + SUBMIT.
  test_prompts.py              # Template rendering.
  test_tools.py                # batch_look (mocked HTTP), search (real bm25s).
  test_reward.py               # ANLS scoring + extra_info passthrough.
  test_agent_loop.py           # Layer-2 end-to-end with scripted server_manager.
  fixtures/
    sample_doc/                # Tiny doc_dir for tests (3 pages, fake OCR, BM25).
    canned_trajectory.json     # Scripted assistant turns for test_agent_loop.

data/
  val/{docs/{doc_id}/{pages,ocr,bm25}, questions.json}    # Built by prepare_data.py.
```

---

## Task 1: Bootstrap package and tests harness

**Files:**
- Create: `recipe/docvqa/__init__.py`
- Create: `tests/recipe/docvqa/__init__.py`
- Create: `tests/recipe/docvqa/conftest.py`
- Create: `tests/recipe/docvqa/fixtures/sample_doc/metadata.json`
- Create: `tests/recipe/docvqa/fixtures/sample_doc/pages/page_0.png`
- Create: `tests/recipe/docvqa/fixtures/sample_doc/pages/page_1.png`
- Create: `tests/recipe/docvqa/fixtures/sample_doc/ocr/page_0.md`
- Create: `tests/recipe/docvqa/fixtures/sample_doc/ocr/page_1.md`

- [ ] **Step 1: Create empty package markers**

```python
# recipe/docvqa/__init__.py
"""DocVQA recipe: agent loop, tools, prompts, reward."""
```

```python
# tests/recipe/docvqa/__init__.py
```

- [ ] **Step 2: Create the sample_doc fixture used by every test that needs a real doc_dir**

```bash
mkdir -p tests/recipe/docvqa/fixtures/sample_doc/{pages,ocr,bm25}
```

Then write `tests/recipe/docvqa/fixtures/sample_doc/metadata.json`:

```json
{
  "doc_id": "sample_doc",
  "doc_category": "business_report",
  "num_pages": 2,
  "source_dataset": "test_fixture"
}
```

`tests/recipe/docvqa/fixtures/sample_doc/ocr/page_0.md`:

```markdown
# Q3 Financial Report

The total revenue in Q3 was $1.2 billion, up 15% year-over-year.
Net income reached $250 million.
```

`tests/recipe/docvqa/fixtures/sample_doc/ocr/page_1.md`:

```markdown
# Forward Guidance

Q4 revenue is projected at $1.4 billion.
```

Generate two trivial 100×100 PNG pages via Python:

```python
# Run once, manually:
from PIL import Image
Image.new("RGB", (100, 100), color="white").save(
    "tests/recipe/docvqa/fixtures/sample_doc/pages/page_0.png")
Image.new("RGB", (100, 100), color="lightgray").save(
    "tests/recipe/docvqa/fixtures/sample_doc/pages/page_1.png")
```

Commit the PNGs as binary (small).

- [ ] **Step 3: Build the BM25 index for sample_doc using bm25s**

Create `tests/recipe/docvqa/fixtures/sample_doc/build_index.py` (run once, output committed):

```python
"""Build BM25 index for sample_doc fixture. Run once, committed output."""
import json
from pathlib import Path
import bm25s
import Stemmer

ROOT = Path(__file__).parent
ocr_dir = ROOT / "ocr"
bm25_dir = ROOT / "bm25"
bm25_dir.mkdir(exist_ok=True)

chunks = []
for p in sorted(ocr_dir.glob("page_*.md")):
    page = int(p.stem.split("_")[1])
    chunks.append({"page": page, "text": p.read_text()})

(bm25_dir / "chunks.json").write_text(json.dumps(chunks, indent=2))

corpus = [c["text"] for c in chunks]
tokens = bm25s.tokenize(corpus, stemmer=Stemmer.Stemmer("english"))
retriever = bm25s.BM25()
retriever.index(tokens)
retriever.save(bm25_dir, corpus=None)
print(f"Wrote {len(chunks)} chunks to {bm25_dir}")
```

Run it: `python tests/recipe/docvqa/fixtures/sample_doc/build_index.py`. Verify `bm25/` contains `chunks.json`, `data.csc.index.npy`, `indices.csc.index.npy`, `indptr.csc.index.npy`, `params.index.json`, `vocab.index.json`.

- [ ] **Step 4: Create conftest.py with the sample_doc path fixture**

```python
# tests/recipe/docvqa/conftest.py
"""Shared pytest fixtures for recipe/docvqa tests."""
from pathlib import Path

import pytest


@pytest.fixture
def sample_doc_dir() -> Path:
    """Path to the committed sample doc_dir fixture."""
    return Path(__file__).parent / "fixtures" / "sample_doc"
```

- [ ] **Step 5: Verify pytest discovers the empty test directory**

```bash
cd /home/baris/repos/docvqa-verl
pytest tests/recipe/docvqa/ -v
```

Expected: "no tests ran" (collected 0 items). No errors.

- [ ] **Step 6: Commit**

```bash
git add recipe/docvqa/__init__.py tests/recipe/docvqa/
git commit -m "[docvqa] feat: scaffold recipe/docvqa package and test fixtures"
```

---

## Task 2: Code-fence parser

**Files:**
- Create: `recipe/docvqa/parser.py`
- Create: `tests/recipe/docvqa/test_parser.py`

The parser must extract the *last* ```` ```python ```` fence in the model's text, ignore fences nested inside `<think>` blocks, and tolerate `python`/no-lang/whitespace variations.

- [ ] **Step 1: Write failing tests covering the contract**

```python
# tests/recipe/docvqa/test_parser.py
"""Tests for code-fence extraction."""
from recipe.docvqa.parser import parse_last_python_fence


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
    # Depending on policy: we choose to ignore the fence inside an unclosed think.
    # If the model never closes <think>, no fence counts.
    assert parse_last_python_fence(text) is None


def test_strips_outer_whitespace():
    text = "```python\n   x = 1\n   y = 2\n```"
    assert parse_last_python_fence(text) == "   x = 1\n   y = 2"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/recipe/docvqa/test_parser.py -v
```

Expected: `ImportError` on `recipe.docvqa.parser`.

- [ ] **Step 3: Implement the parser**

```python
# recipe/docvqa/parser.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/recipe/docvqa/test_parser.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/parser.py tests/recipe/docvqa/test_parser.py
git commit -m "[docvqa] feat: code-fence parser with <think> awareness"
```

---

## Task 3: Subprocess interpreter

**Files:**
- Create: `recipe/docvqa/subprocess_interp.py`
- Create: `tests/recipe/docvqa/test_subprocess_interp.py`

Vendored from `~/repos/docvqa/src/docvqa/rlm/subprocess_interpreter.py` with:
- `display`, `RESET_HISTORY`, `dspy_lm` config removed
- `SUBMIT`, `_make_tool_proxy`, JSON-line IPC kept
- Added `extra_globals: dict[str, str]` to inject env vars for the subprocess

- [ ] **Step 1: Write failing tests for the contract**

```python
# tests/recipe/docvqa/test_subprocess_interp.py
"""Tests for the vendored subprocess interpreter."""
import pytest

from recipe.docvqa.subprocess_interp import (
    SubprocessInterpreter, FinalOutput, CodeInterpreterError,
)


def test_state_persists_across_executes():
    interp = SubprocessInterpreter()
    try:
        interp.execute("x = 7")
        out = interp.execute("print(x + 3)")
        assert "10" in out
    finally:
        interp.shutdown()


def test_submit_returns_final_output():
    interp = SubprocessInterpreter(output_fields=[{"name": "answer", "type": "str"}])
    try:
        result = interp.execute("SUBMIT(answer='42')")
        # SubprocessInterpreter.execute returns (FinalOutput, captured_stdout) on SUBMIT
        assert isinstance(result, tuple)
        final, _captured = result
        assert isinstance(final, FinalOutput)
        assert final.output == {"answer": "42"}
    finally:
        interp.shutdown()


def test_tool_call_round_trip():
    """Host-registered tool should be callable from subprocess code via IPC."""
    calls = []

    def echo_tool(s: str) -> str:
        calls.append(s)
        return s.upper()

    interp = SubprocessInterpreter(tools={"echo_tool": echo_tool})
    try:
        out = interp.execute("print(echo_tool('hello'))")
        assert "HELLO" in out
        assert calls == ["hello"]
    finally:
        interp.shutdown()


def test_runtime_error_returns_error_marker():
    interp = SubprocessInterpreter()
    try:
        out = interp.execute("1 / 0")
        # Expect raised CodeInterpreterError wrapping a Python ZeroDivisionError
        # (the error path returns a string in some flows, raises in others —
        # see the implementation; both forms are acceptable but consistent).
        # We assert: any error surface includes 'ZeroDivisionError'.
        assert "ZeroDivisionError" in str(out)
    except CodeInterpreterError as e:
        assert "ZeroDivisionError" in str(e)
    finally:
        interp.shutdown()


def test_sandbox_code_runs_at_startup():
    interp = SubprocessInterpreter(sandbox_code="GREETING = 'hi'")
    try:
        out = interp.execute("print(GREETING)")
        assert "hi" in out
    finally:
        interp.shutdown()


def test_shutdown_is_idempotent():
    interp = SubprocessInterpreter()
    interp.start()
    interp.shutdown()
    interp.shutdown()  # must not raise
```

- [ ] **Step 2: Run the tests — they fail**

```bash
pytest tests/recipe/docvqa/test_subprocess_interp.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Vendor the interpreter**

Copy `~/repos/docvqa/src/docvqa/rlm/subprocess_interpreter.py` to `recipe/docvqa/subprocess_interp.py`, then make these edits:

1. **Remove DSPy import lines** at top — replace `from dspy.primitives.code_interpreter import CodeInterpreterError, FinalOutput` with our own definitions:

```python
# At top of recipe/docvqa/subprocess_interp.py, after stdlib imports:
class CodeInterpreterError(RuntimeError):
    """Raised by the interpreter when subprocess code fails or IPC misbehaves."""


class FinalOutput:
    """Marker for a successful SUBMIT() call.

    Attributes:
        output: dict[str, Any] — the kwargs passed to SUBMIT (e.g. {"answer": "42"}).
    """
    def __init__(self, output):
        self.output = output
```

2. **Inside `_REPL_SCRIPT`, delete the entire `display(...)` function definition and the `_display_max_pixels` global setup.** Remove the line `_display_max_pixels = config.get("display_max_pixels") or 1_000_000`. Remove the `display` entry from the namespace dict. Keep `SUBMIT`. **Delete** the entire `RESET_HISTORY` function and `_ResetHistorySignal` exception class. Delete the `_handle_display_image` host method, the `pop_images` method, and `self._pending_images`/`self._image_counter` from `__init__`. Delete the corresponding handlers in the host `execute` loop (`if result.get("type") == "display_image": ...`).

3. **Remove `dspy_lm` from the config**: delete the `if config.get("dspy_lm"): ...` block in the REPL script and the `_extract_lm_config` static method on the host class. Remove `dspy_lm` from `__init__`.

4. **Remove `_handle_reset_history`** branch from `execute`.

5. **Add `extra_env: dict[str, str] | None = None`** parameter to `__init__`. In `start()`, after `env = os.environ.copy()`, add `if self._extra_env: env.update(self._extra_env)`.

6. Drop `display_max_pixels` parameter and the `_display_max_pixels` config plumbing.

The resulting file is a self-contained interpreter with: `start()`, `shutdown()`, `execute(code, variables=None)`, `tools` property, `_send`/`_recv` plumbing, `_handle_tool_call`. Roughly 350-400 lines.

- [ ] **Step 4: Run tests — they pass**

```bash
pytest tests/recipe/docvqa/test_subprocess_interp.py -v
```

Expected: 6 passed. If `test_runtime_error_returns_error_marker` fails because the implementation raises rather than returns: keep that branch but adjust — the test accepts either.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/subprocess_interp.py tests/recipe/docvqa/test_subprocess_interp.py
git commit -m "[docvqa] feat: vendored subprocess interpreter (no DSPy)"
```

---

## Task 4: Sandbox startup snippet

**Files:**
- Create: `recipe/docvqa/sandbox.py`
- Create: `tests/recipe/docvqa/test_sandbox.py`

The "sandbox code" is a Python string injected into the subprocess at startup. It reads `DOC_DIR` from env, populates `pages` and `page_texts`, and defines `SUBMIT` (which raises `_FinalOutputSignal`, already in the REPL script). Tool proxies (`batch_look`, `search`) are auto-installed by `SubprocessInterpreter` from `tools=...`.

- [ ] **Step 1: Write tests for sandbox setup**

```python
# tests/recipe/docvqa/test_sandbox.py
"""Tests that the sandbox preloads pages, page_texts, and globals correctly."""
import os
from pathlib import Path

import pytest

from recipe.docvqa.sandbox import build_sandbox_code
from recipe.docvqa.subprocess_interp import SubprocessInterpreter


def test_sandbox_loads_pages_and_page_texts(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print(len(pages), len(page_texts))")
        assert "2 2" in out
    finally:
        interp.shutdown()


def test_sandbox_page_text_content(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print('Q3' in page_texts[0])")
        assert "True" in out
    finally:
        interp.shutdown()


def test_sandbox_pages_are_pil_images(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print(pages[0].size)")
        assert "(100, 100)" in out
    finally:
        interp.shutdown()
```

- [ ] **Step 2: Run tests — fail**

```bash
pytest tests/recipe/docvqa/test_sandbox.py -v
```

Expected: ImportError on `recipe.docvqa.sandbox`.

- [ ] **Step 3: Implement `build_sandbox_code`**

```python
# recipe/docvqa/sandbox.py
"""Inline Python injected into the subprocess at startup.

Reads DOC_DIR env var, loads `pages` (PIL.Image list) and `page_texts`
(list[str]). The IPC tool proxies (batch_look, search) and SUBMIT are
already in the REPL namespace before this code runs.
"""
from __future__ import annotations

_TEMPLATE = '''
import os
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500_000_000

_doc_dir = Path(os.environ["DOC_DIR"])
pages = [
    Image.open(p)
    for p in sorted((_doc_dir / "pages").glob("page_*.png"),
                    key=lambda p: int(p.stem.split("_")[1]))
]
page_texts = [
    p.read_text()
    for p in sorted((_doc_dir / "ocr").glob("page_*.md"),
                    key=lambda p: int(p.stem.split("_")[1]))
]
'''


def build_sandbox_code() -> str:
    """Return the Python source string to inject at subprocess startup."""
    return _TEMPLATE
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_sandbox.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/sandbox.py tests/recipe/docvqa/test_sandbox.py
git commit -m "[docvqa] feat: sandbox startup loads pages and OCR from DOC_DIR"
```

---

## Task 5: Tool host handlers — `search` (BM25)

**Files:**
- Create: `recipe/docvqa/tools.py` (initial — search only; batch_look in next task)
- Create: `tests/recipe/docvqa/test_tools.py` (search tests now; batch_look mocks next task)

- [ ] **Step 1: Write tests for `search`**

```python
# tests/recipe/docvqa/test_tools.py
"""Tests for the host-side tool handlers."""
from pathlib import Path

import pytest

from recipe.docvqa.tools import search, clear_bm25_cache


@pytest.fixture(autouse=True)
def _reset_bm25_cache():
    clear_bm25_cache()
    yield
    clear_bm25_cache()


def test_search_returns_relevant_page(sample_doc_dir):
    results = search(str(sample_doc_dir), "Q3 revenue", k=5)
    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert top["page"] == 0
    assert top["score"] > 0
    assert "Q3" in top["text"] or "revenue" in top["text"].lower()


def test_search_filters_zero_scores(sample_doc_dir):
    # A nonsense query should return either empty or only positive-score hits.
    results = search(str(sample_doc_dir), "zzzzzz nonsense token unlikely", k=5)
    for r in results:
        assert r["score"] > 0


def test_search_caches_index(sample_doc_dir):
    """Second call should not rebuild the retriever (process-local cache)."""
    from recipe.docvqa import tools
    search(str(sample_doc_dir), "Q3", k=3)
    assert str(sample_doc_dir) in tools._BM25_CACHE
    n_before = id(tools._BM25_CACHE[str(sample_doc_dir)])
    search(str(sample_doc_dir), "different", k=3)
    n_after = id(tools._BM25_CACHE[str(sample_doc_dir)])
    assert n_before == n_after  # same cached object
```

- [ ] **Step 2: Run tests — they fail**

```bash
pytest tests/recipe/docvqa/test_tools.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `search` and the cache**

```python
# recipe/docvqa/tools.py
"""Host-side tool handlers: batch_look (HTTP→VLM) and search (BM25)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

# Process-local cache: doc_dir -> (BM25 retriever, list of chunk dicts)
_BM25_CACHE: dict[str, tuple[Any, list[dict]]] = {}


def clear_bm25_cache() -> None:
    """Drop the BM25 cache (used by tests)."""
    _BM25_CACHE.clear()


def _load_bm25(doc_dir: str) -> tuple[Any, list[dict]]:
    if doc_dir in _BM25_CACHE:
        return _BM25_CACHE[doc_dir]
    bm25_dir = Path(doc_dir) / "bm25"
    retriever = bm25s.BM25.load(str(bm25_dir), load_corpus=False)
    chunks = json.loads((bm25_dir / "chunks.json").read_text())
    _BM25_CACHE[doc_dir] = (retriever, chunks)
    return retriever, chunks


def search(doc_dir: str, query: str, k: int = 5) -> list[dict]:
    """BM25 search over the document's per-page OCR.

    Returns: list of {"page": int, "score": float, "text": str}, sorted by
    decreasing score, with score > 0.
    """
    retriever, chunks = _load_bm25(doc_dir)
    tokens = bm25s.tokenize([query], stemmer=Stemmer.Stemmer("english"))
    n = min(k, len(chunks))
    indices, scores = retriever.retrieve(tokens, k=n)
    out = []
    for idx, score in zip(indices[0], scores[0]):
        if score <= 0:
            continue
        c = chunks[idx]
        out.append({"page": c["page"], "score": round(float(score), 2), "text": c["text"]})
    return out
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_tools.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/tools.py tests/recipe/docvqa/test_tools.py
git commit -m "[docvqa] feat: search() BM25 host handler with per-doc cache"
```

---

## Task 6: Tool host handlers — `batch_look` (VLM HTTP)

**Files:**
- Modify: `recipe/docvqa/tools.py`
- Modify: `tests/recipe/docvqa/test_tools.py`

- [ ] **Step 1: Write failing tests using a mocked httpx client**

Append to `tests/recipe/docvqa/test_tools.py`:

```python
import asyncio
import base64

import httpx
import pytest

from recipe.docvqa.tools import batch_look


def _mock_transport(_resps: list[str]) -> httpx.MockTransport:
    """Round-robin through canned VLM completion responses."""
    counter = {"i": 0}
    def handler(request: httpx.Request) -> httpx.Response:
        i = counter["i"] % len(_resps)
        counter["i"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": _resps[i]}}]},
        )
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_batch_look_round_trip(tmp_path, monkeypatch):
    # Create two tiny PNG inputs.
    from PIL import Image
    p0 = tmp_path / "a.png"; Image.new("RGB", (10, 10), "red").save(p0)
    p1 = tmp_path / "b.png"; Image.new("RGB", (10, 10), "blue").save(p1)

    transport = _mock_transport(["red answer", "blue answer"])
    async with httpx.AsyncClient(transport=transport, timeout=5) as client:
        res = await batch_look(
            requests=[{"path": str(p0), "query": "color?"},
                      {"path": str(p1), "query": "color?"}],
            client=client,
            base_url="http://mock",
            model_id="vlm",
        )
    # Order preserved
    assert res == ["red answer", "blue answer"]


@pytest.mark.asyncio
async def test_batch_look_handles_http_error(tmp_path):
    from PIL import Image
    p = tmp_path / "a.png"; Image.new("RGB", (10, 10), "red").save(p)

    def fail_handler(_req):
        return httpx.Response(500, text="boom")
    transport = httpx.MockTransport(fail_handler)
    async with httpx.AsyncClient(transport=transport, timeout=5) as client:
        res = await batch_look(
            requests=[{"path": str(p), "query": "?"}],
            client=client, base_url="http://mock", model_id="vlm",
        )
    assert len(res) == 1
    assert res[0].startswith("[VLM error")
```

Add `pytest-asyncio` to project deps (or add `asyncio_mode = "auto"` to `pyproject.toml`'s `[tool.pytest.ini_options]` and import `pytest`):

Add to `pyproject.toml` under `[tool.pytest.ini_options]`:

```toml
asyncio_mode = "auto"
```

- [ ] **Step 2: Run tests — fail**

```bash
pytest tests/recipe/docvqa/test_tools.py::test_batch_look_round_trip -v
```

Expected: ImportError on `batch_look`.

- [ ] **Step 3: Implement `batch_look`**

Append to `recipe/docvqa/tools.py`:

```python
import asyncio
import base64
from pathlib import Path

import httpx


async def _one_look(
    client: httpx.AsyncClient, base_url: str, model_id: str,
    path: str, query: str,
) -> str:
    img_b64 = base64.b64encode(Path(path).read_bytes()).decode()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": query},
        ]}],
        "max_tokens": 512,
        "temperature": 0.0,
    }
    try:
        resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[VLM error: {type(e).__name__}: {e}]"


async def batch_look(
    requests: list[dict],
    client: httpx.AsyncClient,
    base_url: str,
    model_id: str,
) -> list[str]:
    """Send (path, query) pairs to the VLM in parallel. Returns answers in order."""
    coros = [_one_look(client, base_url, model_id, r["path"], r["query"]) for r in requests]
    return await asyncio.gather(*coros)
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_tools.py -v
```

Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/tools.py tests/recipe/docvqa/test_tools.py pyproject.toml
git commit -m "[docvqa] feat: batch_look() VLM HTTP handler with parallel requests"
```

---

## Task 7: Prompt templates

**Files:**
- Create: `recipe/docvqa/prompts.py`
- Create: `tests/recipe/docvqa/test_prompts.py`

System prompt + first-user-message + per-turn observation. `ANSWER_FORMATTING_RULES` and `get_category_tips` are imported from the docvqa repo if it's importable; otherwise vendored as a string. To keep this repo self-contained, **vendor** them as constants here.

- [ ] **Step 1: Lift the answer-formatting block from the docvqa repo**

```bash
sed -n '/^ANSWER_FORMATTING_RULES/,/^[A-Z_]\+ = /p' \
    /home/baris/repos/docvqa/src/docvqa/prompts.py | head -60
```

Read the value (it's a multi-line string) — copy verbatim into a constant in `recipe/docvqa/prompts.py` (Step 3).

Also lift `get_category_tips`:

```bash
grep -n "def get_category_tips\|^[A-Z_]\+_TIPS" /home/baris/repos/docvqa/src/docvqa/prompts.py
```

Note the function and per-category tip strings. Copy them in.

- [ ] **Step 2: Write tests**

```python
# tests/recipe/docvqa/test_prompts.py
"""Tests for prompt-template construction."""
from recipe.docvqa.prompts import (
    build_system_prompt, build_first_user_message, build_observation_message,
    ANSWER_FORMATTING_RULES, get_category_tips,
)


def test_system_prompt_mentions_tools():
    sp = build_system_prompt(category="business_report")
    for required in ("batch_look", "search", "SUBMIT", "<think>", "```python"):
        assert required in sp, f"missing: {required}"


def test_system_prompt_includes_answer_format_rules():
    sp = build_system_prompt(category="business_report")
    assert ANSWER_FORMATTING_RULES.strip().splitlines()[0] in sp


def test_system_prompt_includes_category_tips_when_present():
    tips = get_category_tips("business_report")
    if tips:
        sp = build_system_prompt(category="business_report")
        assert tips.strip() in sp


def test_system_prompt_no_category_tips_when_unknown():
    # Unknown category should not break.
    sp = build_system_prompt(category="non_existent_category_xyz")
    assert sp  # still produces a prompt


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
    assert "3" in msg  # num_pages
    assert "First page content here." in msg


def test_first_user_message_truncates_long_first_page():
    long = "x" * 1000
    msg = build_first_user_message(
        question="?", category="c", num_pages=1, page_texts=[long],
    )
    assert "…" in msg
    assert "x" * 500 not in msg  # not the full thing


def test_observation_message_includes_turn_counter():
    msg = build_observation_message(turn=3, max_iter=20, output="hello")
    assert "Turn 3/20" in msg
    assert "hello" in msg
```

- [ ] **Step 3: Implement prompts.py**

```python
# recipe/docvqa/prompts.py
"""Prompt templates: system, first user, per-turn observation.

ANSWER_FORMATTING_RULES and category tips are vendored from
~/repos/docvqa/src/docvqa/prompts.py to keep this recipe self-contained.
"""
from __future__ import annotations

# === ANSWER FORMATTING ===
# Vendored verbatim from docvqa repo. Update if the repo's rules change.

ANSWER_FORMATTING_RULES = """\
- Output ONLY the final answer string — no leading/trailing prose, no labels,
  no quotes around the value.
- For numeric answers, output the bare number (e.g. "42", "3.14"); include
  units only if the question asks for them or the document encodes them.
- For dates, match the document's format unless otherwise specified.
- For multi-value questions, separate values with commas in the order they
  appear in the document.
- For "Unknown" — only when the value is genuinely missing/N/A — output
  exactly the string: Unknown
"""

# Per-category tips (lift from ~/repos/docvqa/src/docvqa/prompts.py if you
# want richer guidance). Empty string means "no extra tips".
_CATEGORY_TIPS: dict[str, str] = {
    # Fill from the docvqa repo's get_category_tips(); see lines in that file.
    # Empty entries are equivalent to absent.
}


def get_category_tips(category: str) -> str:
    return _CATEGORY_TIPS.get(category, "").strip()


# === SYSTEM PROMPT ===

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
  BM25 search over `page_texts`. Returns [{page, score, text}, ...]. Useful
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

## ANSWER FORMATTING
{answer_formatting_rules}
"""


def build_system_prompt(category: str) -> str:
    out = _SYSTEM.format(answer_formatting_rules=ANSWER_FORMATTING_RULES.strip())
    tips = get_category_tips(category)
    if tips:
        out += f"\n## CATEGORY TIPS\n{tips}\n"
    return out


# === FIRST USER MESSAGE ===

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


# === PER-TURN OBSERVATION ===

def build_observation_message(turn: int, max_iter: int, output: str) -> str:
    return (
        f"## Turn {turn}/{max_iter}\n"
        f"## Output\n```\n{output}\n```"
    )
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_prompts.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Optionally lift the real category tips from the docvqa repo**

If `get_category_tips` in `~/repos/docvqa/src/docvqa/prompts.py` is non-trivial, copy each tip string into `_CATEGORY_TIPS`. Quick way:

```bash
python -c "
from pathlib import Path
import re
src = Path('/home/baris/repos/docvqa/src/docvqa/prompts.py').read_text()
print(src)
" | less
```

This is a docs-style edit; no test changes needed.

- [ ] **Step 6: Commit**

```bash
git add recipe/docvqa/prompts.py tests/recipe/docvqa/test_prompts.py
git commit -m "[docvqa] feat: prompt templates (system, first user, observation)"
```

---

## Task 8: Reward function (ANLS)

**Files:**
- Create: `recipe/docvqa/reward.py`
- Create: `tests/recipe/docvqa/test_reward.py`

We need the ANLS implementation. Lift `evaluate_prediction` and `compute_anls` from `~/repos/docvqa/src/docvqa/metrics.py`. Keep them as a single self-contained function.

- [ ] **Step 1: Inspect the source metric**

```bash
grep -n "def evaluate_prediction\|def compute_anls\|def normalize" \
    /home/baris/repos/docvqa/src/docvqa/metrics.py
```

Read the bodies to understand:
- normalization (lowercase, strip whitespace, …),
- distance metric (Levenshtein),
- threshold (0.5 or 0.8),
- handling of "Unknown" / multi-answer cases.

- [ ] **Step 2: Write tests**

```python
# tests/recipe/docvqa/test_reward.py
"""Tests for the ANLS reward function."""
from recipe.docvqa.reward import compute_anls, compute_score


def test_anls_exact_match():
    assert compute_anls("42", "42") == 1.0


def test_anls_case_insensitive_and_whitespace():
    assert compute_anls("Hello World", "  hello world ") == 1.0


def test_anls_close_match_above_threshold():
    # ANLS uses normalized Levenshtein with threshold 0.5; small typos pass.
    score = compute_anls("$1.2 billion", "$1.2 billion.")
    assert score > 0.8


def test_anls_far_off_returns_zero():
    assert compute_anls("apple", "battlefield") == 0.0


def test_compute_score_with_no_submission():
    """submitted_answer=None ⇒ score 0."""
    score, info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={"submitted_answer": None, "termination": "iter_cap"},
    )
    assert score == 0.0
    assert info["anls"] == 0.0
    assert info["termination"] == "iter_cap"


def test_compute_score_with_correct_submission():
    score, info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={"submitted_answer": "42", "termination": "submit"},
    )
    assert score == 1.0
    assert info["anls"] == 1.0


def test_compute_score_passes_through_metadata():
    score, info = compute_score(
        data_source="docvqa", solution_str="ignored",
        ground_truth="42",
        extra_info={
            "submitted_answer": "42", "termination": "submit",
            "num_turns": 5, "vlm_calls": 2, "search_calls": 1,
            "messages": [{"role": "system", "content": "..."}],
        },
    )
    assert info["num_turns"] == 5
    assert info["vlm_calls"] == 2
    assert info["messages"][0]["role"] == "system"
```

- [ ] **Step 3: Implement reward.py**

```python
# recipe/docvqa/reward.py
"""ANLS reward for DocVQA. Plugs into verl's custom_reward_function hook."""
from __future__ import annotations

import string
from typing import Any

# Use rapidfuzz if available (faster); else python-Levenshtein; else pure-Python.
try:
    from rapidfuzz.distance import Levenshtein as _LD
    def _lev(a: str, b: str) -> int:
        return _LD.distance(a, b)
except ImportError:
    def _lev(a: str, b: str) -> int:
        # Pure-Python fallback.
        if a == b: return 0
        if not a: return len(b)
        if not b: return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cost = 0 if ca == cb else 1
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))
            prev = cur
        return prev[-1]


_ANLS_THRESHOLD = 0.5  # Standard DocVQA ANLS threshold.


def _normalize(s: str) -> str:
    s = s.strip().lower()
    # Collapse whitespace
    s = " ".join(s.split())
    # Strip surrounding punctuation
    s = s.strip(string.punctuation + " ")
    return s


def compute_anls(prediction: str, ground_truth: str) -> float:
    """Single-answer ANLS. Returns 0..1."""
    p = _normalize(prediction)
    g = _normalize(ground_truth)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    d = _lev(p, g)
    nls = 1.0 - d / max(len(p), len(g))
    return nls if nls >= _ANLS_THRESHOLD else 0.0


def compute_score(
    data_source: str,
    solution_str: str,  # noqa: ARG001 — verl signature
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """verl reward function: returns (score, extra_info_to_dump).

    `extra_info` arrives from `AgentLoopOutput.extra_fields`. We compute
    ANLS against the model's `submitted_answer` (None ⇒ 0 score) and pass
    every other field through to the rollout JSONL dump.
    """
    extra = dict(extra_info or {})
    submitted = extra.get("submitted_answer")
    score = 0.0 if submitted is None else compute_anls(submitted, ground_truth)
    extra["anls"] = score
    return score, extra
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_reward.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/reward.py tests/recipe/docvqa/test_reward.py
git commit -m "[docvqa] feat: ANLS reward function with extra_info passthrough"
```

---

## Task 9: AgentLoop — class skeleton with no per-turn loop

**Files:**
- Create: `recipe/docvqa/agent_loop.py`
- Create: `tests/recipe/docvqa/test_agent_loop.py`

We split the agent loop into three tasks (9, 10, 11) so the diff stays small.

This task: instantiate the class, register it, accept a row, build the initial prompt, build messages list, and return an immediately-terminated `AgentLoopOutput` (no real generation yet). Subsequent tasks add the turn loop and SUBMIT handling.

- [ ] **Step 1: Skeleton test — instantiation + zero-turn run with mocked server_manager**

```python
# tests/recipe/docvqa/test_agent_loop.py
"""End-to-end tests for DocVQAReplAgentLoop with scripted server_manager."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from recipe.docvqa.agent_loop import DocVQAReplAgentLoop


class _FakeTokenOutput:
    def __init__(self, token_ids, log_probs=None):
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.num_preempted = 0
        self.extra_fields = {}
        self.routed_experts = None


class _ScriptedServerManager:
    """Returns scripted token id sequences for each generate() call."""
    def __init__(self, scripted_responses: list[list[int]]):
        self._responses = list(scripted_responses)

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        if not self._responses:
            raise AssertionError("Out of scripted responses")
        return _FakeTokenOutput(self._responses.pop(0))


def _make_trainer_config(**overrides):
    cfg = OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 8192,
                "response_length": 8192,
                "agent": {"agent_loop_config_path": None},
                "multi_turn": {},
                "trace": {"project_name": "test", "experiment_name": "test"},
            },
            "model": {},
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
    })
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


# Task-9 scope: class instantiates and `run` returns immediately when
# scripted_responses is empty (we'll wire the iteration cap path).

@pytest.fixture
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def test_agent_loop_instantiates(tokenizer, sample_doc_dir):
    from recipe.docvqa.agent_loop import DictConfigWrap
    cfg = _make_trainer_config()
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([]),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    assert loop.tokenizer is tokenizer
```

(We expand this test in Tasks 10 and 11 as the loop body grows.)

- [ ] **Step 2: Run test — fail**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py -v
```

Expected: ImportError on `recipe.docvqa.agent_loop`.

- [ ] **Step 3: Skeleton implementation**

```python
# recipe/docvqa/agent_loop.py
"""DocVQA REPL agent loop for verl.

Replicates the deployed `flat_solo` scaffold: a persistent CPython
subprocess with batch_look / search / SUBMIT, model emits <think>...
</think> + ```python ... ``` per turn, ANLS reward end-of-trajectory.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, DictConfigWrap, register,
)

from recipe.docvqa.parser import parse_last_python_fence
from recipe.docvqa.prompts import (
    build_first_user_message, build_observation_message, build_system_prompt,
)
from recipe.docvqa.sandbox import build_sandbox_code
from recipe.docvqa.subprocess_interp import (
    CodeInterpreterError, FinalOutput, SubprocessInterpreter,
)


# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "page_factor": 1.5,
    "max_iterations_base": 20,
    "max_iterations_cap": 30,
    "max_response_tokens_per_turn": 4096,
    "max_obs_chars": 8000,
    "subprocess_timeout_s": 120.0,
    "parse_error_strikes_to_terminate": 3,
}


def _adaptive_max_iter(num_pages: int, knobs: dict) -> int:
    import math
    bonus = knobs["page_factor"] * math.sqrt(max(0, num_pages - 9))
    return min(knobs["max_iterations_cap"],
               knobs["max_iterations_base"] + int(bonus))


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

@register("docvqa_repl")
class DocVQAReplAgentLoop(AgentLoopBase):
    """Per-question REPL agent. One persistent subprocess per rollout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Allow overrides via config.actor_rollout_ref.rollout.agent.docvqa
        agent_cfg = self.rollout_config.agent.get("docvqa", {})
        self._knobs = {**_DEFAULTS, **dict(agent_cfg)}

        # VLM endpoint for batch_look (lifted from rollout config or env).
        self._vlm_base_url = (
            agent_cfg.get("vlm_base_url")
            or os.environ.get("DOCVQA_VLM_BASE_URL")
            or "http://localhost:8928"
        )
        self._vlm_model_id = (
            agent_cfg.get("vlm_model_id")
            or os.environ.get("DOCVQA_VLM_MODEL_ID")
            or "qwen3.6-27b"
        )

        self._response_length_cap = self.rollout_config.response_length

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        # Tasks 10 and 11 fill this in.
        raise NotImplementedError("DocVQAReplAgentLoop.run will be filled in Task 10")
```

- [ ] **Step 4: Run test — pass**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py::test_agent_loop_instantiates -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/agent_loop.py tests/recipe/docvqa/test_agent_loop.py
git commit -m "[docvqa] feat: DocVQAReplAgentLoop skeleton (instantiation only)"
```

---

## Task 10: AgentLoop — turn loop, parsing, observation injection

**Files:**
- Modify: `recipe/docvqa/agent_loop.py`
- Modify: `tests/recipe/docvqa/test_agent_loop.py`

This task implements the meat of `run()`: spawn subprocess, render initial prompt, loop generating + executing + appending observations, with response_mask bookkeeping. SUBMIT handling deferred to Task 11.

- [ ] **Step 1: Tests — single-turn `print(2+2)` then iter cap**

Append to `tests/recipe/docvqa/test_agent_loop.py`:

```python
async def test_run_iter_cap_with_no_submit(tokenizer, sample_doc_dir):
    """Model emits a print but never SUBMITs; loop hits iter cap."""
    from transformers import AutoTokenizer
    tok = tokenizer
    # Build a one-turn assistant text and tokenize.
    text = "<think>Let me check.</think>\n\n```python\nprint(page_texts[0][:50])\n```"
    assistant_ids = tok.encode(text + "<|im_end|>", add_special_tokens=False)
    # Repeat enough times to hit the iter cap.
    scripted = [assistant_ids] * 50

    from recipe.docvqa.agent_loop import DocVQAReplAgentLoop, DictConfigWrap
    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 3,
                                 "max_iterations_cap": 3}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager(scripted),
        tokenizer=tok,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )

    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0",
        question="What was Q3 revenue?",
        doc_dir=str(sample_doc_dir),
        gold_answer="$1.2B",
        category="business_report",
    )
    assert out.extra_fields["termination"] == "iter_cap"
    assert out.extra_fields["submitted_answer"] is None
    assert out.extra_fields["num_turns"] == 3
    assert len(out.response_mask) == len(out.response_ids)
    # All assistant tokens have mask=1; observation tokens have mask=0.
    assert sum(out.response_mask) > 0  # some assistant tokens
    assert any(m == 0 for m in out.response_mask)  # some observation tokens


async def test_run_records_messages_in_extra_fields(tokenizer, sample_doc_dir):
    text = "<think>OK.</think>\n```python\nprint('hi')\n```"
    assistant_ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)

    from recipe.docvqa.agent_loop import DocVQAReplAgentLoop, DictConfigWrap
    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 2,
                                 "max_iterations_cap": 2}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([assistant_ids] * 5),
        tokenizer=tokenizer,
        processor=None, dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0", question="?", doc_dir=str(sample_doc_dir),
        gold_answer="x", category="business_report",
    )
    msgs = out.extra_fields["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and "Question" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user" and "Turn" in msgs[3]["content"]
```

- [ ] **Step 2: Run tests — fail (NotImplementedError)**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py -v -k "iter_cap or messages"
```

- [ ] **Step 3: Implement `run()` (without SUBMIT handling — Task 11)**

Replace the `run()` body in `recipe/docvqa/agent_loop.py`:

```python
    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        wall_start = time.monotonic()

        # 1. Pull dataset row fields.
        question = kwargs["question"]
        question_id = kwargs.get("question_id", "")
        doc_dir = kwargs["doc_dir"]
        category = kwargs.get("category", "unknown")
        gold_answer = kwargs.get("gold_answer")  # for traces; reward fn uses ground_truth

        # 2. Read doc metadata.
        meta = json.loads((Path(doc_dir) / "metadata.json").read_text())
        num_pages = meta["num_pages"]
        page_texts = [
            p.read_text() for p in sorted(
                (Path(doc_dir) / "ocr").glob("page_*.md"),
                key=lambda p: int(p.stem.split("_")[1]),
            )
        ]

        max_iter = _adaptive_max_iter(num_pages, self._knobs)

        # 3. Build initial messages and tokenize.
        sys_prompt = build_system_prompt(category)
        first_user = build_first_user_message(question, category, num_pages, page_texts)
        messages: list[dict] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": first_user},
        ]
        prompt_ids: list[int] = await self.apply_chat_template(messages)
        response_ids: list[int] = []
        response_mask: list[int] = []

        # 4. Spawn subprocess. Tools handled host-side by closures.
        import httpx
        from recipe.docvqa import tools as host_tools

        vlm_client = httpx.AsyncClient(timeout=120)
        try:
            async def _batch_look_host(requests: list[dict]) -> list[str]:
                return await host_tools.batch_look(
                    requests, vlm_client, self._vlm_base_url, self._vlm_model_id,
                )

            def _batch_look_sync(requests):
                """Bridge to async — invoked from a host-side thread by IPC."""
                return asyncio.run_coroutine_threadsafe(
                    _batch_look_host(requests), self.loop,
                ).result(timeout=300)

            def _search_sync(query: str, k: int = 5):
                return host_tools.search(doc_dir, query, k)

            interp = SubprocessInterpreter(
                sandbox_code=build_sandbox_code(),
                tools={"batch_look": _batch_look_sync, "search": _search_sync},
                output_fields=[{"name": "answer", "type": "str"}],
                timeout=self._knobs["subprocess_timeout_s"],
                extra_env={"DOC_DIR": doc_dir},
            )
            interp.start()

            request_id = uuid4().hex
            termination = "iter_cap"
            submitted_answer: str | None = None
            num_turns = 0
            vlm_calls = 0
            search_calls = 0
            parse_error_strikes = 0

            for turn in range(1, max_iter + 1):
                num_turns += 1

                # A. Sample one assistant turn
                token_out = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params={
                        **sampling_params,
                        "max_tokens": self._knobs["max_response_tokens_per_turn"],
                        "stop": ["<|im_end|>"],
                    },
                )
                assistant_ids = list(token_out.token_ids)
                response_ids += assistant_ids
                response_mask += [1] * len(assistant_ids)
                assistant_text = self.tokenizer.decode(
                    assistant_ids, skip_special_tokens=False,
                )
                # Strip a trailing <|im_end|> if the model emitted it.
                clean_text = assistant_text.split("<|im_end|>")[0]
                messages.append({"role": "assistant", "content": clean_text})

                # B. Parse code
                code = parse_last_python_fence(clean_text)
                if code is None:
                    parse_error_strikes += 1
                    observation = "[Error] No `python` code block found. Write a single ```python ... ``` block."
                    if parse_error_strikes >= self._knobs["parse_error_strikes_to_terminate"]:
                        # Append the observation, then terminate.
                        await self._append_observation(
                            messages, response_ids, response_mask,
                            turn, max_iter, observation,
                        )
                        termination = "parse_error"
                        break
                else:
                    parse_error_strikes = 0
                    # C. Execute (may raise / return FinalOutput / return string)
                    try:
                        result = await self.loop.run_in_executor(
                            None, interp.execute, code,
                        )
                    except (CodeInterpreterError, SyntaxError) as e:
                        result = f"[Error] {e}"

                    # SUBMIT handled in Task 11 — for now treat FinalOutput as
                    # the same as a printed observation so iter_cap path works.
                    if isinstance(result, tuple) and isinstance(result[0], FinalOutput):
                        # Defer real handling to Task 11. For now we just turn
                        # it into a string so the loop doesn't crash.
                        observation = f"FINAL: {result[0].output}"
                    elif isinstance(result, FinalOutput):
                        observation = f"FINAL: {result.output}"
                    elif isinstance(result, str) and result.startswith("[Error]"):
                        observation = result
                    elif isinstance(result, list):
                        observation = "\n".join(map(str, result))
                    elif result:
                        observation = str(result)
                    else:
                        observation = "(no output - did you forget to print?)"

                    # Crude bookkeeping for vlm/search call counts (refined later)
                    if "batch_look(" in code:
                        vlm_calls += code.count("batch_look(")
                    if "search(" in code:
                        search_calls += code.count("search(")

                    observation = self._truncate(observation)

                # D. Append observation as a user-role diff
                await self._append_observation(
                    messages, response_ids, response_mask,
                    turn, max_iter, observation,
                )

                # Token-budget guard
                if len(response_ids) >= self._response_length_cap - 256:
                    termination = "token_cap"
                    break

            else:
                termination = "iter_cap"

        finally:
            try: interp.shutdown()
            except Exception: pass
            try: await vlm_client.aclose()
            except Exception: pass

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[:self._response_length_cap],
            response_mask=response_mask[:self._response_length_cap],
            num_turns=num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "messages": messages,
                "termination": termination,
                "submitted_answer": submitted_answer,
                "num_turns": num_turns,
                "vlm_calls": vlm_calls,
                "search_calls": search_calls,
                "wall_clock_s": time.monotonic() - wall_start,
                "doc_id": meta["doc_id"],
                "question_id": question_id,
                "category": category,
            },
        )

    async def _append_observation(
        self, messages: list[dict], response_ids: list[int],
        response_mask: list[int], turn: int, max_iter: int, output: str,
    ) -> None:
        text = build_observation_message(turn, max_iter, output)
        messages.append({"role": "user", "content": text})
        obs_ids = await self.apply_chat_template(
            [{"role": "user", "content": text}],
            remove_system_prompt=True,
        )
        response_ids += obs_ids
        response_mask += [0] * len(obs_ids)

    def _truncate(self, text: str) -> str:
        cap = self._knobs["max_obs_chars"]
        if len(text) <= cap:
            return text
        return text[:cap] + "\n... (truncated)"
```

- [ ] **Step 4: Run tests — pass**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py -v -k "iter_cap or messages"
```

Expected: both tests pass. If `apply_chat_template` complains about `enable_thinking`, ensure `data_config.apply_chat_template_kwargs.enable_thinking=True` is set in `_make_trainer_config`.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/agent_loop.py tests/recipe/docvqa/test_agent_loop.py
git commit -m "[docvqa] feat: agent_loop turn loop with parsing + observations"
```

---

## Task 11: AgentLoop — SUBMIT handling, terminations, sanity round-trip

**Files:**
- Modify: `recipe/docvqa/agent_loop.py`
- Modify: `tests/recipe/docvqa/test_agent_loop.py`

- [ ] **Step 1: Tests — SUBMIT terminates and round-trip identity**

Append:

```python
async def test_run_submit_terminates_with_answer(tokenizer, sample_doc_dir):
    text = '<think>Easy.</think>\n```python\nSUBMIT(answer="$1.2B")\n```'
    assistant_ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
    other_text = '```python\nprint("ignored")\n```'
    other_ids = tokenizer.encode(other_text + "<|im_end|>", add_special_tokens=False)

    from recipe.docvqa.agent_loop import DocVQAReplAgentLoop, DictConfigWrap
    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 5,
                                 "max_iterations_cap": 5}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([assistant_ids, other_ids, other_ids]),
        tokenizer=tokenizer,
        processor=None, dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0", question="?", doc_dir=str(sample_doc_dir),
        gold_answer="x", category="business_report",
    )
    assert out.extra_fields["termination"] == "submit"
    assert out.extra_fields["submitted_answer"] == "$1.2B"
    assert out.extra_fields["num_turns"] == 1


async def test_round_trip_messages_match_recorded_ids(tokenizer, sample_doc_dir):
    """The recorded prompt_ids++response_ids should re-tokenize from `messages`."""
    text = '<think>x</think>\n```python\nprint("hi")\n```'
    assistant_ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)

    from recipe.docvqa.agent_loop import DocVQAReplAgentLoop, DictConfigWrap
    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 2,
                                 "max_iterations_cap": 2}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([assistant_ids] * 5),
        tokenizer=tokenizer,
        processor=None, dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0", question="?", doc_dir=str(sample_doc_dir),
        gold_answer="x", category="business_report",
    )

    # Round-trip: re-render messages and assert tokens match recorded ids.
    from verl.utils.chat_template import apply_chat_template as actl
    rerendered = actl(
        tokenizer, out.extra_fields["messages"], tools=None,
        add_generation_prompt=False, tokenize=True,
        enable_thinking=True,
    )
    recorded = list(out.prompt_ids) + list(out.response_ids)
    # The two should match modulo the trailing generation prompt suffix
    # (we don't add one when finalizing). Allow up to 8 tokens of mismatch
    # at the tail for whitespace differences.
    assert rerendered[: len(recorded) - 8] == recorded[: len(recorded) - 8], \
        "Trajectory token round-trip failed — chat template strips <think>"
```

- [ ] **Step 2: Run tests — fail (SUBMIT not implemented yet, round-trip may fail)**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py -v
```

- [ ] **Step 3: Implement SUBMIT handling**

In `recipe/docvqa/agent_loop.py`, replace the FinalOutput handling block inside the turn loop with:

```python
                    if isinstance(result, tuple) and isinstance(result[0], FinalOutput):
                        final, captured = result
                        submitted_answer = final.output.get("answer")
                        observation = (
                            (captured + "\n" if captured else "") +
                            f"FINAL: {submitted_answer!r}"
                        )
                        await self._append_observation(
                            messages, response_ids, response_mask,
                            turn, max_iter, observation,
                        )
                        termination = "submit"
                        break
                    elif isinstance(result, FinalOutput):
                        submitted_answer = result.output.get("answer")
                        observation = f"FINAL: {submitted_answer!r}"
                        await self._append_observation(
                            messages, response_ids, response_mask,
                            turn, max_iter, observation,
                        )
                        termination = "submit"
                        break
                    elif isinstance(result, str) and result.startswith("[Error]"):
                        observation = result
                    elif isinstance(result, list):
                        observation = "\n".join(map(str, result))
                    elif result:
                        observation = str(result)
                    else:
                        observation = "(no output - did you forget to print?)"
```

(The `if termination == "submit": break` already escapes the for-loop because we use `break`.)

Also remove the for-loop `else: termination = "iter_cap"` line since we now set `termination` explicitly in each branch except when the loop runs to completion. Add a *sentinel default*: at the start of the loop set `termination = "iter_cap"` so if we run to completion without break, that's our value. (Already done in Task 10.) Verify by tracing.

- [ ] **Step 4: Run tests — they pass**

```bash
pytest tests/recipe/docvqa/test_agent_loop.py -v
```

Expected: all green. If the round-trip test fails, the chat template is stripping `<think>` from prior turns. Fall back: in `_make_trainer_config`'s `model_config`, set `tokenizer.chat_template` to the willcb/Qwen3-8B template (download from HF cache). Document this in the task notes.

- [ ] **Step 5: Commit**

```bash
git add recipe/docvqa/agent_loop.py tests/recipe/docvqa/test_agent_loop.py
git commit -m "[docvqa] feat: SUBMIT termination + token round-trip test"
```

---

## Task 12: Hydra registration

**Files:**
- Create: `recipe/docvqa/agent.yaml`

- [ ] **Step 1: Write the registration**

```yaml
# recipe/docvqa/agent.yaml
# verl agent loop registration. See verl/experimental/agent_loop/agent_loop.py:377-382.
- name: docvqa_repl
  _target_: recipe.docvqa.agent_loop.DocVQAReplAgentLoop
```

- [ ] **Step 2: Sanity check by importing**

```bash
python -c "
from omegaconf import OmegaConf
print(OmegaConf.load('recipe/docvqa/agent.yaml'))
"
```

Expected: prints a list with one item containing `name: docvqa_repl` and the FQN target.

- [ ] **Step 3: Commit**

```bash
git add recipe/docvqa/agent.yaml
git commit -m "[docvqa] feat: register docvqa_repl agent loop in Hydra config"
```

---

## Task 13: Data prep for DocVQA-2026 val (Phase 0)

**Files:**
- Create: `recipe/docvqa/scripts/prepare_data.py`

Builds `data/{val,test}/docs/{doc_id}/` from (a) the docvqa repo's already-prepared OCR + BM25 in `~/repos/docvqa/data/{val,test}/`, and (b) HF `VLR-CVC/DocVQA-2026` for page images. Writes `data/{val,test}/questions.json`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
# recipe/docvqa/scripts/prepare_data.py
"""Materialize per-document working directories for DocVQA-2026.

Source 1: ~/repos/docvqa/data/{val,test}/{ocr,bm25}/ (already prepared).
Source 2: HuggingFace dataset `VLR-CVC/DocVQA-2026` for page images.

Output:
  data/{split}/docs/{doc_id}/
    metadata.json
    pages/page_*.png
    ocr/page_*.md       (copied from source 1)
    bm25/...            (copied from source 1)
  data/{split}/questions.json
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image
from datasets import load_dataset


SRC_DOCVQA_DATA = Path.home() / "repos" / "docvqa" / "data"


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    shutil.copytree(src, dst)


def _materialize_split(split: str, out_root: Path) -> None:
    src_split = SRC_DOCVQA_DATA / split
    docs_dir = out_root / split / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("VLR-CVC/DocVQA-2026", split=split)

    questions: list[dict] = []
    seen_docs: set[str] = set()

    for row in ds:
        doc_id = row["doc_id"]
        question_id = row["question_id"]
        question = row["question"]
        gold = row.get("answer")  # None on test
        category = row.get("doc_category", "unknown")

        # Per-doc materialization (idempotent).
        doc_out = docs_dir / doc_id
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            (doc_out / "pages").mkdir(parents=True, exist_ok=True)
            (doc_out / "ocr").mkdir(exist_ok=True)
            (doc_out / "bm25").mkdir(exist_ok=True)

            # Copy OCR + BM25 from docvqa repo
            src_ocr = src_split / "ocr" / doc_id
            src_bm25 = src_split / "bm25" / doc_id
            if src_ocr.exists():
                for p in src_ocr.iterdir():
                    if p.is_file():
                        shutil.copy(p, doc_out / "ocr" / p.name)
            if src_bm25.exists():
                for p in src_bm25.iterdir():
                    if p.is_file():
                        shutil.copy(p, doc_out / "bm25" / p.name)

            # Render page images from HF row's images list
            images = row.get("images") or row.get("page_images") or []
            for i, img in enumerate(images):
                if isinstance(img, Image.Image):
                    img.save(doc_out / "pages" / f"page_{i}.png", format="PNG")

            # metadata.json
            meta = {
                "doc_id": doc_id,
                "doc_category": category,
                "num_pages": len(images),
                "source_dataset": f"docvqa-2026-{split}",
            }
            (doc_out / "metadata.json").write_text(json.dumps(meta, indent=2))

        questions.append({
            "question_id": question_id,
            "doc_id": doc_id,
            "question": question,
            "answer": gold,
            "category": category,
            "source_dataset": f"docvqa-2026-{split}",
            "doc_dir": str(doc_out.resolve()),
        })

    (out_root / split / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False)
    )
    print(f"[{split}] wrote {len(questions)} questions across {len(seen_docs)} docs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("data"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        _materialize_split(split, args.out_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for val only first**

```bash
cd /home/baris/repos/docvqa-verl
python recipe/docvqa/scripts/prepare_data.py --splits val
```

Expected output: `[val] wrote 80 questions across 25 docs`.

- [ ] **Step 3: Smoke-check the output**

```bash
ls data/val/docs | head -5
ls data/val/docs/business_report_1/{pages,ocr,bm25} | head
head -c 800 data/val/questions.json
```

Verify pages are PNGs, OCR is markdown, BM25 has the expected files, questions.json is human-readable JSON.

- [ ] **Step 4: Commit script and minimal data sample**

The full data is too big to commit. Add to `.gitignore`:

```
data/
!data/.gitkeep
```

```bash
touch data/.gitkeep
git add recipe/docvqa/scripts/prepare_data.py .gitignore data/.gitkeep
git commit -m "[docvqa] feat: data prep script for DocVQA-2026 val/test"
```

---

## Task 14: Layer-3 evaluation script

**Files:**
- Create: `recipe/docvqa/scripts/eval.py`

Drives the agent loop end-to-end without verl training. Reports ANLS on val.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
# recipe/docvqa/scripts/eval.py
"""Layer-3 ANLS reproduction. Runs DocVQAReplAgentLoop over a split with a
running vLLM (student) and the running 27B VLM, reports ANLS.

Usage:
    python recipe/docvqa/scripts/eval.py \
        --questions data/val/questions.json \
        --student-base-url http://localhost:8000/v1 \
        --student-model Qwen/Qwen3-8B \
        --vlm-base-url http://localhost:8928 \
        --vlm-model qwen3.6-27b \
        --concurrency 4 \
        --output outputs/eval/val_qwen3_8b.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

import httpx

# Make sure the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from recipe.docvqa.agent_loop import DocVQAReplAgentLoop  # noqa: E402
from recipe.docvqa.parser import parse_last_python_fence  # noqa: E402
from recipe.docvqa.prompts import (  # noqa: E402
    build_first_user_message, build_observation_message, build_system_prompt,
)
from recipe.docvqa.reward import compute_anls  # noqa: E402
from recipe.docvqa.sandbox import build_sandbox_code  # noqa: E402
from recipe.docvqa.subprocess_interp import (  # noqa: E402
    CodeInterpreterError, FinalOutput, SubprocessInterpreter,
)


# --- Minimal "server_manager" + "tokenizer" stubs that hit a real OpenAI endpoint ---


class OpenAIClientServerManager:
    """Implements .generate(prompt_ids, sampling_params, ...) by calling
    a vLLM-served Qwen3-8B at an OpenAI-compatible /v1/chat/completions endpoint.

    Note: For eval, we call /v1/completions with prompt as token_ids OR we
    decode and use /v1/chat/completions. To keep wire format simple we use
    the prompt-string form (decode prompt_ids → text → completions endpoint
    with `prompt` field). vLLM accepts this directly.
    """

    def __init__(self, base_url: str, model: str, tokenizer):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._tokenizer = tokenizer

    async def generate(self, request_id, prompt_ids, sampling_params, **kw):
        prompt_text = self._tokenizer.decode(prompt_ids, skip_special_tokens=False)
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(f"{self._base_url}/completions", json={
                "model": self._model,
                "prompt": prompt_text,
                "max_tokens": sampling_params.get("max_tokens", 4096),
                "temperature": sampling_params.get("temperature", 1.0),
                "top_p": sampling_params.get("top_p", 0.95),
                "stop": sampling_params.get("stop", ["<|im_end|>"]),
            })
            r.raise_for_status()
            text = r.json()["choices"][0]["text"]
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        class Out:
            def __init__(self, ids):
                self.token_ids = ids
                self.log_probs = None
                self.num_preempted = 0
                self.extra_fields = {}
                self.routed_experts = None
        return Out(token_ids)


# --- Minimal trainer/data config wrappers ---

def _build_loop(student_base_url, student_model, vlm_base_url, vlm_model,
                response_length=32768):
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer
    from recipe.docvqa.agent_loop import DictConfigWrap

    tokenizer = AutoTokenizer.from_pretrained(student_model)
    cfg = OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 16384,
                "response_length": response_length,
                "agent": {
                    "agent_loop_config_path": None,
                    "docvqa": {
                        "vlm_base_url": vlm_base_url,
                        "vlm_model_id": vlm_model,
                    },
                },
                "multi_turn": {},
                "trace": {"project_name": "eval", "experiment_name": "eval"},
            },
            "model": {},
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
    })
    return DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=OpenAIClientServerManager(student_base_url, student_model, tokenizer),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=type("Stub", (), {}),
        data_config=DictConfigWrap(cfg.data),
    )


async def _solve(loop, q):
    out = await loop.run(
        sampling_params={"temperature": 1.0, "top_p": 0.95},
        question_id=q["question_id"], question=q["question"],
        doc_dir=q["doc_dir"], gold_answer=q.get("answer"),
        category=q.get("category", "unknown"),
    )
    submitted = out.extra_fields.get("submitted_answer")
    anls = 0.0 if submitted is None else compute_anls(submitted, q["answer"])
    return {
        "question_id": q["question_id"], "doc_id": q["doc_id"],
        "category": q.get("category", "unknown"),
        "submitted_answer": submitted, "gold_answer": q.get("answer"),
        "anls": anls,
        "termination": out.extra_fields.get("termination"),
        "num_turns": out.extra_fields.get("num_turns"),
        "wall_clock_s": out.extra_fields.get("wall_clock_s"),
    }


async def _main_async(args):
    questions = json.loads(Path(args.questions).read_text())
    if args.limit:
        questions = questions[: args.limit]

    loop_obj = _build_loop(
        args.student_base_url, args.student_model,
        args.vlm_base_url, args.vlm_model,
    )

    sem = asyncio.Semaphore(args.concurrency)
    async def _bound(q):
        async with sem:
            try:
                return await _solve(loop_obj, q)
            except Exception as e:
                return {"question_id": q["question_id"], "error": repr(e), "anls": 0.0}

    results = await asyncio.gather(*(_bound(q) for q in questions))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category", "unknown"), []).append(r["anls"])
    print("=== ANLS report ===")
    print(f"  overall: {statistics.mean(r['anls'] for r in results):.4f}  "
          f"(n={len(results)})")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat:20s}: {statistics.mean(scores):.4f}  (n={len(scores)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--student-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--student-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--vlm-base-url", default="http://localhost:8928")
    ap.add_argument("--vlm-model", default="qwen3.6-27b")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Local smoke run on a tiny limit**

This requires a vLLM serving Qwen3-8B locally. If one is not yet running, document the launch:

```bash
# In a fresh tmux: bring up student vLLM (single GPU on, e.g., GPU 0)
# Adjust GPUs to taste; reuse the existing 27B VLM on GPU 2.
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 40960 --gpu-memory-utilization 0.85
```

Then:

```bash
python recipe/docvqa/scripts/eval.py \
    --questions data/val/questions.json --limit 3 \
    --student-base-url http://localhost:8000/v1 \
    --student-model Qwen/Qwen3-8B \
    --vlm-base-url http://localhost:8928 \
    --vlm-model qwen3.6-27b \
    --concurrency 1 \
    --output outputs/eval/smoke3.json
```

Expected: prints an ANLS report; `smoke3.json` exists with 3 entries.

If failures: inspect `outputs/eval/smoke3.json`, look for parse errors, subprocess crashes, VLM errors.

- [ ] **Step 3: Full val eval — Layer-3 gate**

```bash
python recipe/docvqa/scripts/eval.py \
    --questions data/val/questions.json \
    --student-base-url http://localhost:8000/v1 \
    --student-model Qwen/Qwen3-8B \
    --vlm-base-url http://localhost:8928 \
    --vlm-model qwen3.6-27b \
    --concurrency 4 \
    --output outputs/eval/val_qwen3_8b_layer3.json
```

Expected runtime: ~30-90 min. Compare overall ANLS to the reference flat_solo lean+nothink baseline (recorded once, separately, by running the docvqa repo's eval on Qwen3-8B). Pass if within ±5pp.

- [ ] **Step 4: Commit script**

```bash
git add recipe/docvqa/scripts/eval.py
git commit -m "[docvqa] feat: Layer-3 eval script (ANLS reproduction on val)"
```

---

## Task 15: Phase-1 GRPO smoke launcher

**Files:**
- Create: `recipe/docvqa/scripts/run_smoke_grpo.sh`

50-100 step GRPO run on a 200-question slice. Layer-4 sanity gate.

- [ ] **Step 1: Write the launcher**

```bash
#!/usr/bin/env bash
# recipe/docvqa/scripts/run_smoke_grpo.sh
# Layer-4 smoke training: ~100 GRPO steps on 200 train questions.

set -euo pipefail
cd "$(dirname "$0")/../../.."  # repo root

# Ensure VLM is running at localhost:8928 (frozen in `vllm` tmux session, GPU 2).

export DOCVQA_VLM_BASE_URL=${DOCVQA_VLM_BASE_URL:-http://localhost:8928}
export DOCVQA_VLM_MODEL_ID=${DOCVQA_VLM_MODEL_ID:-qwen3.6-27b}

# Build a 200-question subset of train if not already.
python - <<'PY'
import json, pathlib, random
p = pathlib.Path("data/train/questions.json")
q = json.loads(p.read_text())
random.Random(0).shuffle(q)
out = pathlib.Path("data/train/questions_smoke200.json")
out.write_text(json.dumps(q[:200], indent=2))
print(f"Wrote {out} ({len(q[:200])} questions)")
PY

CUDA_VISIBLE_DEVICES=0,1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.lora_rank=16 \
    actor_rollout_ref.actor.lora_alpha=32 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.prompt_length=16384 \
    actor_rollout_ref.rollout.response_length=32768 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/docvqa/agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl \
    data.train_files=data/train/questions_smoke200.json \
    data.val_files=data/val/questions.json \
    data.apply_chat_template_kwargs.enable_thinking=true \
    custom_reward_function.path=recipe/docvqa/reward.py \
    custom_reward_function.name=compute_score \
    trainer.total_epochs=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.rollout_data_dir='${hydra:runtime.output_dir}/rollouts' \
    trainer.log_val_generations=20 \
    trainer.project_name=docvqa-verl \
    trainer.experiment_name=smoke-phase1-grpo \
    "$@"
```

- [ ] **Step 2: Make it executable and dry-run with `--help` parsing**

```bash
chmod +x recipe/docvqa/scripts/run_smoke_grpo.sh
# Don't run yet — requires data/train and a vLLM. Just verify shellcheck.
bash -n recipe/docvqa/scripts/run_smoke_grpo.sh
```

Expected: no syntax errors.

- [ ] **Step 3: Commit**

```bash
git add recipe/docvqa/scripts/run_smoke_grpo.sh
git commit -m "[docvqa] feat: Phase-1 smoke GRPO launcher"
```

---

## Task 16: Run Layer-4 sanity training (manual gate)

This task is **not pure code** — it's the empirical gate. Document the run and capture results so we know the setup is healthy.

**Files:**
- Create: `docs/superpowers/notes/2026-04-30-layer4-results.md` (results capture)

Prerequisites: Tasks 1-15 done; train data prepared; vLLM running.

- [ ] **Step 1: Make sure train data exists**

If `data/train/` is not yet built, extend `prepare_data.py` with a DocVQA / MP-DocVQA adapter (this is *future scope* per the spec — for the smoke run we can synthesize a tiny train set from val splits or from any DocVQA family dataset that's quick to download). Document what was done.

- [ ] **Step 2: Bring up the student vLLM**

```bash
tmux new-session -d -s vllm-student
tmux send-keys -t vllm-student "CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85" C-m
```

- [ ] **Step 3: Kick off the smoke training**

```bash
tmux new-session -d -s smoke-grpo
tmux send-keys -t smoke-grpo "bash recipe/docvqa/scripts/run_smoke_grpo.sh" C-m
```

- [ ] **Step 4: Watch metrics**

Tail the W&B run or the local logs for:
- Loss decreasing (or at worst flat) over 50 steps.
- val ANLS not crashing.
- Reward distribution non-degenerate (some > 0).
- No OOMs.
- `outputs/{date}/{time}/rollouts/{step}.jsonl` getting populated each step.

- [ ] **Step 5: Capture results**

Write `docs/superpowers/notes/2026-04-30-layer4-results.md` with: launch command used, GPU layout, observed metrics across 100 steps, val ANLS at step 0 vs step 100, anything that broke and how it was fixed.

- [ ] **Step 6: Commit results**

```bash
git add docs/superpowers/notes/2026-04-30-layer4-results.md
git commit -m "[docvqa] docs: Layer-4 sanity GRPO run results"
```

---

## Self-Review

Spec coverage:

- §3 architecture → Tasks 1, 9-12
- §4 per-turn protocol → Tasks 9-11
- §4.4 knobs → Task 9 (`_DEFAULTS`)
- §4.5 `<think>` round-trip → Task 11 round-trip test
- §5 prompts → Task 7
- §6 tools → Tasks 5-6 + 9-10 (subprocess wiring)
- §7 data layer → Tasks 1 (fixture), 13 (real prep)
- §8 reward → Task 8
- §9 validation layers → Tasks 1-11 (Layers 1-2), 14 (Layer 3), 15-16 (Layer 4)
- §10 rollout dumps → Task 15 (`trainer.rollout_data_dir`) + Task 8 (extra_info passthrough)
- §11 compute → Task 15 (CUDA_VISIBLE_DEVICES)
- §12 Phase-1 launch → Task 15

No placeholder phrases scanned: each step contains real code or a concrete command.

Type consistency:
- `submitted_answer: str | None` consistent across reward.py and agent_loop.py.
- `extra_info` dict keys consistent: `submitted_answer`, `termination`, `num_turns`, `vlm_calls`, `search_calls`, `wall_clock_s`, `messages`, `doc_id`, `question_id`, `category`, `anls`.
- Tool names consistent: `batch_look`, `search`, `SUBMIT`.
- Agent name consistent: `docvqa_repl` (in `@register`, `agent.yaml`, launch script `default_agent_loop`).
- Knob names consistent: `max_iterations_base`, `max_iterations_cap`, `page_factor`, `max_response_tokens_per_turn`, `max_obs_chars`, `subprocess_timeout_s`, `parse_error_strikes_to_terminate`.

**Empirical pre-flight finding (verified before implementation):** stock
`Qwen/Qwen3-8B`'s chat template **strips `<think>` content from prior
assistant turns** when the full message list is re-rendered through
`apply_chat_template`. The community variant `willcb/Qwen3-8B` (same
weights, modified template) preserves `<think>` across turns. Both are
cached locally at `~/.cache/huggingface/hub/`.

**All tasks default to `willcb/Qwen3-8B`** as the student model. Wherever
the plan text below mentions `Qwen/Qwen3-8B`, use `willcb/Qwen3-8B`
instead (e.g. in `tokenizer = AutoTokenizer.from_pretrained(...)`,
`actor_rollout_ref.model.path=...`, the `--student-model` flag of
`scripts/eval.py`, and the `vllm serve` command). This eliminates the
Task-11 round-trip ambiguity up-front.
