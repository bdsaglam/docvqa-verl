# Training Data Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a method-agnostic pool of short-document DocVQA-family datasets (DocVQA-SP, InfographicVQA, ChartQA, MapQA, MP-DocVQA≤3pg, TAT-DQA, DUDE≤2pg, SlideVQA-1evidence) as new `prepare_data.py` adapters that emit verl-ready `questions.json` prompt sets + materialized page images, plus a pool manifest and an optional teacher-trajectory collection wrapper.

**Architecture:** Each dataset is a new `adapter_<name>(split, split_dir) -> list[dict]` in `docvqa/scripts/prepare_data.py`, registered in `ADAPTERS`, mirroring the existing `adapter_docvqa_2026` / `adapter_mmlongbench_doc`. Adapters load HF → materialize `docs/<doc_id>/pages/page_*.png` + `metadata.json` → return rows via the shared `_build_row()` (already verl-ready). Length/evidence filtering happens inside each adapter. A `pool.yaml` manifest curates which prepared `questions.json` files form the pool. Collection (optional) reuses the existing `docvqa/scripts/eval.py` + `make_clean_sft.py` unchanged.

**Tech Stack:** Python 3, `datasets` (HuggingFace), `pypdfium2` (PDF raster), `PIL`, pytest. Repo-local `.venv` (must use `.venv/bin/python` — bare `python` lacks `ray`/`verl`).

---

## Reference patterns (read before starting)

- **The file you extend:** `docvqa/scripts/prepare_data.py` — `_build_row()` (`:56-94`), `_emit_split()` (`:122-133`), `_materialize_docvqa_2026_doc()` (`:140-163`), `adapter_mmlongbench_doc()` + `_mmlb_render_pdf()` (`:239-309`), `ADAPTERS` registry (`:316-323`), `main()` (`:326-339`).
- **Existing test to mirror:** `tests/docvqa/test_prepare_data_mmlb.py` — unit-tests pure row builders with in-memory `hf_rows` + `tmp_path`, no network.
- **Collection command to mirror:** `outputs/_run_collect_v4.sh` (the `eval.py` invocation: 27B teacher at `:8927`, `--n 4 --temperature 0.8 --concurrency 12 --rollout-timeout`, `--run-dir`, `--resume`).
- **Trajectory builder (unchanged):** `docvqa/scripts/make_clean_sft.py` — keeps `anls==1.0 && termination=="submit"`, first-fence truncation, emits single-`messages`-column parquet that `docvqa/train/run_seqkd.sh` consumes.
- **Row schema produced by `_build_row`:** `record_id, dataset, split, doc_id, question_id, question, answer, category, doc_dir, prompt, data_source, reward_model{style,ground_truth}, extra_info{...}`. Multi-alias gold answers are stored as `repr([...])` so the scorer's `ast.literal_eval` sees all candidates (mirror `mp_docvqa.py` answer handling).

**Run tests with:** `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`

**Category mapping (nearest of DocVQA-2026's 8):** DocVQA-SP→`business_report`, InfographicVQA→`infographics`, ChartQA→`science_poster`, MapQA→`maps`, MP-DocVQA→`business_report`, TAT-DQA→`business_report`, DUDE→`business_report` (multi-domain; generic), SlideVQA→`slide`.

---

## Task 1: Shared single-image materializer + test file

**Files:**
- Modify: `docvqa/scripts/prepare_data.py` (add helper after `_materialize_docvqa_2026_doc`, ~line 164)
- Test: `tests/docvqa/test_prepare_data_pool.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/docvqa/test_prepare_data_pool.py
import json
from pathlib import Path

from PIL import Image

from docvqa.scripts.prepare_data import _materialize_single_image_doc


def test_single_image_doc_saves_one_page_and_metadata(tmp_path):
    docs_dir = tmp_path / "docs"
    img = Image.new("RGB", (8, 8), "white")
    doc_dir = _materialize_single_image_doc(
        doc_id="d1", image=img, category="maps",
        dataset="mapqa", split="train", docs_dir=docs_dir,
    )
    assert doc_dir == (docs_dir / "d1").resolve()
    assert (docs_dir / "d1" / "pages" / "page_0.png").exists()
    meta = json.loads((docs_dir / "d1" / "metadata.json").read_text())
    assert meta["num_pages"] == 1
    assert meta["doc_category"] == "maps"


def test_single_image_doc_is_idempotent(tmp_path):
    docs_dir = tmp_path / "docs"
    img = Image.new("RGB", (8, 8), "white")
    _materialize_single_image_doc(doc_id="d1", image=img, category="maps",
                                  dataset="mapqa", split="train", docs_dir=docs_dir)
    _materialize_single_image_doc(doc_id="d1", image=img, category="maps",
                                  dataset="mapqa", split="train", docs_dir=docs_dir)
    assert list((docs_dir / "d1" / "pages").glob("*.png")) == [docs_dir / "d1" / "pages" / "page_0.png"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`
Expected: FAIL with `ImportError: cannot import name '_materialize_single_image_doc'`

- [ ] **Step 3: Write minimal implementation**

```python
# docvqa/scripts/prepare_data.py  (add after _materialize_docvqa_2026_doc)
def _materialize_single_image_doc(
    *, doc_id: str, image: Image.Image, category: str,
    dataset: str, split: str, docs_dir: Path,
) -> Path:
    """Save a one-page doc dir for a single-image dataset. Returns abs doc_dir."""
    doc_out = docs_dir / doc_id
    pages_dir = doc_out / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    out_path = pages_dir / "page_0.png"
    if not out_path.exists():
        image.convert("RGB").save(out_path, format="PNG")
    (doc_out / "metadata.json").write_text(json.dumps({
        "doc_id": doc_id, "num_pages": 1, "doc_category": category,
        "dataset": dataset, "split": split,
    }, indent=2))
    return doc_out.resolve()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_pool.py
git commit -m "docvqa(data): shared single-image doc materializer for pool adapters"
```

---

## Task 2: DocVQA-SP adapter (single-page core)

HF: `lmms-lab/DocVQA`, config `DocVQA`, split `validation`. Per-row fields (confirm in Step 1): `docId`, `questionId`, `question`, `answers` (list 1–6), `image` (single PIL). Multiple questions share a `docId`/image.

**Files:**
- Modify: `docvqa/scripts/prepare_data.py` (add `adapter_docvqa_sp` + register)
- Test: `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe the HF schema (confirm field names before coding)**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
ds = load_dataset('lmms-lab/DocVQA', 'DocVQA', split='validation', streaming=True)
r = next(iter(ds)); print({k: type(v).__name__ for k,v in r.items()}); print({k:(v if k!='image' else 'PIL') for k,v in r.items() if k!='image'})
"
```
Expected: keys include `docId/questionId/question/answers/image`. **If the field names differ, adjust the test + adapter below to the printed names.**

- [ ] **Step 2: Write the failing test (pure row builder, no network)**

```python
# add to tests/docvqa/test_prepare_data_pool.py
from docvqa.scripts.prepare_data import _gold_answer_str

def test_gold_answer_single_vs_multi():
    assert _gold_answer_str(["paris"]) == "paris"
    assert _gold_answer_str(["paris", "Paris"]) == repr(["paris", "Paris"])
    assert _gold_answer_str([]) is None
    assert _gold_answer_str("paris") == "paris"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py::test_gold_answer_single_vs_multi -v`
Expected: FAIL with `ImportError: cannot import name '_gold_answer_str'`

- [ ] **Step 4: Implement the helper + adapter**

```python
# docvqa/scripts/prepare_data.py
def _gold_answer_str(answers) -> str | None:
    """Single alias -> bare str; multiple -> repr(list) for the ast.literal_eval scorer."""
    if isinstance(answers, str):
        return answers
    if not answers:
        return None
    answers = [str(a) for a in answers]
    return answers[0] if len(answers) == 1 else repr(answers)


def adapter_docvqa_sp(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        doc_id = str(r["docId"])
        if doc_id not in seen_docs:
            _materialize_single_image_doc(
                doc_id=doc_id, image=r["image"], category="business_report",
                dataset="docvqa-sp", split=split, docs_dir=docs_dir)
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="docvqa-sp", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="business_report",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register in `ADAPTERS`: add `"docvqa-sp": adapter_docvqa_sp,`.

- [ ] **Step 5: Run unit test to verify it passes**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`
Expected: PASS

- [ ] **Step 6: Smoke-run the adapter on a few docs**

Run:
```bash
.venv/bin/python -c "
from pathlib import Path
from docvqa.scripts.prepare_data import adapter_docvqa_sp
import tempfile; d = Path(tempfile.mkdtemp())
rows = adapter_docvqa_sp('validation', d)[:0]  # materializes all; for a quick check use a slice via streaming if too slow
print('rows:', len(rows))
" 2>/dev/null || echo "if slow/over-large, gate with a small num via load_dataset(..., split='validation[:20]')"
```
Expected: prints a row count and writes `docs/<id>/pages/page_0.png`. Verify one `questions.json` row has `record_id`, `doc_dir`, `answer`, `reward_model.ground_truth`.

- [ ] **Step 7: Commit**

```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_pool.py
git commit -m "docvqa(data): DocVQA-SP prepare_data adapter (single-page, ANLS)"
```

---

## Task 3: InfographicVQA adapter

Same repo `lmms-lab/DocVQA`, config `InfographicVQA`, split `validation`; identical row schema to Task 2.

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Confirm schema parity**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
r = next(iter(load_dataset('lmms-lab/DocVQA','InfographicVQA',split='validation',streaming=True)))
print([k for k in r]) "
```
Expected: same `docId/questionId/question/answers/image` keys as DocVQA-SP.

- [ ] **Step 2: Implement adapter (reuses helpers from Tasks 1–2)**

```python
# docvqa/scripts/prepare_data.py
def adapter_infographicvqa(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/DocVQA", "InfographicVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        doc_id = str(r["docId"])
        if doc_id not in seen_docs:
            _materialize_single_image_doc(
                doc_id=doc_id, image=r["image"], category="infographics",
                dataset="infographicvqa", split=split, docs_dir=docs_dir)
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="infographicvqa", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="infographics",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"infographicvqa": adapter_infographicvqa,`.

- [ ] **Step 3: Run tests**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`
Expected: PASS (unchanged — adapter is exercised by the smoke run).

- [ ] **Step 4: Smoke-run on a 20-row slice**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
r=next(iter(load_dataset('lmms-lab/DocVQA','InfographicVQA',split='validation[:1]')))
print('ok', r['docId'])"
```
Expected: prints `ok <docId>`.

- [ ] **Step 5: Commit**

```bash
git add docvqa/scripts/prepare_data.py
git commit -m "docvqa(data): InfographicVQA prepare_data adapter"
```

---

## Task 4: ChartQA adapter

HF: `HuggingFaceM4/ChartQA`, splits `train`/`val`/`test` (all have answers). Per-row: `image` (PIL), `query` (question), `label` (list; `[0]`=answer). No native doc id → synthesize.

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe schema**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
r=next(iter(load_dataset('HuggingFaceM4/ChartQA',split='train',streaming=True)))
print({k:(type(v).__name__) for k,v in r.items()}); print('label=',r.get('label'),'query=',r.get('query'))"
```
Expected: `image`, `query`, `label` (list), maybe `human_or_machine`. **Adjust field names if different.**

- [ ] **Step 2: Implement adapter**

```python
# docvqa/scripts/prepare_data.py
def adapter_chartqa(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("HuggingFaceM4/ChartQA", split=split)
    rows: list[dict] = []
    for idx, r in enumerate(ds):
        doc_id = f"chartqa_{split}_{idx}"
        label = r.get("label")
        answer = _gold_answer_str(label if isinstance(label, list) else [label])
        _materialize_single_image_doc(
            doc_id=doc_id, image=r["image"], category="science_poster",
            dataset="chartqa", split=split, docs_dir=docs_dir)
        rows.append(_build_row(
            dataset="chartqa", split=split, doc_id=doc_id, question_id=str(idx),
            question=r["query"], answer=answer, category="science_poster",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"chartqa": adapter_chartqa,`.

- [ ] **Step 3: Run tests + smoke**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py -v`
Then: `.venv/bin/python -c "from datasets import load_dataset; print(next(iter(load_dataset('HuggingFaceM4/ChartQA',split='train[:1]')))['query'])"`
Expected: tests PASS; prints a question string.

- [ ] **Step 4: Commit**

```bash
git add docvqa/scripts/prepare_data.py
git commit -m "docvqa(data): ChartQA prepare_data adapter (numeric, relaxed reward in Task 10)"
```

---

## Task 5: MapQA adapter

HF: `nimapourjafar/mm_mapqa` (single `train` split). Per-row: `image` (PIL) + a `data` field holding the instruction/chat with the question + answer that must be parsed. **The `data` structure is unknown — probe first.**

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe the `data` field structure**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
r=next(iter(load_dataset('nimapourjafar/mm_mapqa',split='train',streaming=True)))
print('keys=',list(r)); print('data=',r.get('data'))"
```
Expected: prints the `data` payload. Identify where the question text and answer live (e.g. a list of `{from:'human',value:...}` / `{from:'gpt',value:...}` turns, or `question`/`answer` keys). **Record the exact access path for Step 3.**

- [ ] **Step 2: Write a failing unit test for the parser (using the probed shape)**

```python
# add to tests/docvqa/test_prepare_data_pool.py
from docvqa.scripts.prepare_data import _mapqa_qa_from_data

def test_mapqa_parser_extracts_question_and_answer():
    # Replace this literal with the EXACT shape printed in Step 1.
    sample = [{"from": "human", "value": "Which state has the highest value?"},
              {"from": "gpt", "value": "Texas"}]
    q, a = _mapqa_qa_from_data(sample)
    assert q.endswith("highest value?")
    assert a == "Texas"
```

- [ ] **Step 3: Run test (fails), then implement parser + adapter against the probed shape**

Run: `.venv/bin/python -m pytest tests/docvqa/test_prepare_data_pool.py::test_mapqa_parser_extracts_question_and_answer -v` → FAIL (ImportError).

Implement (adjust the body to the Step-1 shape):
```python
# docvqa/scripts/prepare_data.py
def _mapqa_qa_from_data(data) -> tuple[str, str]:
    """Extract (question, answer) from mm_mapqa's `data` field.
    ADJUST to the exact shape printed by the Step-1 probe."""
    if isinstance(data, dict) and "question" in data:
        return str(data["question"]), str(data.get("answer", "")).strip()
    # chat-turn list fallback:
    human = next(t["value"] for t in data if t.get("from") in ("human", "user"))
    gpt = next(t["value"] for t in data if t.get("from") in ("gpt", "assistant"))
    # strip any leading image placeholder tokens like '<image>\n'
    question = human.split("</image>")[-1].replace("<image>", "").strip()
    return question, str(gpt).strip()


def adapter_mapqa(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("nimapourjafar/mm_mapqa", split=split)
    rows: list[dict] = []
    for idx, r in enumerate(ds):
        doc_id = f"mapqa_{split}_{idx}"
        question, answer = _mapqa_qa_from_data(r["data"])
        _materialize_single_image_doc(
            doc_id=doc_id, image=r["image"], category="maps",
            dataset="mapqa", split=split, docs_dir=docs_dir)
        rows.append(_build_row(
            dataset="mapqa", split=split, doc_id=doc_id, question_id=str(idx),
            question=question, answer=answer or None, category="maps",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"mapqa": adapter_mapqa,`. Re-run the unit test → PASS.

- [ ] **Step 4: Smoke-run + commit**

Run: `.venv/bin/python -c "from pathlib import Path; from docvqa.scripts.prepare_data import _mapqa_qa_from_data; print('parser import ok')"`
```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_pool.py
git commit -m "docvqa(data): MapQA prepare_data adapter (maps category)"
```

---

## Task 6: MP-DocVQA adapter with ≤3-page filter

HF: `lmms-lab/MP-DocVQA`, split `val`. Per-row: `doc_id`, `questionId`, `question`, `answers`, `page_ids` (list of page ids), `image_1..image_20` columns (image cols are `None` past the doc's page count). One row per (doc, question). **Filter docs to `len(page_ids) <= 3` using the column only, before image decode.**

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe schema + page_ids type**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
ds=load_dataset('lmms-lab/MP-DocVQA',split='val')
print([c for c in ds.column_names]); print('page_ids ex=', ds[0]['page_ids'], type(ds[0]['page_ids']))"
```
Expected: confirms `page_ids` is a list (or a stringified list needing `ast.literal_eval`) and the `image_1..N` columns exist. **Note which.**

- [ ] **Step 2: Write a failing unit test for the page-count parser + filter**

```python
# add to tests/docvqa/test_prepare_data_pool.py
from docvqa.scripts.prepare_data import _page_count

def test_page_count_handles_list_and_stringified():
    assert _page_count(["p0", "p1", "p2"]) == 3
    assert _page_count("['p0', 'p1']") == 2
    assert _page_count(None) == 0
```

- [ ] **Step 3: Run (fail), implement helper + adapter**

```python
# docvqa/scripts/prepare_data.py
import ast  # ensure imported at top of file

def _page_count(page_ids) -> int:
    if page_ids is None:
        return 0
    if isinstance(page_ids, str):
        try:
            page_ids = ast.literal_eval(page_ids)
        except (ValueError, SyntaxError):
            return 0
    return len(page_ids)


def _mp_doc_images(row) -> list:
    imgs = []
    for i in range(1, 21):
        v = row.get(f"image_{i}")
        if v is None:
            break
        imgs.append(v)
    return imgs


def adapter_mp_docvqa(split: str, split_dir: Path, max_pages: int = 3) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("lmms-lab/MP-DocVQA", split=split)
    seen_docs: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        if _page_count(r.get("page_ids")) > max_pages:
            continue
        doc_id = str(r["doc_id"])
        if doc_id not in seen_docs:
            doc_out = docs_dir / doc_id / "pages"
            doc_out.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(_mp_doc_images(r)):
                p = doc_out / f"page_{i}.png"
                if not p.exists() and isinstance(img, Image.Image):
                    img.convert("RGB").save(p, format="PNG")
            (docs_dir / doc_id / "metadata.json").write_text(json.dumps({
                "doc_id": doc_id, "num_pages": _page_count(r.get("page_ids")),
                "doc_category": "business_report", "dataset": "mp-docvqa", "split": split,
            }, indent=2))
            seen_docs.add(doc_id)
        rows.append(_build_row(
            dataset="mp-docvqa", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="business_report",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"mp-docvqa": adapter_mp_docvqa,`. Re-run unit test → PASS.

- [ ] **Step 4: Smoke-run, confirm only ≤3-page docs materialized**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
from docvqa.scripts.prepare_data import _page_count
ds=load_dataset('lmms-lab/MP-DocVQA',split='val')
import collections; c=collections.Counter(_page_count(r['page_ids'])<=3 for r in ds)
print('rows <=3pg vs >3pg:', c)"
```
Expected: prints a nonzero count of `True` (≤3pg) rows.

- [ ] **Step 5: Commit**

```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_pool.py
git commit -m "docvqa(data): MP-DocVQA adapter with <=3-page filter"
```

---

## Task 7: TAT-DQA adapter (manual ZIP + JSON join)

HF: `next-tat/TAT-DQA` does NOT `load_dataset` cleanly. Download `tatdqa_docs_{train,dev}.zip` (rendered page images) + `tatdqa_dataset_{train,dev}.json` (QA), unzip, join by doc id. Financial reports, ~1.1 pages/doc, numeric answers.

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe the repo file list + JSON structure**

Run:
```bash
.venv/bin/python -c "
from huggingface_hub import list_repo_files
print([f for f in list_repo_files('next-tat/TAT-DQA', repo_type='dataset')][:40])"
```
Then download + inspect one JSON record:
```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
import json
p=hf_hub_download('next-tat/TAT-DQA','tatdqa_dataset_dev.json',repo_type='dataset')
d=json.load(open(p)); print(type(d), len(d)); print(json.dumps(d[0], indent=2)[:1500])"
```
Expected: prints the QA JSON schema (doc id, page-image file name(s), and a `questions` list with `question`/`answer`/`answer_type`). **Record: the doc-id key, the image-file path field, and the per-question keys.** Map split name: pool `train`→`tatdqa_dataset_train.json` + `tatdqa_docs_train.zip`; pool `dev` similarly.

- [ ] **Step 2: Implement adapter against the probed schema**

```python
# docvqa/scripts/prepare_data.py
import zipfile

def _tatdqa_download_and_unzip(split: str, docs_dir: Path) -> Path:
    """Download tatdqa_docs_<split>.zip into docs_dir/_raw and return the unzip root."""
    from huggingface_hub import hf_hub_download
    raw = docs_dir / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download("next-tat/TAT-DQA", f"tatdqa_docs_{split}.zip",
                              repo_type="dataset")
    root = raw / split
    if not root.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
    return root


def adapter_tatdqa(split: str, split_dir: Path) -> list[dict]:
    # Pool split names map to TAT-DQA's train/dev (test gold is separate).
    from huggingface_hub import hf_hub_download
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    unzip_root = _tatdqa_download_and_unzip(split, docs_dir)
    qa_path = hf_hub_download("next-tat/TAT-DQA", f"tatdqa_dataset_{split}.json",
                             repo_type="dataset")
    data = json.load(open(qa_path))
    rows: list[dict] = []
    for entry in data:
        # ADJUST these keys to the Step-1 probe output:
        doc_id = str(entry["doc"]["uid"])
        page_files = entry["doc"].get("pages") or [entry["doc"].get("page")]
        doc_out = docs_dir / doc_id / "pages"
        doc_out.mkdir(parents=True, exist_ok=True)
        for i, pf in enumerate(page_files):
            src = unzip_root / pf
            dst = doc_out / f"page_{i}.png"
            if src.exists() and not dst.exists():
                Image.open(src).convert("RGB").save(dst, format="PNG")
        (docs_dir / doc_id / "metadata.json").write_text(json.dumps({
            "doc_id": doc_id, "num_pages": len(page_files),
            "doc_category": "business_report", "dataset": "tatdqa", "split": split,
        }, indent=2))
        for q in entry["questions"]:
            rows.append(_build_row(
                dataset="tatdqa", split=split, doc_id=doc_id,
                question_id=str(q["uid"]), question=q["question"],
                answer=_gold_answer_str(q.get("answer")), category="business_report",
                doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"tatdqa": adapter_tatdqa,`.

- [ ] **Step 3: Smoke-run on dev**

Run: `.venv/bin/python docvqa/scripts/prepare_data.py --dataset tatdqa --splits dev --out-root /tmp/tatdqa_test`
Expected: prints `questions.json: N Qs / M docs`; `/tmp/tatdqa_test/tatdqa/dev/docs/<id>/pages/page_0.png` exists.

- [ ] **Step 4: Commit**

```bash
git add docvqa/scripts/prepare_data.py
git commit -m "docvqa(data): TAT-DQA adapter (financial page images, manual ZIP join)"
```

---

## Task 8: DUDE adapter with ≤2-page filter

HF: `jordyvl/DUDE_loader` needs `trust_remote_code=True` + poppler/`pdf2image` (we use `pypdfium2` instead). Documents are PDF binaries → rasterize with the existing `_mmlb_render_pdf` pattern, filter to `len(pdf) <= 2`. **Per-doc page-count + answer field names unverified — probe first.**

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe schema (splits, per-row fields, where the PDF/answer live)**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
ds=load_dataset('jordyvl/DUDE_loader','Amazon_original',split='val',trust_remote_code=True)
print('cols=',ds.column_names); r=ds[0]; print({k:(type(v).__name__) for k,v in r.items()})
print('answers=',r.get('answers'),'doc/pdf field?')"
```
Expected: prints columns (question, answers, doc id, and a PDF path/bytes or page images). **Record: the PDF/image access, the doc-id key, the answers key. If DUDE ships page images rather than a PDF, save them directly and count them for the ≤2 filter.**

- [ ] **Step 2: Implement adapter (PDF-binary variant; adjust to probe)**

```python
# docvqa/scripts/prepare_data.py
def _render_pdf_capped(pdf_path: Path, out_dir: Path, max_pages: int, dpi: int = 150) -> int:
    """Render up to max_pages; return total page count of the PDF (for filtering)."""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(pdf_path))
    n_total = len(pdf)
    if n_total <= max_pages:
        out_dir.mkdir(parents=True, exist_ok=True)
        scale = dpi / 72.0
        for i in range(n_total):
            png = out_dir / f"page_{i}.png"
            if not png.exists():
                pdf[i].render(scale=scale).to_pil().save(png, format="PNG", optimize=True)
    pdf.close()
    return n_total


def adapter_dude(split: str, split_dir: Path, max_pages: int = 2) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("jordyvl/DUDE_loader", "Amazon_original", split=split,
                      trust_remote_code=True)
    seen: set[str] = set()
    rows: list[dict] = []
    for r in ds:
        # ADJUST keys to Step-1 probe:
        doc_id = str(r["docId"])
        if doc_id not in seen:
            pdf_path = Path(r["document"])  # path to PDF binary (per probe)
            n_total = _render_pdf_capped(pdf_path, docs_dir / doc_id / "pages", max_pages)
            if n_total > max_pages:
                seen.add(doc_id)  # mark seen so we skip its questions too
                continue
            (docs_dir / doc_id / "metadata.json").write_text(json.dumps({
                "doc_id": doc_id, "num_pages": n_total,
                "doc_category": "business_report", "dataset": "dude", "split": split,
            }, indent=2))
            seen.add(doc_id)
        if not (docs_dir / doc_id / "pages").exists():
            continue  # doc was over the page cap
        rows.append(_build_row(
            dataset="dude", split=split, doc_id=doc_id,
            question_id=str(r["questionId"]), question=r["question"],
            answer=_gold_answer_str(r.get("answers")), category="business_report",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"dude": adapter_dude,`.

- [ ] **Step 3: Smoke-run on a small slice**

Run: `.venv/bin/python docvqa/scripts/prepare_data.py --dataset dude --splits val --out-root /tmp/dude_test`
Expected: prints `questions.json` counts; only ≤2-page docs have a `pages/` dir.

- [ ] **Step 4: Commit**

```bash
git add docvqa/scripts/prepare_data.py
git commit -m "docvqa(data): DUDE adapter with <=2-page filter (pypdfium2 raster)"
```

---

## Task 9: SlideVQA adapter (single-evidence-slide)

HF: `NTT-hil-insight/SlideVQA` (license cleared; needs `huggingface-cli login`). Per-row: `deck_name`, `page_1..page_20` (PIL), `qa_id`, `question`, `answer`, `evidence_pages` (list of ints, 1-indexed). **Keep `len(evidence_pages)==1`; materialize only that evidence slide as a single-page doc.**

**Files:** Modify `docvqa/scripts/prepare_data.py`; Test `tests/docvqa/test_prepare_data_pool.py`

- [ ] **Step 1: Probe schema (confirm evidence_pages indexing + page column names)**

Run:
```bash
.venv/bin/python -c "
from datasets import load_dataset
ds=load_dataset('NTT-hil-insight/SlideVQA',split='train',streaming=True)
r=next(iter(ds)); print('keys=',list(r)); print('evidence_pages=',r.get('evidence_pages'))"
```
Expected: confirms `evidence_pages` is a list of ints and `page_1..page_N` columns. **Note whether evidence indices are 1-based (page_1 == index 1).**

- [ ] **Step 2: Write failing unit test for evidence selection**

```python
# add to tests/docvqa/test_prepare_data_pool.py
from docvqa.scripts.prepare_data import _slidevqa_evidence_image

def test_slidevqa_single_evidence_selection():
    from PIL import Image
    row = {f"page_{i}": Image.new("RGB", (4, 4)) for i in range(1, 6)}
    row["evidence_pages"] = [3]
    img = _slidevqa_evidence_image(row)
    assert img is not None  # returns page_3 (1-based)
    row["evidence_pages"] = [2, 4]
    assert _slidevqa_evidence_image(row) is None  # multi-evidence -> skip
```

- [ ] **Step 3: Run (fail), implement helper + adapter**

```python
# docvqa/scripts/prepare_data.py
def _slidevqa_evidence_image(row):
    """Return the single evidence slide PIL (1-based page_N), or None if not single-evidence.
    ADJUST 1-based vs 0-based to the Step-1 probe."""
    ev = row.get("evidence_pages") or []
    if len(ev) != 1:
        return None
    return row.get(f"page_{int(ev[0])}")


def adapter_slidevqa(split: str, split_dir: Path) -> list[dict]:
    docs_dir = split_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("NTT-hil-insight/SlideVQA", split=split)
    rows: list[dict] = []
    for r in ds:
        img = _slidevqa_evidence_image(r)
        if img is None:
            continue
        ev = int(r["evidence_pages"][0])
        doc_id = f"{r['deck_name']}__p{ev}"
        _materialize_single_image_doc(
            doc_id=doc_id, image=img, category="slide",
            dataset="slidevqa", split=split, docs_dir=docs_dir)
        rows.append(_build_row(
            dataset="slidevqa", split=split, doc_id=doc_id,
            question_id=str(r["qa_id"]), question=r["question"],
            answer=_gold_answer_str(r.get("answer")), category="slide",
            doc_dir_abs=(docs_dir / doc_id).resolve()))
    return rows
```
Register: `"slidevqa": adapter_slidevqa,`. Re-run unit test → PASS.

- [ ] **Step 4: Smoke-run + commit**

Run: `.venv/bin/python docvqa/scripts/prepare_data.py --dataset slidevqa --splits train --out-root /tmp/slide_test` (requires `huggingface-cli login`).
Expected: each doc dir has exactly one `page_0.png`.
```bash
git add docvqa/scripts/prepare_data.py tests/docvqa/test_prepare_data_pool.py
git commit -m "docvqa(data): SlideVQA adapter (single-evidence slide, slide category)"
```

---

## Task 10: Relaxed-numeric reward for ChartQA/MapQA/TAT-DQA (optional)

`docvqa/metrics.py:evaluate_prediction` does strict numeric matching (no ±tolerance), so correct ChartQA/MapQA/TAT-DQA numbers in a different format are scored wrong — suppressing rejection-sampling yield and RL reward. Add a ±5% relaxed-accuracy branch selectable per dataset. **Run only if Task 11 shows poor numeric-dataset yield.**

**Files:**
- Modify: `docvqa/metrics.py`
- Test: `tests/docvqa/test_reward.py` (existing) or a new `test_relaxed_numeric.py`

- [ ] **Step 1: Confirm current numeric behavior + find the call site**

Run:
```bash
.venv/bin/python -c "
from docvqa.metrics import evaluate_prediction
print(evaluate_prediction('5.2 million', '5200000'))   # likely (False, ...)
print(evaluate_prediction('52%', '52.0'))"
```
Read `docvqa/metrics.py` around `evaluate_prediction` and `_check_strict_match` to see where to branch.

- [ ] **Step 2: Write failing test**

```python
# tests/docvqa/test_relaxed_numeric.py
from docvqa.metrics import relaxed_numeric_correct

def test_relaxed_within_5pct():
    assert relaxed_numeric_correct("5.2", "5.0") is True     # 4% off
    assert relaxed_numeric_correct("6.0", "5.0") is False    # 20% off
    assert relaxed_numeric_correct("foo", "5.0") is False
```

- [ ] **Step 3: Run (fail), implement**

```python
# docvqa/metrics.py
def relaxed_numeric_correct(pred: str, gold: str, tol: float = 0.05) -> bool:
    """ChartQA-style: True if both parse to numbers within `tol` relative error."""
    import re
    def _num(s):
        m = re.search(r"-?\d[\d,]*\.?\d*", str(s).replace("%", ""))
        if not m:
            return None
        try:
            return float(m.group(0).replace(",", ""))
        except ValueError:
            return None
    p, g = _num(pred), _num(gold)
    if p is None or g is None:
        return False
    if g == 0:
        return abs(p) < 1e-9
    return abs(p - g) / abs(g) <= tol
```

- [ ] **Step 4: Wire it into scoring keyed by dataset**

In the collection/reward scorer (where `evaluate_prediction` is invoked with the record's `dataset`/`category`), for `dataset in {"chartqa","mapqa","tatdqa"}` first try `relaxed_numeric_correct(pred, gold)`; if True, treat as correct, else fall back to `evaluate_prediction`. Show the exact wiring at the call site found in Step 1 (e.g. in `docvqa/reward.py` or `eval.py`'s scoring), and add a test asserting a ChartQA-style record is credited.

- [ ] **Step 5: Run tests + commit**

Run: `.venv/bin/python -m pytest tests/docvqa/test_relaxed_numeric.py tests/docvqa/test_reward.py -v`
```bash
git add docvqa/metrics.py tests/docvqa/test_relaxed_numeric.py
git commit -m "docvqa(reward): ChartQA-style relaxed numeric tolerance for chart/map/table datasets"
```

---

## Task 11: Pool manifest + optional collection driver

**Files:**
- Create: `docvqa/train/pool.yaml`
- Create: `docvqa/train/collect_pool.sh`
- Test: `tests/docvqa/test_pool_manifest.py`

- [ ] **Step 1: Prepare all Phase-1 datasets**

Run (writes under `data/<dataset>/<split>/`):
```bash
.venv/bin/python docvqa/scripts/prepare_data.py --dataset docvqa-sp --splits validation
.venv/bin/python docvqa/scripts/prepare_data.py --dataset infographicvqa --splits validation
.venv/bin/python docvqa/scripts/prepare_data.py --dataset chartqa --splits train
.venv/bin/python docvqa/scripts/prepare_data.py --dataset mapqa --splits train
```
Expected: each prints `questions.json: N Qs / M docs`.

- [ ] **Step 2: Write the manifest**

```yaml
# docvqa/train/pool.yaml
# Curated short-doc DocVQA-family pool. `questions` = prepared prompt set.
# K = sample size per dataset for collection; tier = curriculum difficulty.
datasets:
  - name: docvqa-sp
    questions: data/docvqa-sp/validation/questions.json
    category: business_report
    tier: easy
    K: 150
  - name: infographicvqa
    questions: data/infographicvqa/validation/questions.json
    category: infographics
    tier: hard
    K: 150
  - name: chartqa
    questions: data/chartqa/train/questions.json
    category: science_poster
    tier: easy
    K: 150
  - name: mapqa
    questions: data/mapqa/train/questions.json
    category: maps
    tier: easy
    K: 150
  # Phase 2 (add after Task 6-9 prepared):
  # - {name: mp-docvqa, questions: data/mp-docvqa/val/questions.json, category: business_report, tier: mid, K: 150}
  # - {name: tatdqa,    questions: data/tatdqa/dev/questions.json,     category: business_report, tier: mid, K: 150}
  # - {name: dude,      questions: data/dude/val/questions.json,       category: business_report, tier: hard, K: 150}
  # - {name: slidevqa,  questions: data/slidevqa/train/questions.json, category: slide,           tier: mid, K: 150}
```

- [ ] **Step 3: Write a failing test that the manifest is valid + paths exist for prepared sets**

```python
# tests/docvqa/test_pool_manifest.py
import json
from pathlib import Path
import yaml

def test_manifest_entries_have_required_fields_and_valid_questions():
    man = yaml.safe_load(Path("docvqa/train/pool.yaml").read_text())
    assert man["datasets"]
    for d in man["datasets"]:
        assert {"name", "questions", "category", "tier", "K"} <= set(d)
        qp = Path(d["questions"])
        if qp.exists():  # prepared ones must be well-formed
            rows = json.loads(qp.read_text())
            assert rows and {"record_id", "question", "doc_dir", "answer"} <= set(rows[0])
```

- [ ] **Step 4: Run test**

Run: `.venv/bin/python -m pytest tests/docvqa/test_pool_manifest.py -v`
Expected: PASS (Phase-1 `questions.json` files prepared in Step 1 are validated).

- [ ] **Step 5: Write the optional collection driver**

```bash
# docvqa/train/collect_pool.sh
#!/usr/bin/env bash
# Optional: collect 27B-teacher trajectories over the pool for SFT/OPD warmup
# or difficulty curation. RL does NOT need this. Usage: ./collect_pool.sh <dataset-name>
set -xeuo pipefail
cd /home/baris/repos/docvqa-verl
source .venv/bin/activate
NAME="$1"
QUESTIONS=$(.venv/bin/python -c "
import yaml,sys
m=yaml.safe_load(open('docvqa/train/pool.yaml'))
print(next(d['questions'] for d in m['datasets'] if d['name']=='$NAME'))")
.venv/bin/python docvqa/scripts/eval.py \
  --questions "$QUESTIONS" \
  --base-url http://localhost:8927/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8927 --vlm-model Qwen/Qwen3.5-27B \
  --n 4 --temperature 0.8 --concurrency 12 --rollout-timeout 900 \
  --run-dir "outputs/runs/pool-collect-${NAME}" --resume
echo "POOL_COLLECT_DONE ${NAME}"
```
Make executable: `chmod +x docvqa/train/collect_pool.sh`.

- [ ] **Step 6: Commit**

```bash
git add docvqa/train/pool.yaml docvqa/train/collect_pool.sh tests/docvqa/test_pool_manifest.py
git commit -m "docvqa(data): pool manifest + optional teacher-collection driver"
```

---

## Task 12: Build the combined trajectory parquet (optional — only when doing SFT/OPD warmup)

Run only when a recipe needs teacher trajectories. RL-from-scratch skips this entirely.

**Files:** Create `docvqa/train/build_pool_parquet.sh`

- [ ] **Step 1: Per-dataset build via existing make_clean_sft (after collection runs)**

```bash
# docvqa/train/build_pool_parquet.sh
#!/usr/bin/env bash
set -xeuo pipefail
cd /home/baris/repos/docvqa-verl
source .venv/bin/activate
mkdir -p data/sft
for NAME in docvqa-sp infographicvqa chartqa mapqa; do
  RUN="outputs/runs/pool-collect-${NAME}"
  [ -d "$RUN" ] || { echo "skip $NAME (no run dir)"; continue; }
  .venv/bin/python docvqa/scripts/make_clean_sft.py \
    --in "$RUN" --out "data/sft/pool_${NAME}.parquet" --max-per-question 3
done
```

- [ ] **Step 2: Concatenate per-dataset parquets into one**

```bash
# append to docvqa/train/build_pool_parquet.sh
.venv/bin/python -c "
import pandas as pd, glob
parts=[pd.read_parquet(p) for p in glob.glob('data/sft/pool_*.parquet')]
assert parts, 'no per-dataset parquets found'
df=pd.concat(parts, ignore_index=True)
assert list(df.columns)==['messages'], df.columns
df.to_parquet('data/sft/pool_combined.parquet')
print('combined rows:', len(df))"
echo "POOL_PARQUET_DONE"
```

- [ ] **Step 3: Verify it loads in the trainer's expected format**

Run:
```bash
.venv/bin/python -c "
import pandas as pd
df=pd.read_parquet('data/sft/pool_combined.parquet')
assert list(df.columns)==['messages']
assert isinstance(df.iloc[0]['messages'], (list,)) and df.iloc[0]['messages'][0]['role']=='system'
print('ok', len(df))"
```
Expected: prints `ok <N>`. This parquet is directly consumable by `docvqa/train/run_seqkd.sh <parquet> <exp_name>`.

- [ ] **Step 4: Commit**

```bash
git add docvqa/train/build_pool_parquet.sh
git commit -m "docvqa(data): build combined pool SFT parquet from collection runs"
```

---

## Phasing & execution order

- **Phase 1 (substrate, single-page — Tasks 1–5, 11):** DocVQA-SP, InfographicVQA, ChartQA, MapQA + manifest. Lands an RL/OPD-ready prompt substrate fast.
- **Phase 2 (short multi-page + numeric — Tasks 6–9):** MP-DocVQA≤3pg, TAT-DQA, DUDE≤2pg, SlideVQA.
- **Phase 3 (only if needed):** Task 10 (relaxed reward, if numeric yield poor), Task 12 (trajectory parquet, only for SFT/OPD).

Direct RL needs only Phases 1–2 (the `questions.json` prompt sets). Tasks 10 and 12 are conditional.

## Self-review notes

- **Spec coverage:** adapters (Tasks 2–9) ↔ spec component 1; manifest (Task 11) ↔ component 2; relaxed reward (Task 10) ↔ component 3; collection driver + parquet (Tasks 11–12) ↔ component 4. Length filters: MP-DocVQA (Task 6), DUDE (Task 8), SlideVQA (Task 9). Val-leakage guard: see open item below.
- **Open item not yet a task — val-leakage guard.** The spec calls for dropping training docs whose image hash collides with DocVQA-2026 val/test. Low risk (separate corpora) but add as a Phase-2 task if InfographicVQA/MP-DocVQA are used heavily: hash `docs/*/pages/page_0.png` against the prepared `data/docvqa-2026/{val,test}/docs/*` hashes and drop matches before manifest inclusion.
- **Probe-then-implement** is used for MapQA (`data` field), TAT-DQA (JSON schema), DUDE (row fields), SlideVQA (evidence indexing) because their exact HF schemas were not field-verified — the probe step prints the real schema and the code is adjusted to it. This is deliberate, not a placeholder.
