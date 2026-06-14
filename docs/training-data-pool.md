# Training data pool — dataset card

A **method-agnostic** pool of short-document DocVQA-family data for training the
≤8B agent (SFT, OPD, or GRPO/RL) inside the Perceive-Reason-Code / CodeAct
scaffold. Short docs keep multi-turn `batch_look` rollouts feasible; every row
carries a gold answer + a rule verifier, so the pool doubles as an RL reward
source and a rejection-sampling filter. Built entirely in this repo
(`docvqa/scripts/`); the agent loop reads page images from each row's `doc_dir`.

- **Status:** complete (8 sources prepared; balanced pool built).
- **Layout:** `data/<dataset>/<split>/{questions.json, docs/<doc_id>/pages/page_*.png, docs/<doc_id>/metadata.json}` per source; balanced pool at `data/pool/`.
- **`data/` is gitignored** (regenerable, holds absolute paths) — rebuild from the commands in *Reproduce* below.

---

## 1. Sources & provenance

8 datasets, chosen for short documents + scoreable answers + coverage of the
ICDAR-2026 DocVQA document types. `category` is the per-row label the agent's
per-category prompting reads (nearest of DocVQA-2026's 8, except MMLB which
keeps its own doc-type labels).

| source | HF id (config) | split used | category label | license | notes |
|---|---|---|---|---|---|
| docvqa-sp | `lmms-lab/DocVQA` (DocVQA) | validation | business_report | apache-2.0 | scanned industry docs (orig. SP-DocVQA) |
| infographicvqa | `lmms-lab/DocVQA` (InfographicVQA) | validation | infographics | apache-2.0 | high-res single infographic |
| chartqa | `HuggingFaceM4/ChartQA` | train | science_poster | **GPL-3.0** | chart value-reading; numeric-heavy |
| mapqa | `nimapourjafar/mm_mapqa` | train | maps | unverified | choropleth/bar-map QA; ~13 Q/image |
| mp-docvqa | `lmms-lab/MP-DocVQA` | val | business_report | unverified | multi-page, **filtered to ≤3 pages** |
| tatdqa | `next-tat/TAT-DQA` | dev | business_report | CC-BY-4.0 | financial report pages; numeric reasoning |
| slidevqa | `NTT-hil-insight/SlideVQA` | train | slide | NTT eval license (gated) | **single-evidence slide only** |
| mmlongbench-doc | `yubo2333/MMLongBench-Doc` | train | *(native doc-type)* | unverified | **long** multi-page; added as it helped DocVQA in SFT |

Splits are chosen for **public gold answers** (test answers are hidden for most).
`DUDE` and `AI2D` were evaluated and **excluded** (DUDE: multi-GB + fragile
replicated loader; AI2D: test-only multiple-choice, pollutes an ANLS pool).
Categories `comics`, `engineering_drawing`, `science_paper` have no suitable
short-doc QA source, so the pool covers **5 of DocVQA-2026's 8** label-wise
(+ MMLB's own types).

---

## 2. Statistics (raw, as prepared)

Per-source, over the full prepared `questions.json` (before pool sampling):

| source | docs | questions | pages/doc (min/med/mean/max) | % multi-alias gold | % numeric answer | median answer chars |
|---|---:|---:|---|---:|---:|---:|
| docvqa-sp | 1,286 | 5,349 | 1 (single image) | 45% | 22% | 17 |
| infographicvqa | 500 | 2,801 | 1 (single image) | 25% | 32% | 9 |
| chartqa | 28,299 | 28,299 | 1 (single image) | 1% | 81% | 4 |
| mapqa | 37,417 | 483,416 | 1 (single image) | 0% | 3% | 12 |
| mp-docvqa | 583 | 2,986 | 1 / 1 / 1.6 / 3 | 100% | 21% | 20 |
| tatdqa | 274 | 1,644 | 1 / 1 / 1.1 / 2 | 13% | 16% | 13 |
| slidevqa | 5,962 | 9,381 | 1 (single image) | 0% | 42% | 6 |
| mmlongbench-doc | 116 | 964 | 9 / 28 / 37 / 80 | 14% | 41% | 10 |
| **total** | **74,437** | **534,840** | — | — | — | — |

**Page-count distribution** (multi-page sources; single-image sources are 1 page each):

| source | 1pg | 2–3pg | 4–10pg | 11–30pg | 31+pg |
|---|---:|---:|---:|---:|---:|
| mp-docvqa (≤3 filter) | 315 | 268 | 0 | 0 | 0 |
| tatdqa | 241 | 33 | 0 | 0 | 0 |
| mmlongbench-doc | 0 | 0 | 2 | 66 | 54 |

(MMLB: 122 doc dirs materialized; 116 carry questions in `questions.json` — the 6-doc gap is materialized docs whose questions were filtered out. The raw table counts the 116 with questions.)

So multi-page *navigation* signal comes from mp-docvqa (~46% of its docs are 2–3pg) and the long tail from MMLB (median 28 pages). Everything else is single-image.

**Answer characteristics worth knowing for reward design:**
- **Numeric-heavy** (relaxed-accuracy reward matters): chartqa (81%), slidevqa (42%), mmlb (41%), infographicvqa (32%). **MapQA is mostly categorical** (3% numeric — state names, not numbers), despite being a "chart-like" source.
- **Multi-alias gold** (stored as `repr([...])`; the reward must parse it — see Schema): mp-docvqa (100%), docvqa-sp (45%), infographicvqa (25%), mmlb (14%), tatdqa (13%). chartqa/mapqa/slidevqa are single-answer.

---

## 3. The balanced pool

The raw pool is dominated by MapQA (483k of 535k questions). `sample_pool.py`
draws `n_sample` rows/source (set in `docvqa/train/pool.yaml`) into a balanced
set so no source dominates:

```
data/pool/prompts.json         # combined, shuffled — 11,464 prompts
data/pool/sampled/<name>.json  # per-source sample (collect_pool.sh reads these)
```

**Composition (seed 42, n_sample=1500/source; MMLB capped at its 964 available):**

| by dataset | n |  | by category | n |
|---|---:|---|---|---:|
| docvqa-sp | 1,500 |  | business_report | 4,500 |
| infographicvqa | 1,500 |  | maps | 1,500 |
| chartqa | 1,500 |  | infographics | 1,500 |
| mapqa | 1,500 |  | science_poster | 1,500 |
| mp-docvqa | 1,500 |  | slide | 1,500 |
| tatdqa | 1,500 |  | *(MMLB doc-types)* | 964 |
| slidevqa | 1,500 |  |  |  |
| mmlongbench-doc | 964 |  | **total** | **11,464** |

> **Balanced by dataset, not by category.** Three sources map to
> `business_report` (docvqa-sp + mp-docvqa + tatdqa) so that label is ~3× the
> others; MMLB contributes its own 7 doc-type labels (Academic paper, Financial
> report, Research report/Introduction, Guidebook, Tutorial/Workshop, Brochure,
> Administration/Industry file). To reshape, edit `n_sample` per source and
> re-run `sample_pool.py`.

---

## 4. Row schema

Every row in `questions.json` / `prompts.json` (from `prepare_data.py:_build_row`):

```json
{
  "record_id": "<dataset>:<split>:<doc_id>:<question_id>",
  "dataset": "...", "split": "...", "doc_id": "...", "question_id": "...",
  "question": "...", "answer": "...",        // multi-alias gold stored as repr([...])
  "category": "maps", "doc_dir": "/abs/path/to/docs/<doc_id>",
  "prompt": [{"role": "user", "content": "<question>"}],   // verl prompt-len filter only
  "data_source": "<dataset>-<split>",
  "reward_model": {"style": "rule", "ground_truth": "<answer>"},
  "extra_info": {"record_id","dataset","split","doc_id","question_id","category"}
}
```

`doc_dir` is **absolute**; the agent loop builds the real prompt from
`question`/`category`/`doc_dir` (not from `prompt`). **Multi-alias gold** is a
`repr([...])` string so the scorer's `ast.literal_eval` sees all candidates —
`docvqa/reward.py:compute_score` parses it (continuous max-ANLS over
candidates); a raw `get_anls` call would mis-score these, so don't bypass it.

---

## 5. How to train with it

Use `.venv` for prepare/sample; RL training uses its own env (see the session
registry). The prompt set carries gold + verifier, so **RL needs no teacher**.

### GRPO / RL
```bash
# per source (one --docs-dir per source); concat parquets for a combined run
.venv/bin/python docvqa/scripts/make_rl_parquet.py \
  --questions data/pool/sampled/chartqa.json \
  --docs-dir  data/chartqa/train/docs \
  --out       data/pool/rl/chartqa.parquet --require-answer
```
Point `docvqa/train/run_grpo.sh` at the parquet(s). Reward = `docvqa/reward.py:compute_score`
(continuous ANLS, multi-alias-aware; the project's target metric is binary ANLS@0.9).
`make_rl_parquet.py` takes a single `--docs-dir`, so build per-source parquets and
concatenate (the combined `prompts.json` spans 8 docs dirs).

### SFT / OPD warmup (needs the VLM)
```bash
./docvqa/train/collect_pool.sh chartqa            # 27B rollouts over the sample -> outputs/runs/pool-collect-chartqa
./docvqa/train/build_pool_parquet.sh chartqa ...  # make_clean_sft (anls==1.0 & submit) -> data/sft/pool_combined.parquet
./docvqa/train/run_seqkd.sh data/sft/pool_combined.parquet <exp>
```
`collect_pool.sh`'s `--base-url/--vlm-base-url` point at the 27B VLM — update them
to the current endpoint (local `:8927` may be down; check the session registry for
a remote tunnel). MMLB collection is **expensive** (long docs) — lower its `n_sample`.

---

## 6. Reproduce

`data/` is gitignored. Rebuild (no GPU; downloads ~tens of GB to the HF cache):

```bash
.venv/bin/python docvqa/scripts/prepare_data.py --dataset docvqa-sp        --splits validation
.venv/bin/python docvqa/scripts/prepare_data.py --dataset infographicvqa   --splits validation
.venv/bin/python docvqa/scripts/prepare_data.py --dataset chartqa          --splits train
.venv/bin/python docvqa/scripts/prepare_data.py --dataset mapqa            --splits train
.venv/bin/python docvqa/scripts/prepare_data.py --dataset mp-docvqa        --splits val
.venv/bin/python docvqa/scripts/prepare_data.py --dataset tatdqa           --splits dev
.venv/bin/python docvqa/scripts/prepare_data.py --dataset slidevqa         --splits train   # needs HF login (gated)
# mmlongbench-doc already prepared at data/mmlongbench-doc/train/
.venv/bin/python docvqa/scripts/sample_pool.py --seed 42
```
Per-source split names are **not** uniform — use the ones above (`prepare_data.py`'s
`main()` default `val,test` is wrong for most). Adapters live in
`docvqa/scripts/prepare_data.py:ADAPTERS`; weights in `docvqa/train/pool.yaml`.

---

## 7. Eval-set contamination check

`find_pool_leakage.py` checks pool images against DocVQA-2026 **val** pages
(the eval set), since DocVQA-2026 doc_ids carry no provenance to join on.
Method: **dHash perceptual hash to shortlist near-duplicate pages → exact pixel
diff to confirm** (cheap two-stage; a raw byte hash would miss re-encoded
images).

**Verdict (val): no question-level leakage.** Of 31 dHash candidates, pixel
verification confirmed only **2 truly reused images** (InfographicVQA ↔ DocVQA-2026
`infographics_1/2`); the rest were false positives on low-detail pages (e.g.
slidevqa "matching" science papers). Critically, the 2 reused images carry
**different questions/answers** in val than in InfographicVQA, so **no eval answer
is memorizable** — image overlap ≠ leakage without a shared question. Kept in.

> **Not yet checked: DocVQA-2026 `test`** (not materialized; images are
> downloadable, answers hidden). Re-run `find_pool_leakage.py` against it before
> trusting test-set numbers.

---

## 8. Limitations & caveats

- **Category balance:** balanced by dataset, not category (business_report ~3×; see §3).
- **Tiny per-category eval resolution** does *not* apply here — this is a *training* pool; the eval set is separate (`data/docvqa-2026/val/`, full 80-Q val; plus `docvqa_mini`, a 13-doc/38-Q stratified smoke subset).
- **Licenses:** ChartQA is GPL-3.0; SlideVQA is gated under an NTT eval license; mapqa/mp-docvqa/mmlb licenses are unverified. Fine for generating training trajectories we don't redistribute; verify before redistributing data.
- **Numeric scoring:** chartqa/tatdqa/mmlb numerics need relaxed-accuracy to be credited fairly; strict ANLS under-credits reformatted numbers (lowers RL reward / rejection yield). A `relaxed_numeric_correct` hook is specced but only wire it in if numeric yield is poor.
- **MapQA volume + style:** 483k raw questions (capped to 1,500 in the pool) and mostly *categorical* answers — don't assume it stresses numeric reading.
- **First-run-verified adapters:** tatdqa (ZIP join), slidevqa (single-evidence), mp-docvqa (≤3 filter), mmlb (PDF raster) all materialized correctly in practice; mapqa needed an HF-image-dict→PIL fix (`_coerce_image`).

---

## 9. Provenance

- Code: `docvqa/scripts/{prepare_data.py, sample_pool.py, make_rl_parquet.py, find_pool_leakage.py}`, `docvqa/train/{pool.yaml, collect_pool.sh, build_pool_parquet.sh}`, reward in `docvqa/reward.py`. Committed on `docvqa-stage0` (`b3928a0b..` series).
- Stats in this card were computed from the prepared `data/` on 2026-06-13 via `outputs/_pool_stats.py`.
