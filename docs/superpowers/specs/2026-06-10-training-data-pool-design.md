# Training data pool — design

**Date:** 2026-06-10
**Status:** approved (design); pending implementation plan
**Repos touched:** `~/repos/docvqa` (dataset loaders + metrics), `~/repos/docvqa-verl` (training-data prep)

## Goal

Assemble a **pool** of short-document DocVQA-family datasets as a
**method-agnostic** training-data substrate for whatever we run on the
scaffold — SFT (rejection-sampled), OPD, GRPO/RL, or combinations,
**including direct RL with no SFT warmup**. The pool does not assume or
privilege any training recipe. One pool, multiple independent consumers.

The pool is built from one set of **loaders + profiles/score_fn +
manifest**. The always-available product is the **prompt set**; the
trajectory parquet is an optional add-on only the trajectory-imitation
methods need:
1. **Prompt set** (core): every question as `(document image, question,
   gold answer, verifier=score_fn)`, annotated with dataset, difficulty
   tier, and **empirical teacher solve-rate**. Sufficient on its own for
   GRPO/RL and OPD.
2. **Trajectory parquet** (optional): rejection-sampled (`score_fn == 1.0`)
   teacher (Qwen3.5-27B) trajectories, in the format `make_clean_sft.py`
   already emits so `run_seqkd.sh` consumes it unchanged. Needed only if a
   recipe does SFT/OPD on teacher trajectories.

### What each method consumes (none is the assumed path)
- **GRPO/RL (incl. from scratch, no warmup)** needs **no pre-collected
  trajectories** — it samples on-policy. It consumes only the prompt set:
  loaders/profiles/`score_fn`/manifest. The teacher-rollout collection (if
  run) doubles as **per-prompt difficulty curation** (drop
  always-impossible prompts that would create GRPO cold-start
  zero-variance) and the per-prompt solve-rate signal.
- **SFT / OPD on teacher trajectories** needs the trajectory parquet
  (product 2).
- **OPD (on-policy, teacher-scored)** needs the prompt set + the live
  teacher endpoint.

Because product 2 is optional, the collection run is **decoupled** from the
loader/manifest substrate: the substrate alone unblocks direct RL; the
collection run is only triggered when a recipe needs trajectories or when we
want the difficulty-curation signal.

### RL-specific load-bearing notes
- **`score_fn` is the RL reward**, not just a rejection filter. A loose or
  buggy relaxed-accuracy verifier → reward hacking. Verifier correctness
  matters *more* for RL than for SFT. Target metric is DocVQA-2026's binary
  ANLS@0.9; pool datasets use ANLS / relaxed-acc as proxy rewards — keep
  them consistent with the target.
- **Short docs are doubly justified for RL**: GRPO runs `G` rollouts ×
  many steps, so rollout cost dominates (proposal's top RL risk). The same
  short-doc gate that makes collection feasible makes GRPO feasible.
- **Teacher solve-rate is a prior, not the selection signal.** GRPO needs
  reward variance *within a group* → prompts the **4B student** sometimes
  solves. Teacher (27B) solve-rate pre-drops impossible prompts; the binding
  signal is the student's per-prompt solve-rate, measured once a 4B policy
  exists (re-estimate with the student, or filter online during GRPO).

## Why a pool (context)

- **DocVQA-2026 has no train split** (val 25 docs/80 Q, test 48 docs/160 Q
  only) — transfer from DocVQA-family is the *only* training-data option.
  In-domain/same-lineage is therefore an asset, not a circularity problem.
  The one hard rule: never train on the DocVQA-2026 val/test items.
- **Prior SFT investigation reached a null** (repo registry): SFT on
  27B-teacher CodeAct trajectories never beat the untrained 4B (~19% on
  docvqa_mini), across in-domain/transfer × undertrained/memorized.
  Mechanism: multi-turn observation-shift defeats trajectory imitation.
  → **More/cleaner data will not fix observation-shift.** This is *why the
  pool is method-agnostic rather than an SFT-on-trajectories bet*: the
  on-policy methods (GRPO/RL, OPD) are first-class consumers, usable with or
  without any SFT warmup. The pool stays useful regardless of which recipe
  wins. The solvability spread it provides (easy→hard, student scores
  non-zero) is valuable across methods — it averts GRPO cold-start
  zero-variance and gives SFT an easy floor — but it is a property of the
  pool, not a commitment to a warmup-first recipe.
- **Feasibility gate:** rollout cost ≈ #turns × #`batch_look` × ~136s VLM
  latency. The prior bottleneck was long MMLongBench docs (repeated 0-turn
  1200s timeouts). **Short docs are a hard requirement**, not a preference —
  and doubly so for RL, where many rollouts per prompt × many steps make
  rollout cost dominate training.

## Pool composition (6/8 category coverage + navigation)

True 8/8 is not achievable with short-doc QA sources: **comics** has no QA
dataset, and **engineering_drawing**'s only candidate (AI2D) is test-only +
multiple-choice (pollutes an ANLS pool) — both left uncovered rather than
covered badly.

| Tier | Dataset | HF id (verified 2026-06-10) | Split (gold) | Doc size | score_fn | doc_category |
|---|---|---|---|---|---|---|
| 1 core | DocVQA-SP | `lmms-lab/DocVQA` cfg `DocVQA` | validation | single-page | ANLS | business_report |
| 1 core | InfographicVQA | `lmms-lab/DocVQA` cfg `InfographicVQA` | validation | single-page | ANLS | infographics |
| 1 core | MP-DocVQA ≤3pg | `lmms-lab/MP-DocVQA` | validation | ≤3 pages | ANLS | (navigation) |
| 2 numeric | ChartQA | `HuggingFaceM4/ChartQA` | train/val | single | relaxed-acc | science_poster |
| 2 numeric | MapQA | `nimapourjafar/mm_mapqa` | train | single | relaxed-acc | maps |
| 2 numeric | TAT-DQA | `next-tat/TAT-DQA` (ZIP) | train/dev | ~1.1 pg | relaxed-acc | business_report |
| 3 multipage | DUDE ≤2pg | `jordyvl/DUDE_loader` (trust_remote_code) | train/val | ≤2 pages | ANLS | (diverse) |
| 4 slide | SlideVQA (1-evidence) | `NTT-hil-insight/SlideVQA` (license cleared) | train/val | evidence slide only | EM/F1 | slide |
| optional | DVQA / PlotQA | `vikhyatk/dvqa` / `achang/plot_qa` | train | single | relaxed-acc | (synthetic volume) |

**Covered:** business_report, infographics, maps, science_poster
(science_paper ~partial via charts/figures), slide, + multi-page
navigation. **Uncovered:** comics, engineering_drawing.

### Per-dataset loader notes (from loadability verification)
- **DocVQA-SP + InfographicVQA** — identical schema; one parametrized
  loader (`lmms_lab_docvqa.py`, config param). 1 row = 1 Q = 1 image;
  `answers` is a 1–6 alias list → `repr([...])` for the multi-alias scorer.
  Test answers are RRC-hidden → collect on `validation`. apache-2.0.
- **MP-DocVQA** — loader exists; add `max_pages` filter via
  `len(ast.literal_eval(row["page_ids"]))` (column-only, no image decode).
- **ChartQA** — fields `query`/`label` (label[0]=answer); single chart;
  official metric relaxed accuracy (±5% numeric). **License GPL-3.0**
  (copyleft) — fine for generating trajectories we don't redistribute.
- **MapQA** — mirror `nimapourjafar/mm_mapqa`, train-only, parse the `data`
  instruction field for (question, answer). License unverified.
- **TAT-DQA** — does **not** `load_dataset` cleanly; download + unzip
  `tatdqa_docs_*.zip` (rendered page images) + join `tatdqa_dataset_*.json`
  QA by doc id. Heavy numeric/arithmetic → relaxed-acc, not ANLS.
  CC-BY-4.0.
- **DUDE** — `trust_remote_code=True` + poppler/pdf2image; documents are
  PDF binaries → rasterize via the existing `pypdfium2` path. **Per-doc
  page count must be verified on a loaded example** (main implementation
  risk) for the ≤2-page filter. CC-BY-4.0.
- **SlideVQA** — schema has `evidence_pages` (seq int); keep
  `len(evidence_pages)==1` and slice context to the evidence slide (decks
  avg ~20 slides — do NOT load the whole deck). `page_1..20` PIL columns.
- **DVQA/PlotQA** — synthetic, numeric; optional. Verify the PlotQA
  mirror's `image` column is real PIL before use.

### Dropped
- **AI2D** — HF is test-only + 4-way MC (accuracy, not ANLS). Would
  pollute the rejection-sampling filter. engineering_drawing stays
  uncovered.

## Architecture

Scope split follows the repo rule (loaders/metrics in `~/repos/docvqa`;
training-data prep in `docvqa-verl`).

### In `~/repos/docvqa`
1. **Loaders** in `src/docvqa/datasets/`, each → `Document(doc_id,
   doc_category, images: list[PIL], questions: [{question_id, question,
   answer}])`, dispatched in `load_documents()` like the existing
   `mp_docvqa.py` / `mmlongbench_doc.py`:
   `lmms_lab_docvqa.py` (DocVQA-SP + InfographicVQA), `chartqa.py`,
   `mapqa.py`, `tatdqa.py`, `dude.py`, `slidevqa.py`; extend `mp_docvqa.py`
   with `max_pages`. (`dvqa.py`/`plotqa.py` optional, later.)
2. **Profiles + score_fns**: register one `DatasetProfile` per dataset in
   `profile.py:_PROFILES` keyed by HF id, `doc_category` = nearest of the 8
   (so the scaffold's per-category answer-format tips apply). Add a
   **`relaxed_accuracy` score_fn** (±5% numeric tolerance, ChartQA-style)
   in `metrics.py` for the numeric sets. **Load-bearing**: strict ANLS=1.0
   would wrongly reject correct-but-reformatted numbers during rejection
   sampling.

### In `docvqa-verl`
3. **Pool manifest** `docvqa/train/pool.yaml` — single source of truth:
   per dataset → HF id, split, `max_pages`, sample size `K`, category,
   score_fn, difficulty tier. This + the loaders/profiles is the **primary,
   always-needed deliverable** (the RL/OPD prompt substrate); it stands
   alone with no collection run.
4. **Collection driver** (optional) `docvqa/train/collect_pool.sh` (+ small
   py) — run only when a recipe needs trajectories or the difficulty signal:
   per manifest entry → load (filtered) → sample `K` → run 27B teacher
   through the existing `eval.py` (`DocVQAReplAgentLoop`) → score with the
   dataset's `score_fn` → keep solved → `make_clean_sft.py` → per-dataset
   parquet → concat → trajectory parquet; record per-question solve-rate
   back onto the prompt set.

### Data flow
```
loaders + profiles/score_fn + manifest  ──► prompt set (RL/OPD-ready, no collection needed)

optional collection (when trajectories/difficulty signal wanted):
manifest → filtered load → sample K → teacher rollouts (eval.py)
  → per-dataset score_fn filter (rejection sampling)
  → make_clean_sft → per-dataset parquet → concat → trajectory parquet
  → (+ per-question solve-rate written back onto the prompt set)
```
No changes to the scaffold or trainer — loaders + profiles + a
manifest-driven wrapper around existing `eval.py` and `make_clean_sft.py`.

## Collection plan (optional — only if trajectories/difficulty signal needed)

Direct RL needs none of this; it runs off the prompt substrate. Run
collection when a recipe does SFT/OPD on trajectories, or to curate the RL
prompt set by difficulty.

**Preconditions:** fixed scaffold (first-fence parse, stop seqs) +
`sandbox.py` PNG-leak fix — both already in place.

**Per-dataset sampling:** `K`≈150 default, `n=4` rollouts, `temp=0.8`,
`conc=12`. **Per-rollout timeout** ~900s + **image downscale** (max-dim
~1600px) in the collection config (kills the 0-turn timeout waste).

**Difficulty signal:** record empirical teacher solve-rate per sampled
question (across its `n` rollouts) → difficulty tag + RL prompt-curation
input. Written back onto the prompt set as a first-class field.

**Phased (each phase yields a usable slice — de-risks June 15):**
- **Phase 1 (single-page, fast):** DocVQA-SP, ChartQA, MapQA,
  InfographicVQA.
- **Phase 2 (short multi-page + numeric):** MP-DocVQA ≤3pg, TAT-DQA,
  DUDE ≤2pg, SlideVQA 1-evidence.
- **Phase 3 (optional synthetic):** DVQA/PlotQA, only if more easy
  student-solvable items are needed.

**Val-leakage guard (lightweight):** before building the parquet, drop any
training doc whose image hash collides with a DocVQA-2026 val/test image
(InfographicVQA/MP-DocVQA share lineage; val is only 25 docs → cheap
insurance).

**Budget (if collection is run):** ~150 Q × 9 datasets × 4 rollouts ≈ 5.4k
rollouts; ~1.5–2 days wall-clock for the full pool at conc12 + short-doc
latency; Phase 1 alone lands a usable set in well under a day. Expected
yield ~800–1600 solved trajectories (prior collections kept 80–245).

## Success criteria

- **Substrate (always):** each pool dataset loads to `Document` objects with
  correct images + gold answers; length filters enforced (MP-DocVQA ≤3pg,
  DUDE ≤2pg, SlideVQA single evidence slide); each has a registered profile
  + `score_fn`, so the manifest is usable as a verl RL/OPD prompt dataset
  **with no collection run**.
- Numeric datasets scored by relaxed-accuracy (correct-but-reformatted
  numbers credited) — load-bearing as both rejection filter and RL reward.
- No DocVQA-2026 val/test image leaks into the pool.
- **Collection (if run):** trajectory parquet builds and loads in
  `run_seqkd.sh` unchanged; prompt set carries per-question empirical
  solve-rate.

## Open risks / verify-on-implementation
- DUDE per-doc page-count field (needed for ≤2pg filter) — verify on a
  loaded example.
- PlotQA mirror `image` column is real PIL (if Phase 3 used).
- MapQA / DVQA / PlotQA / TAT-DQA licenses (unverified) — confirm before
  any redistribution (trajectory generation for internal training is fine).
- ANLS-vs-numeric: confirm `relaxed_accuracy` matches ChartQA's official
  tolerance so yield isn't artificially suppressed.

## Out of scope
- comics / engineering_drawing coverage (no suitable short-doc QA source).
- The training runs themselves (SFT / OPD / GRPO/RL) and the choice of
  recipe — this is method-agnostic data/prompt prep + optional trajectory
  collection only.
- Standalone-SFT-beats-baseline claims (settled null; not this pool's job).
