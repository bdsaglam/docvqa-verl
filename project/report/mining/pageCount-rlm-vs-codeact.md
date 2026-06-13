# Page-count: does RLM (compact context) beat CodeAct (append-only) on long docs?

**Question.** Do RLM (rvlm: compact/managed context) and CodeAct (append-only,
growing transcript) behave differently as **document page count** grows?
**Hypothesis.** CodeAct's transcript grows with the trajectory, so on long
documents (more pages → more perception calls → longer transcript) CodeAct
should degrade *more* than RLM. Prediction: **(RLM − CodeAct) grows with pages**
— RLM ≳ CodeAct on long docs, tied on short.

**Verdict: NOT SUPPORTED (at both configs).** The RLM−CodeAct gap does **not**
grow with page count. It is a small, roughly *uniform* RLM edge (~+4pp overall)
that is flat-to-non-monotonic across page buckets, and the per-doc
(RLM−CodeAct, num_pages) correlation is essentially zero (Pearson **+0.16**
27B-homog, **−0.08** cross 4b/27b; Spearman +0.10 / −0.07). Crucially the
**50+pg bucket shows no extra CodeAct collapse** — at 27B the gap there (+6.9pp)
is similar to the ≤5pg gap, and CodeAct's *worst* relative bucket is the
mid-range (6–20pg), not the longest. The mechanism's signature (CodeAct
trajectory length blowing up faster with pages than RLM's) is also absent: both
solvers' iteration counts stay flat (~12–16/q) across all buckets.

---

## Headline findings

1. **No page-count trend in the RLM−CodeAct gap (hypothesis not supported).**
   Per-doc correlation of (RLM−CodeAct accuracy) with num_pages is ≈0 at both
   configs: Pearson **r=+0.156** (27B-homog), **r=−0.084** (4b/27b);
   Spearman +0.095 / −0.070, n=25. A positive, growing gap would be the
   hypothesis's signature; we see flat noise around a small positive offset.

2. **RLM's edge is small and roughly uniform, not concentrated on long docs.**
   27B-homog overall RLM−CodeAct = **+4.4pp** (rvlm-minimal 40.0% vs old-codeact
   35.6%, doc-averaged). By bucket: ≤5pg **+3.5**, 6–20pg **+7.8**, 21–50pg
   **−0.1**, 50+pg **+6.9**. The 50+pg gap is *not* the largest; the largest is
   the mid bucket. Cross 4b/27b is similar and non-monotonic (≤5 **+6.2**,
   6–20 **−2.7**, 21–50 **+2.6**, 50+ **+5.4**; overall +3.8pp).

3. **The mechanism's premise fails: CodeAct's trajectory length does not grow
   faster with pages than RLM's.** Mean iterations/question by bucket are flat
   for both — RLM 27B 13.6→13.9→13.8→12.5, CodeAct 27B 12.6→14.3→15.6→13.9
   across ≤5 / 6–20 / 21–50 / 50+. Both solvers cap effort at a budget
   regardless of document length; the append-only transcript does **not**
   visibly balloon on long docs in this 25-doc val set. So the proposed
   degradation channel never engages.

4. **0-accuracy ("total failure") docs are not concentrated on long docs for
   CodeAct relative to RLM.** 27B 0-acc doc-trial rate by bucket: CodeAct
   24/30/24/31% vs RLM 28/25/20/23% — CodeAct is slightly worse on long docs
   (31 vs 23%) but also on mid docs, and the difference is within the noise of
   a 25-doc set. The cross-config picture is dominated by the weak 4b LLM
   (both solvers 53–68% failures), not by solver×page interaction.

---

## Per-bucket tables

**Page-count distribution (25 val docs).** Strongly bimodal: 8 single-page docs
+ 1 three-page, then a tail to 181pg. Median 19, quartiles {1, 19, 52}.
Buckets chosen to give roughly balanced n while separating short/mid/long/very-long:

| Bucket | n docs | pages |
|---|---|---|
| ≤5pg | 9 | 1×8, 3 |
| 6–20pg | 4 | 6, 7, 18, 19 |
| 21–50pg | 5 | 30, 32, 36, 36, 44 |
| 50+pg | 7 | 52, 60, 69, 89, 105, 110, 181 |

### Accuracy by bucket (%, doc-averaged)

**27B-homog — RLM `rvlm-minimal` (n=8) vs CodeAct `codeact-3_5-27b` OLD (n=5):**

| Bucket | RLM | CodeAct | RLM − CodeAct |
|---|---:|---:|---:|
| ≤5pg | 41.9 | 38.3 | **+3.5** |
| 6–20pg | 39.8 | 32.0 | **+7.8** |
| 21–50pg | 34.0 | 34.1 | **−0.1** |
| 50+pg | 42.1 | 35.1 | **+6.9** |
| **Overall** | **40.0** | **35.6** | **+4.4** |

**Cross 4b-LLM / 27b-VLM — RLM (n=8) vs CodeAct OLD (n=8):**

| Bucket | RLM | CodeAct | RLM − CodeAct |
|---|---:|---:|---:|
| ≤5pg | 25.8 | 19.6 | **+6.2** |
| 6–20pg | 13.3 | 16.0 | **−2.7** |
| 21–50pg | 13.0 | 10.4 | **+2.6** |
| 50+pg | 19.6 | 14.2 | **+5.4** |
| **Overall** | **19.5** | **15.7** | **+3.8** |

(RLM−CodeAct is non-monotonic in both configs; the long-doc bucket is not the
solver gap's maximum. The dip in the 21–50pg bucket is driven by two hard
0-acc docs — science_paper_2/comics_1 sit at 19/36pg with both solvers at 0.)

### Effort by bucket

**Mean iterations / question** (pooled over trials; from summary.md
"Trajectory (N iterations)"):

| Bucket | RLM 27B | CodeAct 27B (OLD) | RLM 4b/27b | CodeAct 4b/27b (OLD) |
|---|---:|---:|---:|---:|
| ≤5pg | 13.6 | 12.6 | 7.7 | 7.1 |
| 6–20pg | 13.9 | 14.3 | 11.2 | 9.3 |
| 21–50pg | 13.8 | 15.6 | 11.6 | 9.0 |
| 50+pg | 12.5 | 13.9 | 9.7 | 9.7 |

Iterations are flat across buckets for every config — no length explosion on
long docs. (4b configs use fewer iterations overall, mildly rising from ≤5pg,
but RLM 4b does *not* out-grow CodeAct 4b; if anything RLM uses slightly more
iterations everywhere.)

**Mean elapsed / doc (seconds):**

| Config | ≤5pg | 6–20pg | 21–50pg | 50+pg |
|---|---:|---:|---:|---:|
| RLM 27B | 2178 | 2386 | 1833 | 2122 |
| CodeAct 27B (OLD) | 3137 | 3448 | 3419 | 2290 |
| RLM 4b/27b | 1449 | 1975 | 1896 | 2525 |
| CodeAct 4b/27b (OLD) | 1361 | 1158 | 1349 | 1827 |

Wall time is dominated by VLM `batch_look` latency (~130s/look), not transcript
length, and shows no clean page-count ramp. (CodeAct 27B is *slower* than RLM
27B at every bucket but that is a fixed offset, not a long-doc effect.)

### 0-accuracy doc-trial rate by bucket

| Config | ≤5pg | 6–20pg | 21–50pg | 50+pg |
|---|---:|---:|---:|---:|
| RLM 27B | 28% | 25% | 20% | 23% |
| CodeAct 27B (OLD) | 24% | 30% | 24% | 31% |
| RLM 4b/27b | 31% | 66% | 65% | 55% |
| CodeAct 4b/27b (OLD) | 53% | 59% | 68% | 66% |

### Per-doc detail (27B-homog), sorted by pages

Deltas are small (|d|≤0.15) and sign-scattered; no visible page trend. RLM wins
the very-longest doc (business_report_1, 181pg, +0.11) but also loses some
mid-length docs (slide_1 36pg −0.04, science_paper_1 44pg −0.07).

| pages | doc | RLM | CodeAct | Δ |
|---:|---|---:|---:|---:|
| 1 | maps_3 | 0.04 | 0.00 | +0.04 |
| 1 | infographics_1 | 0.69 | 0.70 | −0.01 |
| 1 | science_poster_1 | 0.35 | 0.20 | +0.15 |
| 1 | maps_2 | 0.10 | 0.16 | −0.06 |
| 1 | maps_1 | 0.00 | 0.10 | −0.10 |
| 7 | engineering_drawing_1 | 0.54 | 0.40 | +0.14 |
| 30 | science_paper_3 | 0.50 | 0.40 | +0.10 |
| 36 | slide_1 | 0.56 | 0.60 | −0.04 |
| 44 | science_paper_1 | 0.30 | 0.37 | −0.07 |
| 60 | comics_3 | 0.33 | 0.20 | +0.13 |
| 89 | business_report_3 | 1.00 | 0.90 | +0.10 |
| 181 | business_report_1 | 0.31 | 0.20 | +0.11 |

(Abridged; full 25-doc list in the analysis script output.)

---

## Caveats (load-bearing)

- **CodeAct here = the OLD implementation, not the corrected one.** The
  `codeact_chat` *corrected* campaign (true multi-turn chat MDP) ran **remotely**
  and its run dirs are **not on this box**. The local `codeact-3_5-27b-val-t*`
  dirs are the pre-correction `codeact` (`docs/results.md`: old codeact pooled
  36.74% ± 4.29, −2.64pp vs rvlm), whereas corrected `codeact_chat` is **39.53%
  ± 2.83 — tied with rvlm (39.38%)**. So the **+4.4pp overall RLM edge measured
  here is an artifact of the OLD codeact**, and the corrected solver erases it.
  Any page-count comparison built on these dirs inherits that confound: we are
  partly measuring "old vs new codeact implementation," not purely "compact vs
  append-only context." The *shape* finding (no page-count trend in the gap) is
  more robust than the *level*, but a clean test needs the `codeact_chat` dirs.
- **n=25 docs, bimodal page distribution.** Only 7 docs in the 50+pg bucket and
  4 in 6–20pg; the single-page bucket is 8 of 9 ≤5pg docs. Bucket means swing on
  one or two hard 0-acc docs (e.g. science_paper_2, comics_1, business_report_2
  all at 0 for both solvers). Per-doc correlations on n=25 have wide CIs — the
  near-zero r's are "no detectable trend," not a tight null.
- **rvlm variant used = `rvlm-minimal`** (prompt-minimized; ~40.0% doc-avg here,
  ~2–3pp above canonical rvlm 39.38). `rvlm-skeletal` (37.1%) and the canonical
  number are lower; absolute levels shift with variant but the page-count shape
  is the object of interest. CodeAct has no matching "minimal" prompt variant
  locally, so the RLM side is mildly favored on prompt engineering.
- **Trial aggregation.** RLM configs n=8 trials; OLD CodeAct 27B has only **5
  valid trials** (t6 results.json is truncated/unparseable — excluded);
  CodeAct 4b/27b n=8. Per-doc accuracy is the trial-mean; bucket accuracy is the
  unweighted doc-average within bucket; iterations/elapsed pooled across trials.
- **Effort proxy.** "Iterations" counts `### Trajectory (N iterations)` headers
  per question in summary.md; this is turn count, a proxy for transcript length
  but not token length. A token-level measurement could still reveal CodeAct
  context growth that turn-count masks — but turn count is flat, so any growth
  would have to come from longer per-turn observations, not more turns.

---

## Cross-link: consistency with the docvqa-verl Pillar-B runaway finding

The docvqa-verl report §5 (`project/report/sections/05-training-small-agent.md`,
backed by `project/report/mining/pillarB-results.md`) finds that the **base 4B
CodeAct/append-only agent** has a per-document **runaway** (budget-exhaustion,
no-submit) rate that **rises sharply with page count**: 23% at 1–5pg → 59% at
50+pg → 100% on the single 181-page doc, **Pearson r ≈ 0.70** over 25 docs, with
accuracy collapsing 18.4 → 7.9 ANLS short→long.

**Is my RLM-vs-CodeAct finding consistent with that? Yes — they measure
different things and do not conflict.** Pillar-B measures the **absolute
level**: CodeAct *does* degrade with page count, strongly (the append-only
transcript grows until the budget is exhausted). My analysis confirms the
*level* effect is present for both solvers — e.g. 27B accuracy is lower on the
hardest long docs, and 4b CodeAct 0-acc rate climbs to 66% at 50+pg. What my
analysis tests is the **relative** quantity (RLM − CodeAct): whether RLM's
*compact* context lets it **escape** that long-doc degradation more than
CodeAct does. The answer is no — **RLM degrades on long docs too**, by a
similar amount, so the *gap* stays flat. In other words: the Pillar-B runaway
is a property of long documents under *this VLM-latency-bound, budget-capped
scaffold generally*, not a property unique to append-only context that compact
context avoids. Both the compact (rvlm) and append-only (codeact) harnesses sit
under the same per-look VLM-latency wall and the same turn budget, so both run
out of budget on the 181-page doc. The append-only-context-growth hypothesis
predicts CodeAct should pull *further* behind RLM as pages grow; that
divergence is what is absent.

Caveat on the cross-link: Pillar-B's runaway is measured on the **base 4B**
agent in docvqa-verl at n=4 with explicit no-submit accounting, while this
mining cut uses the 27B-homog and 4b/27b-cross *evaluation* runs in the docvqa
repo and infers effort from turn count (not a no-submit flag). The two are
consistent in direction but not the same measurement.

---

## Proposed report use

The clean takeaway is a **null with a caveat**, which is still reportable:

> *"On the 25-doc ICDAR-2026 val set, the RLM (compact-context) vs CodeAct
> (append-only) accuracy gap shows no dependence on document page count
> (Pearson r = +0.16 / −0.08 across two model configs; per-bucket gap
> non-monotonic, 50+pg not the worst bucket for CodeAct). The append-only
> context-growth hypothesis is not supported here. Trajectory length is flat
> in page count for both solvers (~12–16 turns), so the proposed degradation
> channel does not engage at this scale."*

**Suggested figure (only if rerun on corrected `codeact_chat`):** scatter of
per-doc (RLM − CodeAct accuracy) vs num_pages (log-x), one panel per config,
with a fitted line + r annotation — visually shows the flat/no-trend result.
**Before putting any *level* claim (the +4pp) in the report, re-run the
page-count cut on the corrected `codeact_chat` dirs** (fetch from the remote
campaign), since the local CodeAct is the superseded implementation and the
corrected one ties RLM overall.

---

*Analysis script: `/tmp/analyze_pagecount.py` (run in `~/repos/docvqa`).
Page counts from `data/docvqa-2026/val/ocr/<doc>/metadata.json`. Run dirs:
`output/runs/{rvlm-minimal,rvlm-skeletal,rvlm-minimal-4b-llm-27b-vlm,
codeact-3_5-27b,codeact-4b-llm-27b-vlm}-val-t*`.*
