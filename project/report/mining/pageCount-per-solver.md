# Per-solver accuracy vs page count — which harness wins on short vs long documents?

**Question.** For each agentic harness (at the 27B-homogeneous config), what is its
accuracy profile across document page-count buckets? Where does each one win — on
short, visually-dense single-page docs, or on long, navigation-heavy multi-page docs?

**Source.** `~/repos/docvqa/output/runs/<run>/results.json` (per-doc `accuracy`,
`elapsed`) + `~/repos/docvqa/data/docvqa-2026/val/ocr/<doc>/metadata.json`
(`num_pages`) + per-doc `tasks/<doc>/summary.md` (`### Trajectory (N iterations)`).
Val set = **25 docs / 80 questions**. Aggregated over all available trials (t1..t8).

> **Major data caveat — read first.** Of the solvers requested, **only RLM and the
> OLD CodeAct are present locally at the 27B-homogeneous config.** The canonical
> 27B-homog **ReAct** (`react-cmp-val-*`), **raw multi-image VLM**
> (`raw-vlm-multi-cmp-val-*`), **No-scaffold / official** (`official-cmp-val-*`), and
> **OCR-only RLM** (`ocr-only-cmp-val-*`) run dirs were evaluated on a different host
> and are **NOT in this checkout** (the local react families are all *mixed*-perception
> configs — 4B/8B/9B-LM + 27B-VLM, or 27B-LM + 9B-VLM — never 27B-homog). Therefore
> **Analysis 3 as specified (RLM−ReAct, CodeAct−ReAct by bucket at 27B-homog) is not
> computable from local data.** What *is* computable — and reported here — is the full
> per-bucket profile for **RLM (two variants) vs CodeAct**, plus the RLM−CodeAct cut.
> The ReAct/baseline rows are left explicitly blank with a pointer to the host that has
> them. Treat every number as **doc-equal-weighted** unless noted (see caveats).

---

## (a) Headline findings

1. **All harnesses peak on the 2–20pg bucket, not on the dense single-page docs, and
   not on the long docs.** RLM-skeletal: 1pg **33.8%** → 2–20 **46.7%** → 21–50 **34.6%**
   → 50+ **36.7%**. The 2–20 bucket is dominated by **engineering_drawing** (3 of 5 docs),
   which are effectively single dense technical diagrams spread over a handful of pages —
   so the peak is a *density/category* effect, not a "few pages is easy" effect. There is
   **no monotonic accuracy-vs-pages trend** for any solver.

2. **The single-page bucket is internally bimodal and that hides the real story.** Within
   the 8 1pg docs, **maps collapse for every solver** (maps_1 0%, maps_3 4%, maps_2 ~15%)
   while infographics / science_poster / engineering_drawing single-pagers score 50–75%.
   So "1-page" ≈ "hardest dense images (maps) + easiest dense images (infographics)"
   averaged together. **Page count and visual density are fully entangled** here: every
   1pg doc is a large dense image; multi-page only appears in text-heavy categories
   (science_paper, slide, comics, business_report). You cannot separate the two axes in
   this 25-doc set — only describe the joint pattern.

3. **No solver collapses past a page threshold — long docs are NOT where these harnesses
   fail.** The 50+ bucket (7 docs, up to 181pg) holds at 35–42%, *equal to or above* the
   21–50 bucket. Long multi-page business_reports are handled fine (business_report_3 @89pg
   = 100% for both RLM variants); the failures are category-specific outliers that appear
   at *all* lengths: comics_1 (36pg) 0%, science_paper_2 (19pg) 0%, business_report_2
   (105pg) 0%. **Length is not the limiter; specific hard documents are.**

4. **RLM-minimal is the most length-robust harness; it is the only one that *gains* on the
   longest docs.** RLM-minimal: 1pg 38.5 → 2–20 45.6 → 21–50 34.0 → **50+ 42.1%** — its
   second-best bucket is the *longest* one. RLM-skeletal and CodeAct are flatter and dip
   slightly past 20pg. If any harness "wins on long docs," it is RLM-minimal.

5. **CodeAct (OLD impl) is the flattest profile — competitive on 1pg, weakest on the
   engineering-drawing 2–20 bucket.** CodeAct: 1pg 35.6 / 2–20 37.6 / 21–50 34.1 / 50+ 35.1
   — within ~3pp across all buckets. It actually *edges RLM-skeletal on 1pg* (35.6 vs 33.8)
   but loses ~8–9pp to both RLM variants on the dense engineering-drawing 2–20 bucket, where
   the crop/zoom/composite REPL loop pays off most. (OLD-impl caveat below.)

---

## (b) Tables

### Page-count buckets and category composition (the density confound, made explicit)

Buckets chosen at the natural gaps in the bimodal distribution (1 / then 2–20 / 21–50 /
50+). n=25 docs total.

| Bucket | n docs | Page range | Category composition | Density / task character |
|---|---|---|---|---|
| **1pg** | 8 | 1 | maps×3, infographics×2, science_poster×2, engineering_drawing×1 | **All single dense large images.** Pure visual-acuity / cropping; no navigation. maps are the hardest sub-class. |
| **2–20** | 5 | 3–19 | engineering_drawing×3, slide×1, science_paper×1 | Mostly **dense technical diagrams** (eng_drawing) over few pages — still density-dominated, minimal navigation. |
| **21–50** | 5 | 30–44 | science_paper×2, slide×2, comics×1 | Text+figure mix; moderate navigation (find-the-right-page). |
| **50+** | 7 | 52–181 | business_report×4, comics×3 | **Navigation-heavy** long docs; locate the right page among many, then read. |

**The confound, stated plainly:** the two short buckets (1pg, 2–20) are *almost entirely*
the dense-image categories (maps, infographics, posters, engineering_drawing); the two long
buckets (21–50, 50+) are *entirely* the text/sequential categories (science_paper, slide,
comics, business_report). So **"short vs long" in this set is also "dense-image vs
multi-page-text."** Any "short-doc strength" is inseparable from "dense-image strength."

### MAIN TABLE — per-solver accuracy by bucket (27B-homog, doc-equal-weighted %)

| Solver (27B-homog) | run family | n trials | 1pg | 2–20 | 21–50 | 50+ | overall (doc-eq / Q-wt) |
|---|---|---|---|---|---|---|---|
| **RLM — skeletal (canonical)** | `rvlm-skeletal-val-t*` | 8 | 33.8 | **46.7** | 34.6 | 36.7 | 37.3 / 38.62 |
| **RLM — minimal (leaner)** | `rvlm-minimal-val-t*` | 8 | 38.5 | **45.6** | 34.0 | **42.1** | 40.0 / 42.03 |
| **CodeAct — OLD impl** | `codeact-3_5-27b-val-t*` | 5† | 35.6 | 37.6 | 34.1 | 35.1 | 35.6 / 37.50 |
| ReAct (27B-homog) | `react-cmp-val-*` | — | — | — | — | — | *not local* (39.38-era host) |
| Raw multi-image VLM | `raw-vlm-multi-cmp-val-*` | — | — | — | — | — | *not local* |
| No-scaffold / official | `official-cmp-val-*` | — | — | — | — | — | *not local* |
| OCR-only RLM (control) | `ocr-only-cmp-val-*` | — | — | — | — | — | *not local* |

† CodeAct-OLD has only 5 local trials (t1–t5; t6 is empty); RLM variants have 8.

**Read-off.** *Best on 1-page dense docs:* **RLM-minimal (38.5%)**, with CodeAct (35.6) just
ahead of RLM-skeletal (33.8). *Best on long 50+ docs:* **RLM-minimal (42.1%)** by ~6–7pp.
*Ranking changes with page count:* yes — **CodeAct ≈ RLM-skeletal on 1pg, but RLM pulls
ahead by ~8–9pp on the dense 2–20 bucket and RLM-minimal pulls ahead by ~7pp on 50+.**
RLM-minimal is the most consistent winner across the page-count axis.

### Per-doc detail (exposes the within-bucket category drivers)

| doc | pages | bucket | RLM-skel | RLM-min | CodeAct |
|---|---|---|---|---|---|
| maps_3 | 1 | 1pg | 4 | 4 | 0 |
| infographics_1 | 1 | 1pg | 57 | 69 | 70 |
| maps_2 | 1 | 1pg | 15 | 10 | 16 |
| science_poster_2 | 1 | 1pg | 51 | 52 | 44 |
| maps_1 | 1 | 1pg | 0 | 0 | 10 |
| engineering_drawing_2 | 1 | 1pg | 69 | 75 | 70 |
| science_poster_1 | 1 | 1pg | 18 | 35 | 20 |
| infographics_2 | 1 | 1pg | 56 | 62 | 55 |
| engineering_drawing_4 | 3 | 2–20 | 75 | 69 | 60 |
| engineering_drawing_3 | 6 | 2–20 | 67 | 50 | 40 |
| engineering_drawing_1 | 7 | 2–20 | 29 | 54 | 40 |
| slide_3 | 18 | 2–20 | 62 | 55 | 48 |
| science_paper_2 | 19 | 2–20 | 0 | 0 | 0 |
| science_paper_3 | 30 | 21–50 | 38 | 50 | 40 |
| slide_2 | 32 | 21–50 | 38 | 33 | 33 |
| slide_1 | 36 | 21–50 | 62 | 56 | 60 |
| comics_1 | 36 | 21–50 | 0 | 0 | 0 |
| science_paper_1 | 44 | 21–50 | 36 | 30 | 37 |
| comics_2 | 52 | 50+ | 39 | 38 | 40 |
| comics_3 | 60 | 50+ | 12 | 33 | 20 |
| comics_4 | 69 | 50+ | 38 | 50 | 40 |
| business_report_3 | 89 | 50+ | 100 | 100 | 90 |
| business_report_2 | 105 | 50+ | 0 | 0 | 0 |
| business_report_4 | 110 | 50+ | 30 | 42 | 36 |
| business_report_1 | 181 | 50+ | 38 | 31 | 20 |

---

## (c) The active-perception cut — RLM − CodeAct by bucket (ReAct unavailable)

The requested RLM−ReAct / CodeAct−ReAct cut **cannot be computed** (no local 27B-homog
ReAct). The locally-feasible active-perception contrast is **RLM vs CodeAct** — both share
the same `batch_look` REPL crop/zoom tool surface; they differ only in context discipline
(RLM hidden-namespace POMDP vs CodeAct append-only MDP), so their difference is *not* an
active-perception-vs-none contrast. Reported for completeness:

| Bucket | RLM-skel − CodeAct | RLM-min − CodeAct |
|---|---|---|
| 1pg (dense images) | −1.8 | +2.9 |
| 2–20 (dense eng-drawings) | **+9.1** | **+8.0** |
| 21–50 (text+figure) | +0.5 | −0.1 |
| 50+ (long navigation) | +1.5 | **+6.9** |

**Interpretation against the density confound.** Both RLM variants beat CodeAct *most* on
the **2–20 bucket**, which is **dense engineering drawings** — i.e. the RLM advantage over
CodeAct concentrates on **single dense technical images**, consistent with a **density /
fine-grained-cropping** effect, not a long-doc navigation effect. RLM-minimal's *second*
advantage (50+, +6.9) is the only length-flavored signal. Since both harnesses have the
same crop/zoom actuators, this is better read as "RLM's hidden-state context lets it iterate
crops on a dense image more effectively than CodeAct's growing append-only context" than as
an active-perception-vs-none claim.

> The proper active-perception test (vs ReAct, which has *no* crop tool — only whole-page
> `look`) lives at the 39.38-era host: per `docs/results.md`, RLM 39.38 vs ReAct 25.16
> (−14.2pp) and per `harness-axis-summary.md` Finding 3, ReAct is the lone perception-bound
> harness because it cannot crop. **Whether that 14pp gap is concentrated on dense 1pg/2–20
> docs or on long docs requires the local ReAct run dirs and is left as a follow-up to run
> on the host that holds `react-cmp-val-*`.**

---

## (d) Effort by bucket per solver (does anyone scale effort with pages?)

Iterations = mean per-question trajectory length; elapsed = sec/doc (wall, includes VLM
queueing — not pure compute).

| Solver | metric | 1pg | 2–20 | 21–50 | 50+ |
|---|---|---|---|---|---|
| RLM-skeletal | iters/q | 11.7 | 14.0 | 14.2 | 13.9 |
| RLM-skeletal | sec/doc | 2475 | 2299 | 2170 | 2274 |
| RLM-minimal | iters/q | 14.0 | 14.3 | 12.6 | 12.2 |
| RLM-minimal | sec/doc | 2247 | 2234 | 1833 | 2122 |
| CodeAct-OLD | iters/q | 12.8 | 13.8 | 15.5 | 13.9 |
| CodeAct-OLD | sec/doc | 3299 | 3127 | 3419 | 2290 |

**Effort is essentially flat across page count for all three solvers** (~12–15 iters/q
everywhere; no solver spends visibly more turns on a 181pg doc than a 1pg one). This is a
real finding: **none of these harnesses scales its reasoning budget with document length** —
they iterate on the *current crop/page*, not on a page-by-page sweep, so a 181-page doc costs
about the same per-question budget as a 1-page one. CodeAct-OLD is uniformly **~1.3–1.5×
slower in wall-time** than RLM at the same bucket (its append-only growing context inflates
per-turn latency), without buying accuracy.

---

## (d′) Caveats

- **n=25 docs, bimodal, tiny buckets (8/5/5/7).** Each bucket mean rests on ≤8 docs; a
  single 0%/100% outlier (comics_1, business_report_3) moves a bucket several pp. These are
  **directional profiles, not significance-tested**. No CIs reported because doc-level n is
  too small to be meaningful.
- **Doc-equal weighting** (each doc = 1 vote) is used for all bucket numbers, so they differ
  from the question-weighted `overall_accuracy` headline (e.g. RLM-skeletal doc-eq 37.3 vs
  Q-wt 38.62). Buckets mix docs with 1–10 questions; doc-equal is the honest per-doc-difficulty
  read but is not the leaderboard metric.
- **ReAct, raw_vlm, official, OCR-only are absent at 27B-homog locally** — the central
  cross-harness comparison and the requested ReAct cut are **incomplete**. Only RLM (2
  variants) and OLD CodeAct could be profiled.
- **CodeAct here is the OLD `codeact-3_5-27b` implementation** (the corrected `codeact_chat`
  dirs are not local) and has only **5 trials** (vs 8 for RLM). Its profile is the
  noisiest and may not reflect the corrected scaffold.
- **Which RLM variant.** Neither local RLM family equals the canonical 39.38 exactly:
  `rvlm-skeletal` (Q-wt 38.62, n=8) is the closest proxy and is treated as the **canonical
  RLM**; `rvlm-minimal` (42.03) is a leaner sibling reported alongside. The exact 39.38 run
  (`rvlm-cmp-val-*`) is not in this checkout.
- **Density confound is irreducible in this set.** Every 1pg doc is a dense single image;
  every multi-page doc is text/sequential. "Short-doc strength" ≡ "dense-image strength"
  here — they cannot be disentangled with these 25 docs. A clean separation would need
  multi-page dense-image docs and single-page text docs, which the val set lacks.

---

## (e) Proposed report table/figure (if a finding survives)

The one finding clean enough to promote:

> **Figure: accuracy vs page-count bucket, per harness (27B-homog, val).** A 4-point line
> per solver (1pg / 2–20 / 21–50 / 50+). It would show (i) the **non-monotonic, 2–20-peaked**
> shape shared by all harnesses — the peak being the dense engineering-drawing bucket, not a
> "fewer pages = easier" effect; (ii) **RLM-minimal's upward tail on 50+**, the only
> length-robust signal; (iii) **flat per-question effort** as an annotation, showing the
> harnesses don't budget more for longer docs.

**Honest framing for the report:** the headline is **negative on the page-count axis** —
*document length is not what separates these harnesses or what limits them; visual density
(maps ≈ 0 everywhere) and a handful of hard documents are.* This is worth one sentence + the
figure, not a section. **Do not claim "harness X wins on long docs"** beyond the narrow,
noisy RLM-minimal 50+ edge. The real cross-harness story (RLM ≫ ReAct active-perception gap)
needs the missing ReAct run dirs and should be deferred.
