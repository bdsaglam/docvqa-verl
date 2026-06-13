# Pillar A — quantitative slice analysis (agentic-harness evaluation)

**Scope.** Mining of `~/repos/docvqa/output/runs/*/results.json` + per-doc
`tasks/<doc>/summary.md` for the claim that *active perception via a code REPL*
— not model scale, not a fixed tool loop — is the accuracy lever on
document-agent DocVQA. Val set = 25 docs / 80 questions, 8 categories × 10 Q.
All numbers computed live from the run dirs unless marked **[from doc]**.

> **CRITICAL DATA-AVAILABILITY CAVEAT (read first).** The run dirs that back the
> headline 8-solver 27B-homog comparison in `docs/results.md`
> (`*-cmp-val-t*` for `react`, `direct_vlm`, `raw_vlm_multi`, `official`,
> `rlm_ocr`, and the canonical `rvlm`/`codeact`) are **NOT on disk** — only their
> summary numbers survive in the per-solver markdown docs. The live run dirs are a
> *different, newer* set: rvlm prompt-minimization variants
> (`rvlm-minimal/unified/skeletal/hybrid-val`, n=8), `codeact-3_5-27b-val` (n=5),
> and a **complete cross-model matrix** (rvlm/codeact/react × {4b,8b,9b}-LM+27b-VLM
> and the 27b-LM/9b-VLM "v3" corner, n=8 each). So the per-category/per-doc slices
> below are computed on the **cross-model matched-config runs** (same VLM, only
> harness/reasoner differ — methodologically the *cleaner* contrast) plus the
> live `rvlm-minimal-val` (n=8) as the 27B active-perception reference. The
> classic 27B-homog `react`/`raw_vlm`/`direct_vlm`/`official` baselines and the
> `rlm_ocr` control are cited **[from doc]** where needed. Live `rvlm-minimal-val`
> n=8 = **42.0% ± 2.2** (the published `rvlm` is 39.4% ± 1.5 — same family, slightly
> higher; treat ~40% as the 27B active-perception anchor).

---

## (a) Headline findings

1. **Active perception is a per-category lever concentrated on visually dense
   docs, not a uniform lift.** At matched config (27B reasoner / 9B VLM, the
   cleanest harness contrast on disk), **rvlm − ReAct** by category:
   engineering_drawing **+30.0**, business_report **+23.8**, maps **+18.8**,
   infographics **+18.8**, science_poster **+15.0** — but science_paper only
   **+4.9** and slide **+10.0**. The REPL helps *most* exactly where recursive
   crop/zoom perception should matter (drawings, dense graphics) and *least* on
   linear text-heavy pages. **Supports** the active-perception claim and its
   mechanism. (n: rvlm n=3, codeact n=4, react n=8 — small; see gaps.)

2. **The "advantage grows with page count" hypothesis is FALSE — the slice is
   U-shaped or inverted, and the real driver is visual density, not length.**
   - v3 (27bLM/9bVLM): rvlm−ReAct = **+18.3** (≤10pg) / **+4.4** (11–40) /
     **+15.8** (40+) — *weakest in the middle*, not growing with pages.
   - Mixed 9bLM/27bVLM (n=8, strong perception): rvlm−ReAct = **+6.0** (≤10) /
     **+4.4** (11–40) / **−0.6 (40+)** — the advantage *vanishes on long docs*
     when the VLM is strong; ReAct's whole-page reads suffice once the VLM is
     good and the answer page is found.
   The ≤10pg bucket is where rvlm wins biggest, and that bucket is **dominated by
   single-page visually-dense categories** (8 of 11 docs are maps / infographics /
   science_poster / single-page eng_drawings). So page count is a confound for
   category. **Complicates** the naive "more pages ⇒ active perception shines"
   story; **supports** the deeper "visual density, not scale or length" framing.

3. **The OCR-free collapse is category-graded and confirms vision is irreplaceable
   where OCR sees nothing.** rvlm (live, n=8) vs `rlm_ocr` **[from doc]** per
   category — accuracy gap (rvlm − rlm_ocr): engineering_drawing **+60.0**
   (rlm_ocr = **0/10 in all 8 trials**), infographics **+43.8**, science_poster
   **+33.8**, business_report **+27.5**; OCR is **most survivable** on slide
   (+18.8) and science_paper (+17.9) — the text-dense categories. The lone
   exception is **maps (+6.2 only)**, but because rvlm *itself* nearly fails maps
   (6.2%), not because OCR helps. **Supports** the perception-modality claim.

4. **Effort does not buy success — iteration count is negatively correlated with
   accuracy.** rvlm@27B doc-level corr(iterations, accuracy) = **−0.311**
   (n=200 doc-trials). Binned: 0–8 iters → **62.8%**, 8–14 → **43.2%**, 14+ →
   **31.6%**. More REPL turns = the agent is *stuck on a hard doc*, not grinding
   to a win. ReAct spends ~half the iterations (5.4 vs rvlm 10.0 at 9bLM/27bVLM)
   and ~⅓ the wall time (812s vs 1865s/doc) — active perception costs effort, and
   that effort tracks difficulty, not a reliable accuracy return. **Complicates**
   any "just iterate more" reading; the lever is the *quality* of the perception
   loop, not its length.

5. **Harness rank flips with reasoner scale — the REPL is only a lever for capable
   reasoners.** Overall: at 8b-LM ReAct **19.7** > rvlm 13.9 > codeact 12.3 (the
   simplest loop wins for the weakest reasoner); at 9b rvlm 25.3 ≈ codeact 24.2 >
   react 22.7; at the 27b/9b corner rvlm 35.1 / codeact 32.2 ≫ react 18.3.
   **Active perception is not free** — below ~9B the REPL can't be exploited and a
   fixed ReAct loop is as good or better. **Supports** the "not model scale alone"
   nuance: the lever is *REPL × sufficient reasoner*, and it's the harness, not raw
   scale, that converts capability into accuracy.

---

## (b) Computed tables

### Table A1 — Category × harness, matched config (27B-LM / 9B-VLM, v3 corner)
Mean accuracy % across trials. n: rvlm=3, codeact=4, react=8. Same VLM for all
three → the difference is purely harness.

| category | rvlm | codeact | react | **rvlm−react** | codeact−react |
|---|---:|---:|---:|---:|---:|
| business_report | 40.0 | 40.0 | 16.2 | **+23.8** | +23.8 |
| comics | 30.0 | 32.5 | 16.2 | +13.8 | +16.2 |
| engineering_drawing | 50.0 | 42.5 | 20.0 | **+30.0** | +22.5 |
| infographics | 50.0 | 47.5 | 31.2 | +18.8 | +16.2 |
| maps | 20.0 | 5.0 | 1.2 | +18.8 | +3.8 |
| science_paper | 21.1 | 22.5 | 16.2 | **+4.9** | +6.2 |
| science_poster | 30.0 | 35.0 | 15.0 | +15.0 | +20.0 |
| slide | 40.0 | 32.5 | 30.0 | +10.0 | +2.5 |

*Visually dense (eng_drawing, business_report w/ figures, maps, infographics,
poster) gain most from the REPL; text-linear science_paper/slide gain least.*

### Table A2 — Category × harness, 27B-homog (rvlm & codeact live; baselines [from doc])
rvlm/codeact from live dirs (n=8 / n=5). react/raw_vlm/direct_vlm/official **not on
disk** — cite `docs/results.md` overall numbers only (react 25.16, raw_vlm 20.47,
direct_vlm 22.34, official 17.81, rlm_ocr 13.91).

| category | rvlm (n8) | codeact (n5) |
|---|---:|---:|
| business_report | 47.5 ± 12.8 | 40.0 |
| comics | 35.0 ± 5.3 | 30.0 |
| engineering_drawing | 60.0 ± 10.7 | 50.0 |
| infographics | 63.8 ± 7.4 | 58.0 |
| maps | 6.2 ± 7.4 | 10.0 |
| science_paper | 31.2 ± 8.3 | 34.0 |
| science_poster | 43.8 ± 10.6 | 32.0 |
| slide | 48.8 ± 9.9 | 46.0 |

*Per-category trial std is large (5–13pp on 10-Q cells) — single-category claims
are noisy; business_report (±12.8) and maps (±7.4, floor) are the least stable.*

### Table A3 — OCR-free collapse, per category (rvlm vs rlm_ocr)
rvlm = live n=8; rlm_ocr per-category **[from doc]** (avg of reported t1–t3
correct/10; eng_drawing & maps = 0/10 in **all 8** trials per the doc).

| category | rvlm % | rlm_ocr % | **gap (rvlm−ocr)** | OCR survivability |
|---|---:|---:|---:|---|
| engineering_drawing | 60.0 | 0.0 | **+60.0** | total collapse |
| infographics | 63.8 | 20.0 | +43.8 | poor |
| science_poster | 43.8 | 10.0 | +33.8 | poor |
| business_report | 47.5 | 20.0 | +27.5 | moderate |
| comics | 35.0 | 13.3 | +21.7 | moderate |
| slide | 48.8 | 30.0 | +18.8 | **best** |
| science_paper | 31.2 | 13.3 | +17.9 | **best** |
| maps | 6.2 | 0.0 | +6.2 | both fail (rvlm floor) |

### Table A4 — Page-count buckets × harness (per-doc accuracy averaged over trials)
Buckets ≤10 / 11–40 / 40+ pages (justified: the 25 docs split 11/6/8; ≤10 = all
single-page dense docs + small eng_drawings, 40+ = business_reports + long comics).

**v3 corner (27B-LM / 9B-VLM):**
| bucket | ndocs | rvlm | codeact | react | rvlm−react | codeact−react |
|---|---:|---:|---:|---:|---:|---:|
| ≤10 | 11 | 35.7 | 30.8 | 17.4 | **+18.3** | +13.3 |
| 11–40 | 6 | 26.1 | 21.4 | 21.7 | +4.4 | −0.3 |
| 40+ | 8 | 31.6 | 35.9 | 15.9 | +15.8 | +20.0 |

**Mixed 9B-LM / 27B-VLM (n=8 each, strong perception):**
| bucket | ndocs | rvlm | codeact | react | rvlm−react | codeact−react |
|---|---:|---:|---:|---:|---:|---:|
| ≤10 | 11 | 26.7 | 25.9 | 20.7 | +6.0 | +5.3 |
| 11–40 | 6 | 20.3 | 22.5 | 15.8 | +4.4 | +6.7 |
| 40+ | 8 | 24.8 | 23.3 | 25.4 | **−0.6** | −2.1 |

*Page-count distribution of the 25 docs:* ≤10pg = 8 single-page (maps×3,
infographics×2, science_poster×2, eng_drawing_2) + eng_drawing_3/4/1 (3/6/7pg) =
11 docs; 11–40 = slides + science_paper_2/3 + comics_1 = 6; 40+ = science_paper_1
(44), comics_2/3/4 (52/60/69), business_report_3/2/4/1 (89/105/110/181) = 8.

### Table A5 — Effort by harness and page bucket
Iterations from `Trajectory (N iterations)` headers; elapsed from
`documents[].elapsed`.

| config | mean iters | elapsed/doc | iters ≤10 / 11–40 / 40+ |
|---|---:|---:|---|
| rvlm @27B-homog | 13.3 | 2127 s | 14.2 / 12.2 / 12.9 |
| codeact @27B-homog | 13.9 | 3006 s | 13.0 / 14.9 / 14.3 |
| rvlm @9bLM-27bVLM | 10.1 | 1865 s | 9.5 / 9.3 / 11.4 |
| codeact @9bLM-27bVLM | 10.0 | 1582 s | 9.1 / 10.5 / 10.8 |
| react @9bLM-27bVLM | 5.4 | 812 s | 3.6 / 6.6 / 6.9 |

*Active-perception harnesses spend ~2× the iterations and ~2–2.5× the wall time of
ReAct. Iteration count barely rises with page count (rvlm 14.2→12.9), i.e. effort
is roughly flat across doc length — it is **not** scaling up to "find the page" on
long docs.* **Effort↔success: corr(iters, acc) = −0.311** (rvlm@27B, n=200);
0–8 iters → 62.8%, 8–14 → 43.2%, 14+ → 31.6%.

---

## (c) Proposed figures / tables for the report

1. **Figure: "Active perception is a category-specific lever."** Grouped bar chart,
   x = 8 categories sorted by rvlm−ReAct gap, bars = rvlm / codeact / react
   accuracy at the matched 27B-LM/9B-VLM config (Table A1). Annotate the rvlm−ReAct
   delta above each group. Caption: *"At matched perception (9B VLM), the REPL
   harnesses beat ReAct most on visually dense categories (engineering_drawing
   +30, infographics/maps +19) and least on text-linear ones (science_paper +5) —
   the active-perception advantage tracks visual density, not category difficulty."*
   Numbers: Table A1.

2. **Figure: "The OCR-free collapse is graded by visual density."** Horizontal
   lollipop / dumbbell, one row per category, two dots = rvlm% and rlm_ocr%, sorted
   by gap (Table A3). Caption: *"Swapping recursive visual perception for OCR text
   (same LeanRLM scaffold) collapses engineering_drawing to 0/10 in all 8 trials
   and halves infographics/poster; it is most survivable on text-dense slides and
   science papers — vision does work OCR cannot replace, exactly where the page is
   non-textual."* Numbers: Table A3.

3. **Table: "Effort buys difficulty, not accuracy."** Two-panel small table:
   (i) corr(iterations, accuracy) = −0.31 with the 3-bin accuracy ladder
   (62.8 / 43.2 / 31.6%); (ii) iterations & wall-time per harness (Table A5).
   Caption: *"Active-perception harnesses spend ~2× the steps and wall-time of
   ReAct, but more iterations mark harder documents, not a path to the answer
   (negative iteration–accuracy correlation) — the lever is the quality of the
   perception loop, not its length."* Numbers: Tables A5 + the corr/bin block.

   *(Optional 4th, if a scale story is wanted: harness-rank-flip line plot — overall
   accuracy vs reasoner size {8b,9b,27b} for rvlm/codeact/react, showing ReAct wins
   at 8b and falls ~12–18pp behind at 27b. Numbers in the "Overall" block of the
   computed data / harness-axis-summary.)*

---

## (d) Gaps / caveats

- **Missing run dirs (biggest gap).** The canonical 27B-homog baselines
  (`react`, `raw_vlm_multi`, `direct_vlm`, `official`) and the `rlm_ocr` control
  have **no `results.json` on disk** — only overall + sparse per-category numbers in
  `docs/experiments/*-qwen-3_5-27b.md`. So Table A2's right columns and the OCR
  collapse (Table A3) lean on **[from doc]** figures; I could not recompute
  per-doc/per-page slices for them. The clean per-category/per-page harness
  contrast (Tables A1, A4) therefore uses the *cross-model* matched configs
  (27bLM/9bVLM and 9bLM/27bVLM), not 27B-homog. To close: re-run the 5 missing
  27B-homog baselines, or locate the archived `*-cmp-val-t*` dirs.
- **Small n in the v3 corner.** Table A1's rvlm (n=3) and codeact (n=4) columns
  are under-powered; react is n=8. The 8-trial 9bLM/27bVLM rows (Table A4 lower)
  are the most trustworthy. Per-category cells are 10 Q × few trials → wide CIs.
- **rvlm version mismatch.** Live `rvlm-minimal-val` (42.0%) ≠ published `rvlm`
  (39.4%) — same family (prompt-minimized variant), so absolute category numbers in
  Tables A2/A3 are ~+2–3pp vs the paper's canonical rvlm. Deltas are robust;
  absolute values should cite the canonical run.
- **rlm_ocr per-category averaged over t1–t3 only** (the doc tabulates full
  per-category for the first 3 trials; eng_drawing/maps 0/10 is confirmed for all
  8). Treat A3's rlm_ocr column as a 3-trial estimate (the 0/10 categories are
  exact).
- **High trial variance.** Per-category std is 5–13pp on 10-Q cells (Table A2);
  business_report (±12.8) and the maps floor (±7.4) are the least stable. Any
  single-category headline needs the std attached.
- **Page-count ≈ category confound.** Buckets are not independent of category
  (≤10pg = dense single-page; 40+ = business_reports). The U-shaped bucket pattern
  is partly a re-expression of the category pattern — do not read it as a pure
  length effect.
- **codeact_chat (the RL-target transcript, 39.53% [from doc])** has **no run dir
  on disk** under any `*chat*` name — could not slice it. If it's the fine-tuning
  target, its per-category profile is worth recovering.
