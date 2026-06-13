# Pillar A — Active-perception trajectories (mined evidence)

Qualitative evidence that a **code REPL converting reasoning into targeted
perception** is what lifts DocVQA accuracy. Every excerpt below is **verbatim**
(trimmed, with `[...]` marking elisions) from agent trajectories in
`~/repos/docvqa/output/runs/<run>/tasks/<doc_id>/summary.md`. The strong
27B active-perception runs sampled: `rvlm-minimal-val-t1..t4` (RLM harness,
39.4% val) and `codeact-3_5-27b-val-t1..t3` (CodeAct chat MDP, 39.5% val). The
contrast harness is ReAct (`react-cmp-val-*` 27B-homog 25.2%; `react-27b-llm-9b-vlm-*`
v3). Provenance is given per example.

The agent's only perception tool is `batch_look(requests)`, where each request
is an `(image, query)` tuple sent to the frozen 27B VLM; its only terminal
action is `SUBMIT(answer=...)`. Images are PIL pages the agent can `.crop()`,
slice by `.size`, and index — that PIL surface *is* the mechanism.

---

## 1. Canonical active-perception loop — **★ RECOMMENDED MAIN FIGURE**

**Provenance:** `rvlm-minimal-val-t1` / `business_report_1` /
`business_report_1_q1` — **[CORRECT]**, 16 iterations.
**Question:** *"In Fiscal 2025, by how many dollars does NVIDIA's TSR value
exceed the Nasdaq-100 Index TSR value? Do not provide units."* GT `2048.88`,
prediction `2048.88`. Document: a **181-page** NVIDIA annual review.

This single question walks the full *survey → locate → extract → verify →
submit* loop end to end. (Recommended as the centerpiece figure; for a more
compact alternative see example 3, the science-poster contrast.)

**Survey** (iter 2) — batch the document in one VLM sweep:

```python
# Survey multiple pages to find TSR information
survey_pages = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40]
results = batch_look([(pages[i], f'Summarize page {i}: What financial metrics, tables, or sections are present? Look for TSR, Total Shareholder Return, or performance comparisons.') for i in survey_pages])
for i, result in zip(survey_pages, results):
    print(f'Page {i}: {result}')
```

**Locate via a table-of-contents pointer** (iter 7 finds the TOC; iter 8
reasoning): *"Page 15 (Table of Contents) indicated that 'Compensation
Discussion and Analysis' starts on page 44 [...] I need to look at pages around
44-65."* The agent then narrows to pages 71–85 and the table surfaces (iter 11
output): `Page 76: {'graph_title': 'NEO CAP versus TSR', [...] 'fiscal_2025_tsr':
{'nvidia_tsr': '$2,287.07', 'nasdaq100_index_tsr': '$238.19'}}`.

**Verify by re-reading, distrusting precise numbers** (iter 14) —
reasoning: *"the VLM can be unreliable for precise numbers. Let me crop the
relevant region and re-read":*

```python
crop_box = (100, 300, 750, 800)  # left, top, right, bottom
cropped = pages[76].crop(crop_box)
result = batch_look([(cropped, "What are the exact dollar values shown for NVIDIA TSR and Nasdaq-100 Index TSR for Fiscal 2025? Provide the numbers exactly as shown.")])
print(f"Cropped verification: {result[0]}")
```

The crop **disagreed** with the full-page read (`$978.42`/`$190.57` vs
`$2,287.07`/`$238.19`), so the agent broke the tie by re-reading the top and
bottom halves separately (iter 15) before committing:

```python
top_crop = pages[76].crop((0, 0, 850, 550))
result_top = batch_look([(top_crop, "What TSR values for Fiscal 2025 are shown in this region? Look for NVIDIA and Nasdaq-100 Index values.")])
bottom_crop = pages[76].crop((0, 550, 850, 1100))
result_bottom = batch_look([(bottom_crop, "What TSR values for Fiscal 2025 are shown in this region? Look for NVIDIA and Nasdaq-100 Index values.")])
```

**Submit** (iter 16) — the arithmetic the VLM cannot do is done in Python:

```python
nvidia_tsr = 2287.07
nasdaq_tsr = 238.19
difference = nvidia_tsr - nasdaq_tsr
[...]
```
`FINAL: {'answer': '2048.88'}`

**What it demonstrates:** every component of the thesis in one trace —
programmatic survey of a 181-page doc, locating via a TOC pointer, focused
extraction, a *self-distrusting* crop-verify that catches a VLM
misread, and Python arithmetic at submit. The crop disagreement is a bonus: it
shows the verify step is load-bearing, not ceremonial.

---

## 2. Behavioral taxonomy (one verbatim snippet each)

Six distinct code-driven perception moves observed across the 175 sampled
trajectories (25 docs × 7 runs). Frequencies are approximate counts of files
exhibiting the move.

### (a) Multi-page survey via `batch_look` — **very common (~139/175)**

The standard opening move: one batched VLM call over many page indices.
`codeact-3_5-27b-val-t2` / `business_report_3` / `business_report_3_q2`
**[CORRECT]**, iter 2 (89-page report):

```python
results = batch_look([(pages[i], 'Summarize this page and identify any key topics, especially related to unionization, labor rates, or statistics') for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 88]])
print(results)
```

### (b) Cropping / zooming to a sub-region — **ubiquitous (~140/175)**

The dominant localize-then-read move; crop box is `(left, top, right, bottom)`.
`rvlm-minimal-val-t1` / `science_poster_1` / `science_poster_1_q1`
**[WRONG]**, iter — cropping the right half of a 4000×2000 poster:

```python
right_side = pages[0].crop((2000, 0, 4000, 2000))
result = batch_look([(right_side, "Find any table or section that shows rFID metrics, specifically rFID-50k, comparing Baseline and TexTok methods for ImageNet 512x512 with 128 tokens. [...]")])
```

### (c) Coordinate arithmetic / tiling — **common (~91 do dimension-fraction math, ~69 build strips)**

Derives crop boxes from page dimensions (fractions, per-row bands, strips).
**No `.resize()` / `.paste()` / `Image.new()` was used anywhere** across all 7
runs — "compositing" is realized only as *computed sequences of crops*, never as
upscaling or stitching. `rvlm-minimal-val-t1` / `infographics_2` /
`infographics_2_q4` **[CORRECT]** — indexes a per-row band to a city's row
(*"With 10 cities [...] each section is about 210 pixels. Osaka is #6"*):

```python
page_height = pages[0].size[1]
page_width = pages[0].size[0]
section_height = page_height / 10
# Osaka is #6, so roughly at position 5 to 6 sections down
osaka_top = int(5 * section_height)
osaka_bottom = int(6 * section_height)
osaka_crop = pages[0].crop((0, osaka_top, page_width, osaka_bottom))
result = batch_look([(osaka_crop, "What city is shown and what object is depicted with it? Describe the object.")])
```

A tiling variant (`rvlm-minimal-val-t1` / `engineering_drawing_1` /
`engineering_drawing_1_q2` **[WRONG]**) loops a large page into strips:

```python
page_width, page_height = pages[0].size
strip_height = page_height // 4
for i in range(4):
    y_start = i * strip_height
    y_end = (i + 1) * strip_height if i < 3 else page_height
    strip_crop = pages[0].crop((0, y_start, page_width, y_end))
    strip_result = batch_look([(strip_crop, f"This is strip {i+1} of the parts list. Read all rows. [...]")])
```

### (d) Programmatic search / iteration over pages — **common (~110/175)**

A loop/comprehension that sweeps a page range and filters hits in Python.
`rvlm-minimal-val-t1` / `business_report_3` / `business_report_3_q1`
**[CORRECT]** — sweep 49 pages, filter by keyword:

```python
sample_pages = list(range(40, 89))
results = batch_look([(pages[i], "Look for a pictogram of a stroller, baby carriage, pram, or similar icon. [...]") for i in sample_pages])
for i, result in zip(sample_pages, results):
    if "stroller" in result.lower() or "baby" in result.lower() or "carriage" in result.lower() or "pram" in result.lower():
        print(f"Page {i}: {result}")
```

### (e) Cross-page evidence chaining — **moderately common**

Read a value off one page, another off a different page, then combine. Pure
two-different-pages-then-arithmetic is the rarer subset (most multi-value Qs pull
both numbers from one table). `rvlm-minimal-val-t1` / `slide_1` /
`slide_1_q2` **[WRONG]** — cleanest pure cross-page structure (SKJ% from page 7
vs page 19):

```python
# Verify Page 7 - Releases percentage for SKJ
result7_verify = batch_look([(pages[7], "Read all percentage values from the pie chart(s) on this slide. List each species and its percentage.")])
# Verify Page 19 - Recoveries percentage for SKJ
result19_verify = batch_look([(pages[19], "Read all percentage values from the pie chart on this slide. List each species and its percentage.")])
[...]
difference = recoveries_pct - releases_pct
```

A pointer-following flavor also appears (`codeact-3_5-27b-val-t2` /
`business_report_3_q2` **[CORRECT]**): a GRI index on page 80 returned
`'2-30 Collective agreements (49)'`, and the agent followed that `(49)` pointer
to page 48/49 to extract the unionization rates.

### (f) Self-verification before SUBMIT — **common (~48/175 with verify-named re-read + SUBMIT)**

Re-read the same evidence with a second `batch_look` (tighter crop or
confirm-phrased query) immediately before committing; near-ritual on numeric
answers. `codeact-3_5-27b-val-t2` / `business_report_3` / `business_report_3_q2`
**[CORRECT]**, final iter:

```python
verify = batch_look([(pages[48], 'What are the exact unionization rates for 2023 and 2024? Provide the percentage values.')])
print("Verification:", verify)
[...]
rate_2023 = 11
rate_2024 = 15
difference = rate_2024 - rate_2023
SUBMIT(answer=f"{difference}%")
```

A *conditional*-submit idiom (gate `SUBMIT` on the verification string) also
appears — `rvlm-minimal-val-t1` / `science_paper_1`:

```python
verify = batch_look([(pages[13], 'In Section 4.1, what is the exact data amount [...] Confirm the number in Millions.')])
if '20M' in verify[0] or '20 million' in verify[0].lower():
    SUBMIT(answer='20')
else:
    print('Need more verification')
```

---

## 3. Contrast pair: RLM/CodeAct (crop) vs ReAct (whole-page) — **★ strong "REPL converts reasoning into perception" evidence**

**Same question, both harnesses, 27B reasoner+VLM.**
`science_poster_1_q1`: *"What is the percentage score improvement from Baseline
to TexTok in rFID-50k for the ImageNet 512x512 case with 128 tokens?"* GT
`30.2%`. The answer requires reading two cells (`1.49`, `1.04`) from a small
table embedded in a **4000×2000** poster and computing `((1.49-1.04)/1.49)*100`.

**RLM (`rvlm-minimal-val-t4`) — [CORRECT], 6 iters.** Survey locates two tables;
the agent crops the band holding the right table, then reads each operand:

```python
# iter 4 — crop the band, extract the TexTok value
result = batch_look([(pages[0].crop((0, 800, 4000, 1400)), "Describe the 'Tokenization Improvements Translate to Generation' section in detail. [...] find rFID-50k metric values for ImageNet 512x512 with 128 tokens")])
# -> "the rFID-50k metric value for TexTok-128 (w/ text) is 1.04."

# iter 5 — same crop, extract the Baseline operand
result = batch_look([(pages[0].crop((0, 800, 4000, 1400)), "In the 'Tokenization Improvements Translate to Generation' table, what is the rFID-50k value for Baseline on ImageNet 512x512 with 128 tokens?")])
# -> "1.49"

# iter 6 — Python does the percentage
baseline = 1.49; textok = 1.04
improvement = ((baseline - textok) / baseline) * 100
SUBMIT(answer=f"{round(improvement, 2)}%")   # FINAL: {'answer': '30.2%'}
```

**ReAct (`react-27b-llm-9b-vlm-val-t1`) — [WRONG], 2 iters.** Same question, but
the harness has **no REPL**: its only actuators are `look(page, query)` /
`look_many(pages, query)` at whole-page granularity. It reads the whole
4000×2000 poster, cannot zoom to the cell, terminates in 2 iterations, and
predicts **`0.48%`** (vs GT `30.2%`). The other ReAct trials on this question
land at `14.2%` / wrong-cell answers — never the right value.

The mechanism difference is stated in the ReAct solver's own docstring
(`~/repos/docvqa/src/docvqa/solvers/react_baseline_solver.py`, verbatim):

> *"What's intentionally lost vs `rvlm`: No `PIL.crop` on retrieved page images
> — ReAct can't construct PIL ops, so fine-detail extraction degrades. No
> arithmetic / list comprehensions / counting in Python — superlative and
> compound-answer questions must be assembled by the LM itself from the
> trajectory."*

**What it demonstrates:** on a fine-detail, two-operand-plus-arithmetic question
the REPL agent crops to the table band, reads each cell, and computes the answer
in Python; ReAct is pinned at whole-page acuity and fails. This is the
"REPL converts reasoning into perception" claim made concrete on a single
shared question. (It is corroborated at the aggregate level by the v2→v3
factorial: swapping to a stronger 27B reasoner while holding the VLM at 9B
*gains* +10.3pp for RLM and +6.2pp for CodeAct but *regresses* −3.05pp for ReAct
— ReAct alone is perception-bound because it has no actuator to direct
finer-grained perception. See `docs/experiments/harness-axis-summary.md`
Finding 3.)

---

## 4. Failure / over-perception example — runaway map-tracing

**Provenance:** `rvlm-minimal-val-t1` / `maps_2` / `maps_2_q4` — **[WRONG]**,
**25 iterations (hits the cap)**; the `maps_2` doc took **4501s** wall time.
**Question:** *"From Worcester I take highway 140 towards Taunton, I take the
second macadam & gravel road that I encounter, continuing on that road, what
town do I reach?"* GT `Woonsocket`, prediction `Milford`.

This is a multi-hop *spatial-tracing* question (follow a route, count dashed-road
intersections in order, read the destination label). It is not localizable to a
single region, so the agent thrashes: from iter ~20 onward it re-crops nearly the
same Worcester / highway-140 region over and over, re-asking the same query with
slightly perturbed boxes, never converging. The final (capped) iteration fires
three near-duplicate crops at once:

```python
final_check = pages[0].crop((1250, 950, 2650, 2050))
result = batch_look([(final_check, "[...] Count ALL dashed (macadam & gravel) roads that intersect highway 140 in order from Worcester. List them as: 1st dashed road leads to [TOWN NAME], 2nd dashed road leads to [TOWN NAME] [...]")])
verification_crop = pages[0].crop((1400, 1050, 2500, 1900))
result2 = batch_look([(verification_crop, "Find the SECOND dashed (macadam & gravel) road that intersects highway 140 [...] What is the exact name of the town?")])
medway_area = pages[0].crop((1600, 1300, 2400, 1800))
result3 = batch_look([(medway_area, "Is there a dashed road leading to Medway that intersects highway 140? [...]")])
```

The three reads come back **mutually contradictory** in the same turn:
`Final check: 1st [...] Upton, 2nd [...] Milford [...]` · `Verification: W.
Woodstock` · `Medway check: The provided map does not show Medway, Worcester, or
highway 140.` The agent's own iter-25 reasoning concedes: *"I've received many
inconsistent answers about the second macadam & gravel road."*

**What it demonstrates (limitations):** code-driven perception is a *localization*
mechanism — it shines when evidence sits in a croppable region. On dense maps
requiring multi-hop spatial reasoning, the same loop degenerates into
**over-perception**: repeated re-crops of an un-localizable region yield noisy,
contradictory VLM reads, the agent cannot adjudicate them, and it burns the full
iteration budget without converging. `maps` is consistently the worst category,
and this churn (median iters / %@cap on `maps_2`) is the visible signature.

---

## Frequency note (across the 175 sampled trajectories)

- **Ubiquitous / very common:** cropping to a sub-region (b, ~140) and
  multi-page survey (a, ~139) — these are the two reflexive moves on essentially
  every non-trivial doc.
- **Common:** programmatic page-sweep/filter (d, ~110), coordinate-fraction
  arithmetic and strip tiling (c, ~91 / ~69), self-verification re-reads before
  SUBMIT (f, ~48 with an explicit verify-named call; verify/confirm *reasoning*
  is near-universal on numeric answers).
- **Moderately common / rarer:** true cross-page evidence chaining (e) — value
  on page X combined with value on page Y, or pointer-following across pages —
  shows up mainly on multi-part financial / multi-slide questions.
- **Absent:** image *compositing* in the literal sense (`.resize`, `.paste`,
  `Image.new`) — never observed. "Compositing" in this corpus = computed
  *sequences* of crops, not stitched images.

One caveat for the training-data angle: the arithmetic crops in (c) are genuine
but **noisy** — the cleanest correct example (`infographics_2_q4`) still produced
a wrong intermediate read from its computed band and recovered only via a
follow-up verify. Success often comes from the *verify* step after a computed
crop, not the crop alone — consistent with the failure mode in §4.
