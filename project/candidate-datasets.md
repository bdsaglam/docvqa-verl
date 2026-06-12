# Candidate Datasets for Generality Experiments

Evaluation of every dataset in `tmp/datasets.md` as a potential
**additional generality benchmark** for the paper. One section per
dataset with a fit verdict and reasoning.

Grounded in the **D-006 framing** (`decisions.md`): the headline method
is OCR-free **recursive visual perception** (a code-capable LLM in a
REPL whose load-bearing tool is a `batch_look` VLM sub-call). OCR/search
is an extension, not a contribution. So a dataset's relevance now hinges
on the **visual context-budget** mechanism, *not* on OCR retrieval.

## Fit rubric

A dataset is a good generality target only if it passes all of:

1. **QA with measurable accuracy.** Rules out captioning and pretraining
   corpora — no per-question answer to score, no lift to measure.
2. **Perception-budget-bound.** The benchmark must stress a VLM's *finite
   visual context*: fine detail at high resolution, visually dense
   content, and/or many pages. This is the exact mechanism the paper
   claims (rationed recursive VLM perception). If a raw VLM already
   nails it in one forward pass, there is no lift to demonstrate.
3. **Non-circular (no document *reuse*).** The test is literal document/
   image overlap with DocVQA-2026's 8 categories (`business_report,
   comics, engineering_drawing, infographics, maps, science_paper,
   science_poster, slide`, from `VLR-CVC` = CVC/UAB), **not** shared
   authorship. CVC/UAB is central to the entire DocVQA field — MP-DocVQA
   and MMLongBench-Doc share its lineage yet are accepted because their
   documents are separate. Reuse risk is highest for a benchmark that is
   the *canonical source* of a DocVQA-2026 category (infographics →
   InfographicVQA; slide → SlideVQA). Default to verify-then-use, not
   exclude-on-lineage.
4. **Axis coverage (bonus).** Extends one of D-006's three predictions:
   model-size, **document-length** (the thin axis — only 3 benchmarks
   today), or simply adds a distinct document type for the perception
   mechanism.

**Verdict scale:** ✅ strong candidate · ⚠️ conditional/weak ·
❌ exclude — circular · ❌ off-target.

## Summary

| Dataset | QA? | Perception-bound? | Circular? | Verdict |
|---|---|---|---|---|
| InfographicVQA | ✓ | ✓ | **yes** (infographics) | ❌ circular |
| DocVQA (SP) | ✓ | weak | lineage | ❌ lean-exclude |
| ChartQA | ✓ | ✓ | skill overlap | ❌ exclude |
| TextVQA | ✓ | partial | no | ⚠️ conditional |
| MMMU | ✓ | weak | no | ⚠️ weak |
| MRAG-Bench | ✓ | no (retrieval) | no | ❌ off-target |
| VQA v2 | ✓ | no | no | ❌ off-target |
| GQA | ✓ | no | no | ❌ off-target |
| VCR | ✓ | no | no | ❌ off-target |
| NLVR2 | ✓ (NLI) | no | no | ❌ off-target |
| CLEVR | ✓ | no (synthetic) | no | ❌ off-target |
| Visual Genome | partial | no | no | ❌ off-target |
| MindBench | no (parsing) | n/a | no | ❌ off-target |
| INQUIRE | no (retrieval) | n/a | no | ❌ off-target |
| Flickr30K | no | n/a | n/a | ❌ off-target |
| MS COCO Captions | no | n/a | n/a | ❌ off-target |
| Conceptual Captions | no | n/a | n/a | ❌ off-target |
| Conceptual 12M | no | n/a | n/a | ❌ off-target |
| RedCaps | no | n/a | n/a | ❌ off-target |
| WIT | no | n/a | n/a | ❌ off-target |
| YFCC100M | no | n/a | n/a | ❌ off-target |
| COYO-700M | no | n/a | n/a | ❌ off-target |
| LAION-5B | no | n/a | n/a | ❌ off-target |
| Re-LAION-5B | no | n/a | n/a | ❌ off-target |
| DataComp-1B | no | n/a | n/a | ❌ off-target |
| MMC4 | no | n/a | n/a | ❌ off-target |

**Bottom line:** the non-circularity constraint removes every clean
document-VQA fit, because DocVQA-2026 is deliberately comprehensive
(8 doc types). Nothing in this list is a strong candidate. The best
generality evidence comes from the **related-works benchmark pool**
(next section), not from `tmp/datasets.md`.

---

## Benchmarks from the related-works library (the relevant pool)

`docs/paper/related-works/` indexes the document-VQA benchmarks the
field actually uses — a *better* pool than `tmp/datasets.md` for
generality experiments. Two reasons: several are **multi-page** (the
document-length axis, D-006 prediction 2, where we currently have only
3 benchmarks), and several are what our **direct competitors report
on**, which enables head-to-head positioning rather than standalone
lift.

Already handled: DocVQA-SP (lean-exclude, below), InfographicVQA
(circular, below), ChartQA (skill overlap, below), **MP-DocVQA +
MMLongBench-Doc** (in use — the two length-axis legs). New candidates
surfaced here:

| Benchmark | arXiv | Pages | Reuse risk | Competitor reports on it | Verdict |
|---|---|---|---|---|---|
| DUDE | 2305.08455 | multi (→70+) | low | — | ✅ strong |
| MADQA | 2603.12180 | doc collections | low (fresh docs) | MADQA (our planned baseline) | ✅ strong; regime favors OCR ext. |
| SlideVQA | 2301.04883 | multi (slides) | **high** (slide cat.) | SlideAgent | ⚠️ verify reuse |
| VisualMRC | 2101.11272 | single | low | — | ⚠️ conditional |
| ST-VQA | 1905.13648 | single (scene) | low | — | ⚠️ redundant w/ TextVQA |

### DUDE (2305.08455) — ✅ strong candidate

Document Understanding Dataset and Evaluation (Van Landeghem et al.,
ICCV 2023). Multi-domain, multi-industry documents spanning single page
to 70+ pages, with extractive + abstractive + list + unanswerable
answers. **The best generality benchmark available to us:** it hits the
document-length axis *and* the perception mechanism (diverse dense
layouts), it's a recognized standard, and its document collection is
distinct from DocVQA-2026 (shares CVC organizers, but its own docs — low
reuse risk; no single DocVQA-2026 category maps onto it). Caveat:
abstractive + list + unanswerable answers mean it needs the profile
`score_fn`, not plain ANLS (per the cross-benchmark methodology rule in
`CLAUDE.md`). Recommend adding it.

### MADQA (2603.12180) — ✅ strong + strategic, with a framing tension

Borchmann et al., *Strategic Navigation or Stochastic Search?*
**Verified from the PDF (2026-05-27):** 2,250 human-authored questions
over 800 heterogeneous PDFs, framed as document-**collection** QA
(corpus retrieval + cross-page/cross-doc multi-hop subsets). Metric is
LLM-judged **Accuracy** plus a novel **Kuiper effort-calibration**
statistic — not ANLS. Best system *Gemini 3 Pro BM25 MLLM Agent* =
82.2%; ~18% oracle gap; the paper's thesis is that **retrieval, not
reasoning, is the bottleneck**. Reuse risk **explicitly low** — the
paper advertises *fresh documents not recycled from existing
benchmarks*. Collection-scale, so strong on the context-budget
mechanism.

**Two tensions to position against, not gloss (per CLAUDE.md "surface
blind spots"):**

1. **Regime favors our OCR *extension*, not our OCR-free core.** MADQA is
   collection-scale retrieval — exactly where our own data says OCR/search
   helps (MMLongBench-Doc, MP-DocVQA long bucket) and where a pure
   visual-perception core would look retrieval-bound. Leading here with
   the OCR-free method risks a weak number; we'd lead with the extension.
2. **They already tested unconstrained RLM and found it cost-catastrophic.**
   §5: "constrained agency … avoids the catastrophic effort overhead of
   RLMs." They run RLM citing the *same* Zhang et al. 2025 paper we
   instantiate — e.g. Claude 4.5 Sonnet RLM burned 270M input tokens /
   ~$850 and still lost to its BM25-agent counterpart. This **supports**
   our "focused/constrained instantiation" framing (D-005) but also
   **pre-empts any "constraining RLM helps" claim** — that's their result.
   Our defensible delta is the *visual* sub-call specialization + the
   perception-budget hypothesis on benchmarks where visual perception
   (not collection retrieval) is the bottleneck.

Verdict: include as benchmark + baseline (D-005 stands), but lead with
the OCR extension and frame explicitly against their RLM-efficiency
result. Read the full PDF before drafting the positioning paragraph.

### SlideVQA (2301.04883) — ⚠️ candidate, verify slide-document reuse

Tanaka et al. (AAAI 2023). Multi-page slide-deck QA with multi-hop
reasoning — good length + perception fit, and SlideAgent (a related-works
competitor) reports on slide tasks, giving positioning value. **But**
DocVQA-2026 has a `slide` category and SlideVQA is *the* canonical
slide-QA dataset — highest document-reuse risk after InfographicVQA. Use
only after confirming DocVQA-2026's slide documents aren't drawn from
SlideVQA.

### VisualMRC (2101.11272) — ⚠️ conditional

Tanaka et al. (AAAI 2021). Single web-page screenshots, abstractive
machine reading comprehension. Distinct domain (web pages), low reuse
risk. But single-image → no document-length axis, and web pages are only
moderately perception-dense → weak mechanism stress. A domain-breadth
point at best, not headline.

### ST-VQA (1905.13648) — ⚠️ conditional, redundant with TextVQA

Biten et al. (ICCV 2019); the original ANLS source. Scene-text VQA on
natural images — the same out-of-domain "reads fine scene text" probe
role as TextVQA. CVC-authored but natural-scene domain (low reuse risk).
Pick TextVQA *or* ST-VQA, not both.

---

## Document-VQA benchmarks (the real candidates)

### InfographicVQA (2021) — ❌ exclude, circular

Single high-resolution infographic images; dense layout + text +
graphics. On the rubric this is a *textbook* perception-budget benchmark
(fine labels, dense charts) — it would fit the mechanism well. But
DocVQA-2026 has a dedicated `infographics` category, and both come from
the **same group (CVC/UAB, the DocVQA authors)**. Using it tests
generality on data almost certainly drawn from the same pool as the
headline benchmark. Circular. *Action: confirm whether DocVQA-2026's
infographic images are literally reused or merely same-domain before
fully discarding — but default is exclude.*

### DocVQA / SP-DocVQA (2021) — ❌ lean-exclude

The original single-page document VQA benchmark (scanned industry
documents). Two problems: (a) it's the **direct predecessor** of
DocVQA-2026 from the same lab — reviewers read it as the same benchmark
family; and (b) it's **single-page, moderate-resolution scanned text**,
so a raw VLM is only weakly perception-bound on it. Weak mechanism fit
*and* lineage overlap. Low standalone value.

### ChartQA (2022) — ❌ exclude, skill overlap

Chart images requiring value reading + reasoning — genuinely
perception-bound (small tick labels, dense plots), and a *different*
lab from CVC. But chart-reading is the explicit task in DocVQA-2026's
`science_poster` (the sample val question is literally a chart-%
comparison) and recurs in `science_paper`/`infographics`. It wouldn't
demonstrate anything beyond content the headline benchmark already
covers. Exclude on skill overlap.

### TextVQA (2019) — ⚠️ conditional

Scene text in natural photographs; reading small/oblique text. Genuinely
**disjoint domain** from DocVQA-2026 (no natural-scene category), so
non-circular, and it does stress visual perception (reading fine text
in cluttered images) — partial mechanism fit. Caveats: **single image,
no document length axis**, and it's a recognition-heavy task more than a
multi-page-budget task. Best role: an optional *out-of-domain* probe to
show the recursive-perception mechanism isn't document-specific. Keep on
the shortlist; not load-bearing.

### MMMU (2023) — ⚠️ weak

College-level multimodal reasoning (multiple-choice) across disciplines,
including charts/tables/diagrams but also chemistry structures, medical
images, music notation. Mostly disjoint from DocVQA-2026, so
non-circular. But it's a **reasoning** benchmark; visual content is
often a single moderate-resolution panel, so the perception-budget
mechanism is only lightly stressed. Defensible as a "does the scaffold
help hard multimodal reasoning generally" point, but it tests reasoning
breadth, not the paper's mechanism. Weak fit.

### MRAG-Bench (2024) — ❌ off-target

Multimodal RAG benchmark: 1,353 MCQs answered by retrieving from an
**external image corpus** (16,130 images). It's multimodal QA, but the
task is *cross-image retrieval*, not perception-budget allocation within
a document. Different mechanism. Off-target.

---

## Natural-image VQA / reasoning (wrong mechanism)

These are QA benchmarks but on **single natural images** where a raw VLM
is not perception-budget-bound — the bottleneck is recognition or
commonsense reasoning, not rationing visual context across dense/long
documents. The method's machinery (recursive crop/zoom over many dense
pages) doesn't activate, so there's no lift to show.

### VQA v2.0 (2017) — ❌ off-target
Open-ended VQA on COCO images. Recognition + commonsense, single image.

### GQA (2019) — ❌ off-target
Compositional VQA over scene graphs of natural images. Reasoning, not
perception budget.

### VCR (2019) — ❌ off-target
Commonsense reasoning + rationale on movie stills. Single image, reasoning.

### NLVR2 (2019) — ❌ off-target
Visual NLI over image pairs (true/false), not document QA.

### CLEVR (2017) — ❌ off-target
Synthetic compositional reasoning. No real-document perception at all.

### Visual Genome (2017) — ❌ off-target
Primarily a scene-graph / region-annotation resource. Region-QA exists
but the dataset is built for grounding, not perception-budget document QA.

---

## Document parsing / retrieval (not QA-accuracy)

### MindBench (2024) — ❌ off-target
Mind-map / structured-document *parsing* — output is structured
representations, not scored short answers. Document-domain but wrong task
shape; no clean accuracy lift to report.

### INQUIRE (2024) — ❌ off-target
Natural-world image **retrieval** benchmark (250 queries). Retrieval,
not document QA.

---

## Captioning & pretraining corpora (no QA target)

None of these have per-question answers to score, so there is no lift to
measure — they fail rubric criterion 1 outright. Listed for completeness:

- **Flickr30K (2014)** — captioning. ❌
- **YFCC100M (2014)** — multimedia pretraining corpus. ❌
- **MS COCO Captions (2014)** — captioning. ❌
- **Conceptual Captions (2018)** — caption-pretraining pairs. ❌
- **WIT (2021)** — Wikipedia image-text pretraining. ❌
- **Conceptual 12M (2021)** — web captioning. ❌
- **RedCaps (2021)** — user-generated captions. ❌
- **COYO-700M (2022)** — URL-text pretraining. ❌
- **LAION-5B (2022)** — CLIP-filtered pretraining. ❌
- **DataComp-1B (2023)** — curated pretraining. ❌
- **MMC4 (2023)** — interleaved multimodal pretraining docs. ❌
- **Re-LAION-5B (2024)** — safety-updated pretraining corpus. ❌

---

## Recommendation

1. **Drop** InfographicVQA, DocVQA(SP), and ChartQA from consideration —
   circular or skill-overlapping with DocVQA-2026.
2. **Keep TextVQA** as an optional out-of-domain perception probe (shows
   the mechanism generalizes beyond curated documents). Not headline.
3. **MMMU** only if a reviewer wants breadth; weak mechanism fit.
4. **Everything else is off-target.**

**The generality story must lean on off-list benchmarks.** This list
contains no long-document benchmark, which is exactly where D-006
prediction 2 (document-length axis) lives. The strongest non-circular
evidence already in flight:

- **MMLongBench-Doc** — long-doc, the context-budget leg (+16.84pp judge,
  Qwen 27B). Already run (`docs/experiments/mmlongbench-doc-qwen27b.md`).
- **MP-DocVQA** — multi-page, with the per-length-bucket breakdown
  (+13.68pp in the 11–20pp bucket). Already run
  (`docs/experiments/mp-docvqa-qwen27b.md`).

Both are distinct datasets from DocVQA-2026 (non-circular), and both
exercise the document-length axis a single-image benchmark cannot.

**Ranked plan for additional benchmarks** (drawn from the related-works
pool, not `tmp/datasets.md`):

1. **MP-DocVQA + MMLongBench-Doc** — in use, the two length-axis legs.
2. **DUDE** — best new add: multi-page, diverse, standard, low reuse risk.
3. **MADQA** — strategically strongest (competitor benchmark + planned
   baseline), gated on arXiv verification + a read.
4. **SlideVQA** — usable only after confirming no slide-document reuse.
5. **TextVQA** *or* **ST-VQA** — one optional out-of-domain perception
   probe; not headline.

`project/datasets.md` (the generic VLM list, formerly `tmp/datasets.md`)
contributes nothing beyond the optional TextVQA probe.

---

# Discovery pass — recent (2023–2026) document & perception benchmarks

> Added 2026-06-10. The two pools above (`project/datasets.md` + the
> related-works library) predate the 2024–2026 wave of **long-document,
> multimodal-RAG, and high-resolution perception** benchmarks. This pass
> hunts that wave specifically — the niche closest to our mechanism and to
> the under-covered **document-length axis** (D-006 prediction 2). Same
> rubric applies. Every arXiv ID below was resolved against
> `arxiv.org/abs/{id}` (2026-06-10); IDs that could **not** be verified are
> flagged inline — do not cite those without a manual check.

## New-candidate summary

| Benchmark | arXiv | Pages | Metric | Mechanism fit | Reuse risk | Verdict |
|---|---|---|---|---|---|---|
| **LongDocURL** | 2412.18424 | multi (avg 85.6) | answer-match (GPT-assisted) | long-doc ✓✓ | low (Alibaba/CASIA) | ✅ strong (best new add) |
| TAT-DQA | 2207.11871 | single | EM + numeracy-F1 | dense table ✓ | **business_report overlap** | ⚠️ conditional |
| VisualWebBench | 2404.05955 | single | accuracy | web screenshot ✓ | low | ⚠️ domain-breadth |
| MMDocBench | 2410.21311 | single | acc + grounding-IoU | fine-grained ✓ | domain overlap (papers/info/fin) | ⚠️ weak |
| MMSci | 2407.04903 | single | MC acc | hard sci imagery ✓✓ | moderate (Nature, non-arXiv) | ⚠️ best sci-figure option |
| TableVQA-Bench | 2404.19205 | single | accuracy | dense table ✓ | low-mod (table skill) | ⚠️ skill overlap |
| ChartMuseum | 2505.13444 | single | accuracy | visual chart reasoning ✓ | low (web) | ⚠️ chart skill overlap |
| V\*Bench | 2312.14135 | single (natural) | accuracy | **high-res visual search ✓✓** | none (non-doc) | ⚠️ mechanism probe, tiny |
| M3DocVQA / M3DocRAG | 2411.04952 | collection (40k pg) | accuracy | retrieval-regime (ext.) | low (UNC) | ⚠️ extension, not core |
| ViDoSeek / ViDoRAG | 2502.18017 | collection (~6k img) | unique-answer acc | retrieval-regime (ext.) | low (Alibaba) | ⚠️ extension, best agent fit |
| VisDoM / VisDoMBench | 2412.10704 | multi-doc | acc + LLM-judge | retrieval-regime (ext.) | low (Adobe/UMD) | ⚠️ extension, judge metric |
| M-LongDoc | 2411.06176 | multi (hundreds) | **LLM-judge** | long-doc ✓ | low (DAMO) | ⚠️ judge metric |
| DocBench | 2407.10701 | multi | extractive + judge | long-doc ✓ | low | ⚠️ small + judge |
| DocHaystack / InfoHaystack | 2411.16740 | collection (≤1k docs) | accuracy | retrieval needle | **built from DocVQA + InfographicVQA** | ❌ reuse |
| OpenDocVQA / MHDocVQA | 2504.09795 | collection | accuracy | retrieval-regime | **bundles DocVQA/InfoVQA/DUDE** | ❌ partial reuse |
| VisRAG | 2410.10594 | collection | retrieval + acc | retrieval mechanism | reuses DocVQA/InfoVQA | ❌ off-target + reuse |
| CharXiv | 2406.18521 | single | GPT-judge | chart reading | **arXiv → science_paper** | ❌ reuse + skill |
| SPIQA | 2407.09413 | multi-figure | L3Score (judge) | sci figure/table | **arXiv → science_paper** | ❌ reuse + skill |
| SciFIBench | 2405.08807 | single | MC acc | fig↔caption match | **arXiv → science_paper** | ❌ reuse + skill |
| ArXivQA | 2403.00231 | single | MC acc | arXiv figure (synth QA) | **arXiv → science_paper** | ❌ reuse + train-set |
| ChartQAPro | 2504.05506 | single | mixed acc | chart + infographic | low-mod (infographic skill) | ❌ skill overlap |
| EvoChart / ChartX / ChartBench | 2409.01577 / 2402.12185 / 2312.15915 | single | accuracy | chart reading | none (mostly synthetic) | ❌ skill overlap |
| DVQA / PlotQA / FigureQA | 1801.08163 / 1909.00997 / 1710.07300 | single | accuracy | synthetic chart | none | ❌ synthetic, saturated |
| FinChart-Bench | 2507.14823 | single | mixed acc | financial chart | low-mod | ❌ chart + mixed formats |
| PDF-VQA | 2304.06447 | mostly single | accuracy | layout/element recog | low (PMC) | ❌ weak (single-page) |
| PDF-MVQA / "MMVQA" | 2404.12720 | multi | **retrieval metric** | entity retrieval | low (PMC) | ❌ metric mismatch |
| DocCVQA | 2104.14336 | collection (14k docs) | ANLS + retrieval | retrieval | CVC-UAB lab | ❌ ~20 questions (tiny) |
| OCR-VQA | *no arXiv (ICDAR'19)* | single | exact-match | salient text | none (book covers) | ❌ low lift |
| WebSRC | 2101.09465 | single | EM/F1 | HTML-canonical | low | ❌ redundant w/ VisualWebBench |
| ViDoRe / MMDocIR | 2407.01449 / 2501.08828 | collection | nDCG/Recall | **retrieval only** | low | ❌ off-target (retrieval) |
| TAT-QA / FinQA / MultiHiertt | 2105.07624 / 2109.00122 / 2206.01347 | — | exec-acc | **text/HTML, no image** | n/a | ❌ off-target (no perception) |
| WTQ / HiTab / TabFact | 1508.00305 / 2108.06712 / 1909.02164 | — | acc | **text/HTML tables** | n/a | ❌ off-target (no perception) |

*Unverified IDs to confirm before citing:* HR-Bench (2408.15556), HRScene
(2504.18406), ScreenQA (2209.08199); SciVQA has **no standalone arXiv**
(SDP@ACL 2025 only); OCR-VQA has **no arXiv** (ICDAR 2019, IEEE Xplore).

## The one strong new add

### LongDocURL (2412.18424) — ✅ strong

Chao et al., **ACL 2025** (Alibaba + CASIA — a lab with *no* CVC-UAB
lineage, so non-circular by construction). 2,325 questions over 396 PDFs,
**average 85.6 pages** (>33k pages total), spanning Understanding /
Reasoning / Locating across 20 subtasks. This is the **deepest
document-length stress available to us** — far past MP-DocVQA's ≤20 pages
and DUDE's ≤70, and even past MMLongBench-Doc's ~50-page average. The lift
is real and large (best open model ~30.6 vs GPT-4o 64.5 in the paper), so
a perception-budget method has genuine headroom to demonstrate. Caveat:
the answer set is mixed (extractive + reasoning + locating) scored by a
GPT-assisted answer-match, so it needs a profile `score_fn`, not bare
ANLS — same handling as DUDE. **Recommend adding alongside DUDE as the two
new length-axis legs** (DUDE = breadth/diverse-domain, LongDocURL =
extreme depth).

## Collection-scale long-doc (extension regime, not the OCR-free core)

A cluster of strong, recent benchmarks — **M3DocVQA/M3DocRAG** (2411.04952,
UNC, 3k+ PDFs / 40k+ pages, multi-hop), **ViDoSeek/ViDoRAG** (2502.18017,
Alibaba, ~6k-image collection, *unique-answer* so no judge noise),
**VisDoM** (2412.10704, Adobe/UMD, multi-doc), **M-LongDoc** (2411.06176,
hundreds of pages, LLM-judge), **DocBench** (2407.10701) — all test
**corpus-scale retrieve-then-read**. Per the MRAG-Bench rubric call and the
**MADQA framing tension**, this is the regime where our OCR/search
*extension* is the right tool and a pure OCR-free visual core looks
retrieval-bound. They are usable as **extension-regime evidence** (and
ViDoSeek is the cleanest fit for our iterative REPL+`batch_look` agent
loop, with a noise-free unique-answer metric), but lead with the extension
here, exactly as decided for MADQA. Several also carry **judge metrics**
(M-LongDoc, VisDoM, DocBench) — avoid as a headline number.

## Perception-dense single-document options (each with a caveat)

- **MMSci (2407.04903)** — the **best of the scientific-figure group**.
  Graduate-level figures (microscopy, western blots, schematics,
  simulations) sourced from **Nature Communications, *not* arXiv**, which
  is what sidesteps the reuse problem that kills CharXiv/SPIQA/SciFIBench
  (all of which pull figures from arXiv = the same population as
  `science_paper`/`science_poster`). Genuinely hard specialized perception
  and a skill beyond chart-reading. Caveat: MC + caption metrics, still
  "scientific paper figure" by domain.
- **TAT-DQA (2207.11871)** — the only *visual* financial-document option
  (page images, EM + numeracy-F1). But its financial reports overlap the
  `business_report` category by domain, it's single-page, and the
  bottleneck is numerical reasoning over a perceived table more than the
  context-budget mechanism. Conditional.
- **MMDocBench (2410.21311)** / **TableVQA-Bench (2404.19205)** — add a
  fine-grained grounding skill and dense-table perception respectively, but
  both overlap content DocVQA-2026 already covers (papers/infographics/
  financial; tables in business_report/science_paper). Weak generality
  claim.
- **ChartMuseum (2505.13444)** — explicitly targets *non-OCR-extractable*
  visual chart reasoning (humans 93% vs best model 63%), which is the
  cleanest articulation of our `batch_look` thesis in the chart space, and
  it's web-sourced (low reuse). But it's still chart-reading — the exact
  **skill overlap** that excluded ChartQA. A better ChartQA, same objection.

## The cleanest *mechanism* probe (out-of-domain)

**V\*Bench (2312.14135)** — Wu & Xie, CVPR 2024. High-resolution
(~2246×1582), visually-crowded **natural images** with attribute/spatial
questions that require *searching* for a small detail. This is the single
**closest conceptual match to recursive `batch_look`** — "find the small
thing in a big image via iterative visual search" is literally the
mechanism — and it's non-document, so fully non-circular. It would be a
sharper out-of-domain probe than TextVQA for showing the perception
mechanism is task-general. Caveats: only ~191 images (noisy), MCQ, and
non-document (so it's a *mechanism* demonstration, not a DocVQA generality
point). Use as an optional probe in place of, or alongside, TextVQA.

## Excluded — circular / reuse

- **DocHaystack / InfoHaystack (2411.16740)** — built *from* DocVQA and
  **InfographicVQA**; InfoHaystack therefore overlaps our `infographics`
  category at the image level. Reuse. ❌
- **OpenDocVQA / VDocRAG (2504.09795)** — bundles DocVQA + InfoVQA + DUDE.
  Only the **MHDocVQA** multi-hop subset would be defensible, and only
  after checking it excludes those components. ❌ as-is.
- **VisRAG (2410.10594)** — retrieval mechanism *and* its QA eval reuses
  DocVQA/InfographicVQA. Off-target twice over. ❌
- **CharXiv / SPIQA / SciFIBench / ArXivQA** — all draw figures from
  **arXiv papers**, the same population feeding `science_paper` /
  `science_poster`; high document-reuse risk on top of chart/figure skill
  overlap. ❌
- **DocCVQA (2104.14336)** — CVC-UAB, and the answer task is only **~20
  questions** over a 14k-doc collection (a retrieval benchmark). Too small
  for a headline number. ❌

## Off-target (wrong task shape or no perception)

- **Text/HTML-only:** TAT-QA, FinQA, MultiHiertt, WTQ, HiTab, TabFact — the
  input is structured text, not a rendered page; no visual perception to
  exercise. ❌
- **Pure retrieval:** ViDoRe/ColPali (2407.01449), MMDocIR (2501.08828) —
  scored on retrieval (nDCG/Recall), different mechanism. ❌
- **Synthetic / saturated chart:** DVQA, PlotQA, FigureQA, ChartX,
  ChartBench, EvoChart — synthetic or pure chart-reading; ChartQA-exclusion
  logic applies. ❌
- **Low lift:** OCR-VQA (large salient book-cover text a raw VLM reads in
  one pass); PDF-VQA (mostly single-page element recognition);
  PDF-MVQA/MMVQA (retrieval metric); WebSRC (HTML-canonical, redundant with
  VisualWebBench). ❌

## Updated bottom line (after discovery)

The discovery pass changes the recommendation in exactly **one** place:
add **LongDocURL** as a second new length-axis leg next to DUDE. The
revised ranked plan for *additional* benchmarks:

1. **MP-DocVQA + MMLongBench-Doc** — in use, the two length-axis legs.
2. **DUDE (2305.08455)** — multi-page, diverse, standard, ANLS.
3. **LongDocURL (2412.18424)** — *new*; extreme depth (avg 85.6 pg),
   distinct lab, large lift. Pairs with DUDE on the length axis.
4. **MADQA (2603.12180)** — competitor benchmark + planned baseline; lead
   with the OCR extension (see framing tension above).
5. **V\*Bench (2312.14135)** — *new*; optional out-of-domain **mechanism**
   probe, cleaner conceptual match than TextVQA (tiny/non-document).
6. **MMSci (2407.04903)** — *new*; optional "hard scientific perception"
   point, the only sci-figure set without arXiv-reuse risk.

Situational: the collection-scale set (M3DocVQA, ViDoSeek, VisDoM) is
**extension-regime** evidence only — same framing as MADQA, not the
OCR-free headline. Everything else is circular, off-target, or skill-
overlapping with DocVQA-2026's 8 categories. The earlier conclusion holds:
DocVQA-2026's deliberate 8-category breadth means the generality story
must lean on **distinct-corpus long-document** benchmarks, and the
strongest of those are now MP-DocVQA + MMLongBench-Doc (in use) + DUDE +
LongDocURL (to add).
