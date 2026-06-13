<!-- DERIVED: concatenation of sections/ in order. Do not hand-edit; regenerate. -->

# Seeing by Coding: Active Perception Lifts Document Agents Across Model Families

## 1 Introduction

The documents in DocVQA-2026 are long, multi-page, high-resolution, and largely visual. A single question can range over dozens of pages of figures, plots, schematics, maps, and infographics, and the answer-bearing content is frequently non-textual — a value in a chart cell, a label on an engineering drawing, a region of a poster. Because much of the signal never resolves to text, OCR-and-text pipelines cannot reach it. Answering is, first, a problem of *finding*: locating the right page and the right region before any reading can begin, then often compositing or computing over what is found.

Scaling the model does not by itself crack this. On the DocVQA-2026 validation split, bare setups score low — a raw multi-image VLM call, a multimodal agent reading pages in its own context, and a competition-kit prompt with no scaffold all land near 18–22 ANLS. Frontier-scale frozen models do better but only plateau: the official ICDAR-2026 baselines place Gemini 3 Pro at 37.5 and Gemini 3 Flash at 33.75 ANLS on the same validation split, matching or modestly exceeding the bare agents but low for their size on documents this large. Both ends of the scale axis fall short of what the task affords, which points away from model capacity as the binding constraint.

What moves accuracy is *active perception*. Give a code-capable reasoner a persistent Python REPL and a single visual primitive — an on-demand call to a frozen VLM perception tool against an arbitrary image region — and the reasoner can direct perception rather than consume it whole: crop to the evidence, zoom for acuity, composite regions, and do in code the coordinate and numeric arithmetic the VLM cannot. Holding both the reasoner and the frozen VLM fixed and varying only this scaffold spans roughly 25 ANLS points. The REPL active-perception scaffolds reach ~39 ANLS — far above tool-based ReAct, which has the same VLM but no REPL (~25), and above no-REPL VLM agents (~20) — and match or exceed the much larger frontier frozen models above. The result is demonstrated chiefly at a Qwen3.5-27B backbone, but it is not specific to one model: it holds across model families and is gated by base capability rather than size.

A second thread follows from the structure of the winning scaffold. Active perception is a *family* of harnesses, and its append-only member preserves a growing-prefix trajectory — the structure that weight-level training assumes — at no measured accuracy cost, making it the principled target for fine-tuning. A preliminary study asks whether training a small reasoner (Qwen3.5-4B) inside that member can lift a ≤8B agent further toward the gain.

This work makes the following contributions.

1. **Active perception is the dominant lever on document-agent accuracy.** A controlled comparison on DocVQA-2026 val (binary ANLS at threshold 0.9, $n{=}8$, frozen Qwen3.5-27B VLM) shows REPL active-perception scaffolds (~39 ANLS) far surpassing tool-based ReAct (~25) and no-REPL VLM agents (~20), and matching or exceeding the official ICDAR-2026 frontier baselines on the same validation split (Gemini 3 Pro 37.5, Gemini 3 Flash 33.75 — low for their size on documents this large).

2. **The mechanism, isolated.** The code REPL and active visual perception are each load-bearing — dropping either collapses the score. The advantage concentrates on *visually dense* pages, where cropping recovers detail a whole-page read misses (removing crop/zoom costs −11.2 on engineering drawings and −17.5 on science posters, against only −2.5 overall), and OCR-free visual perception is decisive (swapping vision for OCR text costs −25.5 points, with engineering drawings and maps falling to zero). The lever tracks visual density, not document length, and the REPL is what converts reasoning capability into perception quality: a stronger reasoner buys accuracy only when it has a REPL through which to direct perception.

3. **Cross-family, capability-gated generalization.** The advantage holds across model families — sharp on both Qwen3.5-27B and the cross-family Gemma-4-31B (a +14pp gap over ReAct that mirrors Qwen) — and is gated by base capability (reasoning, coding, and vision together), not raw size. A modern Qwen3.5-4B clears the bar, while a same-size Gemma-4-E4B and an older-generation Qwen3-8B serve as clean negative controls where the lift vanishes. `[TODO: second-dataset generalization point — pending]`

4. **The trainable-scaffold argument.** Among the active-perception family, the append-only member maintains a strictly growing token prefix across turns, the invariant on which policy-gradient and per-token distillation losses are built. At the Qwen3.5-27B backbone it ties the context-managing alternative on accuracy (provisional on the older-implementation side), so prefix preservation is free — identifying it as the scaffold to train inside.

5. **A preliminary ≤8B training study.** SFT, GRPO, and on-policy distillation are explored on Qwen3.5-4B inside the trainable scaffold, reported lightly and honestly. The study surfaces one structural ceiling that bounds every method: on long documents the append-only trajectory grows until the agent exhausts its budget before submitting, and this runaway rises with page count (r ≈ 0.70) — the same growing-prefix property that makes the scaffold trainable.

The remainder proceeds from background and task (§2) to the active-perception evaluation and its mechanism (§3), the argument for the append-only scaffold as a trainable target (§4), the preliminary small-agent training study (§5), related work (§6), and conclusions (§7).


## 2 Background

### 2.1 The DocVQA 2026 benchmark

Document Visual Question Answering asks a system to answer natural-language questions about a document presented as page images rather than as parsed text [DocVQA, arXiv:2007.00398]. The ICDAR 2026 DocVQA benchmark targets the hard end of this task. Its documents are long, multi-page, and high-resolution, and they span eight content categories — including dense business reports, scientific posters, maps, and engineering drawings — so that answering a single question often requires locating the relevant page and region before reading it. The benchmark provides no training split: the validation set holds 25 documents and 80 questions, and the test set holds 48 documents and 160 questions. Models must therefore be developed without in-distribution supervised data, which places weight on transfer from related corpora and on the design of the inference-time agent.

Answers are scored with the Average Normalized Levenshtein Similarity (ANLS), introduced for scene-text VQA to give partial credit for near-miss strings while penalizing substantive errors [ST-VQA, arXiv:1905.13648]. ANLS is one minus the normalized Levenshtein edit distance between a prediction and a reference, averaged over questions. The 2026 protocol applies it in a binary form: a per-question similarity at or above a threshold of 0.9 counts as a correct answer and anything below it as wrong, so the reported score is effectively a thresholded accuracy. This binary ANLS@0.9 governs every number in this report; the evaluation protocol is given in full in §3.1.

Two properties of the documents shape the difficulty. First, much of the answer-bearing content is visual rather than textual — figures, plots, schematics, and map labels — where optical character recognition is unreliable or simply inapplicable, so a text-only pipeline cannot reach the relevant evidence. Second, many questions are multi-hop or arithmetic, requiring evidence to be gathered from several locations and then combined or computed rather than read off a single span. Both properties favor a system that can perceive document regions selectively and reason over what it perceives, rather than one that reads a fixed transcription of the page.

### 2.2 Perceive–Reason–Code agents

A Perceive–Reason–Code agent answers document questions by pairing a frozen vision–language model (VLM) for perception with a code-capable language model (LM) as the reasoner. The reasoner does not view the document pixels directly; instead it writes code that issues perception requests against the VLM — cropping, zooming, and querying specific page regions — receives the returned observations into its context, and continues reasoning until it commits a final answer. Perception is thus *active*: rather than consuming a single fixed rendering of the document, the reasoner decides what to look at next based on what it has already seen, iteratively, and on demand. This active visual perception is what lets the agent reach content OCR cannot and chain evidence across pages. The VLM stays a frozen external endpoint throughout; what varies between agents is the reasoning LM and the harness that connects it to perception.

These agents form a family. A harness exposes the same perception interface — most importantly a code REPL in which the reasoner programs its visual queries — but members differ in how they manage the growing context and which tools they surface; the present work refers to the members built around a code REPL and `batch_look`-style perception as the REPL active-perception scaffold family. A prior competition entry instantiates one member of this family: a 27B-parameter reasoner driving an active-perception harness over a frozen VLM, submitted in the 8B–35B parameter tier of the leaderboard. That entry is one instance of the family, not a self-evidently best design, and it is the prior-work boundary for the present study.

What is reused from that prior work is the scaffold — the active-perception harness and its tool surface. What is new here is twofold. First, a controlled *harness evaluation*: rather than taking the active-perception design as given, this work measures it against alternative agent designs to isolate which structural choices drive accuracy (§3). Second, *weight-level training* of a small (≤8B) reasoning LM inside the trainable member of the family, asking whether fine-tuning can lift a small agent further rather than deploying a larger frozen reasoner zero-shot (§5). The scaffold is inherited; both the evaluation of it and the training within it are the contributions of this report.


## 3 Active Perception Lifts Document Agents

A frozen vision-language model and a code-capable reasoner can be wired into a DocVQA agent in many ways, and the wiring — not the choice of either model — is what moves accuracy. Holding both models fixed and varying only the scaffold spans roughly 25 ANLS points on the DocVQA-2026 validation set. This section establishes the challenge (the task resists both small and frontier-scale models), states the active-perception hypothesis, proves it under a controlled comparison, isolates the mechanism, and shows it generalizes across model families.

### 3.1 Evaluation setup

The metric is binary ANLS at threshold 0.9, the official DocVQA-2026 scoring rule: a prediction earns credit only when its normalized Levenshtein similarity to the gold answer reaches 0.9, otherwise zero. All harness numbers below are reported as mean ± std over $n{=}8$ independent trials on the DocVQA-2026 validation split (25 documents, 80 questions), micro-averaged per question, with a frozen Qwen3.5-27B serving as the VLM perception endpoint throughout. The reasoner's native thinking mode is disabled.

Disabling native thinking warrants explanation, because it does not mean the agent answers without reasoning. Three facts make this precise. First, the cost is empirically negligible: a thinking ablation on the strongest scaffold gives no accuracy gain at 27B and is slightly lower elsewhere, while official thinking was also hang-prone at 27B, so no-think is the stable configuration. Second, every solver parses only fenced Python code blocks out of the model's output, so the reasoner still writes free-text reasoning before and between code blocks; disabling the native thinking channel *relocates* reasoning into the body of the turn, it does not remove it. Third, the dspy-implemented solvers carry an explicit `reasoning` signature field, so reasoning is a declared part of each turn's contract regardless of whether native thinking is enabled. Thinking-off is therefore a stability and throughput choice, not an answer-without-reasoning handicap.

### 3.2 The challenge: bare and frontier setups both fall short

DocVQA-2026 documents are long, multi-page, high-resolution, and largely visual — figures, plots, schematics, maps — so the answer-bearing content is often non-textual and a single question usually requires locating the right page and region before reading it. The task resists both ends of the scale axis. In a bare setup, no-scaffold prompting and raw multi-image VLM calls score low: a single raw multi-image call reaches 20.47 ANLS, a single multimodal agent reading pages in its own context 22.34, and a competition-kit prompt with no scaffold 17.81. At the other end, frontier-scale frozen models only plateau — the official ICDAR-2026 baselines place Gemini 3 Pro at 37.50 and Gemini 3 Flash at 33.75 on the same validation split, low for their size on documents this large. Scaling the model alone is not the answer.

### 3.3 Harness design space and the active-perception hypothesis

A document agent's design space spans a taxonomy of scaffolds, all sharing the same frozen Qwen3.5-27B VLM:

- **Raw multi-image VLM** — a raw multi-image VLM call with no scaffold at all.
- **Multimodal REPL agent** — a single multimodal agent that pulls document pages into its own context with `display()` and reasons over them directly, with no VLM-tool call.
- **No-scaffold baseline** — a competition-kit prompt with no scaffold, used as an external anchor.
- **ReAct** — a thought → tool → observation loop with the same VLM perception tools but **no Python REPL**. Its only perception actuators are whole-page VLM queries at fixed page granularity; it cannot crop, zoom, composite, or do coordinate arithmetic.
- **the REPL active-perception family** — scaffolds whose action is Python executed in a persistent REPL, sharing a tool surface, prompt body, and the perception primitive `batch_look`, an on-demand call to the frozen VLM perception tool against cropped or zoomed image regions. The family has two members. **RLM** holds state in a hidden interpreter namespace and compacts the rendered context across turns. **CodeAct** is its append-only twin — identical tools and `batch_look` perception call, but a strictly append-only chat transcript in which each turn extends the previous one and no earlier token is rewritten, the prefix-preserving property §4 requires of a fine-tuning target. Neither reads OCR text; perception is purely visual.
- **OCR-only control** — the RLM scaffold with visual perception swapped for OCR text plus lexical search (OCR-only RLM), isolating the contribution of the visual modality.

The hypothesis follows: what governs accuracy is the **REPL active-perception loop** — code that issues on-demand visual queries (crop, zoom, coordinate arithmetic) against a frozen VLM and continues reasoning over what comes back — not model scale and not a fixed-granularity tool loop.

### 3.4 Result: active perception dominates

Table 1 reports the in-house solvers under the common protocol, alongside two external frontier anchors. The results fall into three tiers separated by gaps that exceed every cell's standard deviation.

**Table 1.** Harness comparison on DocVQA-2026 val (80Q), $n{=}8$, frozen Qwen3.5-27B VLM, no-think. Δ is the difference of means against the RLM reference. CodeAct is the corrected `codeact_chat` MDP loop (final, n=8); its ablation variants (+ OCR, without crop/zoom, + display channel) remain the older implementation.

| Tier | Solver | Mechanism | ANLS (n=8) | Δ vs RLM |
|---|---|---|---|---|
| **REPL active-perception** | RLM | REPL + VLM perception tool `batch_look`, OCR-free | **39.38 ± 1.49** | — |
| | CodeAct *(provisional)* | append-only twin, same tools | **39.53 ± 2.83** | +0.15 |
| | RLM + general sub-agent | `batch_look` generalized to any subtask | 39.22 ± 3.34 | −0.16 |
| | RLM + OCR | adds OCR text + lexical search atop vision | 37.81 ± 3.12 | −1.56 |
| | RLM without crop/zoom | whole-page reads only | 36.88 ± 3.20 | −2.51 |
| | RLM + display channel | direct `display()` atop the perception call | 35.47 ± 4.48 | −3.91 |
| **no-active-perception** | ReAct | VLM tools, no REPL | 25.16 ± 4.60 | −14.22 |
| | Multimodal REPL agent | pages in own context, no VLM-tool call | 22.34 ± 2.79 | −17.03 |
| | Raw multi-image VLM | raw multi-image, no scaffold | 20.47 ± 1.63 | −18.91 |
| | No-scaffold baseline | competition prompt, no scaffold | 17.81 ± 1.86 | −21.56 |
| **OCR-only floor** | OCR-only RLM | RLM scaffold, OCR text, no vision | **13.91 ± 1.56** | **−25.47** |
| *external anchor* | Gemini 3 Pro | — | 37.50 (val) | — |
| *external anchor* | Gemini 3 Flash | — | 33.75 (val) | — |

The REPL active-perception scaffolds occupy 35–39 ANLS, the no-active-perception family 18–25, and the OCR-only control sits alone at 14; each gap is several times the within-tier std. RLM, the load-bearing active-perception result, reaches 39.38 and the append-only CodeAct ties it at 39.53 (within trial-to-trial variance; the CodeAct cell is provisional pending re-run of the older implementation), so the win belongs to the active-perception *family* rather than a single harness — a tie §4 builds on. The family also matches or exceeds the official ICDAR-2026 frontier baselines on the same validation split (Gemini 3 Pro 37.50, Gemini 3 Flash 33.75). The advantage is not scale: an active-perception scaffold on a 27B VLM reaches the accuracy of proprietary models many times larger.

### 3.5 Mechanism: what is load-bearing

Five contrasts isolate why the loop wins.

**The REPL and the VLM perception tool are each load-bearing.** Removing either component of RLM in isolation collapses the score (Table 2). Stripping the REPL but keeping the VLM perception tool (ReAct) costs +14.2 relative to RLM; stripping the VLM perception tool but keeping the REPL — the Multimodal REPL agent, which loads pages into its own context instead — costs +17.0. Neither half alone suffices, and the no-agent baselines that drop both (Raw multi-image VLM 20.47, No-scaffold baseline 17.81) sit lower still.

**Table 2.** Each RLM component removed in isolation, vs RLM (39.38). The no-agent baselines (Raw multi-image VLM, No-scaffold baseline) drop both components and are not single-component ablations.

| Component removed | Resulting solver | Δ vs RLM | Reading |
|---|---|---|---|
| The REPL (perception tool kept) | ReAct (25.16) | +14.2 | the code REPL is load-bearing — crop, zoom, arithmetic, compositing |
| The VLM perception tool (REPL kept) | Multimodal REPL agent (22.34) | +17.0 | a focused VLM perception call beats raw pixels in the agent's own context |
| The visual modality (vision → OCR) | OCR-only RLM (13.91) | +25.5 | swapping vision for OCR text collapses the score |

**Cropping is the active ingredient behind the visual-density effect.** Restricting `batch_look` to whole-page reads costs only −2.5 overall but −11.2 on `engineering_drawing` and −17.5 on `science_poster` — the detail-dense categories where zoom is load-bearing — while leaving text-linear categories near-unchanged. The cost is acuity, not search effort: removing crop/zoom does not raise iteration count (11.8 vs 13.0 steps/Q). The advantage of the REPL family over ReAct mirrors this profile by category. At matched perception (a shared VLM, so the difference is purely harness), the RLM−ReAct gap is largest on visually dense categories — `engineering_drawing` ≈ +30, `business_report` ≈ +24, `maps` and `infographics` ≈ +19, `science_poster` ≈ +15 — and smallest on text-linear ones — `science_paper` ≈ +5, `slide` ≈ +10. The lever tracks **visual density**, recoverable by cropping, not document length: with a strong VLM the advantage is flat to slightly negative on the longest documents, where a whole-page read suffices once the answer page is found. (These per-category deltas are computed on cross-model matched-config runs; the canonical absolutes are the $n{=}8$ numbers in Table 1.)

**OCR-free visual perception is decisive.** Holding the scaffold, tools, and reasoner fixed and swapping only the perception modality — visual `batch_look` for OCR text plus lexical search — drops accuracy by 25.5 points, from 39.38 to 13.91, the matrix floor. The collapse is graded by visual density: the OCR-only agent scores **0/10 in all eight trials** on `engineering_drawing` and `maps`, loses ≈ +44 on `infographics` and ≈ +34 on `science_poster`, and is most survivable (≈ +18) on text-dense `slide` and `science_paper`. Vision does work OCR cannot replace, exactly where the page is non-textual.

**The REPL converts reasoning into perception.** A factorial swap separates the harnesses by what they are bound on. Replacing a weak reasoner on the 27B VLM with a strong 27B reasoner on a weaker 9B VLM *gains* +10.3 for RLM and +6.2 for CodeAct (both reasoning-bound) but *loses* −3.05 for ReAct (perception-bound). The REPL members write Python around `batch_look` — cropping to the evidence region, zooming, compositing crops, doing coordinate arithmetic — so a stronger reasoner produces better-targeted perception queries and extracts more even from a weaker VLM. ReAct has no such actuator: its perception ceiling is the VLM's whole-page acuity, which a stronger reasoner cannot direct. The complementary swap confirms a perception bottleneck for mid and small reasoners: holding the reasoner fixed and upgrading its homogeneous VLM to the 27B VLM lifts accuracy +7.9 at the 9B reasoner (Welch $t{=}3.54$, 95% CI [+3.4, +12.3]) and +8.6 at the 4B reasoner ($t{=}4.96$, 95% CI [+5.2, +12.0]). The crop/zoom loop is the mechanism that turns reasoning capability into perception quality; remove it and reasoning can no longer buy perception.

**Effort is not accuracy.** Iteration count correlates negatively with success: $\mathrm{corr}(\text{iterations}, \text{accuracy}) \approx -0.31$ over 200 doc-trials, with accuracy falling 62.8% → 43.2% → 31.6% across the 0–8 / 8–14 / 14+ iteration bins. The active-perception harnesses spend roughly twice the iterations and wall time of ReAct, but extra turns mark a hard document, not a path to the answer; the lever is the quality of the perception loop, not its length.

**A trajectory.** Figure 1 shows the loop on a single question over a 181-page document, trimmed verbatim from one agent trajectory. The agent surveys the document programmatically, locates the target table through a table-of-contents pointer, *distrusts* the full-page VLM read and re-reads a tighter crop — which disagrees, exposing a VLM misread — adjudicates by reading the region in halves, and finally does the arithmetic the VLM cannot do, in Python. Every component of the mechanism appears in one trace, and the self-distrusting crop-verify is load-bearing rather than ceremonial: it catches a wrong number the whole-page read returned.

> **Figure 1.** Active-perception loop on `business_report_1_q1` (181-page NVIDIA annual review), RLM, 16 iterations, **correct** (gold `2048.88`, predicted `2048.88`). *"In Fiscal 2025, by how many dollars does NVIDIA's TSR value exceed the Nasdaq-100 Index TSR value?"*
>
> ```python
> # iter 2 — survey: one batched VLM sweep over candidate pages
> survey_pages = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40]
> results = batch_look([(pages[i], f'Summarize page {i}: ... Look for TSR, '
>     'Total Shareholder Return, or performance comparisons.') for i in survey_pages])
> ```
> *iter 7–11 — locate via a TOC pointer ("Page 15 ... 'Compensation Discussion and Analysis' starts on page 44"), narrow to pages 71–85, and the table surfaces:* `Page 76: {... 'fiscal_2025_tsr': {'nvidia_tsr': '$2,287.07', 'nasdaq100_index_tsr': '$238.19'}}`
> ```python
> # iter 14 — verify, distrusting precise numbers: crop and re-read
> crop_box = (100, 300, 750, 800)
> cropped = pages[76].crop(crop_box)
> result = batch_look([(cropped, "What are the exact dollar values shown for "
>     "NVIDIA TSR and Nasdaq-100 Index TSR for Fiscal 2025?")])
> ```
> *The crop disagreed with the full-page read (`$978.42`/`$190.57` vs `$2,287.07`/`$238.19`), so the agent broke the tie by re-reading the top and bottom halves separately (iter 15) before committing.*
> ```python
> # iter 16 — the arithmetic the VLM cannot do, in Python
> nvidia_tsr = 2287.07
> nasdaq_tsr = 238.19
> difference = nvidia_tsr - nasdaq_tsr   # FINAL: {'answer': '2048.88'}
> ```

The same mechanism shows as a head-to-head contrast. On `science_poster_1_q1` — "the percentage score improvement from Baseline to TexTok in rFID-50k for ImageNet 512×512 with 128 tokens", answer `30.2%`, which requires reading two cells (`1.49`, `1.04`) from a small table embedded in a 4000×2000 poster and computing $((1.49-1.04)/1.49)\times100$ — the RLM agent crops the table band, reads each operand with `batch_look`, and computes the result in Python, returning `30.2%` (correct) in six iterations. ReAct, with no REPL, reads the whole 4000×2000 poster at page granularity, cannot zoom to the cell, terminates in two iterations, and returns `0.48%` (wrong). The REPL is the difference between reading a cell and reading a page.

### 3.6 Generalization: cross-family and capability-gated

The advantage is not specific to one model. Two sweeps probe its reach without conflating perception with reasoning: one varies the reasoner while holding perception fixed at the 27B VLM (Table 3), the other compares whole homogeneous models, where the reasoner is also its own VLM, across families (Table 4).

**Table 3.** Reasoner scale at fixed perception — each reasoner paired with the 27B VLM. DocVQA-2026 val, $n{=}8$.

| Reasoner (× 27B VLM) | RLM | ReAct | CodeAct |
|---|---|---|---|
| Qwen3-8B (older gen) | 11.73 | 15.79 | 9.50 † |
| Qwen3.5-4B | 21.09 | 15.66 | 22.34 |
| Qwen3.5-9B | 24.54 | 21.01 | 24.26 † |
| Qwen3.5-27B | 39.38 | 25.16 | 39.53 |

With perception held fixed, RLM is at or above ReAct at every Qwen3.5 reasoner size (4B 21.09 vs 15.66, 9B 24.54 vs 21.01, 27B 39.38 vs 25.16), and the corrected CodeAct ties RLM at the two scales where it is complete — 22.34 vs 21.09 at 4B and 39.53 vs 39.38 at 27B. The older-generation Qwen3-8B is the informative exception: given the same strong 27B perception, it still scores higher under ReAct (15.79) than under RLM (11.73) or CodeAct (9.50). A weak coder cannot drive a code REPL, so handing it one hurts; with perception held constant, the binding constraint here is coding capability.

**Table 4.** Cross-family, homogeneous models — the reasoner is also its own VLM. DocVQA-2026 val, $n{=}8$.

| Model (homogeneous) | RLM | ReAct | CodeAct |
|---|---|---|---|
| Qwen3.5-27B | 39.38 | 25.16 | 39.53 |
| Gemma-4-31B | 32.50 | 18.44 | 29.25 † |
| Gemma-4-E4B | 7.34 | 6.09 | 7.66 † |

The harness ordering reproduces on a second model family: Gemma-4-31B shows RLM 32.50 ≫ ReAct 18.44 (a +14pp gap), mirroring Qwen3.5-27B's 39.38 ≫ 25.16. The small homogeneous Gemma-4-E4B is the floor control — all three harnesses land at 6–8, within noise of the ~6.25 no-scaffold baseline, so at this capacity no scaffold pays off.

Together the sweeps fix the gate as base capability — reasoning, coding, and vision jointly — not raw parameter count. A modern Qwen3.5-4B clears it (Table 3); a same-size but weaker Gemma-4-E4B does not (Table 4); and an older Qwen3-8B fails even with strong perception, for lack of coding ability (Table 3). The advantage appears once, and only once, the base can drive the loop. († older CodeAct implementation, provisional pending re-run; the 4B and 27B CodeAct cells are the corrected `codeact_chat` MDP loop, final at $n{=}8$.)

A second-dataset evaluation at 27B is in progress to test whether the advantage holds beyond DocVQA-2026. `[TODO: second-dataset generalization point — pending]`


## 4 From Harness to Trainable Scaffold

The §3 evaluation establishes a *family* of active-perception scaffolds, not a single winning harness: at the Qwen3.5-27B backbone the two strongest members, RLM and CodeAct, attain the highest accuracy and are statistically indistinguishable (39.4 and 39.5 ANLS, $n{=}8$), sharing a tool set, prompt body, and VLM-tool perception. The win is the family. The weight-level training in §5 targets one member, the append-only CodeAct. The two harnesses differ in how each turn's context is formed across turns, and that difference — not accuracy — is what determines whether gradient-based training is well-defined.

### 4.1 The append-only prefix invariant

Policy-gradient multi-turn reinforcement learning (GRPO, PPO, GSPO) and per-token distillation losses are built on an append-only token-prefix invariant: $\text{prompt}_{t+1} = \text{prompt}_t + \text{completion}_t + \text{observation}_t$. Under this invariant the whole rollout is one monotonically growing token sequence — the context at turn $t$ is an exact prefix of the context at turn $t{+}1$ — so the trainer recovers every per-step log-probability from a single masked forward pass over the trajectory, scoring each action in exactly the context that generated it [renderers, prefix-invariant write-up].

CodeAct preserves this invariant. It maintains an append-only chat transcript — a system message followed by alternating user and assistant turns — in which each turn only appends and no earlier token is rewritten. RLM does not. It surfaces a sidecar of metadata describing the variables currently live in the REPL, holds their full values off-context in the interpreter, and re-renders this managed view of state at each turn rather than appending to it. Because the visible context is rebuilt every step, earlier positions change from turn to turn and consecutive turns do not stand in a prefix relation.

Breaking the invariant breaks the policy gradient. Rewriting history makes the observation distribution policy-dependent and non-stationary, violating the assumptions of policy-gradient methods [FoldAct, arXiv:2512.22733], and the rollout is no longer a single growing sequence, so per-step likelihoods must be recovered from separate forward passes over distinct contexts [ReSum, arXiv:2509.13313]. Importance ratios and cross-turn credit assignment are then no longer defined over one coherent sequence. Methods that let an agent compress its own history during reinforcement learning report exactly this and repair it with purpose-built machinery: FoldAct stabilizes summary actions with custom losses, and ReSum segments the trajectory at each compression and broadcasts a trajectory-level advantage across segments. An append-only transcript is the structure these objectives are built for; a context-manipulating one needs bespoke repair.

### 4.2 Prefix preservation is free

Selecting the append-only member concedes no accuracy where it is measured. With the corrected CodeAct loop, CodeAct ties RLM at both scales evaluated under matched perception — 39.5 against 39.4 at the 27B backbone, and 22.3 against 21.1 at the 4B reasoner that the training in §5 actually targets ($n{=}8$, both differences within trial-to-trial variance). The structural property that makes CodeAct trainable thus comes at no measured accuracy cost relative to RLM's managed, re-rendered context — including at the small scale where it matters for training. (On some bases RLM remains the more robust of the two, and the remaining cross-model CodeAct cells await the same corrected re-run.) The conclusion the bridge carries forward is the family-level one — train inside the append-only member of the winning family, paying nothing in accuracy for a trajectory that established training machinery accepts directly.

### 4.3 An accessible ceiling versus a higher one

The two scaffolds differ in how context length scales. RLM's managed view stays compact across arbitrarily many turns, decoupling the number of reasoning steps from the context window, whereas CodeAct's append-only transcript grows monotonically and can approach the window limit on long documents with many perception calls. In principle this gives RLM a higher ceiling on long-horizon tasks. That the advantage does not surface as higher accuracy is consistent with a specific account: exploiting a compact, re-rendered context requires a policy trained to operate on it, but contemporary models are post-trained almost entirely on append-only conversations, leaving a frozen model off-distribution in such a harness. Managed-context systems indeed tend to match or beat append-only baselines only after regime-specific training — an untrained context curator underperforms full-context prompting and overtakes it only once trained with reinforcement learning [ContextCurator, arXiv:2604.11462], and stable context folding requires purpose-built losses [FoldAct, arXiv:2512.22733].

This account is a hypothesis, not a demonstrated result, and its magnitude should not be overstated: frozen models sometimes benefit from compaction with no retraining [ReSum, arXiv:2509.13313], and that benefit shrinks as model capability grows. The conservative conclusion is that an append-only scaffold delivers its full potential under established training, whereas a context-managing scaffold may have a higher but currently less accessible ceiling, reachable only through new and not-yet-established methods. The present work adopts the prefix-preserving scaffold, for which mature training machinery applies directly; training a policy natively for a context-managing harness is left to future work. The cost of this choice is the append-only transcript's monotonic growth, which on long documents becomes a structural ceiling on training the scaffold itself — the runaway behavior quantified in §5.


## 5 Training a Small Agent

The §4 analysis fixes the target of weight-level training: the append-only CodeAct scaffold, whose growing-prefix trajectory is the structure gradient-based objectives are built for. This section reports a preliminary study of whether fine-tuning lifts a ≤8B agent inside that scaffold. The results are not yet decisive, and the section is kept deliberately light; the one firm finding is a structural ceiling that bounds every method considered.

### 5.1 What is trained, and how it is measured

The only weights updated are those of the reasoning policy — the language model that, each turn, emits a thought and a single Python action. It is fine-tuned with LoRA adapters on all linear projections, which keeps a run within a single 80 GB GPU. The perceptual backbone stays frozen: the vision-language model is reached only through an external HTTP endpoint that the policy calls via `batch_look`, and no gradient flows into it. Restricting training to the reasoning policy isolates the question of interest — whether weight-level training can lift the agent's accuracy — from any change in perceptual capability. The same model serves as both the policy under training and the policy under evaluation: trajectories are collected with the CodeAct scaffold, training updates the policy, and evaluation runs the identical scaffold, so the policy that is trained is exactly the policy that is deployed.

The backbone is Qwen3.5-4B, a pivot from the Qwen3-8B model named in the original proposal. The 4B model is the one that holds the ≤8B leaderboard slot inside this agent scaffold, making it the relevant subject for a study aimed at the ≤8B tier; the larger, older-generation 8B model is not the incumbent, and using it would conflate a backbone change with the training intervention. Training uses verl with fully-sharded data-parallel (FSDP) model sharding and a synchronous, colocated GRPO trainer in which rollout generation and policy updates run on the same devices.

Evaluation follows the §3.1 protocol — binary ANLS at threshold 0.9, native thinking disabled (the reasoner still writes its reasoning as free text around the code blocks, as explained in §3.1) — applied to the 4B agent with $n{=}4$ independent samples per question to reduce per-rollout variance. Two scales are used: a 29-question mini-screen for rapid checkpoint comparison and the full 80-question validation set for confirmation. A model and its baseline are compared per question on the same items, so improvements are assessed with paired statistics. Three reference points anchor the trained-model numbers: the untrained Qwen3.5-4B agent in the same CodeAct scaffold, the published ≤8B leaderboard entry at roughly 0.19 ANLS, and the 8B–35B-tier entry at roughly 0.375.

### 5.2 Training signals

Three families of training signal are considered, sitting on a single axis — how much information each rollout returns per unit of generation, and whether that information is on- or off-policy.

**Supervised fine-tuning / sequence-level distillation (SeqKD).** Teacher trajectories are produced by rejection sampling: many rollouts are drawn from a strong solver, and only those scoring above an ANLS threshold are kept; the policy is fit to the accepted trajectories with token-level cross-entropy. The signal is dense in tokens but off-policy — $O(N)$ supervised tokens per trajectory, all generated by another policy. It is the cheapest signal to apply and serves as a cold-start warmup: it moves a frozen model onto the scaffold's turn structure before any on-policy stage, reducing the risk of a zero-reward-variance start in which no sampled rollout ever succeeds. The supervised study draws its trajectories from MMLongBench-Doc [arXiv:2407.01523], a long-document multimodal benchmark disjoint from the DocVQA validation set and therefore leakage-free against the evaluation.

**Reinforcement learning from a verifier reward (GRPO).** Group Relative Policy Optimization [arXiv:2402.03300] samples a group of rollouts per prompt, scores each, and uses the within-group reward spread as a baseline-free advantage. The reward is the answer-level ANLS of the submitted answer, augmented with shaping terms for tool use and code execution that penalize malformed actions that fail to parse or run. The signal is on-policy but sparse: each rollout returns a single scalar, $O(1)$ of information regardless of trajectory length, paid for by generating the full multi-turn rollout.

**On-policy distillation (OPD).** The policy generates its own rollouts, and a per-token KL divergence to the frozen 27B teacher supplies a dense correction at every position — the on-policy, dense-feedback counterpart to RL's sparse scalar [GKD, arXiv:2306.13649]. The signal returns $O(N)$ on-policy per-token feedback, combining RL's distribution match with SeqKD's token density, and is less prone to the catastrophic forgetting of reward-only RL. The signals are not mutually exclusive: the verifier-reward advantage and the per-token teacher KL can be summed into a single update.

A separate, larger prompt substrate was assembled after the supervised runs reported below and has **not** been used for any training reported here; it is the intended substrate for upcoming larger-data SFT, RL, and OPD runs. This pool is method-agnostic — document-question prompts usable by any of the signals above — and draws from seven DocVQA-family datasets: DocVQA [arXiv:2007.00398], InfographicVQA [arXiv:2104.12756], ChartQA [arXiv:2203.10244], MapQA [arXiv:2211.08545], MP-DocVQA [arXiv:2212.05935], TAT-DQA [arXiv:2207.11871], and SlideVQA [arXiv:2301.04883], spanning single- and multi-page documents, infographics, charts, maps, and slide decks. A weighted per-source sampler controls each dataset's contribution rather than pooling raw question counts, which would let the largest source dominate the mixture, and the pool is constructed to be leakage-safe with respect to the DocVQA validation set.

### 5.3 Preliminary results and the structural ceiling

Supervised fine-tuning gives a small, fragile lift on the 4B agent. Trained on the leakage-free transfer trajectories, the model improves over the untrained base on the full validation set, from 15.3 to 20.6 ANLS at 20 epochs (Table 1, $+5.3$ points, $p \approx 0.04$ over the shared question set). The gain is concentrated in extraction quality — submit-only accuracy rises from 25.1 to 32.2 while the submit rate stays flat (61% to 64%) — and is carried almost entirely by the short, visually dense categories (infographics 27.5 to 52.5, slide, science_paper), with maps fixed at 0 and engineering drawings regressing. The lift is fragile: on the 29-question mini-screen the same checkpoints bounce within the ±5-point small-$n$ noise floor. The honest reading is that supervised imitation of teacher trajectories buys a few points at best and is not a reliable lever, consistent with multi-turn observation-shift blunting trajectory imitation — even an in-domain set memorized to near-zero loss only ties the untrained base.

| Stage | ANLS@0.9 | pass@4 | submit-only |
|---|---:|---:|---:|
| Qwen3.5-4B base | 15.3 | 33.8 | 25.1 |
| + SeqKD, 16 epochs | 17.2 | 36.2 | 26.7 |
| + SeqKD, 20 epochs | **20.6** | 41.2 | 32.2 |

*Table 1: Supervised fine-tuning on 27B-teacher CodeAct trajectories (DocVQA-2026 val, 80Q, $n{=}4$). The $+5.3$-point lift is concentrated in extraction quality, not submit rate.*

The direction of the lift reverses with the backbone. Under the same recipe the older-generation Qwen3-8B drops from 5.9 to 4.4 ANLS: it does not inhabit the perception loop — it shortcuts to a fast, shallow submission (median 1–2 turns and a single perception call, against the 4B's nine of each) — and supervised fine-tuning entrenches the shortcut rather than teaching the loop. A larger but older base is not a free capacity upgrade inside a scaffold tuned for the Qwen3.5 family.

Reinforcement learning is, at this stage, a smoke-test rather than a result. The asynchronous GRPO loop runs end to end and produces coherent, diverse rollouts with healthy reward variance; the only available checkpoint (30 steps, partial evaluation, $n{=}2$) has not moved answer accuracy relative to its matched base. No conclusion for or against RL is drawn from it. [TODO: full 80Q, $n{=}4$ GRPO evaluation with a learning curve across steps.]

The load-bearing finding is a structural ceiling shared by every method. On the long, multi-page documents that dominate DocVQA-2026, the append-only trajectory grows until the agent exhausts its turn, token, or wall-clock budget before submitting — a runaway failure independent of answer quality. On the base 4B agent, 38.8% of all rollouts hit a budget cap without ever submitting, and the per-document runaway rate rises sharply with page count: from 23% on 1–5-page documents to 59% on documents over 50 pages, reaching 100% on the single 181-page document (Pearson $r \approx 0.70$ over 25 documents). Accuracy collapses in lockstep, from 18.4 ANLS on the shortest documents to 7.9 on the longest. Supervised fine-tuning trims long-document runaway (59% to 45% at 50+ pages) but does not lift the long-document accuracy floor: even when the trained agent submits on a long document, it does not answer the question. The ceiling therefore caps every method before answer quality enters, and it is precisely the cost of the growing-prefix property that makes the scaffold trainable in the first place (§4) — the same monotonic context growth that licenses gradient-based training on the append-only transcript bounds how far that training can reach on long documents.


## 6 Related Work

**Reinforcement learning for language models.** Group-relative policy optimization (GRPO) replaces PPO's learned critic with a group-of-samples baseline, estimating advantages from the reward spread within a sampled group [GRPO, arXiv:2402.03300]. A line of subsequent work refines its surrogate and normalization. Dr.GRPO removes length and standard-deviation normalizers that bias the gradient toward longer or lower-variance responses [Dr.GRPO, arXiv:2503.20783]; DAPO contributes decoupled clipping, dynamic sampling, and token-level loss for stable long-CoT training [DAPO, arXiv:2503.14476]; GSPO moves importance weighting from the token to the sequence level [GSPO, arXiv:2507.18071]; and CISPO clips importance-sampling weights rather than token updates to retain all-token signal [CISPO, arXiv:2506.13585]. The training in §5 adopts GRPO with an ANLS verifier reward, and these variants delimit the design space for its surrogate.

**Knowledge distillation.** Sequence-level distillation (SeqKD) trains a student on teacher-generated outputs, an off-policy objective dense in tokens but disconnected from the student's own distribution. MiniLLM minimizes a reverse KL to avoid the mode-covering failure of forward KL on generative tasks [MiniLLM, arXiv:2306.08543], and generalized knowledge distillation (GKD) trains on student-sampled sequences to close the train–inference distribution gap [GKD, arXiv:2306.13649]. On-policy distillation extends this lineage to a dense per-token teacher signal computed over student rollouts [OPD, arXiv:2604.00626]. This SeqKD / on-policy-distillation contrast frames the signals explored in §5.

**Reinforcement learning under agent-controlled context.** Policy-gradient training assumes a single growing token sequence; harnesses in which the agent rewrites or compresses its own history break that assumption, and an emerging, not-yet-established line of work targets this regime directly. FoldAct shows that summary actions induce a policy-dependent, non-stationary observation distribution and repairs the gradient with purpose-built machinery [FoldAct, arXiv:2512.22733]; ReSum segments the trajectory at each compression and broadcasts trajectory-level advantage across segments [ReSum, arXiv:2509.13313]; ContextCurator reports that an untrained context manager underperforms full-context prompting and overtakes it only after reinforcement learning [ContextCurator, arXiv:2604.11462]; and AdaCoM adapts the compression policy during agentic reasoning [AdaCoM, arXiv:2605.30785]. Format-sensitivity of frozen models under prompt rewriting motivates the same concern [Sclar, arXiv:2310.11324]. §4 positions the append-only CodeAct scaffold against this frontier, choosing the prefix-preserving path for which mature training applies and leaving context-managing harnesses to future work.

**Agentic document question answering.** Recent systems pair a frozen vision–language model with an orchestrating agent. RVLM drives perception through repeated on-demand calls to a frozen VLM perception tool [RVLM, arXiv:2603.24224], the same perception primitive used by the RLM harness this work reuses — RLM (compact/managed REPL-history context) [RLM, arXiv:2512.24601]. Multi-agent and tool-routing designs decompose document understanding across specialized roles [MADQA, arXiv:2603.12180; MDocAgent, arXiv:2503.13964; ORCA, arXiv:2603.02438; SlideAgent, arXiv:2510.26615], while others add retrieval, visual search, or learned OCR control [ARIAL, arXiv:2511.18192; VISOR, arXiv:2604.09508; DocVStar, arXiv:2604.13731; AgenticOCR, arXiv:2602.24134]. The scaffold here builds on the CodeAct paradigm of expressing agent actions as executable code and on the ReAct interleaving of reasoning and action [ReAct, arXiv:2210.03629].

**DocVQA-family benchmarks.** The training-data pool of §5 and the evaluation of §2 draw on the document-VQA benchmark lineage: single-page DocVQA [DocVQA, arXiv:2007.00398], infographic and chart variants [InfographicVQA, arXiv:2104.12756; ChartQA, arXiv:2203.10244], multi-page and multi-document settings [MP-DocVQA, arXiv:2212.05935; DUDE, arXiv:2305.08455], slide decks [SlideVQA, arXiv:2301.04883], and the long-document MMLongBench-Doc used for transfer trajectories [MMLongBench-Doc, arXiv:2407.01523]. Scoring follows the ANLS metric introduced with ST-VQA [ANLS/ST-VQA, arXiv:1905.13648].


## 7 Conclusion and Future Work

Across two model families and four reasoner scales, what governs document-agent
accuracy on long, visually dense documents is not the size of the reasoner but
whether it can actively control perception: an agent that writes code to crop,
zoom, and query a frozen vision–language model on demand matches or exceeds
much larger frozen models and far surpasses fixed-granularity tool agents. The
advantage is mechanistically located — it concentrates where cropping recovers
detail a whole-page read misses, collapses when visual perception is replaced by
optical character recognition, and appears only once the base model is capable
enough, in both reasoning and code, to drive the loop.

The two strongest scaffolds in this family tie at the 27B backbone, and the
append-only member preserves the growing-prefix trajectory that established
weight-level training assumes; it is therefore the principled target for training
a small agent. A preliminary study at the ≤8B tier finds that supervised
fine-tuning buys a small, fragile lift and that the dominant obstacle is
structural: on long documents the append-only trajectory grows until the agent
exhausts its budget before answering. Training a small agent to realize the
active-perception advantage — past that runaway ceiling, with on-policy methods
that match the deployment distribution — is the open direction this work points
to.


## Limitations

The evaluation is confined to the DocVQA-2026 validation set (25 documents, 80
questions); the test set is held out by the competition portal, and the small
validation size limits the resolution of per-category and per-slice estimates.
The per-category and page-bucket analyses are computed on cross-model
matched-configuration runs rather than the homogeneous baselines, so those
figures are read as deltas, with the headline solver numbers ($n{=}8$) cited for
absolutes. The CodeAct cells throughout are produced by an earlier
implementation of that scaffold and are reported as provisional pending a
re-run; the active-perception conclusions rest on the RLM numbers, which are
unaffected.

Generalization is demonstrated across two model families and is expected, but not
yet shown, to hold across a second document-QA dataset. The training study is
deliberately preliminary: results use $n{=}4$ samples and single checkpoints, the
reported supervised lift is modest and fragile, and the reinforcement-learning
runs are reported only as evidence that the training loop is coherent, not as an
outcome. The long-document runaway that bounds the small-agent results is the
cost of the append-only scaffold's growing context, and the account in §4 of why
a context-managing scaffold has a higher but currently inaccessible ceiling is a
hypothesis rather than a demonstrated result.


