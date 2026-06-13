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
