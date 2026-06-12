## 3 Agentic Harness Evaluation

A frozen vision-language model and a code-capable language model can be wired into a DocVQA agent in many ways. Which wiring matters: holding both models fixed and varying only the scaffold spans roughly 25 ANLS points on the DocVQA-2026 validation set. This section evaluates eight harness designs under a common protocol, isolates the two mechanisms that account for the spread — a code REPL and recursive VLM perception — and shows how the ranking depends on the reasoner's scale.

All numbers below are mean ± std over $n{=}8$ independent trials on the DocVQA-2026 validation split (25 documents, 80 questions), with a frozen Qwen3.5-27B serving as the VLM and `enable_thinking=false` on the reasoner. A thinking ablation on the strongest scaffold gives no gain at 27B, so the reported configuration is no-think throughout.

### 3.1 Harness taxonomy

Six scaffold families are compared, all sharing the same frozen Qwen3.5-27B VLM endpoint:

- **RLM** — a recursive language-model agent whose action is Python executed in a persistent REPL, with the only perception tool being `batch_look`, a recursive sub-call to the frozen VLM against cropped or zoomed image regions. State lives in a hidden interpreter namespace and the rendered context is compacted across turns. No OCR text is ever read; perception is purely visual.
- **CodeAct** — the append-only twin of RLM: identical tools, prompt body, and `batch_look` sub-call, but the transcript is a strictly append-only chat in which each turn extends the previous one and no earlier token is rewritten. This makes the trajectory a fully observable MDP, the property §4 requires of a fine-tuning target.
- **ReAct** — a thought→tool→observation loop with the same VLM perception tools but **no Python REPL**. Its only perception actuators are whole-page VLM queries at fixed page granularity; it cannot crop, zoom, composite, or do coordinate arithmetic.
- **direct-VLM** — a single multimodal agent that pulls document pages into its own context with `display()` and reasons over them directly, with no recursive sub-call.
- **raw-VLM** — a raw multi-image VLM call with no scaffold at all.
- **OCR-only control** — the RLM scaffold with visual perception swapped for OCR text plus lexical search, isolating the contribution of the visual modality (`rlm_ocr`).

A competition-kit prompt with no scaffold (`official`) serves as an external anchor.

### 3.2 The eight-solver matrix

Table 1 reports the eight solvers under the common protocol, alongside two external anchors. The results fall into three tiers separated by gaps that exceed every cell's standard deviation.

**Table 1.** Harness comparison on DocVQA-2026 val (80Q), $n{=}8$, frozen Qwen3.5-27B VLM, no-think. Δ is the difference of means against the RLM reference.

| Tier | Solver | Mechanism | ANLS (n=8) | Δ vs RLM |
|---|---|---|---|---|
| **visual-recursive** | RLM | REPL + recursive VLM `batch_look`, OCR-free | **39.38 ± 1.49** | — |
| | CodeAct | append-only MDP twin, same tools | **39.53 ± 2.83** | +0.15 |
| | sub-call (general) | `batch_look` generalized to any subtask | 39.22 ± 3.34 | −0.16 |
| | sub-call + rationale | sub-call returns answer + uncertainty note | 39.22 ± 2.91 | −0.16 |
| | + OCR | adds OCR text + lexical search atop vision | 37.81 ± 3.12 | −1.56 |
| | no crop/zoom | whole-page reads only | 36.88 ± 3.20 | −2.51 |
| | + display channel | direct `display()` atop the sub-call | 35.47 ± 4.48 | −3.91 |
| **no-recursion** | ReAct | VLM tools, no REPL | 25.16 ± 4.60 | −14.22 |
| | direct-VLM | pages in own context, no sub-call | 22.34 ± 2.79 | −17.03 |
| | raw-VLM | raw multi-image, no scaffold | 20.47 ± 1.63 | −18.91 |
| | official | competition prompt, no scaffold | 17.81 ± 1.86 | −21.56 |
| **OCR-only floor** | rlm_ocr | RLM scaffold, OCR text, no vision | **13.91 ± 1.56** | **−25.47** |
| *external anchor* | Gemini 3 Pro | — | 37.50 (val/test) | — |
| *external anchor* | GPT-5.2 | — | 35.00 (test) | — |

The visual-recursive scaffolds occupy 35–39 ANLS, the no-recursion family 18–25, and the OCR-only control sits alone at 14. Each gap is several times the within-tier std. The append-only CodeAct scaffold (39.53 ± 2.83) is statistically indistinguishable from RLM (39.38 ± 1.49): prefix preservation costs no accuracy, a result §4 builds on. The visual-recursive scaffolds also match or exceed strong proprietary models evaluated zero-shot on the same split (Gemini 3 Pro 37.50, GPT-5.2 35.00).

### 3.3 The OCR-free result

The single most decisive contrast in the matrix is `rlm_ocr`. Holding the scaffold, tools, and reasoner fixed and swapping only the perception modality — visual `batch_look` for OCR text plus lexical search — drops accuracy by **25.5 points**, from 39.38 to 13.91, and lands the agent at the matrix floor. On the `engineering_drawing` and `maps` categories the OCR-only agent scores **0/10 in all eight trials**: visual content with sparse or unreliable extractable text is invisible to OCR, and recursive visual perception does work that OCR text cannot replace.

Both halves of the scaffold are load-bearing, and dropping either collapses the score (Table 2). Removing the recursive sub-call while keeping the reasoner one-shot multi-image (`raw_vlm`, `direct_vlm`) and removing the REPL while keeping the recursive perception (`react`) both fall well below RLM.

**Table 2.** Each component removed, vs RLM (39.38).

| Component dropped | Solver | Δ vs RLM | Reading |
|---|---|---|---|
| Recursive sub-call | raw-VLM (20.47) | +18.9 | recursive agent↔VLM dominates one-shot multi-image |
| The REPL | ReAct (25.16) | +14.2 | the code REPL is load-bearing (crop, arithmetic, compose) |
| Sub-call, pixels kept | direct-VLM (22.34) | +17.0 | raw pixels in-context ≠ a focused VLM sub-call |
| All visual perception | rlm_ocr (13.91) | +25.5 | swapping vision for OCR text collapses the score |

### 3.4 Ablations

Four ablations probe additions and variations on the visual-recursive scaffold; all are measured against RLM at 39.38.

- **Crop/zoom is a category-specific lever, not a global one.** Restricting `batch_look` to whole-page reads costs −2.5 overall but −11.2 on `engineering_drawing` and −17.5 on `science_poster` — the detail-dense categories where zoom is load-bearing — while leaving other categories near-unchanged. It does not raise iteration count (11.8 vs 13.0 steps/Q), so the cost is acuity, not search effort.
- **OCR adds nothing on top of vision.** Layering OCR text and lexical search atop the visual sub-call moves accuracy by −1.6, within noise: once recursive visual perception is present, extracted text is redundant.
- **A direct display channel is mildly harmful and induces churn.** Adding a direct `display()` image channel atop the sub-call costs −3.9 and triples the variance (± 4.48). The extra in-context imagery destabilizes the agent rather than helping it: iterations climb to 18.1 steps/Q (from 13.0) and the cap-hit rate rises to 7%.
- **Generalizing the sub-call is null.** Replacing the perception-only `batch_look` with a general delegation interface that can take any subtask leaves accuracy flat (−0.16), because the agent uses the interface as a perception tool roughly 99% of the time. A single focused perception sub-call already captures the benefit, which bounds the necessary sub-call interface. Adding an uncertainty rationale to every sub-call return is likewise null (−0.16).

### 3.5 Harness × reasoner scale

The harness ranking is not fixed: it flips with reasoner capability. Holding the VLM at 27B and varying only the reasoner (Table 3) reveals that the best scaffold for the weakest reasoner is the worst for the strongest.

**Table 3.** Best harness by reasoner, VLM fixed at 27B, $n{=}8$ val.

| Reasoner | RLM | ReAct | CodeAct | Best |
|---|---|---|---|---|
| Qwen3-8B (older gen, text-only) | 11.73 ± 2.96 | **15.79 ± 2.03** | 9.50 ± 1.44 | ReAct |
| Qwen3.5-4B | **21.09 ± 3.16** | 15.66 ± 4.73 | 15.66 ± 3.00 | RLM |
| Qwen3.5-9B | **24.54 ± 5.30** | 21.01 ± 4.63 | 24.26 ± 4.68 | RLM ≈ CodeAct |
| Qwen3.5-27B | **39.38 ± 1.49** | 25.16 ± 4.60 | 36.96 ± 5.25 | RLM ≳ CodeAct ≫ ReAct |

For the weakest reasoner the simplest scaffold wins — ReAct (15.79) leads RLM (11.73), which leads CodeAct (9.50), whose append-only growing-code context is hardest to drive. By 9B, the REPL scaffolds overtake ReAct, with RLM (24.54) and CodeAct (24.26) tied. At 27B, RLM (39.38) and CodeAct close at the top while ReAct (25.16) falls 12–14 points behind. CodeAct scales hardest of the three: worst at 8B, level with RLM by 9B, and within noise of RLM at 27B. Its reasoner-scaling slope from 4B to 9B (+8.6) is roughly 2.5× steeper than RLM's (+3.5).

Two further factorial swaps locate the bottleneck.

**Perception budget.** Holding the reasoner fixed and swapping its homogeneous VLM up to the 27B VLM lifts accuracy ~8 points at both reasoner sizes — **+7.9 at 9B** (Welch $t{=}3.54$, 95% CI [+3.4, +12.3]) and **+8.6 at 4B** ($t{=}4.96$, 95% CI [+5.2, +12.0]). The lift's consistency across reasoner size is the signature of a perception, not orchestration, bottleneck: the scaffold is perception-budget-bound for mid and small reasoners.

**The REPL converts reasoning into perception.** The complementary swap — a strong 27B reasoner on a weaker 9B VLM, against a weak 9B reasoner on the 27B VLM — separates the harnesses by what they are bound on. With a stronger reasoner and weaker perception, RLM gains **+10.3** and CodeAct **+6.2** (both reasoning-bound), while ReAct *loses* **−3.05** (perception-bound). RLM and CodeAct write Python around `batch_look` — cropping to the evidence region, zooming, compositing, doing coordinate arithmetic — so a stronger reasoner produces better-targeted sub-images and extracts more even from a weaker VLM. ReAct has no such actuator: its perception ceiling is the VLM's whole-page acuity, and a stronger reasoner has no way to direct finer-grained perception. The REPL crop/zoom loop is the mechanism that turns reasoning capability into perception quality; remove it and reasoning can no longer buy perception.
