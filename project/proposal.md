# Term Project Proposal — COGS 560

**Title:** Teaching a Small LLM to Perceive, Reason, and Code: Fine-Tuning an 8B Agent for Document Visual QA

## TL;DR

I propose to push the state of the art in the ≤8B tier of ICDAR 2026 DocVQA by LoRA-fine-tuning an 8B model inside the *Perceive-Reason-Code* agent scaffold — the same REPL-plus-VLM setup I used at 27B in my prior competition entry. Training will explore GRPO-style RL from verifier reward and on-policy distillation from a 27B teacher, individually and combined in a single update by summing their advantages. The target is to clearly surpass the current ≤8B best (0.1875 ANLS) and, as a stretch, to close in on the 8B–35B leader (0.3750) at a fraction of the parameters.

## 1. Problem and Motivation

Document Visual Question Answering (DocVQA) requires answering free-form questions over heterogeneous documents — business reports, engineering drawings, infographics, maps, scientific papers, posters, and slides — that mix text, tables, charts, and figures. Unlike single-image VQA, real documents are long, multi-page, and demand *active perception*: locating the right region, reading fine-grained numbers, and composing evidence across pages. The ICDAR 2026 DocVQA competition formalizes this with 8 document categories and scores submissions with ANLS (Average Normalized Levenshtein Similarity). Crucially, the leaderboard splits entries into three parameter-budget tiers — **>35B**, **8B–35B**, and **≤8B** — so that small-model progress is evaluated on its own merits.

The tiered leaderboard reveals a sharp capability cliff at the small end. The current state of the art is **0.68** (Gemini-based ensembles) in the >35B tier and **0.38** in the 8B–35B tier, but only **0.19** in the ≤8B tier — the best ≤8B entry is a zero-shot pipeline around Qwen3.5-4B, and every fine-tuned 8B entry scores below 0.11. **The goal of this project is to push the state of the art in the ≤8B tier by fine-tuning an 8B model inside a code-executing agent.** Training method is a means to that end, not the object of study.

Closing this gap matters beyond the leaderboard: models at the 27B+ scale require multi-GPU serving that is out of reach for most practitioners and this project specifically, whereas an 8B model with LoRA adapters is trainable and deployable on a single GPU. A strong 8B agent is therefore the configuration most likely to transfer into real use.

## 2. The Benchmark

DocVQA 2026 (`VLR-CVC/DocVQA-2026` on HuggingFace) is designed to be hard in ways that defeat naïve VLM pipelines:

- **Evenly distributed across 8 heterogeneous categories** — business reports, comics, engineering drawings, infographics, maps, science papers, science posters, slides — each with 10 val / 20 test questions. A single strategy rarely works across all categories.
- **Long, multi-page documents.** Val averages 36 pages/doc and test 33 pages/doc, with business reports reaching 280+ pages. Questions often require locating the right page first — full-page visual inspection everywhere is infeasible.
- **High-resolution images.** Test-set pages average ~8.5M pixels, and some maps and posters exceed 240M pixels (`maps_5` reaches 15,695×15,695). Reading small text or thin lines demands targeted crops rather than a single downsampled view.
- **No training split.** The competition provides only `val` (25 docs / 80 questions) and `test` (48 docs / 160 questions); participants must rely on zero-shot prompting, transfer from related datasets, or teacher-generated trajectories — motivating the on-policy distillation approach in this project.
- **Strict answer formatting.** ANLS with threshold 0.80 on short answers penalizes small formatting mistakes (units, decimals, date formats, "Unknown" for unanswerable questions).
- **OCR is unreliable on visual content.** Maps, science posters, and some slides yield little or no extractable text, so the agent cannot shortcut through text search and must actually look.
- **Questions frequently require multi-hop reasoning and arithmetic** — e.g. *"Assuming Q3 revenue is distributed by customer in the same way as Q3 backlog, and customers are equally distributed, what would be the Q3 revenue of the second-largest customer?"* (from the `slide` category).

Together these properties explain why the leaderboard plateaus so sharply at small scale: a single forward pass through an 8B VLM cannot both navigate the document and reason over extracted numbers with high precision. Success requires an agent that actively perceives and computes — which is precisely the scaffold this project fine-tunes.

## 3. Relation to Prior Work

I previously participated in the ICDAR 2026 DocVQA challenge and submitted *Perceive-Reason-Code*, an agentic system built around Qwen3.5-27B that scored 0.3563 in the **8B–35B** tier — tied for second with `ARGUS_Qwen3.5_27B` and behind `MLLM (small)` at 0.3750. That submission used the model **zero-shot** — no fine-tuning — so its score reflects prompting, tool design, and per-category heuristics rather than learned weights. This project reuses the agent scaffolding but addresses a different and harder setting: the **≤8B tier**, with a much smaller backbone and weight-level training as the central contribution. Prior code and prompts serve as the starting point; all novel work (training pipeline, reward design, distillation signal) is done for this course.

## 4. Proposed Method

**Backbone.** Qwen3-8B (or a comparable ≤8B multimodal-capable model), trained with **LoRA** adapters so the experiments fit on a single A100/H100 and remain reproducible for the final report. The 8B size is a deliberate choice constrained by available compute — full fine-tuning of the 27B prior backbone was not feasible in this setting — and it places the contribution squarely in the ≤8B tier where the headroom is largest.

**Agent scaffold.** The Perceive-Reason-Code scaffold is reused: the LLM runs inside a persistent Python REPL with tools for VLM-based visual inspection and BM25 search over OCR text, and solves a question by writing code across an *explore → locate → extract → submit* loop. Exact tool set and iteration budget may be revisited during the project (e.g., collapsing sequential and batched VLM calls into a single tool). Full implementation details are in the reference repository at <https://github.com/bdsaglam/docvqa>.

**Training.** The objective is to maximize ANLS on the ≤8B competition tier. I will explore two on-policy training approaches and their combination, all on top of the same scaffold and LoRA budget, and select the strongest configuration for the final submission:

1. **Reinforcement Learning (GRPO and variants) from Verifier Reward.** Use the answer-level ANLS as a verifiable reward, plus shaping rewards for well-formed tool calls and successful code execution. GRPO is a natural fit because agent rollouts are long, multi-turn, and expensive — group-relative advantages avoid a learned critic. I will explore recent variants (e.g., Dr. GRPO, GSPO, DAPO-style length/diversity controls).  

2. **On-policy distillation (OPD).** Sample trajectories from the *student's own* policy, score each step against the teacher's distribution (token-level KL) or against a final-answer reward from the teacher, and update the student.   

3. **Combination.** RLVR and OPD can be combined: both operate on trajectories sampled from the student, and their advantages can simply be summed (with a tunable weight) before the policy-gradient step — the student is pushed toward high-reward rollouts and toward the teacher's distribution in the same gradient. 

**Tooling.** I will evaluate `trl`, `verl`, and `prime-rl` and pick per-experiment based on multi-turn tool-using rollouts and LoRA support.

**Data.** Primary: the competition's validation split (the only set with public ground truth) for in-distribution evaluation. For training prompts and teacher signals I will use questions drawn from public DocVQA-family datasets (DocVQA, MP-DocVQA, InfographicVQA, ChartQA, SlideVQA, DUDE) to broaden document coverage and to avoid any val leakage. The 27B Perceive-Reason-Code agent (or a Gemini-based agent) serves as the teacher for OPD.

## 5. Experimental Plan

| Stage | Deliverable | Eval |
|---|---|---|
| S0. Baseline | Qwen3-8B zero-shot in the agent scaffold | ANLS on val (8 categories) |
| S1. RL (GRPO) | LoRA checkpoint trained with answer-level reward | ANLS + tool-use statistics |
| S2. OPD | LoRA checkpoint trained with teacher-guided on-policy loss | ANLS + sample efficiency curve |
| S3. RL + OPD | Combined-objective checkpoint (sequential and/or mixed loss) | ANLS + ablations of the two signals |
| S4. Analysis | Per-category breakdown, error taxonomy, ablations | Paper-ready figures |

**Primary success criterion:** achieve state of the art in the ≤8B tier — clearly surpass the current best (0.1875) on the leaderboard and, as a stretch target, close in on or match the 8B–35B tier leader (0.3750, `MLLM (small)`) at 8B parameters. RL, OPD, and their combination are tools toward this end; whichever configuration yields the strongest score is the one the final report will feature.

## 6. Risks and Mitigations

- **Long rollouts make RL expensive.** Mitigation: cap REPL iterations; use OPD as a cheaper bootstrap before running full GRPO; choose a training library that natively supports multi-turn tool rollouts.
- **GRPO cold start — zero reward variance.** If the 8B model never solves any training sample across its rollouts, every sample in the group has the same (zero) reward, GRPO's advantages collapse, and no gradient signal is available. Mitigations: (a) bootstrap first with OPD from the 27B teacher, which provides signal regardless of student success; (b) run SFT with rejection sampling — generate many rollouts from the student and/or teacher, keep only trajectories that score above an ANLS threshold, and SFT on those — before switching to GRPO; (c) curriculum the training prompts from easier DocVQA-family datasets (DocVQA, InfographicVQA, ChartQA, SlideVQA, DUDE) where the student is likelier to score non-zero, then transfer to DocVQA 2026's harder distribution.
- **VLM cost dominates during training.** Mitigation: batch VLM calls; during fine-tuning, freeze the VLM and only train the LLM.
- **LLM variance and small val set.** Mitigation: self-consistency at eval time; report seeds; use per-category scores, not only the headline.

## 7. Deliverables and Timeline

Final report (≥8 pages, ACL format), code, trained LoRA adapters, and a reproducible evaluation script. The validation split (the only split with public ground truth) is the primary, always-available evaluation target. Test-set ANLS will additionally be obtained by uploading predictions to the competition's evaluation interface while it remains open, and — if available — both val and test numbers will be reported side by side. Final report due **June 15, 2026**.

## References (indicative)

1. Mathew et al. *DocVQA: A Dataset for VQA on Document Images.* WACV 2021.
2. Shao et al. *DeepSeekMath: GRPO for Mathematical Reasoning.* 2024.
3. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025.
4. Liu et al. *Understanding R1-Zero-Like Training (Dr. GRPO).* 2025.
5. Agarwal et al. *On-Policy Distillation of Language Models.* 2024.
6. Zhang et al. *Recursive Language Models.* 2025.
7. Qwen Team. *Qwen3 / Qwen3.5 Technical Report.* 2026.
8. Docling Team. *Docling Technical Report.* 2024.
9. ICDAR 2026 DocVQA — Task 1 leaderboard. https://rrc.cvc.uab.es/?ch=34&com=evaluation&task=1
