# Report outline — working document (Level 2)

> Term-project report (COGS 560), ACL format, ~8pp (a bit over OK).
> Locked spine in `throughline.md`. **Two pillars, decoupled in scope:**
> Pillar 1 = active perception lifts document agents (general; scientific arc
> challenge→hypothesis→proof→mechanism→generalization; firm contribution).
> Pillar 2 = can fine-tuning lift a ≤8B agent further (preliminary, **deferred /
> light**). Drafting order: Pillar-1 sections first.
>
> **Confidentiality (hard rule):** cite *public papers* only (arXiv ids), never
> the Obsidian vault.
>
> **Scope (hard rule):** research findings + methodology ONLY — NO
> development/implementation war-stories (bugs, infra/tooling, data-corruption,
> debugging, disk/PNG leaks, eval mistakes-and-corrections). Clean experiment +
> verdict.
>
> **Voice (hard rule):** write like a paper for a cold reader who knows nothing
> of this conversation or the project's history. Declarative, self-contained. No
> paraphrase of our discussion, no dialectic, no meta framing ("two pillars",
> "this section", "as we noted"), no first-person process narrative.
>
> **Banned words / style (hard rule):** NEVER use "load-bearing" or "delve into".
> "Perceive–Reason–Code" is NOT a common term — it is only the NAME of the prior
> competition entry; for the general agent type use "code-REPL agents" /
> "active-perception scaffold". **OPD = future-work mention only** (no time to run
> it). **Ablations are RLM-only and presented SEPARATELY from the solver
> comparison** — Table 1 = distinct harness designs; Table 2 = RLM ablations
> (one lever at a time). Ablations show a component/lever's impact, not a design
> contrast.

## Working title (candidates) — active-perception framed
- "Active Perception with Code: A Harness Evaluation of Active-Perception VLM Agents for
  Document Question Answering"
- "Crop, Look, Reason: Why a Code REPL Is the Lever for Document-VQA Agents"
- "Seeing by Coding: Active Perception Lifts Document Agents Across Model Families"

## Top-line contributions (intro bullet list)
1. **Active perception is the dominant lever on document-agent accuracy.** A
   controlled evaluation (DocVQA-2026 val, n=8, frozen 27B VLM) shows REPL
   active-perception scaffolds (~39 ANLS) far surpass tool-based ReAct (~25) and
   no-REPL VLM agents (~20), and match or exceed much larger frozen models
   (Gemini-3-Pro 37.5, Gemini-3-Flash 33.75 on validation — low for their size).
2. **The mechanism, isolated.** A code REPL and the VLM perception tool are
   each essential; the advantage concentrates on *visually dense* pages because
   cropping recovers detail whole-page reads miss (no-crop −11.2 eng-draw /
   −17.5 poster); OCR-free visual perception is decisive (−25.5pp; eng-draw & maps → 0);
   the REPL converts reasoning capability into perception quality.
3. **Cross-family, capability-gated generalization.** The advantage holds across
   model families (Qwen3.5-27B and Gemma-4-31B) and is gated by base capability
   (reasoning+coding+vision), not size: Qwen3.5-4B clears the bar; Gemma-4-E4B and
   older Qwen3-8B are clean negative controls. [+ second-dataset point, pending]
4. **The trainable-scaffold argument** (the §4 hinge): the append-only member of
   the winning family preserves a growing-prefix trajectory — the structure
   established weight-level training assumes — at no measured accuracy cost,
   making it the principled fine-tuning target. [named contribution]
5. **A preliminary ≤8B training study** (deferred/light): SFT/GRPO/OPD on
   Qwen3.5-4B inside the trainable scaffold; honest preliminary results and one
   structural ceiling — long-document runaway (runaway rate rises with page
   count, r≈0.70) — the same growing-prefix property that makes the scaffold
   trainable.

---

## ⏸️ PARKED / pending (waiting on data or user)
- **§3.6 homog-vs-cross split — DONE (2026-06-13).** Restructured into two
  config-consistent tables: **Table 3** (reasoner scale at fixed 27B perception:
  Qwen3-8B 11.73/15.79/9.50†, 4B 21.09/15.66/22.34, 9B 24.54/21.01/24.26†, 27B
  39.38/25.16/39.53) and **Table 4** (cross-family homogeneous: Qwen3.5-27B
  39.38/25.16/39.53, Gemma-4-31B 32.50/18.44/29.25†, Gemma-4-E4B 7.34/6.09/7.66†).
  8B = coding-gate control (Table 3, perception fixed); E4B = small-model-floor
  control (Table 4, homog). §3.5 audited — perception-budget (v1→v2 swap) and the
  v2↔v3 factorial are already config-explicit. STANDING RULE: never mix homog and
  cross-model numbers in one table.
- **codeact_chat promotion (partly applied):** 27B (39.53) & 4B/27B (22.34) are
  FINAL (codeact_chat MDP loop); 9B / Qwen3-8B / Gemma CodeAct cells still old impl
  (†, provisional, re-run pending). Table 1 + Table 3 4B/27B already updated; the
  rest stays provisional until re-run.
- **Second-dataset generalization point (§3.6)** — pending 27B run.
- **§5 Pillar-B section — REWRITE pending results** (per mining/pillarB-capture.md +
  the new outline §5). The drafted `05-training-small-agent.md` is STALE: (a) lists
  7 pool datasets, missing MMLongBench-Doc (8th); (b) headlines Phase-A 15.3→20.6
  (+5.3) which is OLD-scaffold, non-comparable to corrected baseline 22.34 — Phase-B
  full-val PENDING; (c) frames the data as "MMLongBench transfer" + "pool built-after-
  SFT, not-yet-trained" — now the NEW approach IS rejection-sampling SFT/SeqKD on
  codeact_chat-27B rollouts FROM the pool. Rewrite when Phase-B lands. Add pass@k +
  the rejection-sampling figure (trajviewer /stats).
- **§4.4 + §5.3 prose softening — TODO (page-count finding established).** Section
  drafts still say runaway = "growing-prefix cost" / append-only context growth.
  Decouple: runaway = capability×long-doc (effort length-flat, 27B unaffected);
  §4.4 ceiling = theoretical only → makes "prefix preservation is free" stronger.
- **pass@k** — emitted at n≥2; add column/figure once SFT/RL runs report it.
  Reference base-4B = codeact_chat 4B/27B **22.3 → pass@8 55.0** (docvqa repo).
- **Base-4B not yet evaluated on docvqa-verl (corrected pipeline).** `baseline-
  cleanval-n4` (15.3, thinking-ON, 2026-06-09) and `base-4b-val-n4` (thinking-ON,
  n=2, no results.json) are STALE/pre-correction — IGNORE. Need a corrected
  base-4B eval (thinking-off, n≥4, pass@k) in this repo. ⚠️ The Pillar-B mining
  (runaway 38.8% / r≈0.70 in mining/pillarB-results.md + the pageCount cuts) was
  computed on those stale thinking-on base-4b dirs → RE-VERIFY once the corrected
  base eval lands; treat the runaway magnitudes as provisional.
- **GRPO: RL-train SFT-init AND untrained base (§5.4)** — pending RL (details from user).
- **renderers prefix-invariant citation** — GitHub blog, not arXiv; resolve in bib pass.

## ⚠️ Cross-cutting issues / standing caveats (honesty-critical)
- **TERMINOLOGY (supersedes "recursive" wording below).** The perception
  mechanism is a DEPTH-1 call to a frozen VLM used as a perception tool — NOT
  recursion, NOT delegation. Never call it "recursive." Use "active perception /
  on-demand VLM-tool query." RLM's defining property = **compact/managed
  REPL-history context** (re-rendered each turn), unique to it; CodeAct = append-only
  full REPL history. Reader-friendly solver names: RLM, CodeAct, ReAct, Multimodal
  REPL agent (was direct_vlm), Raw multi-image VLM, No-scaffold baseline, OCR-only
  RLM, RLM + general sub-agent / + OCR / without crop-zoom / + display channel.
  rvlm_rationale is DROPPED from the report. (Older "recursive" phrasings below are
  stale; the drafted sections are already corrected.)
- **ANLS threshold — RESOLVED: binary ANLS @ 0.9, both pillars** (verified
  `docvqa/metrics.py:142` threshold=0.9; `reward.py:4-6`). Numbers comparable;
  define once in setup.
- **Report pass@k alongside ANLS** (user request) for baseline / SFT / RL models —
  already emitted to `results.json` at n≥2; SFT+RL pipelines will report it going
  forward. Add a pass@k column/figure to §5 results + state the protocol in §3.1/§5.1.
- **Visual density, not page length.** The active-perception advantage tracks
  visual density (cropping-recoverable), NOT document length; "advantage grows
  with pages" is empirically false (it vanishes on 40+pg with a strong VLM) —
  must not appear.
- **Model generalization = cross-family + capability-gated.** "Across model
  families (Qwen3.5 + Gemma-4) at capable scale"; negative controls = Gemma-4-E4B
  and older Qwen3-8B. Not "size alone."
- **CodeAct numbers = older impl, re-run pending** (user will re-run). Flag every
  CodeAct cell provisional; lean on clean RLM as the essential active-
  perception number; the Qwen3.5-27B RLM≈CodeAct tie (§4) is provisional on the
  CodeAct side until re-run.
- **Frontier comparison = "matches or exceeds"** (not a hard beat); frame as
  low-for-their-size. **VALIDATION ONLY — never mix val/test:** Gemini-3-Pro 37.5
  + Gemini-3-Flash 33.75 (official ICDAR-2026 val baselines); DROP GPT-5.2 / GPT-5
  Mini (test-only, no val number).
- **Slice-analysis caveat.** Per-category & page-bucket slices computed on
  cross-model matched-config runs (27B-homog baseline run dirs not on disk — only
  doc-level numbers survive); use for *deltas*, cite canonical n=8 for absolutes.
- **Dataset generalization — claimed, evidence pending** (2nd-dataset 27B run in
  progress). If it doesn't land: soften to "one-benchmark evidence; expected to
  generalize." Never assert without data.
- **Thinking OFF in both pillars** (enable_thinking=false) — consistent. **Must be
  explained carefully in the setup (§3.1) so it does NOT read as a methodological
  error:** solvers parse only code blocks, so the model still writes free-text
  reasoning before/between blocks (and dspy solvers carry a `reasoning` signature
  field) → thinking-off *relocates* reasoning, it doesn't remove it; empirically
  no-think is similar/slightly-lower vs thinking-on (at 27B thinking was also
  hang-prone). So thinking-off ≠ answering without reasoning.
- **Backbone pivot.** Proposal said Qwen3-8B; Pillar-2 actual = Qwen3.5-4B (the
  ≤8B leaderboard-slot backbone). State in §5.
- **Don't over-anchor to the original proposal** (its ≤8B-SOTA sprint framing).

---

# §1 Introduction · ~1pp · 🟠 (body 🔵; contributions/title finalize last)
- The challenge: DocVQA-2026 documents are long, multi-page, high-resolution,
  largely visual; OCR-and-text pipelines can't reach the content; locating the
  right page/region precedes reading.
- Scale alone doesn't crack it: bare setups score low and even frontier frozen
  models only plateau (low for their size).
- This work: (Pillar 1) active perception via a code REPL is the lever — isolate
  the mechanism, show it generalizes across families; (Pillar 2, brief) a
  preliminary look at training a ≤8B agent inside the trainable member.
- Contributions (the bullets above). Roadmap sentence.

# §2 Background · ~0.9pp · 🔵 (drafted; needs reframe)
- **2.1 DocVQA-2026 task & why it's hard.** 8 categories; long multi-page high-res
  visual docs; no train split (val 25 docs/80Q, test 48/160); binary ANLS@0.9
  (define, point to setup); OCR unreliable on visual content; multi-hop/arithmetic.
  Cite DocVQA (2007.00398), ANLS/ST-VQA (1905.13648).
- **2.2 Perceive-Reason-Code agent + frozen VLM + prior entry.** The agent concept
  (code-LM reasoner directing a frozen VLM for active perception (a VLM perception tool)). Prior
  competition entry (27B reasoner, active-perception harness, 8B–35B tier) as
  **one instance** of the family — prior-work boundary: scaffold reused, evaluation
  + training are the new work. No RLM-as-solution framing.

# §3 Active Perception: A Harness Evaluation (Pillar 1) · ~3pp · 🟠 (drafted §3 STALE → reorder to this arc)
> Scientific arc. The drafted `03-harness-evaluation.md` has the data but the flat
> matrix order; reorder to challenge→hypothesis→proof→mechanism→generalization.
- **3.1 Evaluation setup** (shared protocol home; covered carefully). Metric =
  binary ANLS@0.9 (official DocVQA-2026, defined once here); val 25 docs/80Q;
  n=8 trials, per-question micro-average; frozen Qwen3.5-27B VLM. **Thinking OFF
  (enable_thinking=false) — explain so it doesn't read as a mistake:** (i)
  empirically similar / slightly-lower accuracy vs official thinking → negligible
  cost; (ii) solvers parse only Python code blocks, so the model still writes
  free-text reasoning before/between blocks → disabling native `<think>`
  *relocates* reasoning, it does not remove it; (iii) dspy-implemented solvers
  carry an explicit `reasoning` signature field → reasoning conceptually on
  regardless. ⇒ thinking-off ≠ answering without reasoning; a stability/throughput
  choice, not a handicap.
- **3.2 The challenge — bare and frontier both fall short.** No-scaffold/raw-VLM
  ~18–22; frontier frozen Gemini-3-Pro 37.5 / Gemini-3-Flash 33.75 on val (low for size).
  Establishes: scaling the model isn't the answer.
- **3.3 Harness design space + the active-perception hypothesis.** Taxonomy:
  raw-VLM / no-scaffold / ReAct (tools, no REPL) / REPL active-perception family
  (RLM, CodeAct) sharing tools + the `batch_look` perception tool. Hypothesis: the REPL
  active-perception loop, not scale or a fixed tool loop, governs accuracy.
- **3.4 Result — active perception dominates** (Table 1, 8-solver matrix + anchors).
  REPL family ~39 ≫ ReAct 25.16 ≫ direct/raw/official 18–22 ≫ OCR-only 13.91;
  matches/exceeds frontier. Three tiers separated by ≫ std. (CodeAct 39.53
  provisional-older-impl; RLM 39.38 reference.)
- **3.5 Mechanism — which components matter.**
  - REPL + the VLM perception tool each essential (drop either → collapses; Table 2).
  - **Cropping = the visual-density lever:** no-crop −2.5 overall / −11.2 eng-draw
    / −17.5 poster; rvlm−react advantage by category (eng-draw ≈ +30 … science_paper
    ≈ +5). Density, not length.
  - OCR-free decisive: −25.5pp; eng-draw & maps 0/10; collapse graded by density.
  - REPL converts reasoning→perception: strong-LM/weak-VLM swap RLM +10.3 / CodeAct
    +6.2 / ReAct −3.05; perception-budget +7.9(9B)/+8.6(4B).
  - Effort ≠ accuracy: corr(iters, acc) ≈ −0.31.
  - **Figure (qualitative active perception):** the 181-pg NVIDIA survey→locate→
    crop-verify→compute→submit trace (catches a VLM misread); + RLM-vs-ReAct
    contrast (science_poster: RLM crop+compute → 30.2 correct vs ReAct whole-page
    → 0.48 wrong). [from mining/pillarA-trajectories.md]
- **3.6 Generalization — cross-family + capability-gated. TWO config-consistent
  tables (never mix homog/cross):** Table 3 = reasoner scale at fixed 27B
  perception (Qwen3.5 4B 21.1 / 9B 24.5 / 27B 39.4, RLM ≥ ReAct; CodeAct ties at
  4B 22.3 & 27B 39.5; Qwen3-8B coding-gate control: ReAct 15.8 > RLM 11.7 w/ strong
  perception). Table 4 = cross-family homogeneous (Qwen3.5-27B 39.4/25.2/39.5,
  Gemma-4-31B 32.5/18.4 +14pp mirrors Qwen, Gemma-4-E4B ~6–8 ≈ baseline =
  small-model-floor control). Gate = base capability (reasoning+coding+vision),
  not size. [+ 2nd-dataset 27B point — pending].

# §4 From Harness to Trainable Scaffold · ~1pp · 🔵 — NAMED CONTRIBUTION (the bridge)
> Drafted `04-trainable-scaffold.md` is the house-style exemplar; needs (a)
> family-not-RLM framing, (b) tie scoped to Qwen3.5-27B, (c) **drop the
> POMDP/observability framing** (§4.1 "not observability" detour) — use the
> multi-turn-RL literature's framing directly.
- Frame via the literature (FoldAct, ReSum, renderers), NOT POMDP/observability
  (both scaffolds are POMDPs — irrelevant). Policy-gradient multi-turn RL assumes
  an **append-only token-prefix invariant**: prompt_{t+1} = prompt_t +
  completion_t + obs, so the rollout is one growing sequence and per-step
  log-probs come from a single masked forward pass. CodeAct preserves it
  (append-only transcript); RLM's context manipulation breaks it — rewriting
  history makes the **observation distribution policy-dependent and
  non-stationary** (FoldAct) and the rollout no longer one growing sequence
  (ReSum), so importance ratios + cross-turn credit become ill-defined and need
  bespoke repair. Cite FoldAct (2512.22733), ReSum (2509.13313), renderers/
  prefix-invariant.
- **Prefix preservation is free:** at Qwen3.5-27B the append-only scaffold ties
  the context-managing one (39.5 vs 39.4 — CodeAct side provisional, older impl).
  → train inside the append-only member of the winning family.
- **Accessible vs higher ceiling (ceiling-suppression, hedged).** RLM's compact
  context → higher *potential* ceiling on long horizons, but a frozen model is
  off-distribution in a context-managing harness → parity, not RLM≫CodeAct.
  Training-unlockable (ContextCurator 2604.11462, FoldAct). Hedge: hypothesis,
  magnitude unproven; ReSum = contrary (frozen compaction can help). **CEILING
  ARGUMENT IS THEORETICAL ONLY** — empirically the append-only context-growth cost
  does NOT materialize at these doc lengths (page-count analysis: effort
  length-flat ~12–16 turns, RLM no long-doc edge, 27B unaffected), which makes
  "prefix preservation is free" STRONGER. Do NOT claim runaway = append-only growth
  (it's capability×long-doc, §5.3). See mining/pageCount-{rlm-vs-codeact,per-solver}.md.

# §5 Training a Small Agent (Pillar 2) · ~1.5pp · 🟠 IN-PROGRESS (results PENDING; fill as they land)
> Full detail in mining/pillarB-capture.md. Honest; no committed training claim
> until Phase-B full-val lands. Pipeline/numbers being finalized by user.
- **Approach = rejection sampling → train.** No DocVQA train set with agent
  reference solutions exists → generate rollouts with the **CodeAct (codeact_chat)
  scaffold + 27B/27B teacher** on the data pool, keep the **correct** rollouts
  (rejection sampling), PEFT/LoRA-train the **4B student** on them. **TERMINOLOGY
  (verified from code = plain masked cross-entropy, no KL/logit loss):** call it
  "**SFT on rejection-sampled teacher trajectories — a form of sequence-level
  knowledge distillation (SeqKD)**" (teacher 27B ≠ student 4B); "RFT" an OK
  synonym; AVOID unqualified "knowledge distillation" (no KL to point to).
- **5.1 What's trained + setup.** Qwen3.5-4B + LoRA (PEFT); VLM frozen external;
  train==deploy scaffold; backbone pivot (8B→4B). **Hyperparameter sweep (LoRA
  rank, learning rate)**; each checkpoint quick-screened on **docvqa-mini (13Q
  representative subset)**, best → **full 80Q val**. Eval per §3.1 (binary
  ANLS@0.9, thinking-OFF). **Report pass@k alongside ANLS** (baseline/SFT/RL;
  already emitted to results.json at n≥2). Baselines: untrained-4B + leaderboard
  anchors. verl/FSDP.
- **5.2 Data pool** (see docvqa `docs/training-data-pool.md` + mining/pillarB-capture.md).
  No DocVQA train split → assembled from **8 sources** (cite each): DocVQA
  (2007.00398), InfographicVQA (2104.12756), ChartQA (2203.10244), MapQA
  (2211.08545), MP-DocVQA (2212.05935), TAT-DQA (2207.11871), SlideVQA
  (2301.04883), **MMLongBench-Doc (2407.01523)** [WAS MISSING — essential
  long-doc source]. Deliberately **page-count-diverse**; balanced sampler
  (~1500/source → ~11,464 prompts, seed 42); leakage-checked (dHash→pixel) vs val.
- **5.3 Results (PENDING).** Phase-B (codeact_chat rollouts, corrected-scaffold
  baseline 22.34) full-val PENDING; preliminary ≈ null. ⚠️ Phase-A's 15.3→20.6
  (+5.3) is on the OLD scaffold — **non-comparable, do NOT headline.** Honest
  obstacle = **long-document runaway** bounding the small agent (38.8% of base-4B
  rollouts hit a budget cap without submitting; r≈0.70 with num_pages). Per the
  page-count analysis this is a **capability × long-doc** effect (weak 4B can't
  navigate long docs under budget; 27B unaffected, effort length-flat) — NOT
  append-only context growth; decouple from the prefix property.
- **5.4 RL (planned, details pending from user).** GRPO; RL-train BOTH the
  SFT-init checkpoint AND the untrained 4B baseline, compared. pass@k reported.
  **RL MOTIVATION = oracle headroom (pass@1↔pass@k gap):** base-4B avg@1 22.3 but
  **pass@8 55.0** (codeact_chat 4B/27B; 27B 39.5→63.8; 4B-homog 16.3→47.5) — the
  capability is already there in *some* trial; RL's job is converting headroom into
  reliable pass@1. SFT moves avg@1 toward the ceiling but does NOT raise pass@k
  (in-domain SFT 15.3→19.0 avg@1, pass@4 flat ~34) = sharpens consistency, doesn't
  expand capability. Frame pass@k as the ceiling, training as closing the gap.
  ⚠️ NO pass@k *harness-ranking* claim (Pillar A) until the 27B matrix per-trial
  artifacts are re-run (deleted; only codeact_chat/vsearch/Gemma-31B retained).
  Pillar A headline (avg@1) unaffected. [detail: docs/results.md pass@k section]
- **Candidate figures (trajviewer /stats):** rejection efficiency (trials-to-first-
  correct), best-ANLS yield/question, page-count diversity, per-category; data in
  outputs/runs/*/trajectories.jsonl (manual screenshot or matplotlib re-derive).

# §6 Related Work · ~0.5pp · 🔵 (drafted; minor reframe)
- RL for LLMs: GRPO (2402.03300) + Dr.GRPO (2503.20783), DAPO (2503.14476), GSPO
  (2507.18071), CISPO (2506.13585).
- Distillation: OPD survey (2604.00626), GKD (2306.13649), MiniLLM (2306.08543), SeqKD.
- RL under agent-controlled / non-prefix context (emerging): FoldAct (2512.22733),
  ReSum (2509.13313), ContextCurator (2604.11462), AdaCoM (2605.30785); Sclar
  (2310.11324) optional.
- Agentic document QA: RVLM (2603.24224), MADQA (2603.12180), ARIAL (2511.18192),
  VISOR (2604.09508), MDocAgent (2503.13964), ORCA (2603.02438), SlideAgent
  (2510.26615), DocVStar (2604.13731), AgenticOCR (2602.24134); RLM (2512.24601);
  CodeAct; ReAct (2210.03629).
- DocVQA-family benchmarks: DocVQA (2007.00398), InfographicVQA (2104.12756),
  ChartQA (2203.10244), DUDE (2305.08455), MP-DocVQA (2212.05935), SlideVQA
  (2301.04883), MMLongBench-Doc (2407.01523), ANLS/ST-VQA (1905.13648).
- **Verify every arXiv id against arXiv before .bib.**

# §7 Conclusion & Future Work · ~0.4pp · 🟠
- Active perception in a REPL is what lets a document agent punch above its model
  scale, across families, gated by capability. The append-only member is the
  trainable target; training a small agent to realize the gain — past the
  runaway ceiling, via on-policy methods — is the open direction.

# Limitations (unnumbered) · ~0.3pp · 🟠
- val-only (small, no test if portal closed); CodeAct numbers older-impl
  (re-run pending); slice deltas on matched-config runs; dataset-generalization
  one-to-two benchmarks; Pillar-2 preliminary (n=4, single-checkpoint, RL
  smoke-test); long-document runaway / context growth bounds the append-only
  scaffold; the §4 ceiling-suppression account is an untested hypothesis.

---

## Drafting order
Pillar 1 first: §3 (reorder + add mechanism/generalization/figure) → §2 reframe →
§4 reframe (family-not-RLM) → §1 body → §6 related (minor). Pillar 2 (§5) light,
fill as results land. §7/Limitations last.

## Mined evidence (draw on when writing)
`project/report/mining/`: pillarA-slices.md (category×harness, page-buckets,
effort, OCR-by-category), pillarA-trajectories.md (curated active-perception
excerpts + RLM-vs-ReAct contrast + the recommended main figure), pillarB-results.md
(master training table, runaway quantification).
