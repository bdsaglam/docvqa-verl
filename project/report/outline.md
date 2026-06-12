# Report outline — working document

> Term-project report (COGS 560), ACL format, ~8pp (a bit over OK).
> Spine: **honest empirical investigation**, two pillars — (A) agentic-harness
> evaluation, (B) weight-level training of a 4B agent. Status: SFT done (modest
> lift), RL/OPD just started (preliminary). Drafting order: all 🔵 sections first.
>
> **Confidentiality:** report cites *public papers* only (arXiv ids), never the
> Obsidian vault. Vault is our context source; citations resolve to arXiv.
>
> **Scope (hard rule):** report covers research findings + methodology ONLY —
> NO development/implementation war-stories (bugs we fixed, infra/tooling issues,
> data-corruption discoveries, parity-debugging, disk/PNG leaks, eval mistakes-
> and-corrections). Present the clean experiment and its verdict, not how we got
> the pipeline working.
>
> **Voice (hard rule):** write like a paper, for a cold reader who knows nothing
> of this conversation or the project's development history. State content
> directly and self-contained. Do NOT paraphrase what we said in discussion, do
> NOT narrate a dialectic ("a natural first hypothesis… this is incorrect"), do
> NOT use meta framing ("this section makes the case", "the two pillars", "as we
> noted"). No first person process narrative. Declarative technical exposition.

## Working title (candidates)
- "Teaching a 4B Agent to Perceive, Reason, and Code: A Harness Evaluation and
  Training Study for Document VQA"
- "Which Scaffold, Which Signal? Evaluating Agentic Harnesses and Weight-Level
  Training for ≤8B Document VQA"

## Top-line contributions (intro bullet list)
1. A controlled **evaluation of agentic harness designs** for DocVQA (8 solvers,
   n=8) isolating the two load-bearing mechanisms: a code REPL + recursive VLM
   perception; the OCR-free result; harness-rank flips with reasoner scale.
2. The **append-only-trajectory argument**: why a prefix-preserving CodeAct
   scaffold — not the (marginally stronger) context-mutating RLM — is the
   principled, TITO-correct target for SFT/distillation/RL. [named contribution]
3. A **controlled SFT study** — transfer-vs-in-domain data and undertraining-vs-
   memorization, on adequately-powered evaluation — establishing that trajectory
   SFT yields a *modest, training-dependent* lift (15.3%→20.6% ANLS, +5.3,
   p≈0.04) and that memorizing teacher trajectories does not transfer.
4. A method-agnostic **DocVQA-family training-data pool** (constructed substrate
   for ongoing work, not yet trained on) + RL/OPD design and preliminary GRPO
   results.

---

## ⚠️ Cross-cutting issues to resolve BEFORE writing results (honesty-critical)
- **ANLS threshold — RESOLVED, both pillars = binary ANLS @ 0.9.** Verified:
  `docvqa/metrics.py:142` scores with `threshold=0.9` (the `0.80` is a stale
  helper default / dataset-doc mention, not the scoring path); `reward.py:4-6`
  confirms "Evaluation stays binary@0.9." Matches the official DocVQA-2026
  metric (binary ANLS@0.9). → A and B numbers ARE comparable; state the single
  metric definition once in §6. (Minor: confirm the §3 harness matrix was run
  through the 0.9 scoring path, not a 0.80 variant.)
- **Number-regime conflict in the harness repo.** Current source of truth =
  `docs/results.md` (rvlm **39.38%±1.49**, n=8, mean-per-trial, thinking-off).
  Stale `CLAUDE.md` / SC-8 submission numbers (48.8 val / 39.0 test) are a
  *different metric* (SC-8 voted) — use only as the labeled competition entry.
- **Competition standing number.** Proposal says the prior 27B entry scored
  **0.3563** (8B–35B tier, tied-2nd, behind 0.3750). rvlm.md SC-8 says 39.0 test.
  Reconcile which is the official leaderboard number before citing.
- **Thinking setting differs across pillars.** Pillar-A 27B CodeAct headline is
  **no-think** (enable_thinking=false; thinking is worse + hang-prone at 27B).
  Pillar-B 4B training/eval runs **thinking ON** (the verl agent_loop has no
  separate reason field, so native thinking is its only reasoning channel).
  → §6 must state each setting explicitly so the pillars don't look contradictory.
- **Backbone pivot.** Proposal said Qwen3-8B; actual = **Qwen3.5-4B** (the model
  holding the ≤8B leaderboard slot). State the pivot + rationale in §5.

---

# §1 Introduction  · ~1pp · 🟠 (body 🔵, contributions finalize last)
- The ≤8B capability cliff (leaderboard: >35B ~0.68, 8B–35B ~0.38, ≤8B ~0.19).
- Reframe: not a SOTA sprint but two questions — (a) *which harness design best
  converts frozen-VLM + code-LLM into DocVQA accuracy?* (b) *can weight-level
  training lift a 4B agent past its own zero-shot self inside that harness?*
- Why honest/eval-forward: small val set, expensive rollouts, training ongoing.
- Contributions (the 4 bullets above). Roadmap sentence.

# §2 Background  · ~1pp · 🔵
- **2.1 DocVQA 2026 benchmark.** 8 categories; long multi-page, high-res docs;
  no train split (val 25 docs/80Q, test 48/160); ANLS metric (threshold — see
  §6); OCR unreliable on visual content; multi-hop/arithmetic questions.
- **2.2 Perceive-Reason-Code idea + prior entry.** The agent concept; the prior
  competition submission (Qwen3.5-27B, RLM harness, 8B–35B tier). This sets the
  prior-work boundary: scaffold reused, weight-training is the new work here.

# §3 Agentic Harness Evaluation (Pillar A)  · ~2pp · 🔵
- **3.1 Harness taxonomy.** RLM (recursive VLM via `batch_look`, OCR-free,
  context-compacted) / CodeAct (append-only twin, identical tools) / ReAct
  (no REPL) / direct-VLM (pixels in own context) / raw-VLM / OCR-only control.
  All share a frozen Qwen3.5-27B VLM endpoint.
- **3.2 Main result — the 8-solver matrix** (Qwen3.5-27B, val 80Q, n=8,
  thinking-off). Three separated tiers (gaps ≫ std):
  - visual-recursive: rvlm 39.38±1.49, codeact_chat 39.53±2.83 (TIED w/ rvlm,
    n=8, no-think — the corrected true-MDP chat loop; supersedes old dspy
    codeact 36.74), subagent 39.22, rationale 39.22, ocr 37.81, nocrop 36.88,
    hybrid 35.47
  - no-recursion: react 25.16, direct_vlm 22.34, raw_vlm 20.47, official 17.81
  - OCR-only floor: rlm_ocr 13.91
  - external anchors: Gemini-3-Pro 37.5 val/test; GPT-5.2 35.0 test.
- **3.3 The OCR-free result.** `rlm_ocr` −25.5pp; engineering_drawing & maps
  0/10 in all 8 trials → visual recursion does what OCR cannot. Both halves
  (REPL + recursive sub-call) load-bearing (dropping either collapses score).
- **3.4 Ablations.** crop/zoom category-specific (−2.5 overall, −11.2 eng-draw,
  −17.5 sci-poster); OCR adds nothing on vision (−1.6); display channel mildly
  harmful + churn (−3.9, 18.1 vs 13.0 iters); sub-call enrichment null (≈−0.16).
- **3.5 Harness × reasoner scale.** Rank-flips: 8B(text-only) ReAct 15.8 > RLM
  11.7 > CodeAct 9.5; 9B RLM 24.5 ≈ CodeAct 24.3 > ReAct 21.0; 27B RLM 39.4 ≳
  CodeAct 37.0 ≫ ReAct 25.2. Perception-budget lift (swap VLM→27B, reasoner
  fixed): +7.9 (9B), +8.6 (4B). The REPL converts reasoning into perception
  (v3 27B-LM/9B-VLM: RLM +10.3, CodeAct +6.2, ReAct −3.05).

# §4 From Harness to Trainable Scaffold  · ~1pp · 🔵 — NAMED CONTRIBUTION, the hinge
- Both RLM and CodeAct are POMDPs (full REPL state off-context). The axis that
  matters for *fine-tuning* is the **token-trajectory shape across turns**.
- **CodeAct = append-only / prefix-preserving:** system→user→asst→user→asst…,
  no earlier token rewritten; turn t is a prefix of turn t+1. Trajectory-as-
  prefix holds.
- **RLM = context-mutating:** history is a manipulable variable; rendered
  context (incl. system prompt) re-written each turn (compaction/RESET_HISTORY/
  variables_info sidecar). t+1 is not an extension of t. Prefix property broken.
- **Why it's load-bearing:** GRPO/PPO/GSPO + per-token distillation assume one
  growing token sequence → all per-step log-probs from a single masked forward
  pass, each action scored in the context that generated it. Context mutation
  breaks the shared prefix → separate forward passes per turn, importance ratio
  + cross-turn credit assignment no longer well-defined over one sequence.
- **No accuracy cost (updated):** the corrected append-only CodeAct (codeact_chat)
  TIES RLM at 27B (39.5 vs 39.4, n=8) — prefix-preservation is free. It is
  trainable with *established* RL; RLM is not. → CodeAct is the fine-tuning scaffold.
- **§4.4 Accessible vs higher ceiling (ceiling-suppression hypothesis).** RLM's
  compact context decouples #steps from window → higher *potential* ceiling on
  long horizons. But a frozen model (post-trained only on append-only chats)
  is off-distribution in a context-managing harness, so RLM's ceiling is
  *untapped* — hence parity, not RLM≫CodeAct. Evidence it's training-unlockable:
  ContextCurator (untrained curator < full-context; RL-trained > full-context),
  FoldAct (folding needs purpose-built losses). HEDGE hard: it's a hypothesis,
  magnitude unproven; contrary evidence = ReSum (frozen compaction can already
  help, benefit shrinks with scale). Verdict: append-only delivers full
  potential under established training; context-managing MAY have a higher but
  less-accessible ceiling → future work. Cost CodeAct pays = context growth
  (ties to §7.4 long-doc runaway + Limitations).
  - Lit verdict (from search): training-side claim [agentic RL assumes
    growing-prefix, breaks under manipulation] = WELL-SUPPORTED (FoldAct, ReSum,
    renderers/prefix-preservation). Inference-side net-underperformance claim =
    NOT supported (ReSum/ContextCurator/AdaCoM: frozen compaction often helps).
    Ceiling-suppression (potential unlocked by training) = the defensible middle.
- **Honest caveat (not "untrainable"):** emerging methods *do* target RL under
  context manipulation — FoldAct (2512.22733), ContextCurator/ActiveContext
  (2604.11462) — but they are new and not yet well-established. We deliberately
  choose the well-trodden prefix-preserving path over that research frontier;
  training RLM-style harnesses directly is future work.
- Cite: RLM (2512.24601), trajectory-as-prefix, FoldAct, ContextCurator.

# §5 Training Methods (Pillar B approach)  · ~1.25pp · 🔵
- **5.1 What's trained.** Qwen3.5-4B + LoRA (all-linear), single GPU; VLM frozen
  external. The 8B→4B pivot rationale (the 4B is the ≤8B SOTA backbone).
- **5.2 Signals explored.** SFT/SeqKD on rejection-sampled teacher trajectories
  (cold-start / warmup); GRPO from ANLS verifier reward (+ tool/exec shaping);
  on-policy distillation (per-token KL to 27B teacher); summed-advantage combo.
  Use the SeqKD(O(N) off-policy) / RL(O(1) scalar, on-policy) / OPD(O(N) dense,
  on-policy) framing.
  - SFT data actually used = MMLongBench-Doc transfer trajectories
    (2407.01523), leakage-free vs DocVQA val.
- **5.3 Training-data pool.** A method-agnostic DocVQA-family prompt substrate.
  **Chronology to state accurately:** assembled *after* the SFT experiments and
  **not yet used for training** — it is the intended substrate for the upcoming
  larger-data SFT run and the RL/OPD runs, not a source of the SFT numbers in
  §7. Sources, each cited: DocVQA (2007.00398), InfographicVQA (2104.12756),
  ChartQA (2203.10244), MapQA (2211.08545), MP-DocVQA (2212.05935), TAT-DQA
  (2207.11871), SlideVQA (2301.04883).
  Weighted per-source sampler (de-dominate MapQA); leakage-safe vs DocVQA val.

# §6 Experimental Setup  · ~0.75pp · 🔵
- Infra: verl (FSDP, sync+colocated GRPO), frozen 27B VLM HTTP endpoint, LoRA.
- Data: MMLongBench-Doc (transfer, leakage-free), DocVQA val, the pool.
- **Eval protocol (define the metric ONCE here):** ANLS @ <threshold-to-verify>;
  n=4; mini-screen (29Q) vs full-80Q confirm; paired per-question stats.
  Baselines: untrained 4B; leaderboard ≤8B 0.1875, 27B 0.375.

# §7 Results: Training Investigation (Pillar B)  · ~1.5pp · 🟠
- **7.1 Zero-shot baseline.** Untrained Qwen3.5-4B in the CodeAct scaffold:
  full-val 15.3% (mini 19.0%). Train==deploy scaffold (policy trained == policy
  evaluated).
- **7.2 SFT design controls.** Transfer (MMLongBench-Doc) vs in-domain (leaked)
  data; training-duration sweep (undertraining vs memorization — 20ep drives
  train loss→0.003 yet in-domain memorization only ties the leaked baseline,
  evidencing that trajectory memorization does not transfer to free rollout).
- **7.3 SFT verdict.** Full-val (n=4, paired): baseline 15.3 → 16ep 17.2 (+1.9,
  n.s.) → 20ep 20.6 (+5.3, p≈0.04). Real but modest and training-monotonic;
  single-checkpoint n=4 caveat.
- **7.4 Error analysis.** Long-document runaway (the agent exhausts its
  turn/time budget before submitting) is the dominant failure mode and bounds
  every method's ceiling — an answer-quality-independent error class.
  Per-category breakdown.
- **7.5 Larger-data SFT (if it lands).** A pool-based SFT run with more data may
  complete before the deadline; slot for its full-val number vs the +5.3 result.
  [fill if available]
- **7.6 RL/OPD preliminary.** GRPO (base 4B, MMLongBench → DocVQA): whatever
  lands by deadline, reported with confidence caveats. [fill late]

# §8 Discussion  · ~0.75pp · 🟠
- Multi-turn observation-shift bounds trajectory imitation: under the student's
  own VLM observations a memorized teacher trajectory cannot be replayed →
  on-policy methods are the lever. Ties back to §4: the same prefix/observation
  structure that makes CodeAct trainable also bounds what offline SFT can do.
- The RL frontier: what GRPO needs here (cold-start via SFT warmup, reward
  shaping for long-document runaway, trainer/inference precision discipline).

# §9 Related Work  · ~0.5pp · 🔵 (methods lineage; benchmark/scaffold already in §2)
- RL for LLMs: GRPO (2402.03300) + variants Dr.GRPO (2503.20783), DAPO
  (2503.14476), GSPO (2507.18071), CISPO/MiniMax-M1 (2506.13585).
- Distillation: OPD survey (2604.00626), GKD (2306.13649), MiniLLM (2306.08543),
  SeqKD. On-policy distillation lineage.
- RL under agent-controlled / non-prefix context (emerging, unestablished):
  FoldAct (2512.22733), ReSum (2509.13313), ContextCurator/ActiveContext
  (2604.11462), AdaCoM (2605.30785) — the frontier §4 cites/avoids; positions our
  append-only choice. (Format-sensitivity mechanism: Sclar 2310.11324 — optional.)
- Agentic document QA: RVLM (2603.24224), MADQA (2603.12180), ARIAL (2511.18192),
  VISOR (2604.09508), MDocAgent (2503.13964), ORCA (2603.02438), SlideAgent
  (2510.26615), DocVStar (2604.13731), AgenticOCR (2602.24134); RLM (2512.24601);
  CodeAct; ReAct (2210.03629).
- DocVQA-family benchmarks (for §2 + §5 data): DocVQA (2007.00398), InfographicVQA
  (2104.12756), ChartQA (2203.10244), DUDE (2305.08455), MP-DocVQA (2212.05935),
  SlideVQA (2301.04883), MMLongBench-Doc (2407.01523), ANLS/STVQA (1905.13648).
- Bibliography source: the vault's `Sources/Papers/` folders hold all of the
  above (arXiv ids are the folder suffixes). **Verify every id against arXiv
  before it lands in the .bib** — several related-works entries are flagged
  "needs verification" by the author.

# §10 Conclusion & Future Work  · ~0.4pp · 🟠
- Two pillars; honest verdict; CodeAct-as-trainable-target; RL/OPD next.

# Limitations (unnumbered)  · ~0.3pp · 🟠
- n=4 / single-seed / single-checkpoint confidence; compute ceiling (≤8B, LoRA);
  long-document runaway / context growth (CodeAct's append-only transcript can
  approach the window limit — the cost of distribution match, §4.4); the §4.4
  distribution-mismatch account is an untested hypothesis; RL incomplete by
  deadline; no test-set numbers if portal closed.

---

## Drafting order (🔵 first, parallelizable via sub-agents)
1. §3 Harness Evaluation  2. §4 Trainable Scaffold  3. §2 Background
4. §5 Methods  5. §6 Setup  6. §9 Related Work  → then §1 body
→ then §7, §8, §10, Limitations as numbers land.
