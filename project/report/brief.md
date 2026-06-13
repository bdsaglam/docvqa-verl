# Brief (Level 0)

**Medium.** Graduate term-project report, COGS 560 (PhD course). ACL LaTeX
two-column, ~8 pages (a bit over OK). Not a paper — balance rigor/depth to the
venue, but as good as it can be.

**Audience.** The course instructor (grader) and ML/DocVQA researchers. A cold,
technical reader who knows nothing of this project's development. Formal academic
voice common in good papers; not heavy.

**Raw material.**
- *Pillar A — agentic-harness evaluation* (from the `~/repos/docvqa` repo): a
  controlled comparison of 8 harness designs on DocVQA-2026 val (n=8, frozen
  Qwen3.5-27B VLM, no-think). Key facts: a code REPL + a depth-1 VLM perception
  tool (active perception) are the two load-bearing mechanisms (~25 ANLS spread);
  active visual perception beats OCR (rlm_ocr −25.5pp); harness ranking flips with
  reasoner scale; **append-only CodeAct ties context-managing RLM at 27B (39.5 vs
  39.4).**
- *Pillar B — weight-level training* (this repo): fine-tune a 4B reasoner (LoRA,
  frozen VLM) inside the append-only scaffold. SFT verdict: modest,
  training-dependent lift (15.3→20.6 ANLS, +5.3, p≈0.04); memorizing teacher
  trajectories does not transfer (multi-turn observation shift). RL/OPD designed,
  preliminary results as they land.
- *The hinge (§4):* the append-only / prefix-preserving scaffold is the
  trainable target; established RL/distillation assume a growing-prefix
  trajectory. The CodeAct≈RLM tie makes choosing it cost-free.

**Constraints (hard rules).**
- Research findings + methodology ONLY — no development/implementation
  war-stories (bugs, infra, data-corruption, debugging, eval-correction arcs).
- Cite public papers via arXiv ids ONLY; never reference any private vault/path.
- Backbone pivot to state: proposal said Qwen3-8B; actual = Qwen3.5-4B (the ≤8B
  leaderboard-slot backbone).

**Success criteria.** A single, honest, defensible core message (not a vague
two-pillar list); rigorous controlled experiments reported with calibrated
confidence; clean structure a cold reader can follow; preliminary RL/OPD slotted
without overclaiming. SOTA is not a deliverable — the instructor values
evaluation contributions, so Pillar A is the solid result and Pillar B is the
first step down the path it identifies.
