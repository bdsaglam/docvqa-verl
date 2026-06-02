# DocVQA-verl: off-policy distillation (CodeAct 4B ← 27B) design

**Date:** 2026-06-02
**Author:** Barış (with Claude)
**Status:** spec, awaiting review
**Supersedes (for the training method):** the GRPO Phase-1 plan in
`2026-04-30-docvqa-verl-training-design.md`. That spec's *scaffold*
(agent loop, subprocess interpreter, prompts, reward, data layer) still
stands and is reused as-is. This spec changes the **backbone** (Qwen3-8B
→ Qwen3.5-4B), the **first training method** (cold-start GRPO →
off-policy distillation), and corrects the **teacher identity**.

## 1. Summary

Fine-tune **Qwen3.5-4B** as an append-only, code-writing DocVQA agent
(the **CodeAct** scaffold already implemented in `docvqa/agent_loop.py`)
via **traditional off-policy knowledge distillation** from a
**Qwen3.5-27B teacher running the same CodeAct scaffold**. Evaluate on
DocVQA-2026 val; train only on DocVQA-family data that does **not**
overlap DocVQA-2026's sources (primary: MMLongBench-Doc).

The 4B is the *agent* (writes Python, reasons in `<think>`); the 27B is
used in two frozen roles: as the `batch_look` **VLM perception tool**
(both student and teacher call it), and as the **policy teacher** whose
CodeAct trajectories the 4B is distilled from. The 27B is never trained.

We pick off-policy distillation over GRPO because a prior smoke run
(2026-05-01, recorded in `project/verl_recipe_survey.md`) confirmed cold
start: 84% of GRPO rollout groups had zero reward variance and one step
dropped val ANLS 0.167→0.083. The teacher succeeds where the cold 4B
fails, so its (rejection-sampled) trajectories give signal without
requiring the student to already solve the task.

This spec covers **Stage 0** (re-point + data prep + teacher trajectory
collection — all runnable now, no training GPU) and **Stage 1** (the
distillation training, a two-rung ladder: SeqKD → forward-KL top-k).
GRPO refinement (Stage 2) and pedagogical RL (Stage 3) are scoped as
future work, not designed here.

## 2. Goals and non-goals

### Goals

- Train a 4B CodeAct agent that beats its own zero-shot CodeAct baseline
  on DocVQA-2026 val, and ideally surpasses the ≤8B-tier reference
  (proposal cites 0.1875).
- Keep **train == deploy**: the policy we train (CodeAct, append-only,
  `batch_look`+`SUBMIT`) is exactly the policy we evaluate and submit.
- Ship a reportable checkpoint by the **June 15** report deadline with
  the lowest-risk method first (SeqKD), then upgrade signal (top-k KD)
  if time allows.
- Do all GPU-free preparation (re-point, data prep, teacher trajectory
  collection) immediately, against the already-running 27B servers,
  while training GPUs are occupied.

### Non-goals (this spec)

- GRPO / GRPO-variant refinement on top of the distilled checkpoint
  (Stage 2 — future spec).
- Pedagogical RL, privileged self-teacher, on-policy distillation,
  combined RL+distillation advantages (Stage 3 — future).
- Training or modifying the 27B (teacher and VLM stay frozen).
- Modifying verl's trainer / dataset / reward internals beyond the
  distillation data-path inversion described in §7. Everything else
  lives in `docvqa/`.
- Replay-buffer RL (no native verl recipe; explicitly deferred).

## 3. Corrected framing (read before implementing)

### 3.1 CodeAct, not rvlm — and what the teacher is

The deployed competition system used the **`rvlm`** solver
(`~/repos/docvqa/src/docvqa/solvers/rvlm_solver.py`): a LeanRLM REPL
that keeps full variable values in a *hidden* interpreter namespace,
shows the model only a `variables_info` sidecar, and lets the model
compact its own history with `RESET_HISTORY`. The observable context is
a lossy view of a larger hidden state — a **POMDP**. rvlm-27B scores
~39% single-trial / 0.512 with SC-8 (Qwen3.6) on val. **rvlm cannot be
a fine-tuning target or a distillation source**: its trajectories are
not an append-only token stream, so a standard LM loss cannot reconstruct
them.

What we train is **CodeAct**
(`~/repos/docvqa/src/docvqa/solvers/codeact_solver.py`, and mechanically
`docvqa-verl/docvqa/agent_loop.py`): a strictly **append-only** loop —
every `(think, code, stdout)` step is appended and re-shown verbatim,
`RESET_HISTORY` is ignored, no hidden sidecar. This is an **MDP** with a
monotonic message list, which is exactly what SFT / KD / RL losses
assume. Tool surface: `batch_look(requests)` (parallel image→text via
the 27B VLM) and `SUBMIT(answer=...)`. No OCR/BM25 `search`.

**Therefore the teacher is the 27B driving our *own* CodeAct
`agent_loop`** — collected by pointing `collect_trajectories.py
--lm-model` at the 27B. Its trajectories are append-only, same action
space as the student, and directly imitable. We do **not** use
docvqa-repo rvlm trajectories.

> `agent_loop.py`'s module docstring currently says it "replicates
> `rvlm_minimal_solver`". This is misleading — it is mechanically
> CodeAct (append-only). Fix the wording in Stage 0.

### 3.2 Baselines — what we actually compare against

| Quantity | Status | Action |
|---|---|---|
| 4B-CodeAct zero-shot val ANLS (**the floor to beat**) | **Not yet measured** (CodeAct sweep queued in docvqa repo) | Measure in Stage 0 (eval run / docvqa C-sweep). |
| 27B-CodeAct val ANLS (**teacher ceiling**) | **Not yet measured** | Falls out of Stage-0 teacher trajectory collection on a val-like probe. |
| rvlm-4B + 27B-VLM | 0.2125 (n=8) | Reference only — different scaffold. |
| rvlm-27B | ~0.39 single / 0.512 SC-8 | Aspirational; *not* a CodeAct number. |

**Open risk:** append-only CodeAct may underperform rvlm because rvlm's
`RESET_HISTORY` is how it survives 36-page documents in a bounded
context. If CodeAct-27B is a weak teacher, the distillation ceiling is
low. Mitigation: trajectory collection *is* the CodeAct-27B
measurement; and we rejection-sample (ANLS-filter) so we only ever
train on the teacher's good rollouts regardless of its mean.

### 3.3 Tokenizer compatibility (verified)

`tokenizer.json` is **byte-identical** across Qwen3.5-4B / 9B / 27B
(same MD5). Token-level KD (forming KL at the student's token positions
against the teacher's logprobs) is therefore well-defined. The 27B
servers at `localhost:8927` and `:8928` both advertise
`allow_logprobs: true`.

## 4. Method: off-policy distillation

Off-policy = the **teacher** generates the trajectories; the student
learns from them (not from its own rollouts). A two-rung ladder, each
rung independently shippable. References to verl distillation code are
to the vendored copy surveyed in
`~/obsidian/Knowledge/Tools/verl/repo/`.

### 4.1 Rung 1 — SeqKD (sequence-level KD = SFT on teacher tokens)

Teacher samples CodeAct trajectories `y ~ π_27B(·|x)`; keep
ANLS-passing ones; the student minimizes token NLL on them over
assistant (`mask=1`) positions only:

```
L_SeqKD(θ) = E_{x, y~π_27B(·|x), ANLS(y)=1} [ Σ_t -log π_4B(y_t | y_<t, x) ]
```

- Teacher signal: **tokens only** (no logprobs).
- Implementation: `verl.trainer.fsdp_sft_trainer` with
  `data.multiturn.enable=true`, `messages_key=messages`. The trajectory
  collector already emits `{messages}`. Near-zero new code.
- This is the guaranteed-shippable checkpoint.

### 4.2 Rung 2 — forward-KL top-k (soft-target off-policy KD)

Same ANLS-filtered teacher trajectories, but now match the teacher's
**distribution** at each assistant token, using its top-k logprobs:

```
L_fKL(θ) = E [ Σ_t  Σ_{k∈topk}  p_27B(k|y_<t,x) · ( log p_27B(k|·) - log p_4B(k|·) ) ]
```

- Teacher signal: **top-k logprobs + token ids** per assistant position
  (vLLM `prompt_logprobs`, `topk` default 128; server `max_logprobs ≥
  topk`).
- Implementation: reuse `verl/trainer/distillation/fsdp/losses.py:
  compute_forward_kl_topk` with `use_policy_gradient=False,
  use_task_rewards=False` (pure supervised distillation; cites GKD
  2306.13649). The FSDP logits-processor hook
  (`workers/engine/fsdp/transformer_impl.py`) computes top-k KL inside
  the forward pass so full-vocab logits never materialize across SP
  shards. Negative-divergence-from-truncation is clamped (`clamp_min`).
- **The one real porting task:** verl's distillation data path is wired
  for *on-policy* (student rolls out → teacher scores). We invert it:
  feed pre-generated teacher trajectories + their top-k teacher logprobs
  into the `TensorDict` where the loss expects `teacher_logprobs /
  teacher_ids` (`distillation/losses.py:153-156,378-379`). Bounded,
  FSDP-native — no Megatron port (the old recipe-survey claim that OPD
  is Megatron-only is outdated).

### 4.3 Optional add-on — surprisal-gated assimilation

A free per-token weight on either rung's loss (borrowed from
Pedagogical RL, `~/obsidian/Sources/Papers/pedagogical-rl/blog.md`,
*without* its RL machinery):

```
ℓ_t  ← w_t · ℓ_t ,   w_t = σ( κ · ( log π_4B(y_t | y_<t,x) - γ ) )
```

down-weights teacher tokens the 4B finds unassimilable (the documented
weakness of off-policy KD: teacher "shortcuts" far from the student's
distribution). Needs only the **student's own** logprobs — no extra
teacher signal. Treat as an ablation knob, not load-bearing.

## 5. Data plan

### 5.1 Sources and the no-leakage rule

- **Primary training source: MMLongBench-Doc (train split)** — long
  multi-page documents, most on-distribution for "VQA over large docs."
- **Hard rule: exclude any dataset DocVQA-2026 draws documents from.**
  Stage-0 gate: confirm MMLongBench-Doc docs do not overlap DocVQA-2026
  (by source corpus and, where feasible, document hashing). This rule
  likely excludes DocVQA / InfographicVQA / etc. as training sources if
  DocVQA-2026 sources from them — verify before adding any.
- Evaluation stays **DocVQA-2026 val** only (the gold-labeled split).

### 5.2 Volume gate (open)

The local `mmlongbench-doc/train/` currently contains only a `pages/`
dir — **no materialized `questions.json`**. MMLongBench-Doc is modest in
size (~o(10²) docs). Stage 0 must (a) materialize its train questions
via a `prepare_data.py` adapter, (b) report the question/doc count. If
volume is too thin for SFT (rule of thumb from ReST^EM: ~1k filtered
examples is a workable floor), expand to other **non-overlapping**
long-doc sources (candidates to vet: DUDE, SlideVQA, TAT-DQA) — each
behind a leakage check.

### 5.3 Row contract and trajectory format (unchanged)

Dataset rows follow the existing contract (`{question_id, question,
doc_dir, gold_answer, category}`; everything per-doc under `doc_dir/`).
`prepare_data.py` materializes `data/<dataset>/<split>/docs/<doc_id>/`
with `metadata.json` + `pages/page_*.png` (no OCR/BM25 sidecars — the
CodeAct scaffold only consumes `pages`). The trajectory collector emits
JSONL of `{record_id, messages, submitted_answer, gold_answer, anls,
termination, num_turns, vlm_calls, ...}` (see
`docvqa/scripts/collect_trajectories.py`).

### 5.4 SFT dataset projection

A small script filters the collected JSONL to `anls == 1.0` and projects
to verl multiturn-SFT rows (`{messages}`, optionally `{messages,
teacher_topk}` for Rung 2). Sampling: collect `n≈4–8` rollouts/question
at `temperature=1.0` so we have multiple positives per solved question.

## 6. Components and files

Reuse (no change beyond §3.1 docstring + §8 defaults):

| Path | Role |
|---|---|
| `docvqa/agent_loop.py` | CodeAct loop. Used by both teacher collection and eval. |
| `docvqa/scripts/collect_trajectories.py` | Teacher (and student) trajectory collection over a split. |
| `docvqa/scripts/eval.py` | Scaffold eval (4B floor, 27B ceiling, post-train). **Extended in Stage 0** to the §9 protocol: `top_k` threading, n=8, mean±std / pass@8 / SC-8, matched defaults. |
| `docvqa/{subprocess_interp,sandbox,tools,prompts,parser,reward,metrics}.py` | Unchanged scaffold + ANLS. |

New / modified:

| Path | Purpose |
|---|---|
| `docvqa/scripts/prepare_data.py` | Add MMLongBench-Doc train adapter (+ leakage check). |
| `docvqa/scripts/make_sft_data.py` | Filter ANLS==1, project trajectories → verl SFT rows (and top-k for Rung 2). |
| `docvqa/scripts/run_sft_seqkd.sh` | Rung-1 launcher (`fsdp_sft_trainer`, multiturn, LoRA). |
| `docvqa/distill/` (Rung 2) | Data-path inversion + config to feed teacher top-k into `compute_forward_kl_topk`. |
| `docvqa/scripts/run_distill_topk.sh` | Rung-2 launcher. |
| `tests/docvqa/test_make_sft_data.py` | Projection/filter correctness; round-trip of `messages`→tokens. |

## 7. Implementation stages

### Stage 0 — re-point + data + teacher trajectories (NO training GPU)

Runnable now against the live 27B servers (8927/8928) + CPU subprocesses.

1. **Re-point defaults** to `Qwen/Qwen3.5-4B` (student) and
   `Qwen/Qwen3.5-27B` (teacher LM + `batch_look` VLM) in `agent_loop.py`,
   `collect_trajectories.py`, `eval.py`. Fix the `agent_loop.py`
   docstring (CodeAct, not rvlm).
2. **Chat-template gate:** verify Qwen3.5-4B's chat template preserves
   `<think>` across turns when re-rendering the full `messages` list to
   tokens (the SFT path renders `messages`→tokens; the old `willcb`
   mitigation was 8B-specific). If it strips `<think>`, source/author a
   `<think>`-retaining template before any SFT.
3. **Data prep:** materialize MMLongBench-Doc train doc-dirs +
   `questions.json`; report volume; run the DocVQA-2026 overlap check.
4. **Teacher trajectory collection:** run 27B-CodeAct over the train
   questions (`n≈4–8`, `temp=1.0`) → JSONL. Simultaneously yields the
   **CodeAct-27B quality number** (teacher ceiling) and the SFT data.
5. **Before-eval (the floor):** run the §9 protocol on zero-shot
   Qwen3.5-4B with **our own `eval.py`** (not the docvqa-repo eval — the
   before/after numbers must share one harness). Needs the 4B served on
   a GPU; if none is free yet, this is the one Stage-0 item that waits,
   but it gates nothing upstream (data prep + teacher collection proceed
   without it). Also record CodeAct-27B on val (teacher ceiling) from the
   same harness.

Stage-0 exit: filtered SFT dataset materialized; `eval.py` extended to
the §9 protocol; teacher ceiling and student floor measured; chat-template
gate passed.

### Stage 1a — SeqKD (training GPU)

`run_sft_seqkd.sh`: LoRA multiturn SFT on the ANLS-filtered teacher
trajectories. Validation gates in §9. Ship checkpoint, eval on val.

### Stage 1b — forward-KL top-k (training GPU)

Re-collect (or augment) trajectories with teacher top-k logprobs; wire
the data-path inversion (§4.2); train with `compute_forward_kl_topk`.
Compare to SeqKD on val. Optionally ablate the surprisal gate.

## 8. verl wiring specifics

- **SeqKD:** `verl.trainer.fsdp_sft_trainer
  data.multiturn.enable=true data.multiturn.messages_key=messages
  model.path=Qwen/Qwen3.5-4B model.lora_rank=16 ...`. Closest existing
  templates: `recipe/retool/run_qwen2_7b_sft.sh`,
  `recipe/open_math_reasoning/run_sft_qwen3_8b.sh`.
- **Top-k KD:** `loss_mode=forward_kl_topk`, `topk=128`,
  `use_policy_gradient=False`, `use_task_rewards=False`
  (`workers/config/distillation.py`). Teacher served via vLLM
  `prompt_logprobs` (feed prompt+response, read top-k per position).
  `DistillationTeacherModelConfig` validates `max_logprobs ≥ topk`.
- **FSDP, LoRA, 3×A100.** No Megatron.

## 9. Evaluation protocol (before/after on DocVQA-2026 val)

The headline result is a **before/after delta on DocVQA-2026 val**, run
through a single harness so the two numbers are directly comparable.

**Harness.** `docvqa/scripts/eval.py` (the CodeAct `agent_loop`), used
identically for both:
- **Before:** zero-shot Qwen3.5-4B served in vLLM.
- **After:** the fine-tuned 4B (LoRA merged into the base, or served as
  an adapter) in vLLM — same endpoint contract.

**Split.** DocVQA-2026 val — 25 docs / 80 questions, all 8 categories.

**Metric.** The official DocVQA-2026 score (`docvqa/metrics.py`):
per-question correct iff `ANLS ≥ 0.80` against any gold (with strict
magnitude/date matching), averaged. This is the *thresholded* metric the
docvqa-repo baselines report (e.g. 17/80 = 0.2125), **not** a soft mean
ANLS — `eval.py`'s `anls = 1.0 if is_correct else 0.0` is therefore
correct, not a bug.

**Trials and reported numbers (n=8 rollouts/question):**
- **mean ± std** of per-question correctness across the 8 rollouts — the
  headline single-sample number with its variance.
- **pass@8** — solved if *any* of the 8 rollouts is correct (capability
  ceiling).
- **SC-8** — self-consistency: majority-vote the 8 normalized
  `submitted_answer`s, score the voted answer (deployment-style number,
  comparable to the rvlm SC-8 0.51).
- All three reported overall **and per-category** (watch `maps`, 0.0 at
  the rvlm-4B baseline).

**Sampling params — matched to the docvqa repo for comparability:**
- Agent LM (the 4B): `temperature=0.6, top_p=0.95, top_k=20,
  enable_thinking=true`
  (`~/repos/docvqa/configs/lm/qwen-3_5-27b-vllm-local.yaml`).
- `batch_look` VLM (frozen 27B): `temperature=0.3, top_k=20,
  max_tokens=16384, enable_thinking=false` (repo vlm config).
- The **27B VLM endpoint and its settings are identical before and
  after** — it is the controlled variable; only the 4B agent weights
  change.

**`eval.py` changes (Stage 0):** thread `top_k` (and the matched
defaults) through `_OpenAIClientServerManager.generate`; run n=8
rollouts/question; compute mean±std, pass@8, and SC-8 (majority vote
over normalized `submitted_answer`s); fix stale model defaults to
`Qwen3.5-4B` / `Qwen3.5-27B`.

> Teacher *trajectory collection* (§5.4) deliberately uses higher
> `temperature` (≈1.0) for positive-sample diversity — a data-generation
> knob, separate from this eval protocol.

## 10. Validation gates

1. **Data integrity:** SFT rows round-trip (`messages`→tokens→decode);
   `<think>` preserved across turns; no DocVQA-2026 doc leakage.
2. **SeqKD sanity:** loss decreases; held-out (train-split) trajectory
   NLL drops; no degenerate outputs.
3. **Val ANLS (the §9 protocol):** distilled 4B-CodeAct beats the
   4B-CodeAct zero-shot floor on DocVQA-2026 val — same harness, n=8,
   reporting mean±std / pass@8 / SC-8, per-category (watch maps=0.0),
   27B VLM frozen and identical before/after. SUBMIT rate ≥ teacher's;
   subprocess-error rate ≤ 2%.
4. **Top-k KD:** beats or matches SeqKD on val; report top-k overlap /
   teacher-mass diagnostics.

## 11. Compute layout

- 27B teacher + VLM: already serving at 8927/8928 (frozen). Used for
  trajectory collection (Stage 0, no training GPU) and as `batch_look`
  during eval/inference.
- 4B student training: LoRA + FSDP on the project A100s when free.
  Trajectory collection and data prep do not need them.

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| CodeAct-27B is a weak teacher (append-only loses to rvlm's RESET_HISTORY on long docs). | Measure it in Stage 0; ANLS-filter so we train only on its good rollouts; raise `n`/turns to lift positive yield. |
| Off-policy KD distribution shift — teacher trajectories use 27B "shortcuts" the 4B can't follow. | ANLS-filter + surprisal-gate (§4.3); prefer forward-KL soft targets over hard SeqKD if shift shows. |
| MMLongBench-Doc train volume too thin for SFT. | Stage-0 volume report; expand to vetted non-overlapping long-doc sources. |
| DocVQA-2026 leakage via training source. | Hard exclusion rule + Stage-0 overlap check; eval stays val-only. |
| Qwen3.5-4B chat template collapses `<think>` across turns (breaks SFT targets). | Stage-0 chat-template gate before any training; author a retaining template if needed. |
| Top-k KD data-path inversion takes longer than expected. | SeqKD (Rung 1) is the deadline-safe fallback and ships first. |

## 13. Future stages (not designed here)

- **Stage 2 — GRPO-variant refinement** on the distilled checkpoint:
  now that success rate is lifted, groups have reward variance. DAPO
  dynamic-sampling (drop zero-variance groups) + clip-higher; GSPO/CISPO
  loss swaps (~one-flag changes). Easy→hard curriculum to keep variance
  alive.
- **Stage 3 — Pedagogical RL** (privileged self-teacher + spike-aware
  reward + surprisal-gated assimilation) and/or on-policy distillation /
  combined advantages. Higher ceiling, higher implementation risk; a
  research arm, not the load-bearing result.

## 14. Open questions

- MMLongBench-Doc train question volume and DocVQA-2026 overlap status
  (Stage-0 outputs).
- Whether Qwen3.5-4B needs a custom `<think>`-retaining chat template.
- CodeAct-27B teacher ceiling — does it justify distillation, or do we
  need a stronger teacher configuration (more turns / SC voting on the
  teacher's answers as an oracle for rejection sampling)?
- Exact LoRA rank / lr / batch / max_length for the 4B SFT (tuned at
  Stage 1, recorded in the launcher).
