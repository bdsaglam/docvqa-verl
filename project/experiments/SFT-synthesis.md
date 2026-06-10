# SFT stage — synthesis (corruption discovery + clean restart)

> Updated 2026-06-07. **Supersedes all earlier per-experiment result cards** (removed):
> they measured models trained on corrupted teacher data — see below.

## Headline
**Every SFT model from the first sweep (v1, v2, Arm A/B, v4–v6) was trained on
~93%-corrupted teacher data, so their numbers are invalid.** A scaffold bug let the
27B teacher hallucinate tool output; we discovered it, fixed it, and restarted clean.
The real "does SFT beat baseline" question is **open** and now runs on clean footing.

## The bug (root cause)
- The agent emits `<think>` + a ```` ```python ```` fence; captured stdout becomes the
  next observation. The **only** generation stop was `<|im_end|>`.
- When the model didn't emit `<|im_end|>` after its code fence, it **free-ran and
  role-played the next turns** — fabricating `\nuser\n## Output\n...` observations and
  *more* code blocks (up to **10 fences in one turn**).
- `_FENCE_RE` matches plain ```` ``` ```` too, and we parsed the **last** fence → the
  executed code (and thus the recorded observation) came from a *hallucinated* block.
- **93.5% of anls==1.0 trajectories** had ≥1 such multi-fence turn. Unpatchable: the
  recorded observation is paired with the last-fence (hallucinated) block, not the
  model's real first action — stripping desyncs (code, observation).

## The fix (`docvqa/agent_loop.py`, `docvqa/prompts.py`)
1. **Stop sequences** `["<|im_end|>", "\nuser\n", "\n## Turn", "\n## Output"]` — halt
   the instant the model starts fabricating an observation.
2. **`parse_first_fence=True` default** (eval/collection/RL) — run the model's real
   first action even if a stray block slips in.
3. **Defensive strip** of any fabrication tail (SFT-text path).
4. **Simplified prompt format** to shrink the hallucination surface: observation is
   `## Output (Turn n/N)\n{output}` (no ```` ``` ```` fence); first-user message drops
   the `Begin.` sentinel and puts the **question last** (recency).

## Verification (clean collection v4, fixed scaffold)
- **Hallucination markers: 0.0%** (was pervasive). ✓ Fix confirmed.
- Multi-fence: 93.5% → ~20%, and the residual is **benign** (genuine multi-block, no
  fabricated observations; first-fence runs the right block; build-time first-fence
  truncation makes SFT data fully single-block).
- **Teacher solve-yield jumped ~22% → ~98%** — the fix also helps the 27B *solve*: it
  no longer derails itself by executing hallucinated blocks. (Early-doc sample; watch.)

## Status — FIRST CLEAN VERDICT IS IN (2026-06-08): transfer-SFT loses to baseline
Clean restart done through the mini eval. `seqkd-clean-v5` (245 clean traj from
MMLongBench-Doc, LoRA r32 all-linear, LR 2e-4 constant, 3 ep) vs untrained 4B on
`docvqa_mini` (29 Q, n=4), fixed scaffold:

| | baseline | clean-v5 | Δ |
|--|--|--|--|
| **overall ANLS@0.9** | **0.1897** | **0.1379** | −0.052 |
| submit-only | 0.286 (77) | 0.216 (74) | −0.070 |
| pass@4 / SC-4 | 0.414 / 0.310 | 0.345 / 0.172 | −0.07 / −0.14 |

**Verdict: clean SFT on out-of-domain MMLongBench transfer data does NOT beat the
fixed-scaffold baseline — it slightly degrades it.** Submission rate is unchanged
(77→74); the loss is in *answer quality among submissions* (0.286→0.216) + a SC-4
diversity collapse (0.31→0.17, mode-collapse toward teacher phrasings). Full-val
(80 Q) was gated on clean-v5 beating baseline → **not run** (would only confirm a
negative). The scaffold/prompt fix (0.042→0.19) remains the only lever that moved
the needle. Full card: `results/clean-restart-mini-n4.md`.

**Open question this raised:** is the failure the *domain gap* (MMLongBench ≠ DocVQA)
or the *SFT setup itself*? → **ANSWERED below.**

## In-domain upper-bound verdict (2026-06-08): the SFT SETUP is the limiter
Trained `seqkd-indomain-v1` on 80 clean 27B-teacher trajectories collected on DocVQA
**full val** (leaked-by-design), eval on `docvqa_mini` (a subset of val → also leaked):

| | baseline | clean-v5 (transfer) | indomain-v1 (in-domain, leaked) |
|--|--|--|--|
| **overall** | **0.1897** | 0.1379 | **0.1466** |
| submit-only | 0.286 (77) | 0.216 (74) | 0.218 (78) |
| SC-4 | 0.310 | 0.172 | 0.241 |

**Even leaked in-domain SFT (0.147) does NOT beat baseline (0.190)** — same mechanism
as transfer (submit rate flat 77→78, answer quality drops 0.286→0.218, diversity
collapses). So the prior failure was **NOT the domain gap**: SFT on 27B-teacher CodeAct
trajectories is net-negative for this 4B **regardless of data source**, even under
maximal favorability (in-domain + leakage). **The scaffold/prompt fix (0.042→0.190) is
the only lever that has moved the metric.** Full card: `results/indomain-upperbound.md`.

## Memorization / undertraining test (2026-06-09): setup OK, v1 undertrained, still ties
Was the in-domain failure undertraining or a broken setup? Retrained the same 80 traj
for **20 epochs** (vs 3) → train loss collapsed **0.37 → 0.003** (fully memorized), eval
on the same leaked `docvqa_mini`:

| | overall | train loss |
|--|--|--|
| baseline | 0.1897 | — |
| indomain-v1 (3 ep) | 0.1466 | 0.22 |
| indomain-v2 (20 ep, memorized) | **0.1897** | 0.003 |

**v2 ties baseline exactly — never beats it.** So (a) the setup is **not broken** (it
fits to ~0 loss), (b) v1 was **undertrained**, (c) even in-domain + leaked + fully
memorized, SFT only **matches** baseline. Confirms the **multi-turn observation-shift**:
memorizing teacher trajectories doesn't transfer to free rollout, where the model faces
its own (different) VLM observations and can't replay the memorized answer. Card:
`results/indomain-upperbound.md`.

## Generalization test (2026-06-09): longer transfer-SFT also ties baseline (mmlb-long)
Was clean-v5's 0.138 undertraining too? Trained the same 245 mmlb traj for **20 epochs,
checkpointing every 5**, eval each on DocVQA (no leakage):

| mmlb training | train loss | overall |
|--|--|--|
| 3 ep (clean-v5) | 0.18 | 0.138 |
| 5 / 11 / 16 / 20 ep | 0.10→0.006 | 0.207 / 0.147 / 0.190 / 0.216 |
| baseline | — | 0.1897 |

**Bounces within the ±5% noise floor (29 Q) = null, ties baseline.** clean-v5's 0.138 was
just the undertrained dip; more epochs recover to ≈baseline, none exceed it. No checkpoint
cleared the bar for a full-val promotion. Card: `results/mmlb-long-generalization.md`.

## ⚠️ CORRECTION (2026-06-10): the "null" was a MINI-SET ARTIFACT — SFT DOES help on full val
All verdicts above were measured on **docvqa_mini (29 Q)**, which is underpowered
(SE≈5%) and uses median-difficulty docs that flatter the untrained baseline. Running the
**full 80-Q val** (the reportable set, n=4, paired) flips the transfer conclusion:

| 80Q full-val | overall | submit-only | pass@4 |
|--|--|--|--|
| baseline (untrained 4B) | **15.3%** | 25.1% | 33.8% |
| mmlb-long ep20 (SFT, 20ep) | **20.6%** | 32.2% | 41.2% |
| **Δ** | **+5.3** | +7.1 | +7.4 |

**Paired t-test Δoverall=+5.3%, SE 2.5%, t=2.09 (79 df), p≈0.04 — significant.**
The SFT model is stable across sets (mini 21.6 ≈ full 20.6); the **baseline collapses on
harder docs** (mini 19.0 → full 15.3). SFT's benefit = robustness on the harder
distribution, invisible on the easy mini screen. Card: `results/mmlb-long-generalization.md`.

## REVISED FINAL VERDICT
| variant (eval set) | result |
|--|--|
| transfer undertrained (clean-v5, 3ep, mini) | 13.8% — hurts (undertrained) |
| transfer long (mmlb-long ep20, **full val**) | **20.6% vs 15.3% baseline, +5.3, p≈0.04 — BEATS** |
| in-domain memorized+leaked (v2, 20ep, mini-leaked) | 19.0% = mini-baseline — ties (leaked) |

- **A sufficiently-trained (20ep) mmlb→DocVQA transfer SFT beats the untrained baseline
  on the full val (+5.3 ANLS, p≈0.04).** SFT is a real, modest lever — NOT a null.
- **Undertraining still hurts** (clean-v5 13.8%); the lift needs the full epoch budget.
- The *in-domain memorization* result (v2 ties leaked-mini-baseline) still shows the
  multi-turn loop limits trajectory-replay — but it did NOT mean "SFT never helps." The
  transfer model lifts the *generalization* number. Earlier over-claim corrected.
- **Caveats:** single eval per model; only ep20 tested on full val. **Confirmatory
  full-val evals of ep5 + ep16 queued** to verify across the epoch curve.

**Redirect (softened):** SFT gives a real +5.3 lift on the reportable set — a reasonable
**warmup**, not a dead end. On-policy methods (RL on ANLS, OPD) remain the next lever to
push further, with SFT as a viable cold-start.

## Methodological lesson (load-bearing for the RL phase)
**Screen on docvqa_mini, but CONFIRM verdicts on the full 80-Q val.** The 29-Q mini
(SE≈5%, median-difficulty docs) flattered the baseline and produced a false null that
survived four experiments. Any "X beats/ties baseline" claim from mini must be confirmed
on full val before it goes in a report or drives a decision.

## Carry-forward lessons (scaffold-independent, still valid)
- **/tmp PNG leak (FIXED `docvqa/sandbox.py`):** `batch_look` wrote a `delete=False` temp
  PNG per image and never removed them → ~985G of `tmp*.png` accumulated across runs and
  filled the disk. Fix = `try/finally os.remove` after the synchronous proxy returns.
- **Parallel eval trick:** during SFT *training* the 27B VLM is idle; re-serve it DP=2
  (free a GPU) to eval checkpoints in parallel with training. Turned a ~10h serial eval
  phase into a few hours. (Eval is VLM-bound, so DP throughput is the cap.)
- **Eval methodology:** train on *other* datasets (MMLongBench-Doc) → the full DocVQA
  val is leakage-free, so **no splits** — `docvqa_mini.json` (29 Q / 8 docs, 1 median
  doc per category) for quick iteration, `questions.json` (80 Q) for the reportable
  number (vs docvqa baselines: ≤8B 0.1875, 27B 0.375).
- **LoRA-without-regret:** apply to **all-linear** (MLP matters most), LR ≈ 10–15×
  FullFT (→ ~2e-4 for these short <100-step runs), rank not capacity-limiting at this
  data scale; constant schedule for short runs.
- **VLM perception:** the concise-answer+rationale instruction did **not** help the 27B
  (submit-only 0.373 vs bare-query 0.40, n=1/80) — but that's no proxy for whether it
  helps the weaker 4B; the clean retrain tests it.
- **Per-turn 4K cap is binding for the student** (clips ~13% of turns post-SFT vs 1% of
  teacher turns) — consider raising `max_response_tokens_per_turn` to 8K.
- **Dominant loss term = long-doc runaway / wall_cap** (lives in the agent scaffold),
  not answer quality — caps every method's ceiling; reward shaping / per-turn cap are
  the levers.
