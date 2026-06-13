# Pillar B — Weight-level training of the small agent (mined results)

> Consolidated from `outputs/runs/<run>/` eval dirs in `docvqa-verl`, recomputed
> from raw data (`results.json` where present, else `tasks/*/trajectories.jsonl`).
> Metric is **binary ANLS@0.9** (the ICDAR-2026 metric), averaged over all n×Q
> samples — this is the `summary.overall_accuracy` field, and it equals the mean
> of the per-record `is_correct` flag. All eval on the **DocVQA-2026 val** set
> (80Q over 25 docs, 8 categories ×10Q) at temp 0.6 / top_p 0.95 / top_k 20,
> thinking ON, VLM = frozen Qwen3.5-27B on :8927, rollout_timeout 1800s.
> Mined 2026-06-13. **Pillar-B story is PRELIMINARY/DEFERRED** — the SFT verdict is
> a measured null, the only RL point is a partial single-checkpoint screen.

## Trajectory schema (probed first, as requested)

Each line of `tasks/<doc_id>/trajectories.jsonl` is one rollout sample. Top-level keys:

```
record_id, question_id, doc_id, category, question, gold_answer, model, vlm_model,
sampling, sample_idx, submitted_answer, extracted_answer, is_correct, anls,
termination, num_turns, vlm_calls, turns_truncated, max_turn_tokens, wall_clock_s,
messages, prompt_ids, response_ids, response_mask
```

Load-bearing fields for this analysis:
- **`is_correct`** (bool) — binary ANLS@0.9 for this sample. **`anls`** is stored as the
  *same* thresholded value (anls == is_correct in every run here, i.e. these dumps carry
  the binarized score, not the raw 0–1 ANLS), so submit-only / per-bucket numbers below
  are all binary@0.9.
- **`termination`** ∈ {`submit`, `token_cap`, `iter_cap`, `wall_cap`, `parse_error`}.
  `submit` = agent called `SUBMIT(...)`. The three `*_cap` values = the rollout exhausted
  its turn-token / iteration / wall-clock budget **without ever submitting** — the
  answer-quality-INDEPENDENT "runaway" failure class. `parse_error` = no parseable code
  fence (rare: ≤1 per run for 4B; a 3.5-family format issue).
- **`num_turns`**, **`vlm_calls`** (= `batch_look` calls), `turns_truncated`,
  `wall_clock_s` — effort / shape signals.
- `result.json` (per-doc) and `results.json` (run-level: `summary.overall_accuracy`,
  `pass@4`, `sc-4`, `by_category`) present for completed runs; the two RL-pair runs and
  `mmlb-long-ep5-cleanval` have **no** usable `summary` (ep5-cleanval was a deliberate
  user skip; the RL pair is mid-flight) — those are computed from trajectories.

---

## (a) Headline findings

1. **Preliminary RL is a wash, not a win — but rollouts are healthy.** The only live RL
   point, `rl-4b-curriculum-val-n4` (GRPO 4B, async curriculum, **global_step_30**), on the
   **33 questions it shares** with its matched base run scores **6.1% vs base 8.5%** ANLS@0.9
   — slightly *worse*, deep inside noise (n=2 samples, 33Q, both runs **incomplete**:
   45/80 and 34/80 Q done). No degeneracy: submitted answers are diverse and sensible
   (`'34'`,`'79'`,`'Hortense the Lovable Brat...'`, not collapsed/gibberish), submit-only
   quality is identical (14.8% vs 14.9%), and runaway actually *drops* (32.7% vs 43.9%).
   **Verdict: at 30 steps GRPO has not moved answer accuracy; treat as a smoke-test that
   the loop runs and rollouts stay coherent, not as evidence for/against RL.**

2. **SFT gives a small, training-dependent lift on the 4B — and it is real on full-val.**
   Base 4B = **15.3%** → mmlb-long **ep16 = 17.2%** → **ep20 = 20.6%** (+5.3 pts, the
   confirmed headline). The gain is concentrated in `pass@4` (33.8%→41.2%) and
   submit-only quality (25.1%→32.2%), i.e. SFT teaches the perception loop to extract
   better, not to submit more often (submit-rate 61%→64%, flat). But it is **fragile**:
   on the 29Q mini screen the curve bounces (ep5 20.7 / ep11 14.7 / ep16 19.0 /
   ep20 21.6) within the ±5pt small-n noise floor, and earlier (mmlb-long-generalization
   card) this same curve read as a NULL. The honest read: **SFT-on-teacher-trajectories
   buys a few points at best and is not a reliable lever; multi-turn observation-shift
   blunts trajectory imitation.**

3. **SFT actively HURTS the Qwen3-8B (Qwen3 series, non-hybrid).** 8B base = **5.9%**;
   after the same mmlb SFT recipe, `seqkd-8b-mmlb` = **4.4%** (−1.5). The 8B's whole
   pathology is the opposite of the 4B's: it **shortcuts perception** — median **2 turns /
   1 VLM call** (vs 4B's 9 / 9), submits fast on 87–91% of rollouts, runaway only 3–6% —
   and SFT pushes it *further* into submit-fast-shallow (91% submit, 1 VLM call). The
   CodeAct scaffold/prompt is tuned to the Qwen3.5 family; the Qwen3-series 8B does not
   inhabit the loop, so it is **not** a free capacity upgrade.

4. **Long-document runaway is the dominant, method-independent error class.** On
   base-4B full-val, **38.8% of all rollouts hit a cap without ever submitting**, and the
   per-doc runaway rate correlates with page count at **Pearson r = 0.70** (n=25 docs):
   1–5pg docs runaway 23%, 50+pg docs runaway **59%**, and the single 181-page doc
   runs away **100%** of the time. Accuracy collapses in lockstep: 1–5pg 18.4% →
   50+pg **7.9%**. This caps every method's ceiling before answer-quality even enters.

5. **In-domain leaked SFT cannot beat untrained, even memorized.** `indomain-v2` (trained to
   loss≈0.003 on 80 in-domain trajectories, leaked) ties baseline on mini (19.0% =
   baseline) while `indomain-v1` (undertrained) sits at 14.7%. Trajectory SFT's upper bound
   under this scaffold is "tie the base model" — confirming the lever is on-policy
   (RL/OPD), not more/cleaner SFT data.

---

## (b) Master results tables

### Full-val (DocVQA-2026 val, 80Q over 25 docs, n=4) — the reportable numbers

| Run | Model / stage | ANLS@0.9 | pass@4 | sc-4 | submit% | submit-only | runaway% | med turns / VLM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-cleanval-n4` | Qwen3.5-4B base | **15.31%** | 33.8% | 27.5% | 61% | 25.1% | 38.8% | 9 / 9 |
| `mmlb-long-ep16-cleanval-n4` | 4B SFT mmlb ep16 | 17.19% | 36.2% | 26.2% | 64% | 26.7% | 34.7% | 9 / 9 |
| `mmlb-long-ep20-cleanval-n4` | 4B SFT mmlb **ep20** | **20.62%** | 41.2% | 28.7% | 64% | 32.2% | 35.6% | 9 / 8 |
| `qwen3-8b-baseline-cleanval-n4` | Qwen3-8B base | 5.94% | 16.2% | 6.2% | 87% | 6.9% | 5.6% | 2 / 1 |
| `seqkd-8b-mmlb-cleanval-n4` | Qwen3-8B SFT mmlb | 4.38% | 7.5% | 5.0% | 91% | 4.8% | 3.1% | 1 / 0 |

SFT 4B full-val curve (the explicit headline): **base 15.3 → ep16 17.2 (+1.9) → ep20 20.6 (+5.3)**.
(ep5/ep11 full-val do not exist — ep5-cleanval was a user opt-out; both are mini-only.)

### Mini screen (29Q, n=4) — denoising screen; baseline column is the full-80 base (no 29Q base run exists)

| Run | ANLS@0.9 | pass@4 | sc-4 | submit% | submit-only | runaway% |
|---|---:|---:|---:|---:|---:|---:|
| 4B base (full-80 ref) | 15.31% | 33.8% | 27.5% | 61% | 25.1% | 38.8% |
| `mmlb-long-ep5-cleanmini` | 20.69% | 44.8% | 27.6% | 69% | 30.0% | 29.3% |
| `mmlb-long-ep11-cleanmini` | 14.66% | 37.9% | 27.6% | 65% | 22.7% | 34.5% |
| `mmlb-long-ep16-cleanmini` | 18.97% | 37.9% | 24.1% | 69% | 27.5% | 31.0% |
| `mmlb-long-ep20-cleanmini` | 21.55% | 37.9% | 27.6% | 71% | 30.5% | 28.4% |
| `indomain-v1-cleanmini` (undertrained, leaked) | 14.66% | 34.5% | 24.1% | 67% | 21.8% | 32.8% |
| `indomain-v2-cleanmini` (memorized, leaked) | 18.97% | 34.5% | 27.6% | 77% | 24.7% | 22.4% |

Mini bounces within ~±5pt → treat the mini curve as a noise screen, full-val as the verdict.

### Preliminary RL pair (DocVQA-2026 val, **n=2**, INCOMPLETE) — compute from trajectories

| Run | Stage | Q done /80 | ANLS@0.9 (own Q) | ANLS@0.9 (common 33Q) | runaway% | submit% | submit-only | med turns / VLM |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `base-4b-val-n4` | Qwen3.5-4B base (n=2) | 34 | 8.33% | **8.46%** | 43.9% | 56% | 14.9% | 9 / 9 |
| `rl-4b-curriculum-val-n4` | GRPO 4B async curriculum, **step 30** (n=2) | 45 | 9.88% | **6.06%** | 32.7% | 67% | 14.8% | 9 / 8 |

The **common-33Q paired** column is the only fair read (different Q-subsets otherwise):
**RL 6.06% vs base 8.46%** — flat-to-slightly-worse, n=2, both incomplete. Checkpoint =
`docvqa-grpo-4b-async-curriculum/global_step_30/merged_hf`. No degenerate rollouts.

### Per-category accuracy (full-val, n=4)

| Category | 4B base | 4B ep16 | 4B ep20 | 8B base | 8B SFT |
|---|---:|---:|---:|---:|---:|
| business_report | 10.0 | 12.5 | 10.0 | 0.0 | 0.0 |
| comics | 5.0 | 5.0 | 10.0 | 0.0 | 2.5 |
| engineering_drawing | 25.0 | 20.0 | 12.5 | 2.5 | 0.0 |
| infographics | 27.5 | 35.0 | **52.5** | 12.5 | 10.0 |
| maps | 0.0 | 0.0 | **0.0** | 2.5 | 0.0 |
| science_paper | 10.0 | 15.0 | 20.0 | 5.0 | 5.0 |
| science_poster | 22.5 | 25.0 | 27.5 | 2.5 | 0.0 |
| slide | 22.5 | 25.0 | 32.5 | 22.5 | 17.5 |

SFT's ep20 gain is **almost entirely infographics (27.5→52.5) + slide + science_paper** —
short-doc, dense-perception categories. **maps = 0% for every 4B run** (hard floor;
engineering_drawing actually *regresses* under SFT). The categories SFT helps are exactly
the ones with low runaway.

---

## (c) Runaway / failure-mode quantification

**Definition:** runaway = `termination ∈ {token_cap, iter_cap, wall_cap}` (capped without a
SUBMIT). Answer-quality-independent.

Runaway and accuracy by page-count bucket (full-val, n=4):

| Bucket (docs) | 4B base runaway | 4B base ANLS | 4B ep20 runaway | 4B ep20 ANLS | 8B base runaway | 8B base ANLS |
|---|---:|---:|---:|---:|---:|---:|
| 1–5 pg (9 docs) | 22.8% | 18.4% | 26.5% | 25.0% | 2.2% | 5.1% |
| 6–20 pg (4) | 37.5% | 25.0% | 33.3% | 27.1% | 4.2% | 16.7% |
| 21–50 pg (6) | 50.0% | 10.0% | 46.7% | 20.0% | 5.0% | 6.7% |
| 50+ pg (6) | **59.2%** | **7.9%** | 44.7% | 9.2% | 13.2% | 0.0% |
| **overall** | **38.8%** | 15.3% | 35.6% | 20.6% | 5.6% | 5.9% |

- **Page-count is the driver:** per-doc runaway rate vs `num_pages`, **Pearson r = 0.696**
  (n=25). 1-page docs runaway ~10–25%; the 181-page `business_report_1` runs away 100% of
  samples, the 105-page `business_report_2` 75%.
- **SFT reduces long-doc runaway** (50+pg: base **59.2% → ep20 44.7%**, overall 38.8→35.6)
  and roughly **doubles** 21–50pg accuracy (10→20%) — consistent with SFT teaching the loop
  to terminate. But it does **not** fix the 50+pg accuracy floor (7.9→9.2%): even when the
  ep20 model submits, it can't answer the long-doc question.
- **The 8B has the opposite failure:** ~3–6% runaway because it submits almost immediately
  (median 1–2 turns). Low runaway here is a *symptom of under-perception*, not of competence
  — its accuracy (5.9%) is far below the 4B's despite never running away.

**Implication for the report:** runaway on long docs is a structural ceiling shared by base,
SFT, and (presumably) RL. Any headline ≤8B number is partly a "did the agent finish in
budget on a 100-page PDF" number. This is a scaffold/budget problem orthogonal to the
weight-training method, and the single biggest lever visible in the data.

---

## (d) Proposed report tables / figures

- **Table B1 — "SFT lift on the 4B agent (DocVQA-2026 val, n=4)."** Rows: base 15.3,
  ep16 17.2, ep20 20.6; cols ANLS@0.9 / pass@4 / submit-only. Caption: "Trajectory SFT on
  27B-teacher CodeAct rollouts adds +5.3 pts at 20 epochs, concentrated in extraction
  quality (submit-only 25→32%), not submit rate (61→64%, flat). Gain is fragile on the 29Q
  screen — read full-val as the verdict."

- **Table B2 — "SFT direction reverses with backbone."** 4B base 15.3 → SFT 20.6 (+5.3);
  8B base 5.9 → SFT 4.4 (−1.5). Caption: "The CodeAct loop is Qwen3.5-tuned; the Qwen3-series
  8B shortcuts perception (median 1–2 turns vs 9) and SFT entrenches the shortcut. Larger
  ≠ better here."

- **Figure B3 — "Runaway is a long-document failure" (scatter + bucket bars).** Scatter:
  x = `num_pages` (log), y = per-doc runaway rate, base-4B, r=0.70 fit line. Companion bars:
  runaway% and ANLS by page bucket for base vs ep20. Caption: "38.8% of base-4B rollouts
  hit a budget cap without submitting; rate rises from 23% (≤5pg) to 59% (50+pg) and 100%
  on the 181-page doc. SFT trims long-doc runaway (59→45%) but not the long-doc accuracy
  floor (~8–9%)."

- **Table B4 — "Per-category profile."** The 8×5 matrix above. Caption: "SFT gains live in
  infographics/slide/science_paper (dense single-/few-page); maps=0 everywhere,
  engineering_drawing regresses."

- **Table B5 (caveated, appendix) — "Preliminary GRPO, 30 steps."** RL 6.1 vs base 8.5 on
  the paired 33Q (n=2, incomplete). Caption: "Smoke-test only: confirms the async-GRPO loop
  produces coherent, diverse rollouts and reduces runaway (44→33%), but has not moved answer
  accuracy at 30 steps. Not evidence for or against RL."

---

## (e) Gaps / caveats

- **RL is one partial checkpoint at n=2.** `rl-4b-curriculum-val-n4` is `global_step_30`,
  n=2, **45/80 Q done**; matched base is **34/80 Q done**; they overlap on 33 Q. The 6.1 vs
  8.5 paired number rests on 33 questions × 2 samples = 66 rollouts per arm. **Do not report
  RL as a result** — report it as "loop runs, rollouts coherent, no movement yet." A proper
  RL point needs the full 80Q at n=4 and ideally a learning curve across steps.
- **Single checkpoint per SFT epoch, n=4.** Every SFT point is one merged checkpoint, one
  eval, n=4 — no seed variance. The 80Q full-val std is ~0.30 per the summary, so the SE on a
  0.20 mean over 320 samples is ~±1.7pt; the ep20 +5.3 is ~2–3 SE (the running notes cite
  p≈0.04). Treat as suggestive, not nailed.
- **Mini (29Q) is a noise screen, not a measurement** — ±5pt swings dominate; the same SFT
  curve reads as both "lift" and "null" depending on slice. Full-val is the only verdict
  surface.
- **`anls` field is pre-binarized** in these dumps (anls == is_correct). All "ANLS" numbers
  here are binary@0.9; raw-ANLS (continuous) is **not** recoverable from these files. The
  ICDAR metric is binary@0.9 so this is the right metric, but soft-ANLS analyses aren't
  possible from this data.
- **`mmlb-long-ep5-cleanval` and `-ep11-cleanval` do not exist** (ep5 was a user skip; ep11
  is mini-only) — the full-val SFT curve has only 3 points (base, ep16, ep20).
- **`docvqa-collect` (28.4%) is NOT an eval** — it is the in-domain teacher *collection* run
  (leaked, the SFT-data source); excluded from all comparisons, listed only for completeness.
- **Page buckets are coarse** (25 docs → 9/4/6/6 docs per bucket); the 6–20pg bucket is only
  4 docs so its numbers are thin.
- Eval set is the **public val** split (the leaderboard target is the hidden test); all
  numbers are val-set proxies.
