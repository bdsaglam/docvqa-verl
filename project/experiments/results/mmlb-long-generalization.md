# mmlb-long: does longer transfer-SFT generalize to DocVQA? (docvqa_mini, n=4)

> 2026-06-09. The generalization counterpart to the in-domain memorization test.
> Run dirs: `outputs/runs/mmlb-long-ep{5,11,16,20}-cleanmini-n4`.

## Question
clean-v5 (mmlb→DocVQA transfer, **3 epochs**) lost to baseline (13.8% < 19.0%). Was
that undertraining (like in-domain v1) or a real transfer failure? Train the **same
245 mmlb trajectories** much longer (20 epochs, checkpoint every 5) and eval each on
DocVQA — **no leakage → real generalization curve**.

## Setup
- Data: `data/sft/clean_v5.parquet` (245 MMLongBench-Doc transfer traj). LoRA r32 α32
  all-linear, LR 2e-4 constant, fp32-master+bf16-compute, GPU3. 300 steps / 20 epochs,
  SAVE_FREQ=80 → checkpoints at steps 80/160/240/300 = ep~5/11/16/20.
- Eval: docvqa_mini (29 Q), n=4, temp 0.6/p0.95/k20, 27B VLM. **Evaluated in parallel
  across 2 GPUs** (27B re-served DP=2 on GPUs 0,1 — idle during training — freeing
  GPU2+GPU3 for the 4B agent servers).

## Results (vs baseline 19.0%, submit-only 28.6%)
| mmlb training | train loss | overall | submit-only |
|---|---|---|---|
| 3 ep (clean-v5) | 0.18 | 13.8% | — |
| 5 ep | 0.097 | 20.7% | 30.0% |
| 11 ep | 0.021 | 14.7% | 22.7% |
| 16 ep | 0.008 | 19.0% | 27.5% |
| 20 ep | 0.006 | 21.6% | 30.5% |
| **baseline** | — | **19.0%** | **28.6%** |

## Mini verdict (29 Q) — looked like a null, but the screen was UNDERPOWERED
On docvqa_mini the curve bounces (20.7 → 14.7 → 19.0 → 21.6) within the ±5% noise floor
(29 Q, σ≈28% → SE≈5%) and ep20 (21.6%) is statistically indistinguishable from baseline
(19.0%). **This led to a premature "SFT ties baseline / null" conclusion — WRONG, see
below.** The mini set is 1 *median*-difficulty doc per category, which flatters the
untrained baseline and is too small to resolve a ~5-pt effect.

## ★ FULL-VAL (80Q) — the reportable result: ep20 SFT BEATS baseline (2026-06-10)
Ran baseline + ep20 on the full 80-Q val (n=4, 27B VLM DP=3), paired per-question:

| 80Q full-val | overall | submit-only | pass@4 |
|---|---|---|---|
| baseline (untrained 4B) | **15.3%** | 25.1% | 33.8% |
| **mmlb-long ep20 (SFT)** | **20.6%** | 32.2% | 41.2% |
| **Δ** | **+5.3** | +7.1 | +7.4 |

**Paired t-test on per-question ANLS: Δ=+5.3%, SE 2.5%, t=2.09 (79 df), p≈0.04 —
statistically significant.** All three metrics move the same way.

**Why mini hid it:** the SFT model is *stable* across sets (mini 21.6% ≈ full 20.6%),
but the **baseline collapses on the harder full distribution** (mini 19.0% → full 15.3%).
SFT's benefit is **robustness on harder docs**, invisible on the easy/median mini set.

## REVISED conclusion — SFT (sufficiently trained) DOES help, modestly
- **A 20-epoch mmlb→DocVQA transfer SFT beats the untrained baseline on the full val:
  20.6% vs 15.3%, +5.3 ANLS, paired p≈0.04.** Not a null.
- **Undertraining still hurts** (clean-v5, 3ep, 13.8% on mini) — the benefit needs the
  full epoch budget.
- **Caveats:** single eval per model (one set of n=4); only ep20 tested on full val so
  far (mini curve was non-monotonic). **Confirmatory full-val evals of ep5 + ep16 are
  queued** to check robustness across the epoch curve.
- The earlier "memorization doesn't transfer" framing was over-stated: it holds for the
  *leaked in-domain* eval (v2 ties baseline on mini) but the *transfer* model does lift
  the *generalization* number on the representative set.

**Redirect still stands but softened:** SFT now gives a real, modest lift (+5.3 on full
val). On-policy methods (RL on ANLS, OPD) remain the next lever to push further — SFT is
a reasonable *warmup*, not a dead end.
