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

## Verdict — null: transfer SFT plateaus at baseline, no generalization gain
- The curve **bounces (20.7 → 14.7 → 19.0 → 21.6) within the ±5% noise floor** (29 Q,
  σ≈28% → SE≈5%). Non-monotonic = noise around a flat line, not an epoch→score trend.
- Pooling the three adequately-trained points (ep5/16/20 = **20.4%** over 348 rollouts,
  SE≈1.5%) gives a ~1.4-pt nominal lift over baseline — **under 1 SE, not significant.**
- The only real effect: clean-v5's **13.8% was undertraining** (loss-0.18 point); enough
  epochs recover to ≈baseline but nothing exceeds it. No checkpoint clears the ~24% bar
  for a full-val (80Q) promotion, so none was promoted.

## Conclusion — SFT investigation complete (consistent null across ALL variants)
| variant | overall | vs baseline 19.0% |
|---|---|---|
| transfer undertrained (clean-v5, 3ep) | 13.8% | hurts |
| transfer long (mmlb-long, 5-20ep) | ≈19-21% | ties |
| in-domain undertrained (v1, 3ep) | 14.7% | hurts |
| in-domain memorized+leaked (v2, 20ep) | 19.0% | ties |

**SFT on 27B-teacher CodeAct trajectories never beats the untrained 4B baseline** —
in-domain or transfer, undertrained (hurts) or fully fit (ties). Memorizing the
teacher's trajectories (train loss → ~0) does not transfer to free-rollout eval,
because the multi-turn agentic loop feeds the model its *own* (different) VLM
observations at inference — it can't replay a memorized trajectory. The scaffold/prompt
fix (0.042→0.19) remains the only lever that has moved the metric.

**Redirect (unchanged): SeqKD/SFT is not the lever.** Move to on-policy methods where
the model learns from its own rollouts: RL on answer-level ANLS reward (GRPO etc.), or
on-policy distillation (dense per-token signal, no answer-distribution collapse). The
clean, comprehensive SFT null is a real result — it redirects the project's effort.
