# Autonomous SFT-improvement sweep (user away 2026-06-06 ~22:45)

**Mandate:** make SFT work — try hyperparams + data combinations, measure honestly.

## The real bottleneck (read first)
1. **Eval is the binding constraint**, not training. Eval is VLM-latency-bound
   (~136s/look) and **serial on the shared 27B server**. Training is cheap (GPU3,
   ~30min–3h). So: train many variants, but **eval only the few most promising**.
2. **Prior SFT verdict:** all variants clustered in the **n=1/24 noise floor**
   (~0.04–0.08) — we could not tell if anything beat baseline. **So step 1 of
   "making SFT work" is a denoised eval (n≥4)** that can actually distinguish configs.
   Without that, sweeping is blind.
3. Dominant loss term = non-submission / long-doc runaway (wall_cap), which SFT
   alone barely moves.

## Hypotheses (each = one model to eval)
- **H-data-scale:** more teacher data helps? (v1 87 → grown 275 → full-grown ~385)
- **H-consistency:** new-scaffold-only data (rationale VLM + print prompt) beats
  the old/mixed-scaffold data? (new-only vs grown-mixed)
- **H-lr:** LR 2e-4 (LoRA-post: we underfit at 1e-4) — Arm A/v3a vs v1.
- **H-overfit:** fewer epochs (2) better on tiny data?

## Fixed recipe (LoRA-post "low-regret" compliant)
LoRA r32 α32 `all-linear`, grad-ckpt on, fp32-master+bf16-compute, batch 16
micro 1, MAX_LENGTH 32768, sdpa, FSDP2, 1 GPU. Vary only LR / epochs / schedule /
data. Default good hparams: **LR 2e-4, WARMUP_STYLE=constant, 3 epochs.**

## Datasets
| id | file | composition |
|----|------|-------------|
| v1 | mmlb_transfer.parquet (87) | old-scaffold, anls=1.0 |
| v2 | mmlb_transfer_v2.parquet (185) | old-scaffold |
| filt-v2 | mmlb_transfer_v2_filt12k.parquet (155) | old, ≤12k tok |
| grown_v4 | grown_v4.parquet (275) | filt-v2 + 120 new (mixed scaffold) |
| **D2 new-only** | sft/collect_v3_newonly.parquet (~160–200) | **new-scaffold only**, anls=1.0, ≤12k, cap3/Q |
| **D3 full-grown** | sft/grown_full.parquet (~385) | v2 + all new, ≤12k |

## Models & training battery (GPU3, serial; cheap)
| model | data | hparams | status |
|-------|------|---------|--------|
| v1 (seqkd-transfer-mp) | v1 | LR1e-4 cos 3ep | DONE (merged) |
| Arm A (v3a) | v2 | LR2e-4 const 3ep | DONE (merged) |
| Arm B (v3b) | filt-v2 | LR2e-4 const 3ep | DONE (merged) |
| **v4-grown** | grown_v4 | LR2e-4 const 3ep | TRAINING |
| **v5 new-only** | D2 | LR2e-4 const 3ep | pending (after collection) |
| **v6 full-grown** | D3 | LR2e-4 const 3ep | pending |
| **v7 full-grown-2ep** | D3 | LR2e-4 const **2ep** | pending (overfit control) |

## Eval methodology (CHANGED 2026-06-07 per user — NO DocVQA splits)
We train on **other datasets only** (MMLongBench-Doc), so the *entire* DocVQA-2026
val is leakage-free eval. Drop the strat24/train/heldout splits. Two eval sets:
- **`docvqa_mini.json`** (29 Q / **8 docs**, 1 median-size doc per category, ALL its
  questions) — **quick** iteration / relative model comparison.
- **`questions.json`** (full **80 Q / 25 docs**) — **reportable**, comparable to the
  docvqa-repo baselines (≤8B best 0.1875; 27B 0.375). No subset.
Protocol: n=4, temp 0.6/p0.95/k20, thinking ON, **new concise+rationale VLM**,
rollout-timeout 1800, serve merged_hf on GPU3 :8930, 27B VLM :8927, serial per model.

Two-tier run:
1. **MINI:** v1, v4-grown, v5-newonly, v6-fullgrown, **baseline (last)** on docvqa_mini
   (run-dir `<name>-mini-n4`) → rank SFT + the standing baseline anchor for the quick
   set (one-time, reused by all future quick evals — worth the VLM time).
2. **FULL-VAL reportable:** baseline + best + v1 on questions.json
   (run-dir `<name>-val-n4`) → the number we compare to docvqa baselines.

(Old strat24-n4 partials kept for reference: baseline-strat24 = 0.1667. Not comparable
to baselines; superseded by the above.)

## State machine (driver cron advances this)
1. COLLECT — collection running + v4-grown training. (now)
2. BUILD — collection done → build D2, D3.
3. TRAIN — v4 done → train v5, v6, v7 (GPU3 serial) + merge each.
4. EVAL — all trained + collection stopped → serve+eval battery n4 strat24.
5. SYNTH — write per-model result cards + update SFT-synthesis.md with the
   denoised verdict: does any SFT config beat baseline? which lever mattered?

## Results — docvqa_mini n=4 binary@0.9 (CLEAN scaffold, 2026-06-08)
| model | overall | submit-only | pass@4 | SC-4 | notes |
|-------|---------|-------------|--------|------|-------|
| **baseline (untrained 4B)** | **0.1897** | 0.286 (77) | 0.414 | 0.310 | fixed scaffold; the anchor |
| **seqkd-clean-v5** | **0.1379** | 0.216 (74) | 0.345 | 0.172 | transfer-SFT (mmlb 245 traj) — **WORSE than baseline** |

**VERDICT: transfer-SFT loses.** Same submit rate (77→74), worse answer quality
(0.286→0.216) + SC-4 diversity collapse. Card: `results/clean-restart-mini-n4.md`.
Full-val skipped (was gated on beating baseline). Old corrupted-scaffold v1–v6
numbers are void.

## Honesty guardrail
If the denoised eval shows **no SFT config clears baseline**, that is the finding —
report it plainly (SFT at ceiling for this data scale; the runaway/data bottleneck
dominates) rather than cherry-picking noise. A null result, well-measured, is a result.

## NEXT PHASE — in-domain DocVQA trajectories (user, 2026-06-08)
Goal: does training on **in-domain** (DocVQA) trajectories help — vs the MMLongBench
out-of-domain transfer (clean_v5)? Isolates the domain gap. "See if it learns."

Plan (after the current clean eval finishes):
1. **Collect 27B-teacher trajectories on DocVQA val** (eval.py = collection, fixed
   scaffold, n4) → outputs/runs/docvqa-collect. The 27B (≈0.4 on docvqa) yields more
   anls=1.0 in-domain trajectories than the 4B would.
2. Build clean SFT (make_clean_sft.py) → data/sft/indomain_v1.parquet.
3. Train seqkd-indomain-v1 (LR2e-4 const), merge.
4. Eval vs baseline + clean_v5 on docvqa_mini / full val.

**LEAKAGE CAVEAT:** training on val trajectories + eval on val = optimistic (memorization),
so the val number is a learnability UPPER BOUND, not generalization. Clean options:
 (a) train on val/train.json (56Q/17 docs) → eval on val/heldout.json (24Q/8 docs, disjoint);
 (b) train on all val → report only the official TEST submission.
**CONFIRMED BY USER (2026-06-08): leaked full-val is intended — a learnability
UPPER-BOUND of the current SFT setup** (train on the target's own trajectories, eval on
the same val). Interpretation: if the 4B barely improves even with in-domain val training,
the SFT *setup* is the limiter, not the domain gap. Generator = **27B teacher on full val**
(highest-quality in-domain trajectories → strongest achievable upper bound; clean_v5
self-RFT would be a weaker lower variant). NOT a generalization number — that needs test.
