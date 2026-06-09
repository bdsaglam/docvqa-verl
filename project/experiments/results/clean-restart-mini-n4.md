# Clean-restart eval — baseline vs seqkd-clean-v5 (docvqa_mini, n=4)

> First eval under the **fixed scaffold** (stop seqs + parse_first_fence + no-fence
> prompt). All earlier SFT cards measured corrupted-scaffold models and are void.
> Date 2026-06-08. Run dirs: `outputs/runs/{baseline,clean-v5}-cleanmini-n4`.

## Setup
- **Eval set:** `data/docvqa-2026/val/docvqa_mini.json` — 29 Q / 8 docs (1 median-size
  doc per category). Leakage-free: clean-v5 trained on MMLongBench-Doc, not DocVQA.
- **Protocol:** n=4, temp 0.6 / p0.95 / k20, thinking ON, concise+rationale VLM,
  rollout-timeout 1800, conc 8. Agent LM served on GPU3:8930 (bf16, len 65536,
  enforce-eager); VLM = 27B on :8927. Metric = binary ANLS @ 0.9.
- **Models:** baseline = untrained `Qwen/Qwen3.5-4B`; clean-v5 =
  `checkpoints/docvqa-verl/seqkd-clean-v5/merged_hf` (LoRA r32 α32 all-linear,
  LR 2e-4 constant, 3 ep, fp32-master+bf16-compute, on `data/sft/clean_v5.parquet`
  = 245 clean traj / 89 Q from mmlb-collect-v4, anls==1.0, first-fence-clean).

## Results (116 rollouts each)
| metric | baseline (untrained) | clean-v5 (SFT) | Δ |
|--------|---------------------|----------------|---|
| **overall ANLS@0.9** | **0.1897** | **0.1379** | **−0.052** |
| submit-only ANLS | 0.2857 (n=77) | 0.2162 (n=74) | −0.070 |
| pass@4 | 0.4138 | 0.3448 | −0.069 |
| SC-4 (self-consistency) | 0.3103 | 0.1724 | −0.138 |
| submit / token_cap / iter_cap / wall_cap | 77 / 30 / 9 / 0 | 74 / 31 / 9 / 2 | — |
| turns med / mean / max | 10 / 11.2 / 30 | 9 / 9.9 / 29 | — |

### Per-category overall (acc, n rollouts)
| category | baseline | clean-v5 |
|----------|----------|----------|
| business_report (2) | 0.500 | 0.375 |
| comics (3) | 0.000 | 0.000 |
| engineering_drawing (3) | 0.000 | 0.083 |
| infographics (8) | 0.3125 | 0.219 |
| maps (3) | 0.000 | 0.000 |
| science_paper (2) | 0.125 | 0.000 |
| science_poster (5) | 0.250 | 0.250 |
| slide (3) | 0.167 | 0.000 |

## Verdict — SFT (transfer) does NOT beat baseline; it slightly DEGRADES it
- clean-v5 is **worse on every aggregate** (overall, submit-only, pass@4, SC-4).
- The drop is **NOT a termination/runaway effect**: submission rate is essentially
  unchanged (77→74 submits, token/iter caps near-identical). The loss is in
  **answer quality among submitted answers** (submit-only 0.286 → 0.216).
- So SFT on **out-of-domain MMLongBench-Doc** transfer trajectories taught the 4B a
  style/answer-format that *transfers negatively* to DocVQA — it submits at the same
  rate but answers slightly worse. The SC-4 collapse (0.31→0.17) suggests SFT also
  **reduced answer diversity** (mode-collapsed toward teacher phrasings that miss
  the binary@0.9 bar).
- **The real lever was the scaffold fix** (untrained 4B: 0.042 corrupted → 0.190
  fixed). Transfer-SFT on top is net-negative here.

## What this does NOT settle
Whether SFT *can* help at all, or whether the failure is the **domain gap**
(MMLongBench ≠ DocVQA) vs the **SFT setup itself**. That is the next experiment:
the in-domain learnability upper-bound (27B-teacher trajectories on DocVQA val →
SFT → eval on val). If even in-domain val-leaked SFT can't beat 0.19, the setup is
the limiter; if it can, the problem was the transfer gap.
