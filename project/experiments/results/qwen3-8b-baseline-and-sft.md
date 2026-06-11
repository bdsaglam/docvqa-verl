# Qwen3-8B: baseline + SFT on DocVQA full val (80Q, n=4)

> 2026-06-11. Motivation: Qwen3.5 (our 4B agent) is a **hybrid-reasoning** model that
> vLLM mishandles in RL, so we tested **Qwen3-8B** (non-hybrid, RL-friendly) as the agent
> LM — baseline, then SFT on teacher data, to carry into the RL phase.
> Runs: `outputs/runs/qwen3-8b-baseline-cleanval-n4`, `seqkd-8b-mmlb-cleanval-n4`.

## Results (full val, 80Q, n=4, 27B VLM DP=3, conc24)
| model | overall | submit-only | median turns | median batch_look |
|---|---|---|---|---|
| **4B baseline** (Qwen3.5-4B) | **15.3%** | 25.1% | 9 | 50 |
| 4B-SFT (mmlb ep20) | 20.6% | 32.2% | — | — |
| **Qwen3-8B baseline** | **5.9%** | 6.9% | 2 | 7 |
| **Qwen3-8B SFT** (mmlb ep20, loss 0.007) | **4.4%** | 4.8% | 1 | 8 |

## Findings
1. **Qwen3-8B fits the scaffold far worse than the 4B** (5.9% vs 15.3% baseline). It
   **shortcuts perception** — submits after ~1-2 turns / ~7 `batch_look` calls vs the
   teacher/4B's ~9 turns / ~50 looks. The CodeAct prompt was developed for the **Qwen3.5**
   family; **Qwen3** (different/older series) doesn't follow the survey→locate→extract→
   verify loop and jumps to `SUBMIT`.
2. **SFT did NOT help the 8B.** 4.4% is statistically tied with (marginally below) its
   5.9% baseline (±~1pt SE). Train loss converged to 0.007 (fully memorized the teacher
   trajectories), but that didn't transfer to free-rollout eval — same phenomenon as the
   4B in-domain memorization (the multi-turn loop feeds the model its own observations,
   so a memorized trajectory can't replay).
3. **It "half-took," bimodally.** ~48% of 8B-SFT rollouts DO run the multi-turn loop
   (up to 9+ turns) and those score **~7.5%**, vs ~52% that shortcut to 1 turn and score
   **1.8%**. So the right behavior IS in the distribution — it's just inconsistent, which
   drags the aggregate to ≈baseline. (SFT did clean up format: parse-errors 25→18.)

## Implication for the RL phase
A real fork:
- **4B**: much better at the task (15.3%/20.6%) but **hybrid → vLLM/RL-problematic.**
- **Qwen3-8B**: **RL-friendly** (non-hybrid) but weak (5.9%), and SFT can't lift it.

**RL is the natural lever for the 8B**, *because* the failure mode is inconsistency, not
inability: the 8B already produces multi-turn rollouts that score (7.5%); RL on
answer-level ANLS directly reinforces those and suppresses the 1-turn shortcut — exactly
what SFT (trajectory imitation) failed to make consistent. Options also include screening
another non-hybrid ≤9B model, or unblocking the 4B's vLLM-hybrid issue. SFT-as-warmup
bought ~nothing here, so RL-from-(near-)scratch on the 8B is reasonable.
