# In-domain learnability upper-bound — seqkd-indomain-v1 (docvqa_mini, n=4)

> The decisive SFT experiment. Date 2026-06-08. Run dir:
> `outputs/runs/indomain-v1-cleanmini-n4`. (Full-val 80Q run optional/separate.)

## Question
Transfer-SFT (clean-v5, MMLongBench→DocVQA) lost to baseline. Is that the **domain
gap** or the **SFT setup itself**? Test: collect 27B-teacher trajectories on DocVQA
**full val** (leaked-by-design), SFT on them, eval on the **same** val. If even this
can't beat baseline, the SFT setup is the limiter; if it can, the prior failure was
the domain gap.

## Setup
- **Data:** `data/sft/indomain_v1.parquet` — 80 clean trajectories / 35 Q, built by
  `make_clean_sft.py` from `outputs/runs/docvqa-collect` (27B teacher on the 80-Q val,
  320 rollouts, 91 anls=1.0; first-fence-clean, stray-`</think>`=0). **Leaked by
  design** (train on val trajectories, eval on val) → a learnability UPPER BOUND, not
  generalization.
- **Recipe:** identical to clean-v5 — LoRA r32 α32 all-linear, LR 2e-4 constant, 3 ep,
  fp32-master+bf16-compute, FSDP2, GPU3. Final train loss ~0.225.
- **Eval:** `docvqa_mini.json` (29 Q, subset of val → also leaked), n=4, temp 0.6/
  p0.95/k20, thinking ON, concise+rationale VLM, timeout 1800, 4B on GPU3:8930, 27B
  VLM :8927. Metric = binary ANLS @ 0.9.

## Results (116 rollouts) — vs the two prior models on the same set
| metric | baseline (untrained) | clean-v5 (transfer SFT) | **indomain-v1 (in-domain, leaked)** |
|--------|----------------------|-------------------------|-------------------------------------|
| **overall ANLS@0.9** | **0.1897** | 0.1379 | **0.1466** |
| submit-only ANLS | 0.2857 (77) | 0.2162 (74) | 0.2179 (78) |
| pass@4 | 0.4138 | 0.3448 | 0.3448 |
| SC-4 | 0.3103 | 0.1724 | 0.2414 |
| submit / token_cap / iter_cap | 77 / 30 / 9 | 74 / 31 / 9 | 78 / 30 / 8 |
| turns med / mean / max | 10 / 11.2 / 30 | 9 / 9.9 / 29 | 9 / 9.7 / 26 |

## Verdict — the SFT SETUP is the limiter, NOT the domain gap
- **In-domain leaked SFT (0.1466) does NOT beat baseline (0.1897).** It edges out
  transfer-SFT (0.1379) but stays well below the untrained model.
- **Same failure mechanism as transfer:** submission rate is unchanged (77→78); the
  loss is entirely in **answer quality among submissions** (submit-only 0.286→0.218).
  SFT on 27B-teacher CodeAct trajectories degrades the 4B's answers **regardless of
  data source** — out-of-domain or in-domain-with-leakage alike.
- **Diversity, not just accuracy:** SFT narrows the answer distribution (SC-4
  0.310→0.241; pass@4 0.414→0.345). The untrained 4B's own sampling diversity is more
  useful for binary@0.9 than the SFT-collapsed distribution toward teacher phrasings.
- **Therefore:** the prior transfer failure was NOT (mainly) a domain gap. The SFT
  recipe as configured — imitating 27B teacher trajectories — is net-negative for this
  4B agent even under maximal favorability (in-domain + leakage). The **scaffold/prompt
  fix (0.042→0.190) is the only lever that has moved the metric.**

## Follow-up: memorization test (seqkd-indomain-v2, 20 epochs)
To test whether v1 was undertrained / the setup is broken, retrained the **same 80
trajectories for 20 epochs** (vs 3) → train loss collapsed **0.37 → 0.003** (fully
memorized). Eval on the same leaked docvqa_mini:

| | overall (mini n4) | train loss |
|--|--|--|
| baseline (untrained) | 0.1897 | — |
| indomain-v1 (3 ep) | 0.1466 | 0.22 |
| **indomain-v2 (20 ep, memorized)** | **0.1897** | 0.003 |

**v2 (memorized) exactly TIES baseline — never beats it.** So: (a) the training setup
is **not broken** (it provably fit the data to ~0 loss); (b) v1 was **undertrained**
(more steps lifted 0.147→0.190); (c) but even *in-domain + leaked + fully memorized*,
SFT only **matches** the untrained model. This is strong confirmation of the
**multi-turn observation-shift**: memorizing teacher trajectories does not transfer to
free-rollout eval, where the model faces its own (different) VLM observations and can't
replay the memorized answer. The agentic loop structurally defeats SFT-on-trajectories.
(Partial eval read was 0.279 at 68/116 but regressed to 0.190 as the slow big-doc tail
completed — early-completion bias, not signal.)

## Caveats
- Small data (80 traj / 35 Q) — the in-domain teacher yield is low because DocVQA at
  binary@0.9 is hard for the 27B (~0.4 ANLS). But "not enough solvable in-domain data
  to SFT on" is itself part of the finding.
- 29-Q mini is noisy; the qualitative verdict (SFT ≤ baseline) is robust — it holds for
  **both** transfer and in-domain on the same set, via the same mechanism. A full-val
  (80 Q) run would tighten the point estimate but cannot flip the direction.

## Implication for the project
SFT (SeqKD) on teacher trajectories is not the lever here. Next levers worth more than
more-SFT: (1) the agent scaffold / per-turn cap / runaway control (where the loss
actually lives); (2) RL with answer-level ANLS reward (learns the policy directly
rather than imitating a teacher whose style doesn't transfer); (3) on-policy
distillation (dense per-token signal, less prone to the answer-distribution collapse
seen here). The clean negative on SFT is a real, useful result — it redirects effort.
