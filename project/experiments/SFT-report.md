# SFT Phase — Comprehensive Report

> Authoritative, consolidated writeup of the supervised-fine-tuning (SFT) phase of
> the DocVQA-2026 ≤9B project. Consolidates and supersedes the scattered artifacts:
> `results/sft-sweep-2026-06-14.md` (current teacher-pool sweep card),
> `project/experiments/SFT-synthesis.md` (earlier investigation), and
> `project/experiments/SFT-teacher-gen-handoff.md` (rejection-sampling pipeline).
> ⚠️ **2026-06-14 — merge/serve bug FOUND & FIXED; SFT re-evaluated.** Every SFT eval
> *before* this fix silently ran the **base model**: `verl.model_merger merge` writes base
> `model.safetensors` + a separate `lora_adapter/` subdir (adapter never folded), and `vllm
> serve` without `--enable-lora` dropped the adapter. So the original "SFT ≈ base" headline
> was tautological (base vs base) and the base-model "cap-relief" numbers (16.6→19.1%) were a
> base-4B measurement. **Fix:** PEFT `merge_and_unload()` + assert merged≠base
> (`outputs/_fold_lora.py`). **§5.2 and §6 carry the CORRECTED, properly-folded evals.**
> **FINAL matched-power result: SFT (sft-r64) TIES base — 23.0% avg@8 vs 22.34% (n=8),
> +0.7pp within noise; worse on SC@8.** (An intermediate n=1 "+6.4pp" was a favorable-draw
> artifact, retracted.) See `results/sft-sweep-2026-06-14.md`. RL is from base → unaffected.

---

## 0. TL;DR

- **Method.** SFT a Qwen3.5-4B *agent* (the LM inside a CodeAct REPL scaffold) on
  **rejection-sampled 27B-teacher trajectories** — generate many teacher rollouts on a
  document pool, keep only the ones that solve the question (ANLS==1.0), and LoRA-SFT
  the 4B on those verbatim trajectories.
- **Goal.** Beat the untrained-4B baseline *in our own scaffold* on the DocVQA-2026
  validation set, as a warm start for on-policy RL.
- **Verdict (FINAL, corrected folded evals, matched n=8).** After fixing the merge/serve
  bug, the properly-folded best config (`sft-r64`, rank 64, LR2e-4) at **matched n=8 ties
  the base**: 23.0% avg@8 vs 22.34% (+0.7pp, within ±3.4pp), pass@8 53.8% ≈ 55.0%, and
  **SC@8 17.5% < 26.25% (worse)**. An intermediate n=1 read (28.7%, "+6.4pp") was a
  favorable-draw artifact (n=1 sampled only the cleaner sample_idx=0) — retracted. So
  SFT-on-teacher-trajectories **ties base and loses self-consistency; it is a safe
  warm-start, not a standalone win** — confirming the prior investigation with a correctly-
  folded checkpoint. The scaffold/prompt design remains the largest established lever
  (~+7pp on the untrained model); on-policy RL is the next one.
- **Key mechanism behind the ceiling.** This is a **multi-turn, partially-shifted**
  setting: at deploy/eval the 4B faces its *own* VLM observations, which differ from the
  teacher's. Imitating teacher trajectories therefore transfers weakly — even an
  in-domain, leaked, fully-memorized SFT only *ties* the baseline.

---

## 1. What we fine-tune (the target)

The agent scaffold is **CodeAct**: a strictly append-only, multi-turn REPL agent
(`docvqa/agent_loop.py::DocVQAReplAgentLoop`). Each turn the LM emits **free-text
reasoning followed by a single ```python``` fenced block**; the block runs in a
persistent CPython subprocess whose only tools are `batch_look(requests)` (perception
against the frozen 27B VLM) and `SUBMIT(answer=...)` (terminal). Captured stdout is
appended verbatim as the next turn → a fully-observable, append-only sequence.

- **Native thinking is DISABLED for both the LM and the VLM** (`enable_thinking=False`;
  eval passes `--no-thinking`, the VLM call sets it in `chat_template_kwargs` and strips
  any leaked reasoning). Reasoning survives as plain text before the fence; only the
  fenced code is parsed/executed. (`<think>` tags appear in <0.5% of turns as rare model
  leakage.)
- **We fine-tune only the LM weights**, via **LoRA adapters** (all-linear). The VLM
  stays frozen as an external HTTP endpoint.
- **Train == deploy.** Both trajectory collection and evaluation run the *same*
  `agent_loop`, so the policy we train is the policy we deploy. SFT trains on raw-text
  completions over the chat-message list (verl multi-turn SFT).
- **Fence semantics — `concat_fences` (deploy parity).** The deployed leaderboard solver
  concatenates and runs *every* complete ```python``` block in a turn (after stripping
  paired `<think>`). We match it (`agent_loop._extract_code_concat`, byte-identical) and
  `eval.py` sets it on by default. Consequence for data: assistant turns are kept
  **verbatim** (multi-fence included) — they are valid actions under concat, not
  contamination.

---

## 2. Data — rejection-sampling teacher-trajectory generation

The SFT data is produced by **rejection sampling** (a.k.a. best-of-n filtering / "SeqKD-style"
expert-trajectory collection): run the strong 27B teacher through the *same agent loop*
many times per prompt, then keep only the trajectories that actually solved the task.

### 2.1 Prompt pool

- **Pool file:** `data/pool/teacher_gen_pool.json` (rebuild: `python -m docvqa.scripts.make_teacher_gen_pool`,
  which draws from `data/pool/curriculum_rl.parquet`).
- **Composition: 405 prompts**, deliberately multi-page-inclusive so the teacher
  demonstrates *navigation*, not just single-page reads:
  - 250 single-page (62%)
  - 110 at 2–3 pages (27%)
  - 45 at 11–30 pages (11%)
  - → **38% multi-page**, mean 3.7 pages, max 30.
  - 31–89-page monsters excluded (≈800s-tail rollouts; throughput-capped).
- Rationale: DocVQA-2026 val is ~45% multi-page (34% ≥10 pages). A single-page-only
  corpus would not teach the 4B to navigate; the 27B *can* navigate multi-page docs, so
  those trajectories are the valuable signal.

### 2.2 Rollout generation (the long pole — hours)

Generation uses the eval harness itself (`docvqa/scripts/eval.py`) — an eval run *is* a
trajectory collection, since each per-sample record carries the full `messages`
trajectory + ANLS. Command shape:

```bash
python docvqa/scripts/eval.py \
  --questions data/pool/teacher_gen_pool.json \
  --base-url http://localhost:8932/v1 --model Qwen/Qwen3.5-27B \   # 27B as the AGENT
  --vlm-base-url http://localhost:8928 --vlm-model Qwen/Qwen3.5-27B \  # 27B as perception
  --concurrency 24 --n 8 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --no-thinking --rollout-timeout 1200 \
  --run-dir outputs/runs/teacher-gen-pool --resume
```

- **`--n 8`** rollouts/prompt: with 8 independent attempts at temp 0.6, a solvable prompt
  has a high chance of ≥1 ANLS==1.0 trajectory (the essence of rejection sampling). Yield
  was ~75–88% on easy single-page, lower on multi-page.
- **`--no-thinking`** matches the agent design and what the 4B is trained/deployed with.
- **Sampling** temp 0.6 / top-p 0.95 / top-k 20 — enough diversity across the 8 attempts
  to find a solving path without going incoherent.
- **Per-rollout wall-clock cap** 1200s — a timed-out rollout is recorded `wall_cap` and
  scored 0 (these are simply filtered out by the ANLS==1.0 keep rule).
- **Resumable** at per-question granularity (`--resume` skips questions already having
  ≥n streamed samples), so generation can be stopped/restarted across GPU reconfigs
  without losing completed work.

**GPU layout for generation.** The 27B *agent* generation (long multi-turn sequences) is
the bottleneck, so all 4 local GPUs ran the agent (DP4, `:8932`) and perception used the
remote 27B VLM (`:8928`). Observed throughput ≈ 5.8 rollouts/min on single-page-dominant
heads, ≈4/min blended with the multi-page tail.

### 2.3 Rejection-sample → SFT parquet

```bash
python docvqa/scripts/make_sft_data.py \
  --in outputs/runs/teacher-gen-pool --out data/sft/teacher_pool.parquet \
  --max-per-question 2
```

- **Keep rule:** `anls==1.0` AND terminated via `SUBMIT` (a verified-correct trajectory).
- **`--max-per-question 2`:** cap near-duplicate rollouts of the same easy prompt so the
  set isn't dominated by trivially-solved single-page docs.
- Assistant turns kept **verbatim** (multi-fence preserved — valid under `concat_fences`).

### 2.4 Final dataset

- **`data/sft/teacher_pool.parquet` — 426 trajectories** (strict ANLS==1.0).
- Page mix: 308 × 1pg / 80 × 2–3pg / 42 × 11–30pg (real multi-page coverage).
- Token lengths: median ~2.3K, max ~23.3K, **none > 32K** (fits the context budget).
- Schema: chat-message list + token-level `prompt_ids` / `response_ids` / `response_mask`
  (assistant-only loss mask), the basis for SFT-on-exact-tokens and later KD/RL.

### 2.5 Other SFT datasets used in the investigation (`data/sft/`)

| file | rows | role |
|---|---|---|
| `teacher_pool.parquet` | 426 | **current** — teacher pool, the 2026-06-14 sweep dataset |
| `clean_v5.parquet` | 245 | clean-scaffold MMLongBench-Doc transfer set (mmlb-long trained on this) |
| `indomain_v1.parquet` | 80 | in-domain DocVQA-val trajectories (leaked-by-design upper-bound test) |
| `mmlb_transfer*.parquet` | 87–185 | early transfer sets (pre-corruption-fix variants abandoned) |

`data/` is gitignored (regenerable; absolute paths baked in) — these files exist only on
the project box. Raw (unpacked) teacher trajectories live under
`outputs/runs/teacher-gen-pool/tasks/*/trajectories.jsonl`.

### 2.6 The corruption bug (a load-bearing data-quality fix)

The first generation sweep produced **~93%-corrupted** teacher data, and every model
trained on it was invalid. Root cause + fix (full detail in `SFT-synthesis.md`):

- The agent emits reasoning + a ```python``` fence; the *only* generation stop was
  `<|im_end|>`. When the teacher didn't emit it, it **free-ran and role-played the next
  turns** — fabricating `\nuser\n## Output\n…` observations and *more* code blocks (up to
  10 fences in one turn). The fence regex matched these, and parsing the *last* fence ran
  a *hallucinated* block → recorded observation desynced from the real action.
- **Fix** (`agent_loop.py`, `prompts.py`): stop sequences
  `["<|im_end|>", "\nuser\n", "\n## Turn", "\n## Output"]`; default to the model's real
  first action; defensive strip of any fabrication tail; simplified observation/prompt
  format. After the fix: hallucination markers 0.0% (was pervasive), and teacher
  **solve-yield jumped ~22% → ~98%** (it no longer derails itself by executing
  hallucinated code). The later `concat_fences` parity decision (§1) made the residual
  benign multi-fence turns valid rather than something to truncate.

---

## 3. Training methodology

### 3.1 Framework & recipe

- **verl FSDP SFT trainer** (`docvqa/train/run_seqkd.sh`), 4-GPU FSDP.
- **LoRA, `target_modules=all-linear`** (attention + MLP). LoRA-Without-Regret guidance:
  apply to all linear layers (MLP matters most), LR ≈ 10–15× full-FT (→ ~2e-4 for these
  short <250-step runs), rank not capacity-limiting at this data scale, constant schedule
  fine for short runs.
- **`alpha = rank`** (scaling factor α/r = 1, held constant across the rank sweep so LR
  is the controlled knob).
- **Packing:** `flash_attention_2` + remove-padding + dynamic-bsz, `max_token_len_per_gpu
  = 24576`. Global `train_batch_size = 16`, `micro_batch_size_per_gpu = 1` (CodeAct
  trajectories are long).
- **Optimizer:** FSDP AdamW, `weight_decay = 0`, constant LR (`warmup_style=constant`,
  warmup ratio 0.03, `min_lr_ratio 0.1`).
- **Precision:** fp32 master + bf16 compute (the documented-correct default). SFT is
  immune to the trainer/inference precision-mismatch RL failure (no clipped surrogate, no
  closed-loop data), so this is correct as-is.

### 3.2 The GDN-kernel throughput fix (12–18×)

Qwen3.5 is a **GDN (Gated-DeltaNet) hybrid**; its linear-attention path defaulted to a
slow pure-torch fallback, making SFT ~130s/step (MFU≈0, ~250 tok/s/GPU — ~20× too slow).
Installing the fast-path kernels (`flash-linear-attention` + `causal-conv1d`) dropped step
time to **~7–11s/step steady** (after a brief triton autotune warmup). These deps are
committed to `setup.py [gpu]` extras. This is what made the rank×LR×epoch sweep feasible
(~26 min/config instead of ~8h).

### 3.3 Why 4-GPU FSDP (keep-all-data decision)

We chose to **keep all trajectories** (no dropping long multi-page ones). The longest
(~23K-token) trajectory forces a dynamic-bsz budget that intermittently OOMs a single
80GB GPU; FSDP sharding across 4 GPUs gives the headroom (40–60GB/GPU, no OOM) and also
lets higher LoRA ranks (64) fit. (Small-model SFT is overhead-bound, so 4-GPU ≈ 2-GPU in
throughput — for the sweep we used all 4 for wall-clock per config.)

### 3.4 Exact hyperparameters (teacher-pool sweep, `_run_sft_4gpu_cfg.sh`)

| param | value |
|---|---|
| base model | `Qwen/Qwen3.5-4B` |
| LoRA target | `all-linear` |
| LoRA rank (swept) | 16 / 32 / 64 |
| LoRA alpha | = rank |
| LR (swept) | 1e-5 / 5e-5 / 1e-4 / 2e-4 / 4e-4 |
| schedule | constant, warmup ratio 0.03, min_lr_ratio 0.1 |
| epochs | 8 (ckpt every 2 → ep2/4/6/8 = global_step 52/104/156/208) |
| steps/epoch | 26 (208 total) |
| global batch | 16 |
| micro batch / GPU | 1 |
| dynamic-bsz budget | 24576 tok/GPU |
| attention | flash_attention_2 + remove-padding |
| precision | fp32 master + bf16 compute |
| throughput | ~7s/step → ~26 min/config |

**LoRA parameter fraction of base (Qwen3.5-4B, 4.539B params):** rank16 ≈ **0.859%**,
rank32 ≈ **1.718%**, rank64 ≈ **3.436%**.

(The earlier investigation used the same recipe at **20 epochs** — see §5.1; the sweep
caps at 8 because training loss plateaus ~ep5 and 8 is past it without full memorization.)

### 3.5 Checkpoint → HF → serve

- SFT trainer saves a **FLAT** layout (`global_step_N/model_world_size_*.pt` +
  `huggingface/`, *no* `actor/` subdir — unlike RL checkpoints which have `actor/`).
- Merge: `python -m verl.model_merger merge --backend fsdp --local_dir global_step_N
  --target_dir <…>/merged_hf` (run on GPU — CPU merge fails because FlashAttention2 can't
  init on CPU). Then copy `preprocessor_config.json` + `video_preprocessor_config.json`
  from the Qwen3.5-4B HF snapshot into `merged_hf` (the merge omits the VL image-processor
  config needed to serve).

---

## 4. Evaluation methodology

- **Metric:** ANLS @ 0.9 (DocVQA-2026 official binary threshold), averaged per question
  then over questions (avg@1). We also report **pass@k** (oracle: any of k rollouts
  correct) and **SC@k** (self-consistency / majority vote). *SC@2 is meaningless* (a
  2-vote majority) and is never reported; SC is only used at k≥4.
- **Two-tier screen → confirm.** Screen on `docvqa_mini` / `docvqa_rank13` for ranking,
  **confirm the headline on the full 80-Q val** (`data/docvqa-2026/val/questions.json`).
  This is load-bearing: the mini set (SE≈±5–10pp, median-difficulty docs) produced a
  **false null** that survived four experiments in the earlier investigation (§5.1). Any
  "X beats/ties baseline" claim from mini must be re-checked on full val.
- **No leakage by construction** for the transfer setting: training on MMLongBench /
  pooled non-DocVQA data means the full DocVQA val is leakage-free, so no train/val split
  is needed.
- **Eval is VLM-perception-bound** (see §7): the 27B VLM saturates its GPUs at modest
  concurrency; the 4B agent GPUs sit underutilized. Throughput is set by perception, not
  the policy.

---

## 5. Experiments & results

There were two distinct campaigns. **Phase A** (the earlier investigation) was on an
*older scaffold/implementation* whose untrained baseline was ~15.7%. **Phase B** (the
current teacher-pool sweep) is on the *improved scaffold*, whose untrained baseline is
**22.34% ± 3.44 (n=8)**. The scaffold improvement itself lifted the untrained model ~+7pp,
so cross-phase numbers are NOT directly comparable — each result is only meaningful
against its own-phase baseline.

### 5.1 Phase A — initial investigation (older scaffold)

Sources: `SFT-synthesis.md` + `results/{clean-restart-mini-n4,indomain-upperbound,mmlb-long-generalization}.md`.

1. **Corruption discovery + clean restart** (§2.6). All pre-fix models (v1, v2, ArmA/B,
   v4–v6) invalid; the scaffold/stop-sequence fix was the only change that moved the
   metric (0.042 → 0.19 on mini).
2. **Transfer SFT (clean-v5, 245 MMLongBench traj, 3 ep).** docvqa_mini: 13.8% vs baseline
   19.0% — *hurts* (undertrained; loss only 0.18).
3. **In-domain upper-bound (indomain-v1/v2, 80 leaked DocVQA-val traj).** Even leaked
   in-domain SFT does not beat baseline on mini: v1 (3 ep) 14.7%; v2 (20 ep, train loss
   0.37→**0.003**, fully memorized) = **19.0% = baseline exactly**. → the SFT *setup* is
   not broken (it fits to ~0), but trajectory replay doesn't transfer through the
   multi-turn observation shift.
4. **Generalization (mmlb-long, 245 traj, 20 ep, ckpt every 5).** On the **full 80-Q val**
   (older scaffold, n=4, paired), training-monotonic lift emerged that was invisible on
   mini:
   | full-val | overall | Δ vs base | paired t |
   |---|---|---|---|
   | baseline (untrained 4B) | 15.3% | — | — |
   | mmlb-long ep16 | 17.2% | +1.9 | 0.74 (n.s.) |
   | mmlb-long ep20 | **20.6%** | **+5.3** | **2.09 (p≈0.04)** |
   → On the *older* scaffold, 20-epoch transfer SFT **significantly beat** its baseline
   (+5.3, p≈0.04), but the effect is modest and training-dependent (3 ep hurts, 16 ep
   n.s.). **Methodological lesson:** the 29-Q mini flattered the baseline and masked this.

**Phase A takeaway:** SFT is a *real but modest* lever on the older scaffold — a
reasonable warm start, not a dead end — and undertraining hurts.

### 5.2 Phase B — teacher-pool sweep (current, improved scaffold)

Source: `results/sft-sweep-2026-06-14.md`. Dataset = `teacher_pool.parquet` (426 traj).
Baseline to beat = **22.34% (n=8 full val)**. **All numbers here are the CORRECTED,
properly-folded evals** (the pre-fix ranking/ladder tables were base-model — deleted).

**Final training loss (ep8):** r64 0.040 · r64-lr1e4 0.044 · lr4e4 0.046 · r32 0.051 ·
r16 0.065 · lr1e4 0.070 · r16-lr1e4 0.108 · lr5e5 0.132 · lr1e5 0.301. (Lower loss =
harder fit: higher rank + higher LR drive loss down; loss ≠ eval quality.)

**Config screen — `docvqa_rank13` (13 Q, 1/doc, full-category), n=1, CORRECTLY FOLDED**
(selection-grade only; SE ≈ ±14pp). 6 of 9 configs evaluated — 3 skipped by **loss-curve
diversification** (near-duplicate training curves, to save VLM-bound eval budget):

| config | rank | LR | train loss | rank13 mean (n=1) | cap |
|---|---|---|---|---|---|
| **sft-r64** | 64 | 2e-4 | 0.040 | **30.8%** | 0% |
| sft-r32 | 32 | 2e-4 | 0.051 | 23.1% | 0% |
| sft-r16 | 16 | 2e-4 | 0.065 | 23.1% | 8% |
| sft-r32-lr1e4 | 32 | 1e-4 | 0.070 | 23.1% | 0% |
| sft-r32-lr1e5 | 32 | 1e-5 | 0.301 | 23.1% | 23% |
| sft-r32-lr4e4 | 32 | 4e-4 | 0.046 | 19.2% (n=4) | 2% |
| base-4B (comparator) | — | — | — | ~19–22% | — |
| skipped (loss-dedup) | — | — | — | r64-lr1e4 (≈r64), r16-lr1e4 & lr5e5 (gentle band) | |

**Read:** most configs cluster at 23.1% ≈ base; **`sft-r64` is the sole clear outlier at
30.8% (0 cap)**. At n=1/13Q (±14pp) that is a *candidate*, not significance — 30.8% = 4/13
questions. The loss→eval relation is **noisy, not a clean inverted-U**: lr4e4 has nearly the
lowest loss (0.046) but a low eval (19.2%), while r64 has *the* lowest loss (0.040) and the
*best* eval — so loss is a screening proxy only. Rank is eval-irrelevant at LR2e-4 (r16 and
r32 differ in loss 0.065 vs 0.051 but both eval 23.1%), which justified the loss-dedup skips.

**Full-val (headline) — leader sft-r64:**

| model | full-val ANLS | notes |
|---|---|---|
| base-4B baseline | **0.2234 ± 0.0344 (n=8)** | the bar |
| sft-r64 ep8 (n=1) | 0.287 | favorable single draw (sample_idx=0); retracted as the headline |
| **sft-r64 ep8 (n=8, matched)** | **0.230** | pass@8 53.8% · SC@8 17.5% · submit-only 34.9% · iter_cap 30% |

**Phase B verdict (final):** the pre-fix "≈ base" was a measurement bug (base-vs-base); the
fix is in, but the corrected **matched-power** answer is **still a tie**: r64 = **23.0% avg@8
vs base 22.34%** (+0.7pp, within ±3.4pp), **pass@8 53.8% ≈ 55.0%**, **SC@8 17.5% < 26.25%
(worse)**. The intermediate n=1 "+6.4pp" (28.7%) was a favorable-draw artifact — n=1 used only
sample_idx=0 (26% iter_cap); the 8-sample average regresses to base (30% iter_cap). So
**SFT-on-teacher-trajectories ties base at matched power and loses self-consistency** — it does
not beat the untrained 4B. Consistent with the entire prior investigation, now confirmed with a
correctly-folded checkpoint. On-policy RL is the lever.

---

## 6. Headline number

> **Status 2026-06-15 (matched-power, FINAL):** With the merge bug fixed, the corrected
> **matched n=8** answer is **SFT ≈ base, a tie** — `sft-r64` = **23.0% avg@8 vs base 22.34%**
> (+0.7pp, within ±3.4pp). The intermediate n=1 "28.7% / +6.4pp" was a favorable-draw artifact
> (retracted). The old base-model "lr4e4 19.1%" row was deleted (it was base-4B).

| model | mean ANLS | pass@k | SC@k | cap rate | n |
|---|---|---|---|---|---|
| base-4B baseline | **22.34%** | 55.0% (k=8) | 26.25% (k=8) | unknown (deleted) | 8 |
| sft-r64 ep8 — screen (rank13) | 30.8% | — | — | 0% | 1 |
| sft-r64 ep8 — full-val (n=1, favorable draw, retracted) | 28.7% | — | — | 5%+26% iter | 1 |
| **sft-r64 ep8 — full-val (n=8, matched)** | **23.0%** | **53.8%** | **17.5%** | 3.9%+30% iter | 8 |
| 27B teacher ceiling | ~39.5% | ~63.75% | ~45.0% | — | — |

**Headline verdict (FINAL): SFT (r64) ties base at matched power; does not beat it.** At n=8,
avg 23.0% vs 22.34% (+0.7pp, within noise) and pass@8 53.8% vs 55.0% (tied), while **SC@8
17.5% < 26.25% (worse)** — SFT-on-teacher-trajectories ties the mean and *loses self-
consistency*. The n=1 "+6.4pp" was an unmatched-n / favorable-draw artifact (n=1 sampled only
the cleaner sample_idx=0). Net: SFT is a safe warm-start, **not a standalone win** — confirming
the entire prior investigation with a correctly-folded checkpoint. Headroom = iter_cap (30%,
turn-budget exhaustion). **On-policy RL (GRPO from base) is the lever, launching next.**

> Residual 6.9% cap = genuinely pathological heavy docs where the scaffold fans out unbounded
> `batch_look` calls (up to 374) and never terminates; not fixable by wall-clock — needs a
> perception-call budget. **This will bite RL** (capped rollouts = full-cost, 0-reward,
> zero-variance groups), so the RL config must bound perception calls + filter dead groups.

---

## 7. Bottleneck & infra findings (load-bearing for RL)

- **Perception (27B VLM) is the throughput ceiling**, not the 4B agent. At eval/RL
  concurrency the VLM GPUs run ~100% while the agent GPUs sit ~30–60%. Adding agent GPUs
  or raising concurrency past VLM saturation does not help; the lever is *more VLM
  capacity* (e.g. pooling a local 27B with the remote one). (Earlier per-phase RL
  profiling showed the binding constraint *migrates* as each fix lands — profile, don't
  assume.)
- **VLM endpoint pool** (`tools.py::EndpointPool`, spec `url@w|url@w`, least-`inflight/weight`
  routing, 60s health-bench). Lets a local + remote VLM serve perception together; a
  remote that comes up mid-run is absorbed without restart.
- **Connect to the remote VLM directly by IP** (`http://144.122.52.7:8927`), not through
  an SSH tunnel — concurrent multi-MB image POSTs storm a single tunnel (CLOSE-WAIT
  pileup → all rollouts hang). Direct connection removes that stall and lets concurrency
  scale.
- **VLM throughput tricks:** client-side image downscale to the processor's max area
  (lossless w.r.t. what the model sees, saves transfer/CPU); image prep off the event
  loop; prefix caching (payload order system→image→query so re-looks hit the ~16K-token
  image prefill); thinking-off perception (short answers; thinking dominated wall-clock);
  serve flags `--max-num-batched-tokens 32768 --async-scheduling --limit-mm-per-prompt
  '{"image":1}'`.
- **eval.py per-rollout incremental write** (added 2026-06-14): trajectories stream to
  `tasks/<doc>/trajectories.jsonl` the moment each rollout finishes (was per-question,
  after all n) — crash-safe + live progress; `_load_done_results` dedups by `sample_idx`
  for clean `--resume`. (Rollouts run *sequentially within a question*, so a question can
  take ~80 min at n=8; per-rollout writes make progress visible immediately.)
- **/tmp PNG leak (fixed in `docvqa/sandbox.py`):** `batch_look` wrote a `delete=False`
  temp PNG per image; cleanup via `try/finally os.remove`. Still worth watching disk
  (purge: `find /tmp -maxdepth 1 -user baris -iname 'tmp*.png' -delete`).

---

## 8. Decisions made along the way (chronological)

1. **Scaffold correctness before data quantity.** Discovered the teacher-hallucination /
   multi-fence corruption (~93% of data), fixed stop sequences + fence parsing, restarted
   clean. Single biggest quality lever.
2. **`concat_fences` deploy parity.** Match the leaderboard solver's "run every fenced
   block" semantics in both generation and eval, so train == deploy; keep assistant turns
   verbatim in SFT data.
3. **Thinking OFF** for both the LM and the VLM (deploy parity; perception thinking
   dominated rollout wall-clock).
4. **Rejection sampling at n=8, keep ANLS==1.0, cap 2/question.** Best-of-8 finds solving
   paths; the cap prevents easy single-page prompts from dominating.
5. **Multi-page-inclusive pool.** Include 2–30-page docs (38% multi-page) so the teacher
   demonstrates navigation; exclude 31–89-page tails for throughput.
6. **Keep all data → 4-GPU FSDP.** Don't drop the long multi-page trajectories; shard to
   get the memory headroom (and to fit rank-64).
7. **GDN fast-path kernels.** Root-caused the 12–18× SFT slowdown to the missing
   linear-attention kernels; installing them made the sweep feasible.
8. **Screen on mini, confirm on full val.** After a mini-set false-null burned four
   experiments, full-val confirmation became mandatory for any verdict.
9. **8-epoch cap for the sweep** (loss plateaus ~ep5; 20 ep was the earlier deep run).
10. **LoRA-Without-Regret knobs:** all-linear, α=rank, LR the controlled variable, rank
    not capacity-limiting → sweep LR primarily.
11. **Baseline recalibration:** 15.7% (old scaffold) → 22.34% (improved scaffold, n=8) —
    the bar SFT must clear is ~22%, and our *own* base-4B eval (same agent_loop) is the
    valid comparator.
12. **Eval is VLM-bound → pool local+remote VLM, agent on fewer GPUs, connect direct.**

---

## 9. Verdict & implications

- **SFT-on-teacher-trajectories ≈ untrained baseline** on the improved scaffold; a safe
  warm start / cold-start avoider for RL, not a standalone win. The configs are
  statistically indistinguishable at the eval power we can afford; no LR/rank/epoch trend.
- **Why it caps out:** the multi-turn observation shift — at deploy the 4B faces its own
  VLM observations, so memorized teacher trajectories don't replay. Even in-domain +
  leaked + memorized SFT only ties the baseline. The dominant error mode is long-doc
  runaway / wall_cap (a scaffold property), which caps every method's ceiling.
- **What actually moved the metric:** the scaffold/prompt design (~+7pp on the untrained
  model).
- **Next lever: on-policy RL** (GRPO on ANLS reward, one-step-off-policy disaggregated),
  initialized from a SFT checkpoint (warm start). SFT's value here is providing a sane,
  format-correct initialization that already saturates the easy curriculum, giving RL
  reward variance to work with rather than a zero-reward cold start.

---

## 10. Reproduction quick-reference

```bash
# 1. Generate teacher trajectories (27B agent + 27B VLM perception)
python docvqa/scripts/eval.py --questions data/pool/teacher_gen_pool.json \
  --base-url http://localhost:8932/v1 --model Qwen/Qwen3.5-27B \
  --vlm-base-url http://localhost:8928 --vlm-model Qwen/Qwen3.5-27B \
  --concurrency 24 --n 8 --temperature 0.6 --top-p 0.95 --top-k 20 \
  --no-thinking --rollout-timeout 1200 --run-dir outputs/runs/teacher-gen-pool --resume

# 2. Rejection-sample -> SFT parquet
python docvqa/scripts/make_sft_data.py \
  --in outputs/runs/teacher-gen-pool --out data/sft/teacher_pool.parquet --max-per-question 2

# 3. SFT a config (4-GPU FSDP, LoRA all-linear; args: EXP [LR] [RANK])
bash outputs/_run_sft_4gpu_cfg.sh sft-r32-lr4e4 4e-4 32

# 4. Merge LoRA -> HF (on GPU), copy VL preprocessor configs, then eval
python -m verl.model_merger merge --backend fsdp \
  --local_dir checkpoints/docvqa-verl/sft-r32-lr4e4/global_step_208 \
  --target_dir checkpoints/docvqa-verl/sft-r32-lr4e4/global_step_208/merged_hf

# 5. Eval (screen on rank13/mini, confirm on full val; VLM-pooled, agent on its own GPU)
MODEL=<merged_hf> QFILE=data/docvqa-2026/val/questions.json NROLL=8 TAG=<tag> CONC=32 \
  VLM_URL='http://localhost:8927@2|http://144.122.52.7:8927@3' bash outputs/_eval_pool22.sh
```

---

### Pointers
- Current sweep card: `results/sft-sweep-2026-06-14.md`
- Earlier investigation synthesis: `project/experiments/SFT-synthesis.md`
- Rejection-sampling pipeline handoff: `project/experiments/SFT-teacher-gen-handoff.md`
- Per-experiment cards: `results/{clean-restart-mini-n4,indomain-upperbound,mmlb-long-generalization}.md`
- Report section (polished, light): `project/report/sections/05-training-small-agent.md`
