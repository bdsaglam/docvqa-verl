# Pillar B — capture notes (small-model training of the CodeAct agent)

> Consolidated grounding for report §5 ("Training the small agent"). Read-only
> mining from the docvqa-verl + docvqa repos, 2026-06-14. Pillar B is **in
> progress** — the SFT verdict is a measured near-null; the RL stage is planned
> with only a partial single-checkpoint screen so far. Companion file
> `pillarB-results.md` holds the mined eval numbers; this file holds the
> approach, terminology decision, data-pool, setup, figures, and pass@k.
>
> Primary sources (cite these in the draft):
> - `project/experiments/SFT-report.md` (the authoritative SFT writeup)
> - `docs/training-data-pool.md` + `docvqa/train/pool.yaml` (the pool)
> - `verl/workers/utils/losses.py:28` (`sft_loss`) + `docvqa/train/run_seqkd.sh` (the objective)
> - `tools/trajviewer/{app.py,static/app.js}` (figures)
> - `~/repos/docvqa/docs/pass-at-k.md` + `docvqa/scripts/eval.py` (pass@k)

---

## 1. Approach — rejection sampling → train (accurate terms)

The pipeline that produces the trained 4B agent, end to end
(`SFT-report.md` §2–§3):

1. **Teacher rollout generation.** Run the **27B model as the agent** (the LM in
   the CodeAct REPL scaffold) with the **27B model as the frozen perception VLM**
   ("27B/27B") through the *same* `agent_loop` used at deploy, **n=8 rollouts per
   prompt** at temp 0.6 / top-p 0.95 / top-k 20, thinking OFF, 1200s per-rollout
   cap. Generation is literally an eval run (`docvqa/scripts/eval.py`); each
   per-sample record carries the full `messages` trajectory + ANLS.
   (`SFT-report.md:84-114`.)
2. **Rejection sampling (best-of-n filtering).** Keep only trajectories with
   **ANLS == 1.0 AND terminated via `SUBMIT`** (verified-correct), capped at
   **2 per question** (`--max-per-question 2`) so trivially-solved single-page
   prompts don't dominate. Assistant turns kept **verbatim** (multi-fence
   preserved — valid under `concat_fences` deploy parity). (`SFT-report.md:116-128`.)
3. **PEFT/LoRA train the 4B student** on those kept trajectories with token-level
   cross-entropy (assistant-only loss mask). (`SFT-report.md:170-241`.)

Because the kept trajectories come from a **larger (27B) teacher** and the
**student is a smaller (4B) model**, this is *cross-capacity behavioral cloning*,
i.e. distillation at the **sequence level** — not same-model self-training.

Final dataset actually used for the current sweep:
**`data/sft/teacher_pool.parquet` — 426 trajectories** (strict ANLS==1.0),
page-mix 308×1pg / 80×2–3pg / 42×11–30pg, token lengths median ~2.3K / max ~23.3K
(none >32K). (`SFT-report.md:129-136`.)

> Note: an *earlier* investigation (Phase A) trained on **MMLongBench-Doc**
> transfer trajectories (`clean_v5.parquet`, 245 traj). The report §5 SFT
> *results* (15.3→20.6) come from that Phase-A MMLB transfer run; the current
> `teacher_pool.parquet` sweep (Phase B, improved scaffold, baseline ~22.34) is
> the in-progress headline. Keep the two phases distinct — their baselines differ
> (~15.7 vs ~22.34) and are NOT comparable. (`SFT-report.md:264-351`.)

---

## 2. TERMINOLOGY decision (with file:line evidence)

**Question:** is the 4B training plain cross-entropy on kept teacher trajectories
(→ SeqKD / rejection-sampling SFT), or is there a logit/KL distillation loss
(→ true logit-KD)?

**Answer (from code): plain token-level cross-entropy. No logit/KL distillation.**

Evidence chain:
- `docvqa/train/run_seqkd.sh:70` invokes `verl.trainer.sft_trainer` (the standard
  supervised trainer), passing only a `messages` parquet and a LoRA config — no
  teacher model, no teacher logits, no temperature/KL knobs.
- `verl/trainer/sft_trainer.py:161-163` sets the loss to
  `sft_loss` from `verl.workers.utils.losses` (`partial(sft_loss, config=None)`).
- `verl/workers/utils/losses.py:28-54` — `sft_loss` is exactly:
  `loss = -masked_sum(log_prob, loss_mask) / batch_num_tokens` — the negative
  log-likelihood of the response tokens under the **hard** target token ids,
  masked to assistant tokens. There is **no second (teacher) distribution, no KL
  term, no soft targets, no temperature**. (A separate `ppo_loss` exists in the
  same file for RL; SFT does not use it.)

So the objective is **behavioral cloning by maximum likelihood on the kept
trajectories**. Since teacher (27B) ≠ student (4B), this is **sequence-level
knowledge distillation (SeqKD)** in the Kim & Rush (2016) sense — "distillation"
because a larger teacher's *sequences* (not its logits) supervise the student;
"sequence-level" because the supervision is the full discrete token sequence,
filtered by a quality oracle (rejection sampling), not a per-token soft
distribution.

**Recommended primary term + phrasing:**

> **"Supervised fine-tuning on rejection-sampled teacher trajectories — a form of
> sequence-level knowledge distillation (SeqKD)."**

Rationale for the paper:
- It is **defensible and literally accurate**: the loss is CE on kept sequences
  (verifiable in code), and "SeqKD = train the student with MLE on
  teacher-generated outputs" is the standard definition; rejection sampling is
  the data-filtering variant ("RFT" / best-of-n SFT).
- **Avoid claiming logit-KD / on-policy distillation** — neither is implemented
  (no teacher forward pass at train time, no KL). Calling it "knowledge
  distillation" *unqualified* would invite the (correct) objection "where is the
  KL/logit term?"; the "sequence-level (SeqKD)" qualifier preempts it.
- The internal script name (`run_seqkd.sh`) and the report section already use
  "SeqKD" consistently, so this term keeps the writeup self-consistent.
- If a single shorter label is needed in a table, **"SFT (rejection-sampled
  teacher traj.)"** is fine; "RFT" (rejection-sampling fine-tuning) is an
  acceptable synonym but less recognizable than "SFT/SeqKD" — mention it once in
  parentheses, don't lead with it.

(The report's §5.0 / `05-training-small-agent.md:17` already phrases it as
"Supervised fine-tuning / sequence-level distillation (SeqKD)… fit to the accepted
trajectories with token-level cross-entropy" — this capture confirms that phrasing
against the actual loss code; keep it.)

---

## 3. Data pool (sources, page diversity, sizes, leakage)

Source: `docs/training-data-pool.md` + `docvqa/train/pool.yaml`. There is **no
DocVQA-2026 train split** (val gold is public, train/test are not usable), so a
method-agnostic training-prompt pool was assembled from other DocVQA-family
datasets. Every row carries a gold answer + a rule verifier, so the same pool
serves SFT (rejection-sampling filter), RL (reward), and OPD.

### 3.1 Sources — RECONCILE: 8 in code, 7 in the report outline

The report §5.3 (`05-training-small-agent.md:23`) lists **seven** datasets:
DocVQA(-SP), InfographicVQA, ChartQA, MapQA, MP-DocVQA, TAT-DQA, SlideVQA.

The actual pool (`pool.yaml` + `training-data-pool.md` §1) has **EIGHT** sources —
the 8th is **MMLongBench-Doc** (long multi-page; the only source of >10-page
navigation signal; added because it helped DocVQA in the SFT experiments).
**Action for §5.3:** either add MMLongBench-Doc (making it eight) or explicitly
note it as the long-doc add-on. The report's *results* (Phase-A transfer SFT)
were in fact trained on MMLongBench-Doc, so it is load-bearing and should be
named.

| source | HF id | split | category label | pages (min/med/mean/max) | docs | questions |
|---|---|---|---|---|---:|---:|
| docvqa-sp | `lmms-lab/DocVQA` (DocVQA) | validation | business_report | 1 (single img) | 1,286 | 5,349 |
| infographicvqa | `lmms-lab/DocVQA` (InfographicVQA) | validation | infographics | 1 | 500 | 2,801 |
| chartqa | `HuggingFaceM4/ChartQA` | train | science_poster | 1 | 28,299 | 28,299 |
| mapqa | `nimapourjafar/mm_mapqa` | train | maps | 1 | 37,417 | 483,416 |
| mp-docvqa | `lmms-lab/MP-DocVQA` | val | business_report | 1/1/1.6/3 (≤3 filter) | 583 | 2,986 |
| tatdqa | `next-tat/TAT-DQA` | dev | business_report | 1/1/1.1/2 | 274 | 1,644 |
| slidevqa | `NTT-hil-insight/SlideVQA` | train | slide | 1 (single-evidence) | 5,962 | 9,381 |
| **mmlongbench-doc** | `yubo2333/MMLongBench-Doc` | train | (native doc-type) | 9/28/37/80 | 116 | 964 |
| **total** | | | | | **74,437** | **534,840** |

`DUDE` and `AI2D` were evaluated and excluded (DUDE: multi-GB + fragile loader;
AI2D: test-only multiple-choice, pollutes an ANLS pool). The pool covers **5 of
DocVQA-2026's 8 categories** (business_report, infographics, maps, science_poster,
slide) — comics / engineering_drawing / science_paper have no suitable short-doc
source. (`training-data-pool.md` §1.)

### 3.2 Page-count / multi-page diversity effort

- Single-image sources dominate raw counts; **navigation signal** comes from
  mp-docvqa (~46% of its docs are 2–3pg, filtered to ≤3pg) and the long tail
  from MMLongBench-Doc (median 28 pages, max 80). (`training-data-pool.md` §2.)
- For **teacher generation specifically**, the prompt pool was deliberately
  built multi-page-inclusive: 250×1pg / 110×2–3pg / 45×11–30pg = **38% multi-page,
  mean 3.7pg, max 30**; 31–89pg "monsters" excluded for throughput
  (≈800s-tail rollouts). Rationale: DocVQA-2026 val is ~45% multi-page (34% ≥10
  pages), so a single-page-only corpus would not teach navigation.
  (`SFT-report.md:68-81`.)

### 3.3 Sizes / balancing

Raw pool = 534,840 questions (MapQA alone is 483k). `sample_pool.py` draws
`n_sample` rows/source (set in `pool.yaml`; 1500/source, MMLB capped at its 964)
into a **balanced pool of 11,464 prompts** (`data/pool/prompts.json`, seed 42).
Balanced **by dataset, not category** → `business_report` is ~3× (docvqa-sp +
mp-docvqa + tatdqa all map there). (`training-data-pool.md` §3.)

> Naming nuance: this 11,464-prompt balanced pool is the **substrate for upcoming
> RL/OPD/larger-SFT**; it was assembled *after* the SFT runs reported in §5.3 and
> has **not** been used for any reported training yet (the report says so at
> `05-training-small-agent.md:23`). The current SFT sweep's 426 trajectories were
> rejection-sampled from a separate 405-prompt teacher-gen pool
> (`data/pool/teacher_gen_pool.json`, drawn from `data/pool/curriculum_rl.parquet`).

### 3.4 Leakage-safety

`find_pool_leakage.py` checks pool images against DocVQA-2026 **val** pages via a
two-stage **dHash perceptual-hash shortlist → exact pixel diff** (a raw byte hash
would miss re-encoded images). **Verdict: no question-level leakage.** Of 31
dHash candidates, only 2 were truly reused images (InfographicVQA ↔ DocVQA-2026
`infographics_1/2`), and those carry **different questions/answers** in val, so no
eval answer is memorizable. **Caveat: DocVQA-2026 `test` not yet checked**
(re-run before trusting test numbers). MMLongBench-Doc is disjoint from DocVQA val
by construction → the transfer-SFT results are leakage-free.
(`training-data-pool.md` §7; `SFT-report.md:255-258`.)

---

## 4. SFT setup + hyperparameter sweep + eval protocol + current results

### 4.1 Setup (`SFT-report.md` §3, `run_seqkd.sh`)

- **Base:** `Qwen/Qwen3.5-4B` (4.539B params). Train **only the LM**; VLM frozen.
- **Trainer:** verl FSDP SFT (`verl.trainer.sft_trainer`, FSDP2), 4-GPU.
- **LoRA:** `target_modules=all-linear` (attn + MLP), **α = rank** (held so LR is
  the controlled knob). LoRA param fraction: rank16 ≈ 0.86%, rank32 ≈ 1.72%,
  rank64 ≈ 3.44% of base.
- **Packing:** flash_attention_2 + remove-padding + dynamic-bsz,
  `max_token_len_per_gpu=24576`; global batch 16, micro-batch/GPU 1 (long traj).
- **Optimizer:** FSDP AdamW, weight_decay 0, **constant LR** (warmup ratio 0.03,
  min_lr_ratio 0.1).
- **Precision:** fp32 master + bf16 compute (SFT is immune to the RL
  trainer/inference precision-mismatch failure).
- **Throughput fix (load-bearing):** Qwen3.5 is a **GDN (Gated-DeltaNet) hybrid**;
  installing `flash-linear-attention` + `causal-conv1d` cut step time ~130s →
  ~7–11s (12–18×), which is what made the sweep feasible (~26 min/config).
- **Checkpoint → serve:** FLAT layout (no `actor/` subdir, unlike RL); merge via
  `verl.model_merger` *on GPU*, then copy `preprocessor_config.json` +
  `video_preprocessor_config.json` from the Qwen3.5-4B snapshot into `merged_hf`.

### 4.2 Hyperparameter sweep (teacher-pool, Phase B; `SFT-report.md:207-230`)

| param | value(s) |
|---|---|
| LoRA rank (swept) | **16 / 32 / 64** |
| LoRA alpha | = rank |
| LR (swept) | **1e-5 / 5e-5 / 1e-4 / 2e-4 / 4e-4** |
| schedule | constant, warmup 0.03, min_lr_ratio 0.1 |
| epochs | 8 (ckpt every 2 → ep2/4/6/8 = step 52/104/156/208), 26 steps/ep |
| global batch / micro-batch-per-GPU | 16 / 1 |
| dynamic-bsz budget | 24576 tok/GPU |

(The earlier Phase-A investigation used the same recipe at **20 epochs**; the
sweep caps at 8 because training loss plateaus ~ep5.)

### 4.3 Eval protocol — two-tier screen → confirm (`SFT-report.md` §4)

- **Metric:** ANLS @ 0.9 (DocVQA-2026 official binary threshold), avg@1 (mean per
  question then over questions). Also report **pass@k** (oracle) and **SC@k**
  (self-consistency, k≥4 only — SC@2 meaningless).
- **Two-tier:** screen on **`docvqa_mini` / `docvqa_rank13`** (13-Q, 1/doc,
  full-category — selection-grade only, SE≈±10pp) for ranking, then **confirm the
  headline on the full 80-Q val** (`data/docvqa-2026/val/questions.json`).
  > NB the task brief says "docvqa-mini 13Q"; the report variously calls it
  > `docvqa_mini` / `docvqa_rank13` (13 Q) and once "29-Q mini" (Phase A used a
  > 29-Q variant). The current screen set is **13 Q**. The mini-screen is
  > explicitly flagged as **unreliable** — a mini false-null survived four Phase-A
  > experiments — so any "beats/ties baseline" claim must be re-checked on full val.
- **Leakage-free** by construction for transfer (MMLB / pooled non-DocVQA data).
- **Throughput is VLM-perception-bound** (27B VLM saturates its GPUs; 4B agent
  GPUs idle) — see §7 of the report for the infra levers.

### 4.4 Current results (IN PROGRESS)

**Phase B (current, improved scaffold, baseline = 22.34% ± 3.44 n=8 full-val):**
- `docvqa_rank13` n=2 LR ladder (means): 19.2 → 11.5 → 23.1 → 7.7 → 26.9% —
  **no monotonic LR/rank/epoch structure**; every config within ~1 SE of the
  base-4B level. A "gentle vs aggressive SFT" story was **retracted** (the two
  most-aggressive configs land at opposite extremes = noise).
- **Full-val headline: PENDING** — `lr4e4 ep8` running at n=4 (→ n=8), to be
  reported as mean ANLS / pass@k / SC@k vs base-4B (avg@1 22.34, pass@8 55.0,
  SC@8 26.25; 27B ceiling ≈ 39.5 / 63.75 / 45.0). The only landed number is a
  lower-bound n=1 (0.151, 73/80Q, ~47% capped) — understates vs n=8.
- **Phase-B verdict so far:** SFT ≈ baseline, statistically indistinguishable at
  the affordable eval power. (`SFT-report.md:302-351`, §6.)

**Phase A (earlier, older scaffold, baseline 15.3% full-val):** mmlb-long 20-epoch
transfer SFT **significantly beat** its baseline on full val: ep20 **20.6%
(+5.3, p≈0.04)**, ep16 17.2 (n.s.), 3-ep undertrained hurts (13.8). This is the
+5.3 number in report Table 1. In-domain leaked + memorized SFT (loss→0.003) only
**ties** baseline → confirms multi-turn observation-shift blunts trajectory
imitation. (`SFT-report.md:273-300`.)

**Overall (both phases):** SFT-on-teacher-trajectories is a **safe warm start, not
a standalone win**; the metric lever is the scaffold/prompt design (~+7pp on the
untrained model) and prospectively on-policy RL.

---

## 5. RL plan (per user — details pending)

- **Method:** GRPO on ANLS reward (the report's stated next lever;
  `SFT-report.md:444-447`). Variants on the menu (CLAUDE.md): Dr.GRPO / GSPO /
  CISPO / DAPO-style controls; also OPD (on-policy distillation from the 27B
  teacher) and combinations.
- **Two initializations to train and compare (per user):** **(a) SFT-init
  (warm-started from an SFT checkpoint)** and **(b) untrained-base init**. SFT's
  value here is a sane, format-correct init that already saturates the easy
  curriculum, giving RL reward variance rather than a zero-reward cold start.
- **Infra (from the session registry — bleeding-edge, not yet a clean result):**
  verl `one_step_off_policy` (AReaL-style **disaggregated async**: dedicated
  rollout GPU + train GPU, gen overlaps train, NCCL weight sync). Data =
  `data/pool/curriculum_rl.parquet` (11,464 prompts), curriculum by `num_pages`
  easy→hard. Reward = `docvqa/reward.py:compute_score` (continuous, multi-alias
  max-ANLS). The Qwen3.5 GDN-hybrid weight-sync into vLLM needed a fix
  (`rollout.load_format=safetensors`). **Only a partial single-checkpoint screen
  exists; full RL results are pending.** (Mark §5 RL as planned/in-progress.)

---

## 6. pass@k reporting (how / where / what's available)

Two independent implementations exist — confirm both are wired:

1. **docvqa-verl eval.py (the one used for baseline/SFT/RL checkpoints in THIS
   repo).** `docvqa/scripts/eval.py` runs **`--n` rollouts/question (default 8)**
   and writes pass@k + SC@k directly into `results.json`'s `summary`:
   - `eval.py:270` → `summary["pass@{n}"]` = mean over questions of `passk`
     (oracle: any of the n correct).
   - `eval.py:271` → `summary["sc-{n}"]` = mean of `sc` (self-consistency vote).
   - per-record `passk` / `sc` aggregated at `eval.py:250`; printed at
     `eval.py:424`. Comment at `eval.py:148`: "always runs all n (pass@n / SC
     metrics need every sample)."
   - **So pass@k is available for any baseline/SFT/RL eval run in docvqa-verl,
     for free, as long as `--n ≥ 2`** (the project evals at n=4→8).
2. **docvqa repo `scripts/pass_at_k.py`** (`~/repos/docvqa`) — a *separate*
   post-hoc tool over the **dspy-runner** `output/runs/<run_id>-t1..-tN/` layout
   (trial-suffix cells + `config.yaml`), reusing `docvqa.metrics.evaluate_prediction`
   (binary ANLS@0.9) + `vote_submissions.py` for SC. It backs `docs/pass-at-k.md`'s
   full solver matrix (e.g. `codeact-chat-4b-llm-27b-vlm-val` avg@1 22.34 / pass@8
   55.0 / SC@8 26.25; 27B `codeact-chat-val` 39.53 / 63.75 / 45.0). It **drops
   incomplete trials** (only trials at max question coverage) so pass@k/SC@k are
   over the full 80-Q set. This is the source of the cross-solver comparison
   tables.

**What's needed to report pass@k:** simply **n samples per question** (n≥2; the
project standard is n=8 for headline, n=4 for faster screens). The docvqa-verl
`eval.py` does this automatically; the docvqa-repo tool needs the run laid out as
`<prefix>-t1..-tN` trial cells. **Caveat for matched comparison
(`SFT-report.md:357-362`):** the base-4B per-trial data was deleted, so a
matched-n baseline pass@k can't be recomputed — compare mean ANLS directly and
note the n difference for pass@k/SC@k.

> One subtlety worth a footnote: in the docvqa-verl dumps, `anls` is stored as the
> already-**binarized** @0.9 value (anls == is_correct), per `pillarB-results.md`.
> So pass@k there = "any of n with binary-correct", consistent with the metric.

---

## 7. Candidate figures from trajviewer (+ data locations)

`tools/trajviewer/` is a FastAPI + vanilla-JS SPA that reads eval run dirs live
(`outputs/runs/<run>/tasks/<doc>/trajectories.jsonl`). The **`/run/<run>/stats`
view** (`static/app.js:500-760`, hand-rolled SVG histograms/bar charts — no plot
lib) already renders exactly the rejection-sampling / data figures we'd want.
Each chart can be filtered to **all / correct / incorrect** subsets (the
`statsState.subset` toggle) — directly useful for "what do *kept* (correct)
trajectories look like."

Charts produced (and their report use):

**Per-trajectory (sample) distributions** (`app.js:707-714`):
- **total tokens (prompt+response)** and **response tokens (SFT target)** — the
  sequence-length budget figure (justifies the 24576-tok packing / 32K cap).
- **# turns** and **# VLM looks** (`vlm_calls`) — agent behavior / perception
  budget; pairs well with the Pillar-A page-count analysis.
- **wall-clock (s)** — exposes the long-doc runaway / `wall_cap` tail.
- **ANLS** (0–1, 20 bins) — the reward/score distribution.

**Per-question distributions** (`app.js:717-720`):
- **# pages (document length)** — the **page-count diversity** figure for the
  data section (joins `metadata.json` num_pages).
- **samples per question** (n distribution).
- **best ANLS of question** (= oracle/pass@n per question) — the **rejection-yield
  / solvability** figure: fraction of questions with ≥1 correct rollout.

**Sampling efficiency** (`app.js:733-735`):
- **trials to first correct rollout** (1..n) — *the* rejection-sampling figure:
  how many of the n attempts it takes to get a keeper. Strong candidate for §5.2.

**Category** (`app.js:737`):
- **questions per category** bar chart (filterable by solve status) — the
  per-category breakdown / coverage figure.

**Candidate report figures (priority):**
1. *trials-to-first-correct* (rejection-sampling efficiency) — §5.2 data figure.
2. *best-ANLS-of-question* (per-question oracle / rejection yield) — pairs with
   the pass@k headroom argument.
3. *# pages per question* (page-count diversity of the pool/teacher-gen set) —
   §5.3 data figure; reconciles with the multi-page-inclusion rationale.
4. *response-tokens (SFT target)* — sequence-length budgeting figure for §5.4.
5. *questions per category* — coverage figure.

**Underlying data locations:**
- Rollout records: `outputs/runs/<run>/tasks/<doc_id>/trajectories.jsonl` (one
  JSON/line; fields incl. `is_correct`, `anls`, `num_turns`, `vlm_calls`,
  `wall_clock_s`, `prompt_ids`/`response_ids` lengths, `sample_idx`).
- Page counts: each doc's `metadata.json` `num_pages` (fallback: count
  `pages/page_*.png`), joined via the run's `config.json` `questions` path
  (`app.py:151-184`).
- The teacher-gen run that produced `teacher_pool.parquet`:
  `outputs/runs/teacher-gen-pool/` (raw trajectories under `tasks/*/trajectories.jsonl`).
- Note: figures must be **exported manually** — the viewer renders SVG in-browser;
  there is no figure-export endpoint, so a report figure means screenshotting the
  stats view or re-deriving the histogram from the jsonl. (Flag: a small
  matplotlib re-derivation script would give publication-quality versions.)

---

## 8. Open / pending items (flag in the draft)

- **§5.3 dataset count: 7 vs 8.** Report lists 7; the pool is 8 (add or footnote
  **MMLongBench-Doc** — it is the source of the long-doc navigation signal AND the
  actual training data behind the §5.3 transfer-SFT results).
- **Phase B headline full-val number is PENDING** (`lr4e4 ep8`, n=4→8). The only
  landed Phase-B SFT full-val point is an n=1 lower bound (0.151). Report §5.3's
  15.3→20.6 numbers are **Phase A** (older scaffold) — label phase explicitly;
  the two baselines (15.7 vs 22.34) are not comparable.
- **Mini-screen set size ambiguity:** brief says 13Q; report uses 13-Q
  (`docvqa_mini`/`docvqa_rank13`) currently but 29-Q in Phase A. State "13-Q
  stratified screen, full-80-Q confirm" and note the screen is unreliable
  (false-null history).
- **RL is planned, not done.** Only a partial single-checkpoint RL screen exists;
  the async-disaggregated infra is bleeding-edge (per session registry). Mark §5
  RL as in-progress; the SFT-init-vs-base comparison is the planned design.
- **DocVQA-2026 `test` leakage not yet checked** — only val is verified
  leakage-free. Caveat any future test-set claim.
- **Matched-n baseline pass@k not recomputable** (base-4B per-trial data deleted)
  — compare mean ANLS; note n difference for pass@k/SC@k.
- **trajviewer figures need manual export** (no export endpoint) — consider a
  matplotlib re-derivation from the jsonl for publication quality.
- **`anls` stored binarized @0.9** in docvqa-verl dumps (anls == is_correct) — any
  "continuous ANLS distribution" figure from these dumps is actually the binary
  histogram; use the continuous `reward.py:compute_score` if a soft distribution
  is wanted.
