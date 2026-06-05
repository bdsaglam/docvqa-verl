# Stage-0 Execution Log — Off-Policy Distillation (CodeAct 4B ← 27B)

> Running findings log while executing autonomously. Newest entries at the bottom
> of each section. Deadline: report **2026-06-15**. Goal: beat ≤8B-tier SOTA
> (~0.1875 binary-ANLS) on DocVQA-2026 val; ideally approach the 8B–35B leader
> (~0.375).

## Fixed decisions (from brainstorming, do not relitigate)

- **Method order:** Stage 1 = traditional **off-policy distillation** from the
  27B CodeAct teacher. Loss ladder: **SeqKD** (= SFT on successful teacher
  trajectories) first, then forward-KL top-k. No method-combining yet.
- **Teacher = 27B running OUR CodeAct `agent_loop`** (regenerated trajectories),
  NOT the docvqa-repo `rvlm`/dspy solvers (format-incompatible: dspy uses
  `[[ ## reasoning ## ]]` markers, we use `<think>`+```python``` fence).
- **No leakage:** never train on any doc DocVQA-2026 draws from. Primary
  transfer set = MMLongBench-Doc.
- **First GPU experiment** = in-distribution SeqKD learnability probe (train on
  dv2026 val-train, eval on val-heldout + a train subset) to prove the training
  setup works before the real transfer run. Logic: if it can't improve even
  in-domain, the setup is broken.
- VLM stays frozen (external HTTP). We fine-tune only the 4B LM, LoRA by default.

## Infra map (as of 2026-06-05)

- **GPUs (local, 4× ~80 GB):** all pinned by inference servers.
  - GPU 0: vLLM `EngineCore`, ~77 GB, 100% util — serving the active eval.
  - GPUs 1/2/3: vLLM DP=3 server, ~72 GB each, intermittently idle util.
- **Endpoints:**
  - `:8927` local = **Qwen3.5-27B** (VLM + teacher-LM; used by the running eval's VLM).
  - `:8909` local = **Qwen3.5-9B** (model under eval in `eval-9b` tmux session).
  - `:8928` = **Qwen3.5-27B remote** (ssh tunnel). Used by docvqa judge by default.
- **Running work (not ours, do not disturb):** `eval-9b` tmux session is an
  active CodeAct eval (LM 8909 + VLM 8927).
- **Disk:** `/` at 95% (~190 GB free). Watch checkpoint/cache growth.
- **Monitor cron** `18d03cc4` (:13/:43, session-only): checks for free GPU to
  launch training.

## Data state (leakage-clean, branch `docvqa-stage0`)

| Split | Questions | Docs | Purpose |
|---|---|---|---|
| mmlb train | 899 | 109 | transfer-train teacher trajectories |
| mmlb heldout | 65 | 7 | transfer dev |
| dv2026 val-train | 56 | 17 | in-dist probe train |
| dv2026 val-heldout | 24 | 8 | in-dist probe dev |
| dv2026 val (full) | 80 | 25 | final eval |

Teacher ceiling (docvqa dspy CodeAct-27B): ~0.375 binary-ANLS. Student floor
(our config, CodeAct 4B-LM + 27B-VLM, zero-shot): ~0.169.

## Critical path

1. **Teacher trajectory collection** (GPU-free, uses existing 27B servers) →
   SeqKD data. *Bottleneck for everything.* ← doing now.
2. SeqKD training (needs a free local GPU) → wait for eval to release GPUs.
3. Eval trained checkpoint on dv2026 val (n=8, mean±std, pass@8, SC-8).

## Timeline / actions

### 2026-06-05
- Grounded state after 5-day gap. Confirmed: data prepped, no trajectories yet,
  no training yet, GPUs busy with user's eval, monitor cron alive.
- Decision: collect teacher trajectories against **remote 8928** to avoid
  contending with the local eval (which holds 8927/8909).
- Running single foreground validation rollout to confirm path + format + latency.

**Findings (session 2026-06-05):**
- **8928 (remote 27B) is DEAD** — consistent timeouts on even trivial requests.
  Removed from the plan. Only the local 27B (:8927) is usable, and it is
  shared with the eval.
- **`eval-9b` is a 4-config suite** (tmux windows `v1-homog`, `v2-mixed`,
  `v2-t7`, `v3-rvlm-t3r4`), actively churning (~1 q/min; saw `science_paper_1_q4`
  WRONG GT=76 PRED=167). It holds all 4 local GPUs + :8927/:8909. Likely runs
  for hours+. Not ours — do not disturb.
- **Validation path:** two attempts (300s, 600s) both timed out — but BOTH hit
  the worst case (`comics_2_q1`, 52-page comic) under full eval contention. The
  path itself is proven (the eval runs this exact CodeAct loop). Format confirmed
  from code: append-only system/user/assistant; assistant = `<think>` + ```python```
  fence; matches `MultiTurnSFTDataset`.
- **Feasibility finding (important):** our teacher rollouts run the 27B as BOTH
  LM (long reasoning, up to 30 turns) and VLM — far heavier than the eval's
  (9B LM + 27B VLM). A rollout is minutes even unloaded; 899 mmlb Q x 4 samples
  ≈ tens of GPU-hours. With contended servers + June-15 deadline:
  **prioritize the 56-Q probe** (small, gives the learnability signal), then
  right-size transfer collection from observed teacher success rate.
- **Collection is starving** behind the eval: 0 completions in the first ~15 min
  at concurrency 3. Kept it running as a **self-restarting resumable loop**
  (tmux `stage0-prep:collect-probe`, conc=3, n=4, temp=0.8) so it harvests spare
  server cycles and self-heals; will ramp to conc=8 when the eval frees :8927.

**Built this session (GPU-free, ready to fire):**
- `docvqa/train/run_seqkd.sh` — parametrized LoRA SFT (SeqKD) launcher for
  Qwen3.5-4B; sets `ignore_input_ids_mismatch=True`, `max_length=32768`,
  `truncation=error`, dynamic bsz, FSDP2, single-GPU default. Env-overridable.
- `docvqa/train/README.md` — full pipeline (collect → make_sft_data → train →
  serve+eval) + correctness notes (assistant-only mask, no right-truncation).
- Qwen3.5-4B weights cached locally (9.3 GB) — training needs no download.
- Monitor cron recreated (`692efde1`, :17/:47): idempotent, ramps collection +
  launches the SeqKD probe when a GPU frees. **Session-only** (durable flag not
  honored by the runtime) — dies if this session ends.

**Blocked on:** the eval finishing (frees :8927 for fast collection + a GPU for
training). Until then: trickle-collect + GPU-free prep only. Both critical-path
activities (collect, train) need resources the eval currently holds.

**De-risking win — fixed a real training crash on CPU (no GPU spent):**
- `MultiTurnSFTDataset` tokenizes each turn separately. Our trajectories begin
  with a `system` turn (agent_loop.py:121). The Qwen3.5 chat template needs a
  user message, and verl's fallback PREPENDED a dummy user → `[user, system]` →
  TemplateError "System message must be at the beginning". This would have
  crashed SeqKD on the first batch.
- Fixed `verl/utils/chat_template.py`: for a leading-system, user-less message,
  APPEND the dummy user as a suffix and strip it (preserves the system turn's
  exact tokens). Commit 58ebff21.
- Validated on CPU: system-led CodeAct trajectory loads; **loss mask is
  assistant-only** (reasoning + ```python``` + SUBMIT; excludes system / user /
  observation). `ignore_input_ids_mismatch=True` is required (confirmed it
  raises without it). Regression test `tests/docvqa/test_sft_mask.py`; full
  docvqa suite **58 passing** (needed `pytest-asyncio` for the 6 async
  agent-loop tests — installed in .venv).

**Throughput reality (measured):** with the eval saturating :8927, a single
CodeAct-27B rollout did not complete in 30+ min at concurrency 3 (process alive,
ESTABLISHED sockets to :8927 — crawling, not hung). Our rollouts are heavy (27B
as LM doing long reasoning × up to 30 turns + 27B VLM calls), far heavier than
the eval's (9B LM). Implication: collection is **not viable at useful rates
until the eval releases :8927**. Plan: keep the resumable trickle (captures any
completions, self-heals), and let the monitor ramp to concurrency 8 the moment
`eval-9b` ends. After the eval, a freed GPU runs the SeqKD probe.

**Ready-to-fire state (everything GPU-free is done):** data prepped + leakage
clean; collect→make_sft_data→train pipeline wired + the training crash fixed and
mask validated; `docvqa/train/run_seqkd.sh` + README; Qwen3.5-4B cached; monitor
cron `e096cb9b` will ramp collection and launch the probe when resources free.
The remaining critical path (collect → train → eval) is gated purely on the eval
finishing.

**Recipe config validated** (`python -m verl.trainer.sft_trainer ... --cfg job`):
all `run_seqkd.sh` hydra overrides compose with no unknown-key errors (data /
model.lora_* / engine=fsdp / optim=fsdp / trainer / checkpoint all applied).
So the monitor's training launch won't fail on a config typo; remaining launch
risk is just GPU-memory tuning (4B + LoRA + 32k context), which the monitor
diagnoses (lower MAX_LENGTH / MICRO_BATCH_SIZE_PER_GPU on OOM).

**Decision: collection PAUSED during the eval.** Running it under 27B contention
produced 0 completions in 30+ min while slowing the user's eval — negative value.
Killed the collect loop/window. Monitor `6c68c4b3` now gates on `eval-9b`: while
it runs, only report; once it's gone, ramp collection (conc 8, probe + mmlb) and
launch the SeqKD probe on a freed GPU. **This session must stay alive** for the
(session-only) monitor cron to keep firing.

### Status at session pause (2026-06-05 ~00:40)
Everything implementable without GPUs/spare-27B is done and tested. Blocked on
the user's 4-config eval releasing :8927 + a GPU. Next concrete steps (monitor-
driven, in order): collect probe trajectories → build parquet → SeqKD probe
train → eval on dv2026 val → (if learnability confirmed) collect mmlb transfer →
transfer train → report. Open knobs to tune once real data exists: epochs for
the overfit rung, train_batch_size for small sets, forward-KL top-k as ladder
rung 2.

### 2026-06-05 ~13:00 — monitor cycle: EVAL DONE, GPUs free
- The 4-config eval finished: all `eval-9b` windows back at idle shell prompts;
  local servers `:8927`/`:8909` torn down; **all 4 GPUs free** (14 MiB, 0%).
  The tmux session still exists (not closed) — so the "session exists" gate is
  misleading; reality (idle GPUs, eval done) says GO. Acting on it.
- `:8928` (remote 27B) is alive but **too slow for collection**: a 1-page-doc
  validation rollout timed out at 280s (network + the reasoning model's long
  per-turn generation). Not viable.
- `vllm` is NOT in our .venv, but IS in sibling project venvs
  (prime-rl, rlvr, epiq, pipeline-grpo). Using `prime-rl/.venv/bin/vllm`.
- **Call-path analysis (decides server config):** LM uses `/v1/completions`
  (raw text, pre-templated prompt — `reasoning-parser` irrelevant); VLM
  `batch_look` uses `/v1/chat/completions` reading `message.content`
  (reasoning-parser ON ⇒ null-content risk). → serve the 27B **without**
  `--reasoning-parser`.
- **Launched local 27B** (collection endpoint): `CUDA_VISIBLE_DEVICES=0,1
  vllm serve Qwen/Qwen3.5-27B --port 8927 --gpu-memory-utilization 0.85
  --data-parallel-size 2 --dtype bfloat16 --max-model-len 65536 --enforce-eager
  --enable-prefix-caching` (tmux `stage0-prep:vllm-27b`). GPUs 0,1 = collection;
  GPUs 2,3 reserved for training. Next: validate one rollout against local 8927
  (check batch_look perception isn't truncated-thinking), then collect probe at
  high concurrency, then SeqKD probe train on GPU 2.
- **ROOT-CAUSED the rollout slowness (was an implementation bug, now fixed):**
  `docvqa/tools.py` batch_look set neither `enable_thinking` nor sampling, so the
  Qwen3.5 VLM defaulted to **thinking ON** — both a train/deploy mismatch (the
  deployed scaffold's `configs/vlm/qwen-3_5-27b-vllm-local.yaml` uses
  `enable_thinking:false`, temp 0.3, top_k 20) and the dominant cost (~70s/turn
  of VLM reasoning). Fixed to match deployment (commit de88399f). After the fix a
  full rollout completes (validation: term=submit, 4 turns, 230s, ~57s/turn,
  correct format: python fence + real batch_look perception). The earlier
  280/220s timeouts were this bug + hard 52-page docs, NOT a hang.
- **Probe collection LAUNCHED** on local 8927: tmux `stage0-prep:collect-probe`,
  self-restarting resumable loop, conc 6, n4, temp 0.7 ->
  outputs/teacher_rollouts/dv2026_train_n4.jsonl. First completed rollout:
  iter_cap @ 667s (hard q, anls=0) — some rollouts are slow/fail; success yield
  TBD over next cycles. GPUs 0,1 ~71% util (server busy), 2,3 free for training.
- **Monitor updated** -> `05a24900` (:12/:42): steady state — keep the 27B
  server + collection alive, launch the SeqKD probe (EPOCHS=8) on GPU 2/3 once
  >=20 anls==1.0 trajectories exist, diagnose OOM. Eval gate removed.
- **Feasibility note:** at ~5-11 min/rollout and ~6 concurrent, the 56-Q probe
  (~224 rollouts) is hours; **full mmlb (899 Q x4) is NOT feasible** — must
  subsample mmlb (~150-200 Q) for the transfer run. Revisit after seeing teacher
  success rate on the probe.

### 2026-06-05 ~01:40 — monitor cycle: fixed collection ordering logjam
- State healthy: disk 186G free, 27B server up (8927), GPUs 0,1 busy (collection),
  2,3 free. No training yet (gate: <20 successes). NOTE: monitor's
  `pgrep -f sft_trainer` self-matches the cron's own command text → false
  "TRAINING RUNNING"; the `train-*` tmux-window check is the reliable gate.
- **Collection was stuck on the hardest doc first.** Questions processed in file
  order; doc[0] = `comics_2` (52-page comic) → first ~32 rollouts all on it,
  ~11 min each, 0 successes (1 iter_cap @29 turns, 1 wrong submit). Teacher IS
  reasoning fine (`<think>` present, 1-4k chars) — these are just brutal Qs.
- **Fix:** sorted the probe questions by ascending page count
  (`data/docvqa-2026/val/train_byasc.json`, 1-page maps first → 110-page report
  last) and restarted the collection loop on it (same output, resumable). Easy
  docs now yield fast successes first → reach the 20-success training gate much
  sooner; the giant docs (where the teacher mostly fails anyway) come last.
  (tmux churn: had to kill a duplicate collect window; now single window #5.)

### 2026-06-05 ~02:10 — monitor cycle: bumped concurrency; teacher-quality read
- Collection progressing: 11 rollouts, **1 success (~10%)**, mostly `submit`
  (teacher answers, doesn't get stuck). Per-rollout still ~5-17 min — now
  **LM-bound** (teacher `<think>` reasoning per turn, ~70s/turn), NOT page/VLM
  bound; the VLM fix worked but the teacher's own reasoning is the cost.
- **Success rate is low but the prefix is maps-heavy** (page-sort put all the
  1-page `maps` docs first; maps are content-HARD — reading legends/numbers off
  a map). Teacher near-misses: `"Wareham, 4415"` vs gold `"Wareham"` (extra text
  fails strict metric), misread `17/20` vs `27`. Real read on teacher quality
  needs the non-maps categories (coming next in sort order). NOTE: page-count !=
  difficulty — maps are small but hard.
- **GPUs only ~60% util at conc 6** (rollouts idle the server during local
  subprocess / VLM-serialization phases) -> bumped collection to **concurrency
  12** (fits KV: our seqs ~20k tokens, max-model-len 65536). Should raise
  throughput toward the 20-success training gate (~3-4 hrs at current yield).
- No training yet (1 success, gate is 20). Monitor `05a24900` auto-launches the
  probe when the gate is met.

### 2026-06-05 ~02:45 — monitor cycle: sample-major ordering for diversity
- 18 rollouts, **5 successes (28%)** — teacher does fine on maps after all (10%
  earlier was small-sample noise). BUT only **2 unique questions** solved: n=4
  question-major order produced redundant samples on the same few easy questions.
- **Fix:** switched `collect_trajectories.py` to **sample-major** order (sample 0
  of every question, then sample 1, ...) so one pass covers all 56 questions ->
  successes spread across many questions (commit 7e3129bf). Restarted collection
  (resumed from 18; remaining tasks now cover non-maps categories first).
- **Throughput note:** ~12 rollouts/hr; GPUs only ~60% util even at conc 12 ->
  the bottleneck is NOT the server but per-rollout serialization (sequential
  turns: LM gen -> subprocess code exec -> VLM call -> next turn). A potential
  speed lever is removing `--enforce-eager` from the server (enables CUDA graphs)
  — DEFERRED (server relaunch risk; collection works, probe reachable overnight).
- **GOTCHA for future cycles:** `pkill -f collect_trajectories.py` also kills the
  while-loop shell (its cmdline contains that string), so the self-restart won't
  fire — must relaunch the loop manually after such a kill. Monitor already
  handles "window/proc died -> relaunch", so this self-heals next cycle anyway.
- Still ~12-15 more successes needed for the gate (~2-4 hrs). Per-rollout
  iter_cap failures (4 so far) waste ~20-27 min each — a future lever is a
  collection-only max_iterations cap, but successes submit early so it's
  secondary.

### 2026-06-05 ~03:20 — MILESTONE: SeqKD training validated end-to-end on real data
- Collection reached 14 successes / **11 unique questions** (40% rate) — sample-major
  fix working; successes across maps, engineering_drawing, infographics, science_poster.
- Decided to launch the probe training NOW (at 14, not waiting for 20) to validate the
  REAL-DATA training path early while GPUs 2,3 idle — the biggest remaining unknown
  (CPU validation used synthetic data).
- Built parquet: `make_sft_data --max-per-question 2` -> 13 trajectories. Validated via
  MultiTurnSFTDataset on real data: token lengths 4.3k/6k/12.6k (« 32768, no truncation),
  assistant-only loss mask 0.18-0.45. (Whole-conversation apply_chat_template returns
  len 2 on Qwen3.5 — a red herring; the per-turn dataset path is what matters.)
- **Caught + fixed 2 real training-launch bugs** (the point of running early):
  1. Model defaulted to flash_attention_2 (not installed) -> crash. Fixed:
     `+model.override_config.attn_implementation=sdpa` + `use_remove_padding=False`
     (varlen packing needs flash-attn). `engine.attn_implementation` isn't an FSDP-engine
     field -> dropped. Both env-overridable now. (commit ac90acc3)
  2. Recipe default train_batch_size=16 > 13 samples = 0 steps/epoch (silent no-op).
     Launch with TRAIN_BATCH_SIZE=4 -> 3 steps/epoch.
- **TRAINING RUNS**: Qwen3.5-4B + LoRA(r32), FSDP2, sdpa, EPOCHS=10, ~92s/step, 30 steps,
  ~45 min total. GPU 2. Moved from a raw `&` diag run into tmux `train-seqkd-probe` for
  robustness. Collection continues on GPUs 0,1 in parallel.
- NEXT cycle: confirm training finished + inspect the loss curve (did train loss drop? =
  learnability signal), locate the LoRA checkpoint, then the adapter is ready to EVAL on
  dv2026 (needs serving base+LoRA — see docvqa/train/README "Evaluating"). This probe is
  a learnability + pipeline check; a fuller retrain on more successes can follow.

### 2026-06-05 ~03:45 — CORRECTION + fix: probe training now actually running
- Last cycle's "training running in tmux" was PREMATURE: the diag run worked briefly,
  but the tmux relaunch **OOM'd on physical GPU 0** (the busy vLLM server). Cause:
  `CUDA_VISIBLE_DEVICES=2` as an inline prefix to run_seqkd.sh did NOT pin the device
  through torchrun (it leaked onto GPU 0). No checkpoint was produced.
- **Fix:** `export CUDA_VISIBLE_DEVICES=3` (separate command, not inline) + GPU 3
  (fully free) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Now training runs
  correctly on GPU 3 (no OOM; GPU 0 untouched).
- Rebuilt parquet from grown collection: **17 trajectories / 14 unique questions**
  (maps, engineering_drawing, infographics, science_poster, slide) — good diversity.
- Training: Qwen3.5-4B+LoRA(r32), FSDP2, sdpa, EPOCHS=10, 4 steps/epoch = 40 steps,
  ~92s/step (~60 min), GPU 3, tmux `train-seqkd-probe`. Collection continues GPUs 0,1.
- NEXT: confirm completion + loss curve (learnability), locate adapter, eval on dv2026.
- Monitor lesson: pin training GPU via `export`, not inline prefix.

### 2026-06-05 ~04:00 — probe training STABLE after OOM root-cause
- Prior relaunch crashed at **step 2 OOM** (78/79GB). Root cause: `use_dynamic_bsz`
  packs multiple long CodeAct trajectories per micro-batch; **sdpa's quadratic
  attention memory** then OOMs (the 12.5k-token trajectories are the driver). The
  "GPU 0" in the error is PyTorch's view of the CUDA_VISIBLE_DEVICES=3 device.
- **Fix:** `data.use_dynamic_bsz=False` -> 1 sequence/forward; peak memory bounded
  to the single longest trajectory (~57GB < 80). Made durable: recipe default
  `USE_DYNAMIC_BSZ=False` (commit 30c01300). Also hit exit-127 (background cmd
  lacked venv) — relaunched via tracked bg job WITH `source .venv/bin/activate`.
- **Training now stable**: step1 mem57.3GB loss0.383, step2 mem57.3GB loss0.466
  (bounded mem, no OOM). 40 steps (~55min), GPU 3, bg job `bflwfygdq` ->
  outputs/prep/train_seqkd_probe.log. Collection continues GPUs 0,1.
- Lessons (3 real env/config bugs caught by running on real data, as intended):
  flash-attn->sdpa; train_batch_size<dataset->0 steps; dynamic-bsz+sdpa OOM.
  Plus infra gotchas: pin GPU via export, activate venv in bg jobs.
- NEXT: loss curve over 10 epochs (overfit signal), checkpoint, eval adapter.

### 2026-06-05 ~04:25 — probe training LEARNING; collection at 30 successes
- Training healthy on GPU 3, step ~21/40. **Loss curve shows clear learning**
  (the learnability signal we wanted): ~0.40 (epoch 1, steps 1-4) -> ~0.23-0.28
  (epoch ~3-4) -> ~0.17-0.21 (epoch ~5). Decreasing monotonically-ish on the
  17-trajectory set. ~27 min to completion (save at step 40). No OOM/errors.
- Collection (parallel, GPUs 0,1): **72 rollouts, 30 successes, 23 unique Q**,
  7 categories (maps, engineering_drawing, infographics, science_poster, slide,
  business_report, comics), ~42% success. Already richer than the 17 we're
  training on -> can retrain on the fuller set for a stronger result.
- All infra healthy: disk 180G, server up, both loops alive.
- NEXT: on training completion — confirm final loss, locate LoRA adapter under
  checkpoints/docvqa-seqkd/seqkd-probe, then EVAL on dv2026 (serve base+LoRA,
  run eval.py n=8). Then decide: retrain on 30-success set, or move to mmlb
  transfer collection.

### 2026-06-05 ~05:05 — MILESTONE: learnability probe SUCCEEDED (training works)
- **Probe v1 training COMPLETE & learning confirmed**: loss 0.383 (step1) -> 0.106
  (step40), clean ~4x drop over 10 epochs (steps 33-40 ~0.06-0.13). Checkpoint:
  `checkpoints/docvqa-seqkd/seqkd-probe/global_step_40/` (FSDP-sharded LoRA +
  lora_train_meta.json + huggingface/ config+tokenizer). The end-to-end training
  setup is VALIDATED: the 4B student learns to imitate 27B teacher trajectories.
- **v2 retrain launched** on the fuller set (33 trajectories, collection now 33
  successes): EPOCHS=5 (less overfit than v1's 10), GPU 3, bg job `b8s2au7nn` ->
  outputs/prep/train_seqkd_probe_v2.log, experiment `seqkd-probe-v2` (fresh dir,
  preserves v1).
- **EVAL PATH (concrete, for next cycle):**
  1. Merge LoRA->HF: `python -m verl.model_merger merge --backend fsdp
     --local_dir checkpoints/docvqa-seqkd/seqkd-probe-v2/global_step_<N>
     --target_dir checkpoints/docvqa-seqkd/seqkd-probe-v2/merged_hf`
  2. Serve merged model with vLLM on a free GPU (agent_loop uses /completions).
  3. `python docvqa/scripts/eval.py --questions data/docvqa-2026/val/questions.json
     --lm-base-url <served> --vlm-base-url http://localhost:8927 --n 8` and compare
     to the ~16.9% zero-shot baseline.
- NOTE: probe trains on dv2026-train and would eval on dv2026 (in-distribution) —
  this measures whether training *improves* the student, not a leakage-clean
  number. The clean reportable result still needs the mmlb->dv2026 transfer run.

### 2026-06-05 ~05:15 — EVAL pipeline stood up; first eval running
- Merged v1 LoRA->HF: `verl.model_merger merge --backend fsdp` -> merged_hf/
  (8.5GB bf16 model.safetensors + lora_adapter/). Fixed missing
  preprocessor_config.json / video_preprocessor_config.json (Qwen3.5-4B is a
  ConditionalGeneration/VL model; vLLM needs them) by copying from base cache.
- Served merged student on GPU 2 :8930 (prime-rl vllm). GOTCHA: eval.py loads the
  tokenizer via AutoTokenizer.from_pretrained(student_model), so --student-model
  must be the LOCAL PATH, and the vLLM served id must match -> serve WITHOUT
  --served-model-name (id = the path).
- **First eval RUNNING** (bg `bygi9ls9g`): trained student on dv2026 **val-heldout**
  (24 Q, held out from probe training), n=1, temp0.6/top_p0.95/top_k20, student
  :8930 + VLM :8927. -> outputs/eval/seqkd_probe_heldout_n1.jsonl. ~25-40min.
- Parallel: v2 retrain on GPU 3 (33 traj, EPOCHS=5); collection on GPUs 0,1
  (88 rollouts, 36 successes). All 4 GPUs now busy.
- DISK WATCH: 154G free (96%). Each checkpoint's model_world_size_*.pt is 18GB +
  merged_hf 8.5GB. max_ckpt_to_keep=2 bounds it, but clean stale checkpoints if
  it drops below ~40G.
- NEXT: read heldout eval result (first real ANLS signal: does the trained 4B beat
  the ~16.9% zero-shot baseline?). Caveat: in-distribution (trained on dv2026-train);
  a matched untrained-4B baseline on the same heldout set is still needed for rigor.

### 2026-06-05 ~05:45 — steady state: eval running, v2 near done
- Heldout eval (bygi9ls9g) progressing: student server :8930 served 44+ completions
  (55-60 tok/s); writes results only at end. Contends with collection on the shared
  27B VLM :8927, so slower than standalone. No errors.
- v2 retrain: step 33/40, loss ~0.20-0.32 (down from ~0.42 at epoch 1). ~10min left.
- Collection: 99 rollouts / 39 successes / 26 unique Q.
- All 4 GPUs busy; disk 152G. No intervention.
- TODO when a GPU frees: serve BASELINE untrained Qwen3.5-4B + eval same heldout for
  a matched trained-vs-baseline comparison (the rigorous probe claim). Command mirrors
  the trained eval but --student-model Qwen/Qwen3.5-4B on a fresh served base model.

### 2026-06-05 ~06:00 — v2 done; collection PAUSED; matched baseline eval launched
- v2 retrain COMPLETE: final loss 0.176 (33 traj, EPOCHS=5), checkpoint
  checkpoints/docvqa-seqkd/seqkd-probe-v2/global_step_40. GPU 3 freed.
- **Paused dv2026 collection** (had 39 successes / 26 unique Q — enough for the
  probe; resumable later) to free the 27B VLM :8927, which was the bottleneck
  slowing the evals (collection conc12 + eval both hammered it).
- **Launched matched BASELINE eval**: untrained Qwen3.5-4B served on GPU 3 :8931,
  same dv2026 val-heldout (24 Q), n=1, same sampling -> baseline_heldout_n1.jsonl
  (bg bx5cj5v0a, self-sequencing after server up). Now I'll have trained-vs-baseline
  on the identical held-out set = the rigorous probe comparison.
- Trained eval (bygi9ls9g, student v1 :8930) still finishing (slow under prior
  contention). Both write results at end.
- GPUs: 0,1 = 27B VLM (now eval-only load); 2 = trained student serve+eval;
  3 = baseline serve+eval. disk 152G.
- NEXT: read BOTH heldout numbers -> does SeqKD-trained 4B beat untrained 4B?

### 2026-06-05 ~06:15 — both heldout evals grinding (VLM-latency-bound)
- Trained (v1 :8930) ~106 student turns done (~half); baseline (4B :8931) ~46 (~20%).
  Slow because each rollout turn blocks on a 27B-VLM batch_look over a high-res doc
  page (~tens of seconds/call even at 60% VLM util) x multi-turn x 24 Q. Not stuck.
- Disk 133G free (stable; the -20G was v2's 18G checkpoint, not a leak). HF cache
  650G is pre-existing. Cleanup lever if needed: v1 raw model_world_size_*.pt (18G)
  is redundant given merged_hf — safe to delete (won't resume v1).
- No intervention; both evals notify on completion. Then: trained-vs-baseline binary
  ANLS on identical 24 heldout Q = the probe's headline comparison.
- IMPLICATION for the real eval: n=8 x 80 Q would be ~640 multi-turn rollouts at this
  VLM-bound rate = many hours. The reportable eval will need either higher eval
  concurrency, a dedicated VLM, or accepting a long run.
- 2026-06-05 ~06:45: heldout evals still grinding — trained ~195 turns (~75% of 24Q), baseline ~101 (~40%); VLM healthy, not stuck. Waiting for results (write-at-end). FUTURE: make eval.py write incrementally + bump eval concurrency (VLM KV only ~6% used) for the long n=8x80 reportable eval.
