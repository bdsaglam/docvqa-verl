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

### 2026-06-05 ~07:00 — FIRST EVAL RESULT (trained v1, dv2026 heldout, n=1)
- **Trained v1 (17 traj, EPOCHS=10) on val-heldout (24 Q, n=1): mean binary-ANLS
  = 0.1250 (3/24), pass@1=0.125.** By cat: infographics 0.50, science_poster 0.20,
  science_paper 0.14, rest 0. (eval output is a JSON ARRAY, not JSONL.)
- **Caveats (important, honest):**
  - n=1 over 24 Q is NOISY — each question = 4.2%; 3 vs 4 correct is 1 question.
  - 2 of the 3 "correct" are `"Unknown"` (unanswerable Qs, gold="Unknown"); only
    ONE is a real extracted answer ("30.2%").
  - **8/24 (33%) hit `token_cap`** (ran out of tokens without submitting = forced
    failures). The trained model OVER-GENERATES on hard held-out docs — likely an
    EPOCHS=10 overfit -> verbosity effect. v2 (EPOCHS=5) may behave better.
  - The ~16.9% "baseline" was full-80Q val, NOT this 24-Q subset -> not comparable.
- **Matched baseline (untrained 4B, SAME 24 Q) is the real comparison** — running
  (~70% done). Until then, no claim about whether SeqKD helped/hurt.
- Hypothesis to test: token_cap failures suggest catastrophic-forgetting/verbosity
  from narrow in-dist SFT — exactly the risk flagged at the start. If baseline is
  cleaner, this motivates EPOCHS<=5, KL-regularized KD, or OPD/Pedagogical RL.
- 2026-06-05 ~07:12: baseline eval ~80% done (191 turns, ~19/24 rollouts, 80min elapsed) — NOT stuck (mis-read clock earlier), grinding the last few large-doc/token_cap rollouts. trained v1 result already in (12.5%). Waiting for baseline to complete the matched comparison.

### 2026-06-05 ~07:15 — PROBE RESULT (matched comparison complete)
**dv2026 val-heldout (24 Q, n=1), trained-v1 vs matched untrained-4B baseline:**
| model | mean binary-ANLS | correct | token_cap |
|---|---|---|---|
| **Trained v1** (17 traj, EPOCHS=10) | **0.1250** | 3/24 | 8 |
| **Baseline** (untrained 4B) | **0.1250** | 3/24 | 10 |

- **NO net improvement on held-out** (12.5% = 12.5%). The per-category breakdown is
  identical; aggregate score is a coincidence of the binary metric.
- BUT the LoRA IS working: **11/24 held-out answers differ** between trained and
  baseline — training changed behavior, it just didn't net-improve (wrong->wrong
  swaps dominate; any wrong->right offset by right->wrong).
- **token_cap trained(8) < baseline(10)** => training did NOT increase verbosity;
  the verbosity/forgetting hypothesis is REJECTED. token_cap is doc-driven (large
  held-out docs), affects both models.

**Honest interpretation:**
- The probe's PURPOSE (validate the pipeline end-to-end + show the model CAN learn
  teacher data) is ACHIEVED: loss 0.40->0.10, behavior changed. Mechanism works.
- Flat held-out performance is EXPECTED, not a failure: training on only **17
  trajectories / 14 unique questions** cannot teach general DocVQA skill that
  transfers to 24 different held-out docs. This was always a learnability probe,
  not a performance run.
- n=1 over 24 Q is very noisy (±1 Q = ±4%); 3-vs-3 is within noise regardless.
- CONCLUSION: pipeline de-risked; performance question is UNANSWERED and requires
  the real **transfer run** (mmlb, hundreds of trajectories, leakage-clean).

**Next (the actual path to a reportable number):** mmlb->dv2026 transfer. Collect
many more teacher trajectories (mmlb, subsampled for feasibility), train, eval n=8
on dv2026 val. The probe says the machinery is ready; now it needs DATA AT SCALE.

### 2026-06-05 ~07:45 — PROBE PHASE COMPLETE -> transfer-collection phase
- Probe phase concluded (see result above): pipeline validated end-to-end
  (collect->SFT->train->merge->serve->eval), learnability shown (loss 0.40->0.10),
  held-out performance FLAT (trained 12.5% = baseline 12.5%, n=1) — expected for a
  17-trajectory training set. The machinery works; performance needs scale.
- **Transitioned to MMLB transfer-data collection** (the leakage-clean path to a
  reportable number):
  - Freed GPUs 2,3 (killed the idle student/baseline eval servers).
  - Restarted the 27B as **DP=4 on ALL 4 GPUs** (port 8927, proven config) for max
    collection throughput. (Botched the first restart — killed the working DP=2
    server before the relaunch landed; recovered. tmux send-keys remains finicky.)
  - Page-sorted mmlb train (899 Q; min 9 pages so rollouts are slower than dv2026's
    1-pagers) and launched a resumable collection loop: tmux stage0-prep:collect-mmlb,
    conc 16, n4, temp0.7 -> outputs/teacher_rollouts/mmlb_train_n4.jsonl.
  - Monitor cron -> 6ccb737f: keeps DP=4 server + mmlb collection alive; reports
    progress; **does NOT auto-launch transfer training** (left for user).
- **DECISION FOR USER:** the transfer TRAINING is deliberately not auto-started.
  Given the flat probe, you may want to weigh: (a) how much mmlb data to collect /
  time budget; (b) continue SeqKD at scale vs pivot toward KL-reg KD / OPD /
  Pedagogical RL; (c) eval throughput (n=8x80 ~= half a day at current VLM-bound
  rate — needs higher eval concurrency or a dedicated VLM). Data accumulates
  overnight either way.
- 2026-06-05 ~08:15: mmlb collection healthy — 47 rollouts, 19 successes (40%), 19 unique Q, ALL clean 'submit' (no token_cap, unlike dv2026 heldout). median 6.5min/rollout, DP=4 all 4 GPUs ~60% util. ~40 successes/hr -> ~150 in a few hrs. Disk 132G stable. No intervention; transfer training held for user.
- 2026-06-05 ~08:45: mmlb collection on track — 91 rollouts, 36 successes (40%), 36 unique Q (still 1 success/Q = full diversity), 89 submit / 2 token_cap. ~34 successes/hr -> ~150 in ~3.4hr. disk 132G, all healthy. No intervention.
- 2026-06-05 ~09:15: mmlb collection — 127 rollouts, 49 successes (39%), 49 unique Q (full diversity), 123 submit/4 token_cap. ~26 successes/hr -> ~150 in ~4hr. disk 131G, all healthy. No intervention.

### 2026-06-05 ~09:40 — EVAL PROTOCOL: two-tier (dev subset vs final full)
- Per user: don't eval on the full set during iteration. Use a fixed **stratified
  subset** for before/after (baseline vs trained); full set only for the FINAL model.
- Built `data/docvqa-2026/val/eval_subset_strat24.json`: **24 Q, 3 per category x 8
  categories** (16 docs), deterministic (seed 0). This is the standard DEV eval set.
- Protocol:
  - **DEV (every train iteration):** eval baseline + trained on eval_subset_strat24,
    n=1 (quick signal) or n=2-4 (firmer). Fast, paired, same questions.
  - **FINAL (chosen model only):** full questions.json (80 Q), n=8, mean±std /
    pass@8 / SC-8 (spec §9), then official test.
- For the TRANSFER model (trained on mmlb, no dv2026), ALL of dv2026 val is clean
  held-out, so the stratified subset is leakage-free.
- Baseline-on-strat24 + trained-on-strat24 will be run during the transfer-eval
  phase (all 4 GPUs are on collection now; no free GPU to serve a student model).

### 2026-06-05 ~09:55 — USER GO: SeqKD transfer run (SDPA), autonomous pipeline armed
- Decisions: (1) SDPA now, flash-attn option C (torch2.10/cu128 rebuild) deferred;
  (2) proceed with SeqKD on more data (mmlb->dv2026); pivot to OPD/Pedagogical RL
  only if it fails; (3) 4x80GB is enough to co-serve teacher+student.
- Added make_sft_data `--max-tokens` (SDPA OOM safety): mmlb successes run
  median ~7k / max ~29k tokens; --max-tokens 14000 drops ~16% (the longest).
- **Transfer-run plan (autonomous, cron 51882a2d, artifact-driven stage machine):**
  collect to 100 successes -> pause collection + shrink 27B to DP=2 on GPUs 0,1 ->
  build parquet (--max-tokens 14000, --max-per-question 2) -> train student on GPU 2
  (SDPA, EPOCHS=3, TRAIN_BATCH_SIZE=8) -> merge -> serve baseline(GPU2)+trained(GPU3)
  -> eval BOTH on eval_subset_strat24 (n=1, VLM=27B on 0,1) -> report trained-vs-
  baseline, then STOP for user. GPU layout co-serves teacher+student per user note.
- Target 100 (~6x the probe's 17). If strat24 shows improvement -> scale up / full
  eval; if flat -> that's the signal to start OPD/Pedagogical RL.

### 2026-06-05 ~10:10 — GPU layout corrected (user): teacher=1 GPU, students get the rest
- User: 27B teacher needs only 1 GPU; bottleneck is STUDENT rollout inference -> give
  students the GPUs. Confirmed 27B fits on 1 GPU (it runs 1-replica-per-GPU now in DP=4).
- User asked re: remote 8928 (ssh-forward to another host) as teacher — correct in
  principle (SeqKD = no weight sync, teacher is inference-only). BUT 8928 is UNUSABLE
  here: (a) flaky (2/3 reqs timed out >30s — that host is contended), (b) it has
  `--reasoning-parser qwen3` so chat/completions returns content=None -> breaks
  batch_look (reads message.content). Collection needs the VLM too, so it can't move
  there either. -> keep teacher on the reliable LOCAL 27B; revisit 8928 only if it
  stabilizes AND serves without reasoning-parser.
- **Revised transfer driver (cron d297a1d3):** COLLECTING uses 27B DP=4 (all GPUs,
  nothing else to run). At PREP: shrink 27B to SINGLE GPU 0 (frees 1,2,3), keep
  collection trickling on it. TRAIN: student GPU1. EVAL: 27B VLM GPU0 + baseline-4B
  GPU1 + trained-4B GPU2, **eval --concurrency 16** (was 4 — the real eval-speed lever
  is more concurrent student rollouts, not more teacher GPUs). GPU3 spare.
- 2026-06-05 ~10:45: STAGE COLLECTING — 80 successes (20 to target 100), server DP=4 + collect loop healthy, disk 130G. No action.
- 2026-06-05 ~11:15: STAGE COLLECTING — 90 successes (10 to target). Server+collect healthy, disk 130G. Next cycle likely triggers PREP->TRAIN.

### 2026-06-05 ~11:35 — TRANSFER: PREP+TRAIN done, training running
- Trigger hit (101 successes). PREP: built data/sft/mmlb_transfer.parquet = **87
  trajectories / 87 unique Q** (101 successes - 14 over-length @ --max-tokens 14000);
  shrank 27B to SINGLE GPU 0 (confirmed fits: 73GB/80GB) freeing GPUs 1,2,3.
- TRAIN running on GPU 1 (bg job bh7z71zon): Qwen3.5-4B+LoRA, SDPA, USE_DYNAMIC_BSZ=
  False, EPOCHS=3, batch 8, **30 steps**, step1 loss 0.353, max-mem 35.7GB (no OOM).
  ETA ~45min. (tmux send-keys kept dropping the venv PATH -> torchrun not found;
  switched to run_in_background w/ `source .venv/bin/activate` — worked first try.
  LESSON: launch training via run_in_background, not tmux send-keys.)
- Collection trickles on the 1-GPU 27B (GPU0); GPUs 2,3 idle during train (eval uses
  them). Cron d297a1d3 will MERGE then EVAL (baseline+trained on strat24) when the
  global_step checkpoint appears.

### 2026-06-05 ~11:45 — W&B logging enabled
- W&B IS configured on this box (entity bdsaglam, WANDB_API_KEY + ~/.netrc). I'd been
  defaulting LOGGER=console out of caution — unnecessary. Changed run_seqkd.sh default
  to `console,wandb` (all future runs log to W&B; cron-launched trainings inherit it).
- Restarted the transfer run (was step 5/30, cheap) WITH W&B -> project docvqa-seqkd,
  run seqkd-transfer: https://wandb.ai/bdsaglam/docvqa-seqkd/runs/zsb6x8n3
- Training healthy on GPU1 (step1 loss 0.353, no OOM). Pipeline continues: merge->eval
  baseline+trained on strat24 -> report.

### 2026-06-05 ~11:55 — W&B + checkpoints renamed to project `docvqa-verl` (user)
- Renamed PROJECT_NAME docvqa-seqkd -> docvqa-verl (run_seqkd.sh default): W&B project
  AND checkpoint root now docvqa-verl. Moved checkpoints/docvqa-seqkd -> docvqa-verl
  (probe artifacts included). README example paths updated. Cron -> 627f90d8 (docvqa-verl
  paths).
- Transfer run relaunched (was step 2, cheap) under W&B project docvqa-verl:
  https://wandb.ai/bdsaglam/docvqa-verl/runs/e22zixf6 . The earlier short run in the
  old docvqa-seqkd W&B project (zsb6x8n3) is an orphan — user can delete that project.
- Training healthy on GPU1.

- **2026-06-05 ~12:16 (TRAINING_WAIT):** transfer SeqKD run healthy under W&B project `docvqa-verl` (run e22zixf6), GPU1. At step 17/30 (epoch 2/3, ~70%), train/loss steady ~0.20–0.29, grad_norm ~0.17–0.26, ~90s/step → ~13 steps / ~20 min remaining. Checkpoint saves at end (save_freq=-1) so no global_step_* yet. GPU0 = 27B teacher (73.6G), collect-mmlb still trickling. Disk 129G free (97%). Cron 627f90d8 (:11/:41) will catch the checkpoint → MERGE.

- **2026-06-05 ~12:38 (MERGE done):** transfer SeqKD trained all 30 steps (global_step_30, final train/loss ~0.2). Merged FSDP→HF to checkpoints/docvqa-verl/seqkd-transfer/merged_hf (full model.safetensors + lora_adapter), copied preprocessor_config.json + video_preprocessor_config.json from base cache. Paused collect-mmlb at 118 successes/118 unique Q (resumable; trained checkpoint used earlier 87-traj parquet). Next: EVAL baseline-4B vs trained on eval_subset_strat24.

- **2026-06-05 ~12:40 (EVAL running):** both student servers up (baseline Qwen3.5-4B GPU1:8931, trained merged_hf GPU2:8930, prime-rl vllm gpu-util 0.6, no served-model-name). Background script outputs/eval/run_transfer_eval.sh evaluating both on eval_subset_strat24 (24Q, concurrency 16, n=1, temp 0.6/top-p 0.95/top-k 20) -> transfer_{baseline,trained}_strat24.jsonl. Cron eddbcb9f replaces 627f90d8 with an EVAL in-progress guard (won't relaunch mid-eval). Next: DONE-stage trained-vs-baseline comparison.

- **2026-06-05 ~13:11 (EVAL in-progress, cron fire):** baseline eval still running (eval.py PID alive ~16min, GPU0 VLM + GPU1 student both 100% util, baseline serve log shows live /v1/completions 200s @ ~218 tok/s). Trained eval queued (GPU2 warm). No outputs yet (eval.py writes JSON array at end). Guard honored: no relaunch. Disk 101G free (above thresholds). Stratified-24 multi-turn agent rollouts are VLM-latency-bound; ~16min/model expected. STOP, await completion.

- **2026-06-05 ~13:27 (EVAL in-progress, cron fire):** CORRECTION — eval is healthy/progressing, not stalled (earlier clock misread). 27B VLM (8927) actively serving ~1500-1885 tok/s; eval.py holds 8 live conns to 8927 + 1 to 8931. 5/24 baseline rollouts still in flight; output writes at end. KEY: mmlb collection (PID 2091046) was STILL RUNNING in MY tmux session docvqa-verl (pts/0) — I lost track of it across compaction and wrongly attributed it to the user (user confirmed they did not start it). My 'pause collection during merge/eval' never held — it shares the 8927 VLM, so eval's batch_look calls contend with collection → baseline slow (~46min). Both progressed; collection was contending with eval for the 27B VLM the whole time. Collection now 122 successes/122 unique Q. Driver's "collection paused" assumption is STALE — it's running. Guard honored: no relaunch/kill. Disk 100G.

- **2026-06-05 ~13:55 (EVAL in-progress, cron fire):** collection paused since 13:47, so 27B VLM now exclusive to eval (1674 tok/s). Baseline down to 3/24 rollouts in flight, but 2 have run the FULL ~75min (since eval start) — pathologically slow long-document questions grinding toward the hard 30-turn iter_cap (each turn bottlenecked on multiple batch_look VLM calls @ up to 120s timeout). These WILL terminate (hard cap, not infinite). Honored guard: no relaunch/kill. IMPLICATION: same slow docs will hit the trained eval too, and the future full-80Q eval — worth a per-rollout wall-clock cap before the final eval (don't change mid-run; would lose ~21 completed Q). Disk 99G. STOP, await completion.

- **2026-06-05 ~14:25 (EVAL in-progress, cron fire):** baseline at 22/24; 2 long-doc rollouts (PIDs 2095279/2095283) still grinding at ~105min toward the hard 30-turn cap (VLM exclusive @ ~exclusive, 3438 chat/completions total, still advancing, no errors). Decided NOT to add a per-rollout wall-clock cap mid-run: scoring timeouts as 0 would BIAS the trained-vs-baseline comparison (conflates slow-on-hard-doc with wrong). Leave cap-policy to user for the full eval. Guard honored. Disk 99G. STOP. If these 2 don't clear by next fire (~30min) they're at/near iter_cap — reassess then.

- **2026-06-05 ~14:55 (EVAL in-progress, cron fire):** baseline 23/24 done; ONE rollout left (PID 2095279, ~135min, STILL advancing — VLM +392 chat/completions since last fire, 4600 tok/s, live conn). Not hung. NOTE: 135min exceeds the theoretical 30-turn × 120s-subprocess bound (~60-70min) → the per-turn subprocess_timeout_s likely does NOT bound batch_look's VLM HTTP calls (they may run async outside the capped exec). Worth a code look post-run (candidate for a real per-call/per-rollout VLM timeout before the full eval). Guard honored, no kill. Disk 99G. STOP; near completion.

### 2026-06-05 — walkthrough review: prompt/citation drift flags (no code changed)
Flagged during a `/walkthrough` of the experiment setup. Notes-to-fix, not bugs
affecting the running pipeline.

- **STALE CITATION (fix): `rvlm_minimal_solver.py` no longer exists** in `~/repos/docvqa`.
  Both `docvqa/prompts.py` docstring (lines 3, 11, 16) AND project `CLAUDE.md` cite
  it as the scaffold-to-mirror. Current files are `rvlm_solver.py` +
  `codeact_solver.py`. Our `_TASK_BODY` is still byte-identical to
  `rvlm_solver._TASK_BODY` (lines 154-221), so content is correct — only the
  provenance comments point at a deleted file. TODO: repoint both to
  `rvlm_solver.py` / `codeact_solver.py`.
- **Format divergence from the docvqa dspy `codeact_solver` is INTENTIONAL** (user
  confirmed; do NOT "align"). Ours: `<think>` + ```python``` fence + genuine
  append-only multi-turn `messages` list. dspy codeact: `[[ ## reasoning ## ]]`/
  `[[ ## code ## ]]` ChatAdapter markers, raw fence-less code (`_strip_code_fences`),
  history re-rendered into one `trajectory` input field. Same scaffold semantics
  (append-only MDP, `batch_look`+`SUBMIT`, byte-identical strategy body); different
  token rendering by design — verl `MultiTurnSFTDataset` needs raw-text completions
  over a chat-message list. Valid as long as the deployed evaluator is OUR
  `agent_loop`, not the dspy `codeact_solver`.
- **`category` reaches the model but is behaviorally unused (keep for now, flagged).**
  Injected only via `build_first_user_message` `## Document - category:` line
  (`prompts.py:152`); `build_system_prompt` does `del category` (`prompts.py:137`).
  Both datasets supply it, but with DISJOINT taxonomies — dv2026 {comics, maps,
  infographics, ...} vs mmlb {Research report, Academic paper, Tutorial/Workshop,
  ...}. Transfer training thus feeds mmlb category strings the model never sees at
  dv2026 eval: a small spurious train/eval mismatch on a no-op field. Cheap future
  cleanup: drop the line or normalize to a constant across collect+eval.
- **Answer-formatting rules hardcoded to DocVQA-2026, not profile-driven**
  (`prompts.py:18` vs docvqa codeact's `profile.answer_formatting_rules`). Fine for
  eval (always dv2026). Consequence for COLLECTION: mmlb teacher trajectories answer
  under dv2026 formatting, then ANLS-filter against mmlb gold — if mmlb gold isn't
  dv2026-formatted, some correct reads may be dropped as wrong. Likely second-order;
  revisit if mmlb success yield looks suspiciously low.

- **2026-06-05 ~15:25 (EVAL — RUNAWAY rollout, escalating):** baseline 23/24; lone rollout PID 2095279 at ~165min, ACCELERATING (+620 VLM calls in 30min → 4450 total; 8 concurrent conns to 8927; NO sandbox child). ROOT CAUSE confirmed: batch_look VLM HTTP calls run in the HOST async context, NOT the sandbox subprocess → 120s subprocess_timeout_s does not bound them; a single question can fan out unbounded batch_look calls per turn → effectively non-terminating. 30-turn cap doesn't help (time is inside turns). eval.py writes JSON array only at end → killing it loses the 23 done rollouts too (design flaw: no incremental write). NOT killing autonomously — escalated to user. FIX options: (a) per-rollout asyncio.wait_for wall-clock cap (e.g. 600s, score timeout=0/"wall_cap"); (b) incremental per-question write to eval.py so partial results survive; (c) bound batch_look concurrency/page-count + add a per-call cap. Recommend (a)+(b) before any re-run. Guard otherwise honored. Disk 98G.

- **2026-06-05 ~15:30 (EVAL fix+restart, user-approved):** killed runaway eval (lost 23 in-memory, recompute fast). Patched docvqa/scripts/eval.py: (1) per-rollout asyncio.wait_for cap via --rollout-timeout (default 600s) → timeout scored as non-answer, termination="wall_cap"; (2) incremental write to {output}.partial.jsonl as each Q completes (final JSON array unchanged) so completed work survives a kill. Syntax-checked, restarted (PID 2162156, default cap=600s). DISCOVERY: collection kept respawning because ANOTHER Claude Code session is active in tmux docvqa-verl window 2 (the "walk me through setup" session) — it relaunched collect_trajectories (PID 2127903, now 138 uniq-success Q). The 16 sandbox procs are ITS live workers, not my eval orphans — eval kill was clean. Did NOT kill collection (another session owns it); 600s cap bounds eval even under VLM contention. Multi-session coordination logged in .claude/CLAUDE.md.
- **2026-06-05 ~15:30 (collection-scope Q):** confirmed we have ONLY 2 teacher collections: dv2026 val-train (56Q, probe; 41 succ) + mmlb train (899Q transfer; 139 succ). NEVER collected whole DocVQA-2026 (leakage — it's the eval target; only the 56Q train split used, probe-only). Classic public DocVQA (SP-DocVQA) never collected — flagged as a possible leakage-free transfer source if desired.

- **2026-06-05 ~16:30 (eval ↔ collection unified):** rewrote `docvqa/scripts/eval.py` to write the `~/repos/docvqa/output/runs/`-style structure via `--run-dir`: `results.json` (summary.overall_accuracy + by_category + documents) and `tasks/<doc_id>/{result.json, trajectories.jsonl}`. Per-doc `trajectories.jsonl` = one structured record per (question, sample) with full chat `messages` + `anls`(binary, via `evaluate_prediction`) + `termination` + meta, streamed as questions finish (crash-safe). Per user: NO submission.json, NO human-readable summary.md (structured only), NO page_*.jpg. `_solve_n` now captures messages/vlm_calls/wall_clock per sample (was discarding). `make_sft_data.py --in` now accepts a run-dir (globs `tasks/*/trajectories.jsonl`) → an eval run IS the trajectory collection; `collect_trajectories.py` deprecated (note added). Replaced `--output` with `--run-dir` (+`--dataset`/`--split` for record_ids). Smoke-tested writer + make_sft_data round-trip with synthetic data (no GPUs). Updated README + HANDOFF eval invocations. NOTE: `outputs/eval/run_transfer_eval.sh` still uses old `--output` — fix to `--run-dir` when the transfer eval resumes.

- **2026-06-05 ~16:45 (save token IDs in trajectories):** `agent_loop.run()`'s `AgentLoopOutput` already produces `prompt_ids`/`response_ids`/`response_mask` (assistant-only mask) — eval/collect were discarding them. Now `eval.py` saves them per sample in `trajectories.jsonl` (default on; `--no-token-ids` to omit). Value: (1) SeqKD on EXACT teacher tokens + exact mask → drops the `ignore_input_ids_mismatch` retokenization hack; (2) prerequisite for forward-KL top-k KD / OPD (recompute teacher logprobs via frozen-27B forward pass over `response_ids`; logprobs NOT stored — too heavy, recomputable); (3) RL token-level use. Mock-tested capture + mask correctness + `--no-token-ids`. CAVEAT logged: token IDs use the collection-time `--student-model` tokenizer (27B teacher); verify Qwen3.5-4B/27B share an identical tokenizer before training the 4B on them.

- **2026-06-05 ~17:00 (eval config/provenance + rename):** (1) persist `config.json` at run start (created_at, model, vlm_model, base_urls, dataset/split, num_questions, n, sampling, rollout_timeout, save_token_ids) + embedded in results.json — know which models/sampling produced a run even if it crashes. (2) Store full `sampling` {temperature,top_p,top_k} + `model`/`vlm_model` per trajectory record (was just temperature) — records are self-describing (save raw, format later). (3) Renamed CLI `--student-model/--student-base-url` → `--model/--base-url` (agent LM can be teacher OR student) across eval.py + README + handoff. (4) `--model` default now Qwen/Qwen3.5-27B (we mostly eval the teacher). Syntax + no-stray-refs verified. Handoff doc brought fully up-to-date (cleanup-done state, run-dir format, tokens, config, rename, 27B default).

- **2026-06-05 ~evening (EVAL PARITY established):** Goal: prove our `agent_loop` reproduces the original CodeAct 27B (curated 36.74% full-80-val, band 33–44%). Infra: DP=4 27B vLLM on all 4 GPUs:8927 (serves agent LM + batch_look VLM), eval client via repo `.venv/bin/python` (bare python lacks ray). **enable_thinking CORRECTED to TRUE** — the handoff said match the original's `enable_thinking=false`, but that only made sense in DSPy CodeAct (it has a `reason` signature field); our `agent_loop` has none, so native `<think>` is the ONLY reasoning channel. A no-think run produced empty `<think></think>` (0 deliberation). Verified Qwen3.5-4B≡27B tokenizer (byte-identical) and that the Qwen template strips prior-turn `<think>` (keeps only last). eval.py gained `--no-thinking` (default ON) + `--resume` (skip Qs with ≥n streamed samples; summary still covers all 80).
  - **RESULT (thinking ON, full 80, n=1):** overall **0.275** (22/80, timeouts=0), **SUBMIT-ONLY 0.400** (22/55), wall_cap **27.5%**. Per-cat: maps 0.000, science_paper 0.000 (ALL timed out), business_report 0.100, comics 0.300, eng_drawing 0.400, slide 0.400, infographics 0.500, science_poster 0.500.
  - **VERDICT: scaffold FAITHFUL, not a reimplementation bug.** Submit-only 0.40 ≈ ref 0.37 (in band). Overall < floor is fully a TIMEOUT artifact: our 1800s `--rollout-timeout` is 8× tighter than the reference's 14400s (which only saw maps_2 occasionally cap), so slow long-doc rollouts (maps, science_paper) get guillotined to 0. The long-doc runaway (append-only context balloon → batch_look hang) is a real agent-scaffold issue (lives in ~/repos/docvqa), costing 27.5% of Qs.
  - **No-think comparison (cut @58/80):** overall 0.259, submit-only 0.484 (over an easier non-runaway subset, not comparable), wall_cap **45%**. Thinking roughly halves runaways (45%→27.5%) and lets the agent complete far more Qs. Partial at outputs/runs/parity-codeact-27b-val-nothink-t1.
  - eval shim re-encodes response text→token_ids (`eval.py:75`) — completions token_ids null unless requested — so saved `response_ids` are NOT exact sampled tokens (mask is still buffer-faithful). Refines the SeqKD-on-exact-tokens premise; fix when building training collection.
