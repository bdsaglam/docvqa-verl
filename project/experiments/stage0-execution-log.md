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
