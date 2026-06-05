# docvqa-verl

Fork of `verl-project/verl` for fine-tuning a small (≤8B) language model on the
**ICDAR 2026 DocVQA** benchmark. This repo is the **training side** of the
project; the **agent scaffold and evaluation** live in `~/repos/docvqa`.

> Upstream verl's contribution-policy / agent guidance has been preserved at
> `AGENTS.md` (read it before opening a PR back to `verl-project/verl`).
> The original repo had `CLAUDE.md → AGENTS.md` as a symlink; we replaced
> `CLAUDE.md` with this project-specific orientation and left `AGENTS.md`
> intact.

## Project context

- Term project for COGS 560 (PhD course). Full proposal: `project/proposal.md`.
- **Goal:** push the state of the art in the **≤9B tier** of ICDAR 2026 DocVQA
  by fine-tuning the LM inside the existing Perceive-Reason-Code agent
  scaffold. Current ≤8B leaderboard best is 0.1875 ANLS; 8B-35B leader is
  0.3750 (held by the same agent scaffold at 27B, zero-shot).
- **Methods to explore** (rough priority order):
  1. **SFT with rejection sampling** — generate many rollouts (student and/or
     teacher), keep trajectories above an ANLS threshold, SFT on those.
     Useful as a warmup before GRPO to avoid zero-reward-variance cold start.
  2. **On-policy distillation (OPD)** — sample-efficient, dense per-token
     signal from a 27B teacher, less prone to catastrophic forgetting than
     RL-only. Start here.
  3. **GRPO and variants** (Dr. GRPO, GSPO, CISPO, DAPO-style controls) on
     answer-level ANLS reward + tool-use shaping.
  4. **Off-policy distillation** - Off-policy distillation with various recipes such as SeqKD and Pedagogical RL
  5. **Combinations** — OPD + RL advantages summed in a single update.
- We do **not** fine-tune the VLM. It stays frozen as an external HTTP
  endpoint (Qwen3.5-27B at `localhost:8927` on the project box).

## Architecture

- **Agent scaffold = CodeAct (fixed design)**: a strictly **append-only**,
  multi-turn REPL agent, implemented for this repo in `docvqa/agent_loop.py`
  (`DocVQAReplAgentLoop`). Each turn the LM emits `<think>...</think>` + a single
  ```` ```python ```` fenced block; the code runs in a persistent CPython
  subprocess whose only visual tool is `batch_look(requests)` against the frozen
  VLM, and whose only terminal action is `SUBMIT(answer=...)`. Captured stdout is
  appended verbatim as the next turn → a fully-observable MDP (the property SFT /
  distillation / RL losses assume). It mirrors the docvqa repo's
  `codeact_solver.py` — the append-only twin of the `rvlm_solver.py` LeanRLM
  solver (which uses a hidden REPL namespace + `RESET_HISTORY`, a POMDP, and is
  NOT a fine-tuning target). Tool surface is just `batch_look` + `SUBMIT` (an
  older BM25/OCR `search` tool was dropped — recursive VLM perception is the
  load-bearing mechanism). Solves via a *survey → locate → extract → verify →
  submit* loop. In this repo BOTH trajectory collection and eval run this same
  `agent_loop`, so the policy we train == the policy we deploy. The `<think>` +
  fence text format is **intentionally different** from the docvqa dspy
  `codeact_solver`'s `[[ ## reasoning ## ]]` ChatAdapter markers — verl SFT
  trains on raw-text completions over a chat-message list.
- **What we fine-tune**: only the LM weights. LoRA adapters by default so
  experiments fit on a single A100/H100.
- **What stays external/frozen**: the VLM (HTTP endpoint).
- **Evaluation lives in `~/repos/docvqa`**: ANLS metric, eval harness, and
  dataset loaders. Trained checkpoints from this repo are evaluated there.

## Pointers to the docvqa repo (`~/repos/docvqa`)

For agent and evaluation details, refer to that repo rather than
reimplementing. Start here:

- `~/repos/docvqa/CLAUDE.md` — best results table, infra, key commands.
- `~/repos/docvqa/src/docvqa/solvers/codeact_solver.py` — the **CodeAct
  append-only scaffold** our `docvqa/agent_loop.py` replicates during training
  rollouts (append-only MDP; the fine-tuning target).
- `~/repos/docvqa/src/docvqa/solvers/rvlm_solver.py` — the LeanRLM `rvlm`
  solver. `codeact_solver.py` imports its `_TASK_BODY` / `_build_task_instructions`
  / `_create_tools` / `_build_sandbox_code` (our prompt body is vendored from
  here). The LeanRLM loop itself (hidden namespace + `RESET_HISTORY`, a POMDP) is
  NOT a fine-tuning target.
- `~/repos/docvqa/src/docvqa/solvers/direct_vlm_solver.py` — the DeepEyes-like
  VLM-only path (no REPL). Open option if we ever switch from the LLM-as-agent
  path to the VLM-as-agent path.
- `~/repos/docvqa/src/docvqa/prompts.py` — answer formatting rules and
  per-category tips (the tips unused by the CodeAct / `rvlm` task body).
- `~/repos/docvqa/src/docvqa/datasets/profile.py` — `DatasetProfile`
  (answer-formatting-rules + score_fn) per dataset.
- `~/repos/docvqa/src/docvqa/data.py` — dataset loading.
- `~/repos/docvqa/src/docvqa/runner.py` — concurrent, resumable eval runner.
- `~/repos/docvqa/src/docvqa/metrics.py` — ANLS implementation (the reward).
- `~/repos/docvqa/evals.py` — Hydra entry point used after every training run.
- `~/repos/docvqa/docs/` — dataset, results, solver, and experiment-history
  docs. `docs/dataset.md` and `docs/solvers/*.md` are the most useful.

## Working with verl

- Upstream is `verl-project/verl` (configured as `git remote upstream`).
  Sync periodically; rebase project branches on upstream `main` rather than
  merging.
- `recipe/` in upstream verl has reference training recipes; new recipes
  for this project go in `recipe/docvqa/` (created on first use).
- Other reference repos already cloned locally for cross-pollination:
  - `~/repos/verl` — clean upstream clone for diffing / cherry-picking.
  - `~/repos/prime-rl` — alternative RL training stack; useful for recipe
    inspiration and comparison.

## Scope

**Do here:** training pipelines, reward design, OPD/RL/SFT recipes, data
prep for training prompts (DocVQA-family datasets), checkpoint management.

**Don't do here:** the agent scaffold itself (lives in `~/repos/docvqa`),
evaluation runs (also there), one-off prompting tweaks unrelated to
training.

## RL-training practices (read before the RL stages)

LLM RL/distillation has a set of non-obvious practice gotchas that don't appear
in the algorithm descriptions and cause *silent* failures (training runs, looks
stable, learns the wrong thing). **Consult the RL-training-practices guide
before the forward-KL-KD / GRPO rungs and read it in full** — don't rely on this
summary. Topics it covers, all load-bearing for this project:

- **Trainer/inference precision matching** — the dominant silent RL failure;
  a precision gap zeros ~18% of token gradients via phantom clipping. FP16
  everywhere preferred; fp32 LM-head logits; batch-invariant kernels for the
  residual kernel-level mismatch.
- **TITO (token-in-token-out) for agentic RL** — our scaffold is exactly the
  multi-turn tool-use case this is about: never re-encode decoded tokens, or the
  trainer's reconstructed IDs diverge from what was sampled and corrupt the
  importance ratio. Directly relevant to how we build training sequences.
- **Which metrics to watch** — clip fraction, advantage mean/std, generation
  length, KL, prefix-break rate, token-level |δ|.
- **Loss aggregation vs length dynamics; batch size for RL gradient noise;
  off-policy staleness budget; advantage whitening; the silent-bug checklist**
  (prompt tokens in loss, reward scattered to padding, stale log-probs).

Current stage caveat: **SeqKD = SFT is immune** to the precision-mismatch
failure (no clipped surrogate, no closed-loop data), so the existing
`fp32 master + bf16 autocast + bf16 inference` runs are correct as-is. The guide
matters once we add a clipped surrogate + rollouts.

Public sources behind the guide: "Defeating the trainer-generator precision
mismatch in TRL" (Diroh/HF 2026); the TITO write-up (Gallouédec & Rasul,
`github.com/PrimeIntellect-ai/renderers`); ScaleRL (Khatri 2025); TIM/VeXact
(Zhong 2026); RLHF Book Lecture 4 (Lambert).

## Running things

verl's environment setup (uv, hydra, pre-commit) is documented in
`AGENTS.md`. 
