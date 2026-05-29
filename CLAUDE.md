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
- **Goal:** push the state of the art in the **≤8B tier** of ICDAR 2026 DocVQA
  by fine-tuning the LM inside the existing Perceive-Reason-Code agent
  scaffold. Current ≤8B leaderboard best is 0.1875 ANLS; 8B-35B leader is
  0.3750 (held by the same agent scaffold at 27B, zero-shot).
- **Methods to explore** (rough priority order):
  1. **On-policy distillation (OPD)** — sample-efficient, dense per-token
     signal from a 27B teacher, less prone to catastrophic forgetting than
     RL-only. Start here.
  2. **SFT with rejection sampling** — generate many rollouts (student and/or
     teacher), keep trajectories above an ANLS threshold, SFT on those.
     Useful as a warmup before GRPO to avoid zero-reward-variance cold start.
  3. **GRPO and variants** (Dr. GRPO, GSPO, DAPO-style controls) on
     answer-level ANLS reward + tool-use shaping.
  4. **Combinations** — OPD + RL advantages summed in a single update.
- We do **not** fine-tune the VLM. It stays frozen as an external HTTP
  endpoint (Qwen3.5-27B at `localhost:8927` on the project box).

## Architecture

- **Agent scaffold (frozen design)**: same REPL agent as the
  `rvlm_minimal_solver` in `~/repos/docvqa` (the rename of what was
  `flat_solo_solver`). The LM works in a persistent Python REPL whose only
  visual tool is `batch_look(requests)` against the frozen VLM, and the only
  terminal action is `SUBMIT(answer=...)`. The earlier BM25/OCR `search`
  tool was deliberately removed when the scaffold was renamed — recursive
  VLM perception is the load-bearing mechanism. Solves a question across a
  *survey → locate → extract → verify → submit* loop. Training rollouts must
  replicate this scaffold so the policy we train is the policy we deploy.
- **What we fine-tune**: only the LM weights. LoRA adapters by default so
  experiments fit on a single A100/H100.
- **What stays external/frozen**: the VLM (HTTP endpoint).
- **Evaluation lives in `~/repos/docvqa`**: ANLS metric, eval harness, and
  dataset loaders. Trained checkpoints from this repo are evaluated there.

## Pointers to the docvqa repo (`~/repos/docvqa`)

For agent and evaluation details, refer to that repo rather than
reimplementing. Start here:

- `~/repos/docvqa/CLAUDE.md` — best results table, infra, key commands.
- `~/repos/docvqa/src/docvqa/solvers/rvlm_minimal_solver.py` — the
  category-agnostic agent scaffold to replicate during training rollouts
  (mirrored by our `docvqa/agent_loop.py`).
- `~/repos/docvqa/src/docvqa/solvers/rvlm_unified_solver.py` —
  category-tipped variant; shares `_create_tools` / `_build_sandbox_code` /
  `_build_signature` with `rvlm_minimal_solver`.
- `~/repos/docvqa/src/docvqa/solvers/direct_vlm_solver.py` and
  `direct_vlm_minimal_solver.py` — the DeepEyes-like VLM-only path
  (no REPL). Open option for fine-tuning if we ever switch from the
  LLM-as-agent path to the VLM-as-agent path.
- `~/repos/docvqa/src/docvqa/prompts.py` — answer formatting rules and
  per-category tips (the latter unused by `rvlm_minimal`).
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

## Running things

verl's environment setup (uv, hydra, pre-commit) is documented in
`AGENTS.md`. 
