# docvqa-verl — Training a small document-VQA agent

Fork of [`verl-project/verl`](https://github.com/verl-project/verl) used as the
**training side** of a COGS 560 term project on the **ICDAR 2026 DocVQA**
benchmark. The goal is to push the **≤8B tier** of the leaderboard by fine-tuning
a small reasoner (Qwen3.5-4B) *inside* a code-executing, active-perception agent
scaffold — not by training a monolithic VLM.

> The agent scaffold and the evaluation harness live in a **separate repo**
> (`~/repos/docvqa`). This repo holds the training pipelines (SFT / GRPO /
> distillation), reward + data prep, and the project write-up. Upstream verl's
> own README is preserved at [`README-verl.md`](README-verl.md), and its
> contribution guide at [`AGENTS.md`](AGENTS.md).

## What this project is

The DocVQA-2026 documents are long, multi-page, high-resolution, and largely
*visual* — answers live in chart cells, map labels, and engineering-drawing
annotations that OCR cannot reach. The central finding (from the evaluation repo)
is that **active perception is the dominant lever**: give a code-capable reasoner
a persistent Python REPL plus one visual primitive — an on-demand call to a
frozen VLM against an arbitrary image region — and it can crop, zoom, composite,
and compute over what it sees. Holding the reasoner and the frozen VLM fixed and
varying only the scaffold spans ~25 ANLS points.

This repo asks the follow-on question: **can we train a small (≤8B) reasoner to
be a better agent inside that scaffold?** We fine-tune only the LM (LoRA); the
VLM stays frozen as an external HTTP endpoint (Qwen3.5-27B).

### Headline results (DocVQA-2026 val, binary ANLS@0.9)

- **Base Qwen3.5-4B in the scaffold:** 22.34% ± 3.44 (n=8) — the baseline to beat.
- **SFT on rejection-sampled 27B-teacher trajectories (LoRA r64):** 28.7%
  (n=1, a lower bound vs the n=8 base) — **SFT helps** (+6.4pp). See
  [`results/sft-sweep-2026-06-14.md`](results/sft-sweep-2026-06-14.md).
- **GRPO / on-policy distillation:** designed and run preliminarily; see the
  experiment notes below. Reported lightly and honestly in the write-up.

The remaining headroom is structural: on long documents the append-only
trajectory grows until the agent runs out of turn budget before submitting
(iter-cap), and this scales with page count.

## Architecture (one paragraph)

The fine-tuning target is the **CodeAct** scaffold: a strictly **append-only**,
multi-turn REPL agent ([`docvqa/agent_loop.py`](docvqa/agent_loop.py),
`DocVQAReplAgentLoop`). Each turn the LM emits free-text reasoning plus a single
fenced `python` block; native thinking is disabled, so only the fenced code is
parsed and executed. Code runs in a persistent CPython subprocess whose only
visual tool is `batch_look(requests)` (against the frozen VLM) and whose only
terminal action is `SUBMIT(answer=...)`. Captured stdout is appended verbatim as
the next turn — a fully-observable, growing-prefix MDP, which is exactly the
structure SFT / distillation / RL losses assume. The same `agent_loop` is used
for **both** trajectory collection and evaluation, so the policy we train is the
policy we deploy. Full design rationale is in [`CLAUDE.md`](CLAUDE.md).

## Repository map

### Project (proposal, experiments, report)

| Path | What |
|---|---|
| [`project/proposal.md`](project/proposal.md) | Original COGS 560 proposal — problem, method, plan. |
| [`project/report/`](project/report/) | The write-up. `acl_latex.pdf` (compiled), `tex/` (per-section LaTeX), `sections/` (Markdown source), `references.bib`, `figs/`. |
| [`project/report/brief.md`](project/report/brief.md) · `throughline.md` · `outline.md` | The report's framing and structure ladder. |
| [`project/experiments/`](project/experiments/) | Experiment notes: `SFT-report.md`, `RL-OPD-design.md`, `RL-async-findings.md`. |
| [`project/datasets.md`](project/datasets.md) · [`candidate-datasets.md`](project/candidate-datasets.md) | DocVQA-family dataset survey. |
| [`results/`](results/) | Result cards. `sft-sweep-2026-06-14.md` is the current SFT verdict. |

### Agent + reward (the trainable scaffold)

| Path | What |
|---|---|
| [`docvqa/agent_loop.py`](docvqa/agent_loop.py) | `DocVQAReplAgentLoop` — the append-only CodeAct agent (training + eval). |
| [`docvqa/reward.py`](docvqa/reward.py) | ANLS reward (multi-alias gold → max-ANLS). |
| [`docvqa/prompts.py`](docvqa/prompts.py) · [`parser.py`](docvqa/parser.py) | Prompt format and turn/fence parsing. |
| [`docvqa/sandbox.py`](docvqa/sandbox.py) · [`subprocess_interp.py`](docvqa/subprocess_interp.py) · [`tools.py`](docvqa/tools.py) | Persistent REPL, `batch_look` VLM tool, `SUBMIT`. |
| [`docvqa/metrics.py`](docvqa/metrics.py) · [`eval_metrics.py`](docvqa/eval_metrics.py) | ANLS implementation and eval aggregation. |
| [`docvqa/tests/`](docvqa/tests/) | Unit tests for the above. |

### Data preparation

| Path | What |
|---|---|
| [`docvqa/scripts/prepare_data.py`](docvqa/scripts/prepare_data.py) | DocVQA-family dataset adapters → `data/<ds>/<split>/{questions.json,docs/}`. |
| [`docvqa/train/pool.yaml`](docvqa/train/pool.yaml) | Manifest of the training-prompt pool (7 sources). |
| [`docvqa/scripts/sample_pool.py`](docvqa/scripts/sample_pool.py) | Balanced sampling across sources → `data/pool/prompts.json`. |
| [`docvqa/scripts/make_sft_data.py`](docvqa/scripts/make_sft_data.py) · [`make_clean_sft.py`](docvqa/scripts/make_clean_sft.py) | Rejection-sample teacher trajectories (ANLS==1.0) → SFT parquet. |
| [`docvqa/scripts/make_rl_parquet.py`](docvqa/scripts/make_rl_parquet.py) · [`make_curriculum_parquet.py`](docvqa/scripts/make_curriculum_parquet.py) | Build RL prompt parquets (incl. easy→hard curriculum). |
| [`docvqa/scripts/check_leakage.py`](docvqa/scripts/check_leakage.py) · [`find_pool_leakage.py`](docvqa/scripts/find_pool_leakage.py) | Guard against val leakage in the prompt pool. |

> `data/` is gitignored (large, regenerable, absolute paths). Regenerate via the
> scripts above.

**Training-prompt pool.** DocVQA-2026 ships no training split, so training
prompts are drawn from public DocVQA-family datasets (no val leakage). The pool
is defined in [`docvqa/train/pool.yaml`](docvqa/train/pool.yaml) and sampled
balanced-by-source by `sample_pool.py`. Each source is mapped to one of the
DocVQA-2026 categories so the agent's per-category prompting applies; 5 of the 8
categories are covered (comics, engineering drawings, and science papers have no
suitable short-doc source).

| Source | Split | Mapped category | Tier | Sampled |
|---|---|---|---|---|
| docvqa-sp | validation | business_report | easy | 1500 |
| infographicvqa | validation | infographics | hard | 1500 |
| chartqa | train | science_poster | easy | 1500 |
| mapqa | train | maps | easy | 1500 (capped from ~483k) |
| mp-docvqa | val | business_report | mid | 1500 |
| tatdqa | dev | business_report | mid | 1500 |
| slidevqa | train | slide | mid | 1500 |
| mmlongbench-doc | train | mixed (long, multi-page) | hard | 1000 |

DUDE has an adapter but is excluded (heavy downloads). Teacher trajectories for
SFT are then **rejection-sampled** from these prompts: the 27B agent rolls out,
and only trajectories scoring ANLS==1.0 are kept and turned into SFT data.

### Training recipes

| Path | What |
|---|---|
| [`docvqa/train/run_seqkd.sh`](docvqa/train/run_seqkd.sh) | SFT / SeqKD launcher (LoRA, FSDP). `run_seqkd.sh <data> <exp> <nproc>`. |
| [`docvqa/train/run_grpo.sh`](docvqa/train/run_grpo.sh) | GRPO launcher (`one_step_off_policy`, disaggregated rollout+train + frozen VLM). |
| [`docvqa/train/collect_pool.sh`](docvqa/train/collect_pool.sh) · [`build_pool_parquet.sh`](docvqa/train/build_pool_parquet.sh) | Teacher-trajectory collection and parquet build drivers. |
| [`docvqa/train/README.md`](docvqa/train/README.md) | Recipe-level notes. |

> Project recipes live in `docvqa/train/` (not `recipe/`, which is an
> uninitialized submodule that conflicts on upstream rebases).

### Evaluation

Evaluation is run with [`docvqa/scripts/eval.py`](docvqa/scripts/eval.py) (the
client) against a served checkpoint + the frozen VLM. The `outputs/_eval_*.sh`
helpers wrap common layouts (merge LoRA → serve agent → eval a question file):

- `outputs/_eval_pool.sh` / `_eval_pool22.sh` — eval with a pooled (local +
  remote) VLM endpoint.
- `outputs/_eval_rank.sh` — fast ranking screen on the 13-question subset.
- `outputs/_eval_fullval.sh` — full 80-question validation.
- `outputs/_eval_rl.sh` — merge + serve + eval an RL checkpoint.
- [`outputs/_fold_lora.py`](outputs/_fold_lora.py) — **fold a LoRA adapter into
  base weights** (PEFT `merge_and_unload` + assert merged≠base). `verl.model_merger`
  does *not* fold the adapter — always fold and weight-diff before trusting a
  fine-tuned eval. See the eval-methodology section in `CLAUDE.md`.

## Setup & running

For installing verl itself (engine dependencies, supported backends, hardware
notes), see the upstream verl README preserved at
[`README-verl.md`](README-verl.md); the dev workflow (uv, hydra, pre-commit) is
in [`AGENTS.md`](AGENTS.md). GPU-layout defaults per use case (SFT vs eval vs RL on
the 4×80GB box), the VLM-perception throughput ceiling, RL-training precision
gotchas, and the evaluation methodology (submit-only ANLS × cap-rate) are all in
[`CLAUDE.md`](CLAUDE.md) — **read it before comparing eval numbers or starting an
RL run.**

## Relationship to upstream verl

- Upstream is `verl-project/verl` (git remote `upstream`); rebase project
  branches on upstream `main` rather than merging.
- This is a research fork — the training recipes, reward, agent loop, and data
  prep under `docvqa/` are the project-specific additions on top of verl's RL
  infrastructure.
