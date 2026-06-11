# Training data pool

A **method-agnostic** pool of short-document DocVQA-family data for training the
≤8B agent (SFT, OPD, or GRPO/RL). Short docs = feasible rollouts; gold answers +
a verifier = usable as RL reward or rejection-sampling filter. Built entirely in
this repo (`docvqa/scripts/`), no `~/repos/docvqa` changes.

Design rationale / decisions: `docs/superpowers/specs/2026-06-10-training-data-pool-design.md`.

## What's in it (7 sources)

Each source is materialized to `data/<dataset>/<split>/`:
- `questions.json` — every question as a verl-ready row (schema below)
- `docs/<doc_id>/pages/page_*.png` + `metadata.json` — the page images the agent's `batch_look` reads

| dataset | split | docs | questions | category | notes |
|---|---|---|---|---|---|
| docvqa-sp | validation | 1,286 | 5,349 | business_report | scanned industry docs |
| infographicvqa | validation | 500 | 2,801 | infographics | high-res single image |
| chartqa | train | 28,299 | 28,299 | science_poster | numeric; relaxed-acc reward |
| mapqa | train | 37,417 | 483,416 | maps | numeric; ~13 Q/image |
| mp-docvqa | val | 583 | 2,986 | business_report | filtered to ≤3 pages |
| tatdqa | dev | 275 | 1,644 | business_report | financial tables; numeric |
| slidevqa | train | 5,962 | 9,381 | slide | single-evidence-slide only |

Categories are the nearest of DocVQA-2026's 8 (drives the agent's per-category
prompting). **comics / engineering_drawing / science_paper** have no suitable
short-doc source. **DUDE is excluded** (adapter exists but heavy/risky).

## The balanced pool

`sample_pool.py` draws `n_sample` rows/source (set in `docvqa/train/pool.yaml`)
into a balanced set so no single source dominates (MapQA's 483k would otherwise
swamp everything):

```
data/pool/prompts.json         # combined, shuffled (default 7×1500 = 10,500)
data/pool/sampled/<name>.json  # per-source sample (used by collect_pool.sh)
```

Rebuild / reweight: edit `n_sample` per source in `pool.yaml`, then
`.venv/bin/python docvqa/scripts/sample_pool.py --seed 42`.

> Balanced **by dataset**. By category, `business_report` = 3× (three sources map
> there). Adjust `n_sample` if you want category balance.

## Row schema (what a trainer consumes)

Each row in `questions.json` / `prompts.json` (from `prepare_data.py:_build_row`):

```json
{
  "record_id": "<dataset>:<split>:<doc_id>:<question_id>",
  "dataset": "...", "split": "...", "doc_id": "...", "question_id": "...",
  "question": "...", "answer": "...",        // multi-alias gold stored as repr([...])
  "category": "maps", "doc_dir": "/abs/path/to/docs/<doc_id>",
  "prompt": [{"role": "user", "content": "<question>"}],   // verl prompt-len filter only
  "data_source": "<dataset>-<split>",
  "reward_model": {"style": "rule", "ground_truth": "<answer>"},
  "extra_info": {"record_id","dataset","split","doc_id","question_id","category"}
}
```

`doc_dir` is **absolute**. The agent loop builds the real prompt from
`question`/`category`/`doc_dir` (not from `prompt`).

## How to train with it

Use `.venv` for prepare/sample; RL training uses its own env (`.venv-rl2`, see the
session registry). If `data/` is missing (it's gitignored — regenerable, holds
absolute paths), rebuild first: run `prepare_data.py --dataset <name> --splits <split>`
for each row in the table above (mind the split names), then `sample_pool.py`.

### GRPO / RL (no teacher needed)
The prompt set *is* the training input — each row carries the gold answer + a
rule verifier. Convert to a verl RL parquet:

```bash
# per source (clean: docs-dir matches that source)
.venv/bin/python docvqa/scripts/make_rl_parquet.py \
  --questions data/pool/sampled/chartqa.json \
  --docs-dir  data/chartqa/train/docs \
  --out       data/pool/rl/chartqa.parquet --require-answer
```
Then point `docvqa/train/run_grpo.sh` at the parquet(s). Reward at training time =
`docvqa/reward.py:compute_score` (continuous ANLS, multi-alias-aware; target metric
is binary ANLS@0.9). **Note:** the combined `prompts.json` spans 7 docs dirs;
`make_rl_parquet.py` takes one `--docs-dir`, so either build per-source parquets and
concatenate, or confirm the converter honors each row's absolute `doc_dir` before
passing the combined file.

### SFT / OPD warmup (needs the VLM)
Collect 27B-teacher trajectories, keep the solved ones, train:

```bash
./docvqa/train/collect_pool.sh chartqa          # 27B rollouts over the sample -> outputs/runs/pool-collect-chartqa
./docvqa/train/build_pool_parquet.sh chartqa ... # make_clean_sft (anls==1.0 & submit) -> data/sft/pool_combined.parquet
./docvqa/train/run_seqkd.sh data/sft/pool_combined.parquet <exp_name>
```
`collect_pool.sh` points at the VLM endpoint — update the `--base-url/--vlm-base-url`
in it to the current 27B (the local `:8927` server may be down; the VLM may be a
remote tunnel — check the session registry).

## Gotchas
- **Splits differ per source** — use the table; `prepare_data.py`'s `main()` default (`val,test`) is wrong for most.
- **MapQA images** are HF `{bytes,path}` dicts (decoded via `_coerce_image`); other sources are PIL.
- **Reward + multi-alias:** golds with several aliases are `repr([...])`; `compute_score` parses them (`get_anls` alone does not — don't bypass it).
- **Not yet done:** val-leakage guard (drop pool docs whose images collide with DocVQA-2026 val/test) — add before using infographicvqa/mp-docvqa for anything scored on DocVQA-2026.
- **Reproducible:** `sample_pool.py --seed 42` is deterministic per source.
