# DocVQA-verl: agent scaffold + GRPO training design

**Date:** 2026-04-30
**Author:** Barış (with Claude)
**Status:** spec, awaiting implementation

## 1. Summary

Build a verl-native RL training stack for fine-tuning Qwen3-8B on the
ICDAR 2026 DocVQA benchmark, replicating the deployed `flat_solo` REPL
agent scaffold so that *the policy we train is the policy we deploy*.

The agent operates in a persistent CPython REPL with three tools
(`batch_look`, `search`, `SUBMIT`), reasons inside Qwen3 `<think>`
blocks, emits a single `python` code fence per turn, and terminates by
calling `SUBMIT(answer="...")`. Reward is end-of-trajectory ANLS.

This spec covers the **scaffold** (custom verl `AgentLoopBase`,
subprocess interpreter, tool plumbing, prompts, dataset layer,
rollout dumps) and the **Phase-1 training setup** (GRPO with ANLS
reward only). SDPO-style peer-PI distillation and the external 27B
teacher are explicitly out of scope for this spec — they are future
phases, listed but not designed here.

## 2. Goals and non-goals

### Goals

- Produce a verl AgentLoop that reproduces `flat_solo`'s behaviour
  closely enough that zero-shot Qwen3-8B in the new scaffold scores
  within ±5 ANLS-pp of `flat_solo` on DocVQA-2026 val.
- Make the dataset row contract trivially simple: `{question_id,
  question, doc_dir, gold_answer, category}`. Everything per-document
  lives under `doc_dir/`.
- Dump full per-trajectory rollouts to disk every training step via
  verl's built-in `trainer.rollout_data_dir` + `reward_extra_info`
  hook.
- Be ready to run a small GRPO sanity training run (Phase 1) once the
  scaffold passes its validation gates.

### Non-goals (this spec)

- SDPO-style peer-PI self-distillation loss (Phase 2).
- External 27B teacher OPD (Phase 3).
- Rejection-sampled SFT warmup (only if Phase 2 cold-starts poorly).
- Curriculum across DocVQA-family datasets (Phase 3+).
- Training the VLM. The 27B VLM at `localhost:8928` stays frozen.
- Modifying verl's trainer, AgentLoopWorker, AgentLoopManager,
  RewardManager, or the dataset class. Everything below sits in
  `recipe/docvqa/`.

## 3. Architecture

### 3.1 Component map

```
┌─────────────── verl AgentLoopWorker (Ray actor) ──────────────┐
│  DocVQAReplAgentLoop.run()                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Turn loop (max_iterations):                            │  │
│  │    1. server_manager.generate(prompt_ids,...) → resp    │  │
│  │       (verl student vLLM, sampled on-policy)            │  │
│  │    2. parse <think>...</think> + ```python ...```       │  │
│  │    3. interp.execute(code) → stdout / FinalOutput / err │  │
│  │    4. tokenize observation as `user` role diff,         │  │
│  │       append to running prompt_ids/response_ids,        │  │
│  │       mask=0 for observation, mask=1 for assistant      │  │
│  │  Exits on SUBMIT, iter cap, token cap, or parse error   │  │
│  └─────────────────────────────────────────────────────────┘  │
│              │ stdin/stdout JSON IPC                          │
│              ▼                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Persistent Python subprocess (SubprocessInterpreter)   │  │
│  │  Globals: pages, page_texts, search, batch_look, SUBMIT │  │
│  │  search → BM25 index in doc_dir/bm25/                   │  │
│  │  batch_look → IPC back to host → HTTP to localhost:8928 │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘

Final output: AgentLoopOutput(prompt_ids, response_ids, response_mask,
                              reward_score=None,
                              extra_fields={messages, num_turns,
                                            termination, submitted_answer,
                                            vlm_calls, search_calls,
                                            wall_clock_s, doc_id, ...})
```

### 3.2 Files

All new code lives under `recipe/docvqa/`. Tests under
`tests/recipe/docvqa/`. Example dataset prep under
`recipe/docvqa/scripts/`.

| Path | Purpose |
|---|---|
| `recipe/docvqa/agent_loop.py` | `DocVQAReplAgentLoop(AgentLoopBase)`. The turn-loop, prompt construction, response_mask bookkeeping. |
| `recipe/docvqa/subprocess_interp.py` | Vendored from `~/repos/docvqa/src/docvqa/rlm/subprocess_interpreter.py`, stripped of DSPy + `display` + `RESET_HISTORY` + `dspy_lm` config. |
| `recipe/docvqa/sandbox.py` | Inline Python startup snippet for the subprocess: reads `DOC_DIR`, populates `pages`/`page_texts`, defines `SUBMIT` and the IPC tool proxies. |
| `recipe/docvqa/tools.py` | Host-side handlers: `batch_look` (HTTP to localhost:8928 VLM) and `search` (BM25 over `doc_dir/bm25/`). Async. |
| `recipe/docvqa/prompts.py` | System prompt template + first-user-message template + per-turn observation message template. Lifts `ANSWER_FORMATTING_RULES` and `get_category_tips` from the docvqa repo. |
| `recipe/docvqa/dataset.py` | Builds verl-compatible dataset rows from `data/{split}/questions.json` + `data/{split}/docs/{doc_id}/`. |
| `recipe/docvqa/reward.py` | ANLS reward function. Computes ANLS, returns `(score, extra_info)` so the agent loop's `extra_fields` flow into the rollout dump. |
| `recipe/docvqa/agent.yaml` | One-line Hydra registration: `_target_: recipe.docvqa.agent_loop.DocVQAReplAgentLoop` under name `docvqa_repl`. |
| `recipe/docvqa/scripts/prepare_data.py` | Materializes per-document working directories from source datasets (DocVQA-2026 first; DocVQA / MP-DocVQA / etc. behind feature flags for Phase 1). |
| `recipe/docvqa/scripts/eval.py` | Layer-3 evaluation runner — instantiates the agent loop without verl training, runs over a split, computes ANLS. |
| `recipe/docvqa/scripts/run_smoke_grpo.sh` | Layer-4 smoke-training launcher. |
| `tests/recipe/docvqa/test_subprocess_interp.py` | Layer-1 unit tests. |
| `tests/recipe/docvqa/test_parser.py` | Layer-1 unit tests for code-fence extraction. |
| `tests/recipe/docvqa/test_agent_loop.py` | Layer-2 integration test with a scripted `server_manager`. |
| `tests/recipe/docvqa/fixtures/` | Canned trajectory fixtures for Layer-2. |

## 4. Per-turn protocol

### 4.1 Initial setup

1. Worker receives a dataset row: `{question_id, question, doc_dir,
   gold_answer, category}`.
2. Build BM25 retriever for `doc_dir` (host-side; cached in
   `_BM25_CACHE` per Ray actor across rollouts of the same doc).
3. Spawn the subprocess. Send config containing the sandbox startup
   code; pass `doc_dir` via the `DOC_DIR` env var.
4. Subprocess startup loads `pages` (lazy `PIL.Image.open`),
   `page_texts` (`page_*.md` read into a list of strings),
   defines `SUBMIT(answer=...)` and the IPC proxies for
   `batch_look(requests)` and `search(query, k)`.
5. Render the first prompt — system message + first user message
   (Section 5). Tokenize with
   `apply_chat_template(messages, add_generation_prompt=True,
   enable_thinking=True)`. This becomes `prompt_ids`. (`response_ids`
   starts empty; the running rollout buffer is `prompt_ids ++
   response_ids`.)

### 4.2 Turn loop

```python
for turn in range(1, max_iterations + 1):
    # A. Sample one assistant turn (think + code, ends at <|im_end|>)
    assistant_ids = await server_manager.generate(
        prompt_ids + response_ids,
        sampling_params=SamplingParams(
            temperature=1.0, top_p=0.95,
            max_tokens=max_response_tokens_per_turn,
            stop=["<|im_end|>"]),
    )
    response_ids  += assistant_ids
    response_mask += [1] * len(assistant_ids)

    # B. Parse the model's output
    text = tokenizer.decode(assistant_ids, skip_special_tokens=False)
    code = parse_last_python_fence(text)
    if code is None:
        observation = "[Error] No `python` code block found. Write a single ```python ... ``` block."

    else:
        # C. Execute in subprocess (state persists across turns)
        result = await loop.run_in_executor(None, interp.execute, code)
        if isinstance(result, FinalOutput):
            submitted_answer = result.data.get("answer")
            observation = f"FINAL: {submitted_answer!r}"
            _append_observation(observation, turn, max_iterations)
            termination = "submit"
            break
        elif isinstance(result, str) and result.startswith("[Error]"):
            observation = result
        elif isinstance(result, list):
            observation = "\n".join(map(str, result))
        else:
            observation = str(result) if result else "(no output - did you forget to print?)"
        observation = _truncate(observation, max_obs_chars=8000)

    # D. Append the observation as a `user` role diff (mask=0)
    _append_observation(observation, turn, max_iterations)

    # Token-budget guard
    if len(response_ids) > max_response_length - safety_margin:
        termination = "token_cap"; break
else:
    termination = "iter_cap"
```

`_append_observation(text, turn, max_iter)` constructs

```python
[{"role": "user",
  "content": f"## Turn {turn}/{max_iter}\n## Output\n```\n{text}\n```"}]
```

then tokenizes with `apply_chat_template(msg, add_generation_prompt=True,
remove_system_prompt=True, enable_thinking=True)` and appends the
resulting ids to both `response_ids` and `prompt_ids` (purely for
clarity — really, the running buffer is one stream), with
`response_mask += [0] * len(obs_ids)`.

### 4.3 Termination contract

| Reason | Action | `submitted_answer` |
|---|---|---|
| `SUBMIT(answer=...)` called | Tail observation `FINAL: <answer>` appended (mask=0); break. | The submitted string. |
| Iteration cap reached | Break. | `None`. |
| Token-budget cap reached | Break. | `None`. |
| Repeated parse error (e.g. no code block N turns in a row) | Break. | `None`. |

`submitted_answer = None` ⇒ ANLS = 0. The model must explicitly
`SUBMIT("Unknown")` to claim the unanswerable case — "Unknown" is a
valid DocVQA answer, so the absence of a submission must be
distinguishable from an explicit "Unknown".

### 4.4 Knobs and defaults

| Knob | Default | Rationale |
|---|---|---|
| `max_iterations` | `20 + 1.5 * sqrt(max(0, num_pages - 9))`, capped at 30 | Matches `flat_solo`'s adaptive cap (`page_factor=1.5`). Per-question. |
| `max_response_tokens_per_turn` | 4096 | Allows enough budget for `<think>` + a non-trivial code block. |
| `max_response_length` (entire trajectory) | 32768 | Sized for ~8 turns at full per-turn budget; configurable per launch. |
| `max_obs_chars` | 8000 | Same as `_format_output` in `lean.py`. |
| `subprocess_timeout` (per `execute`) | 120 s | Same as flat_solo. |
| `parse_error_strikes_to_terminate` | 3 | If the model emits no code block N turns in a row, end the rollout. |
| Sampling | `temperature=1.0`, `top_p=0.95`, `stop=["<\|im_end\|>"]` | Standard on-policy GRPO sampling. |

### 4.5 `<think>` preservation across turns

verl tokenizes new turns as a *diff* — `apply_chat_template` runs only
on new messages with `remove_system_prompt=True`, then the resulting
ids are concatenated to the existing `response_ids`. Prior assistant
tokens (including their `<think>` content) stay as raw token ids and
are never re-rendered. This naturally preserves thinking across turns.

**Sanity check (Layer-2 test).** At end of trajectory, decode every
role-tagged segment from the recorded `prompt_ids ++ response_ids`,
re-render the message list through `apply_chat_template`, and assert
byte-for-byte equality with the original ids. If divergent on stock
Qwen3-8B, swap to `willcb/Qwen3-8B` (already cached locally) and
rerun.

## 5. Prompt structure

### 5.1 System prompt (rendered once)

```
You are a Document Visual Question Answering agent. You answer
questions about a document by writing Python code in a persistent
REPL, calling vision tools iteratively, and reasoning programmatically.

## ENVIRONMENT
You operate in a Python REPL. Each turn you write Python code; it
executes; you see its stdout; then you write more code. State
persists across turns — variables defined in one turn are available
in the next.

## REPL VARIABLES (preloaded)
- `pages`  — list[PIL.Image]; one image per page (0-indexed). Pass to
  `batch_look`, e.g. `batch_look([(pages[0], "describe layout")])`.
  Full pages are large — for fine details, crop first via
  `pages[i].crop((l, t, r, b))`.
- `page_texts` — list[str]; OCR-extracted text per page (Markdown).
  May be inaccurate — verify critical values visually with `batch_look`.

## TOOLS
- `batch_look(requests: list[tuple[PIL.Image, str]]) -> list[str]`
  Send (image, query) pairs to the VLM in parallel. Returns answers
  in the same order. Use it for ALL visual inspection.
- `search(query: str, k: int = 5) -> list[dict]`
  BM25 search over `page_texts`. Returns
  [{page, score, text}, ...]. Useful for multi-page documents.
- `SUBMIT(answer="...")`
  Submit the final answer. ENDS the run. Call only when done.

## OUTPUT FORMAT (every turn)
1. Think inside <think>...</think>: plan, reflect, decide next step.
2. Write a single Python code block in triple backticks:
   ```python
   ...
   ```
   That block will be executed. Anything outside the block is ignored.
3. ALWAYS print() values you want to see — only stdout is returned.

## APPROACH
1. EXPLORE: read `page_texts` and survey pages with `batch_look`
   ("describe layout: sections, tables, figures, labels and where
   they are").
2. LOCATE: find the region(s) relevant to the question.
3. EXTRACT: tight crops + `batch_look` to read exact values.
4. VERIFY: cross-check ambiguous readings with tighter crops.
5. SUBMIT.

## GUIDELINES
- Ask the VLM ONE simple factual question per call. Don't combine
  questions or ask it to reason. Extract raw facts; count, compare,
  and compute in Python.
- For "largest / first / last / only" questions, enumerate ALL
  candidates first, then select programmatically.
- Answer "Unknown" only when (a) a named entity does not exist after
  thorough search, or (b) a chart/table explicitly shows N/A. Do NOT
  invent values.
- NEVER use outside knowledge. All answers must come from the
  document.

## ANSWER FORMATTING
{ANSWER_FORMATTING_RULES verbatim from
 ~/repos/docvqa/src/docvqa/prompts.py}

## CATEGORY TIPS  (only when get_category_tips(category) is non-empty)
{get_category_tips(category)}
```

### 5.2 First user message (rendered once)

```
## Question
{question}

## Document
- category: {category}
- num_pages: {num_pages}

## Variable preview
- pages: list[PIL.Image], length {num_pages}
- page_texts: list[str], length {num_pages}
  page_texts[0] preview (first 400 chars):
  ```
  {page_texts[0][:400]}{"…" if len(page_texts[0]) > 400 else ""}
  ```

Begin.
```

The variable preview only includes the first page's prefix. The agent
can `print(page_texts[i][:N])` itself to inspect more.

### 5.3 Per-turn observation message

```
## Turn {turn}/{max_iter}
## Output
```
{captured_stdout_or_error}
```
```

Truncated at `max_obs_chars=8000` with a trailing `... (truncated)`
marker, mirroring `_format_output` in
`~/repos/docvqa/src/docvqa/rlm/lean.py`.

## 6. Tools spec

### 6.1 `batch_look(requests)`

**Subprocess proxy (sandbox.py):**

```python
def batch_look(requests):
    """Send (image, query) pairs to the VLM in parallel. Returns list[str]."""
    paths = []
    for image, query in requests:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image.save(tmp, format="PNG"); tmp.close()
        paths.append({"path": tmp.name, "query": query})
    return _ipc_call("batch_look", paths)
```

**Host-side handler (tools.py):**

```python
async def batch_look(doc_dir: str, requests: list[dict]) -> list[str]:
    """requests: [{"path": str, "query": str}, ...]"""
    async with httpx.AsyncClient(timeout=120) as client:
        return await asyncio.gather(*[_one_look(client, r) for r in requests])

async def _one_look(client, r):
    img_b64 = base64.b64encode(Path(r["path"]).read_bytes()).decode()
    payload = {
        "model": VLM_MODEL_ID,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": r["query"]},
        ]}],
        "max_tokens": 512, "temperature": 0.0,
    }
    try:
        resp = await client.post(f"{VLM_BASE_URL}/v1/chat/completions", json=payload)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[VLM error: {e}]"
```

`VLM_BASE_URL` and `VLM_MODEL_ID` are configurable via Hydra
(default: `http://localhost:8928`, model id discovered from the
running tmux session).

### 6.2 `search(query, k=5)`

Per-doc BM25 index built ahead of time during data prep (see
Section 7) and stored in `doc_dir/bm25/`. Host handler caches per
`doc_dir` in a process-local dict. Subprocess proxy is a thin
`_ipc_call("search", query, k)`. Returns `[{page, score, text}, ...]`,
filtered to `score > 0`.

### 6.3 `SUBMIT(answer="...")`

Sandbox-side only. Raises `_FinalOutputSignal({"answer": ...})`. The
interpreter catches it and signals "FinalOutput" to the host. The
agent loop reads the answer, appends a tail `FINAL: <answer>`
observation (mask=0) for trace readability, and breaks the loop.

### 6.4 IPC contract

JSON-line over the subprocess's stdin/stdout. Same protocol as
`~/repos/docvqa/src/docvqa/rlm/subprocess_interpreter.py`.

```json
// subprocess → host
{"type": "tool_call", "id": 7, "name": "batch_look",
 "args": {"args": [[{"path": "/tmp/.../x.png", "query": "..."}]],
          "kwargs": {}}}

// host → subprocess
{"type": "tool_response", "id": 7, "result": ["answer1", ...],
 "error": null}
```

## 7. Data layer

### 7.1 Per-document working directory

```
data/{split}/
  docs/{doc_id}/
    metadata.json    # {doc_id, doc_category, num_pages, source_dataset}
    pages/page_0.png, page_1.png, ...
    ocr/page_0.md, page_1.md, ...
    bm25/chunks.json, data.csc.index.npy, indices.csc.index.npy,
         indptr.csc.index.npy, params.index.json, vocab.index.json
  questions.json
```

`questions.json` is a flat list of objects, human-readable:

```json
[
  {"question_id": "doc_001_q0", "doc_id": "business_report_1",
   "question": "What was the Q3 revenue?",
   "answer": "$1.2B", "category": "business_report",
   "source_dataset": "docvqa-2026-val"},
  ...
]
```

### 7.2 verl dataset row

The verl dataset wraps each `questions.json` row plus a resolved
`doc_dir` path:

```python
{
    "question_id": str,
    "question": str,
    "doc_dir": str,        # absolute or repo-relative path
    "gold_answer": str | None,
    "category": str,
}
```

Everything else (page paths, OCR paths, BM25 index) is derived from
`doc_dir` at agent-loop startup.

### 7.3 Phased data prep

1. **Phase 0 (smoke):** copy
   `~/repos/docvqa/data/{val,test}/{ocr,bm25}/` into
   `data/{val,test}/docs/{doc_id}/{ocr,bm25}/`. Render page images
   from HF `VLR-CVC/DocVQA-2026` into `pages/`. Build
   `questions.json` from the HF metadata. This is the only data
   needed to clear the validation gates (Section 9).

2. **Phase 1 (training data):** add `data/train/` populated from
   DocVQA + MP-DocVQA via adapters in `prepare_data.py`. Same shape
   as `data/val/`.

3. **Phase 2+ (broader):** InfographicVQA, ChartQA, SlideVQA, DUDE.
   One adapter per source dataset; flagged in the launch script.

### 7.4 Migration of the eval repo

The `~/repos/docvqa` eval harness currently builds `Document.images`
from HF on the fly. After this spec lands, point its `data.py` at
`doc_dir/pages/` for offline evaluation. Single source of truth for
both repos. (This change is small and is recommended but not
required for this spec.)

## 8. Reward function

`recipe/docvqa/reward.py` computes ANLS via `evaluate_prediction`
(lifted from `~/repos/docvqa/src/docvqa/metrics.py`). The function
signature matches verl's reward-fn contract:

```python
def compute_score(
    data_source: str, solution_str: str, ground_truth: str,
    extra_info: dict | None = None,
) -> tuple[float, dict]:
    submitted = (extra_info or {}).get("submitted_answer")
    score = 0.0 if submitted is None else evaluate_prediction(submitted, ground_truth)[0]
    out_extra = dict(extra_info or {})
    out_extra["anls"] = score
    return score, out_extra
```

`extra_info` is fed in from `AgentLoopOutput.extra_fields`, which
contains `messages`, `num_turns`, `termination`, `submitted_answer`,
`vlm_calls`, `search_calls`, `wall_clock_s`, `doc_id`, etc. Verl
threads anything the reward fn returns in `extra_info` into the
rollout dump (see Section 10).

No shaping rewards in this spec. A small SUBMIT-vs-fallback bonus may
be added later if rollouts show the policy never learns to terminate.

## 9. Validation strategy

Four layers, increasing in cost. Each must clear before the next.

### Layer 1 — unit/integration smoke

`tests/recipe/docvqa/test_subprocess_interp.py` — exercises the
vendored interpreter without verl: state persistence across
`execute()` calls; `SUBMIT` semantics; IPC round-trip with a fake
host handler; sandbox crash recovery (the host receives `[Error]`
and can keep going).

`tests/recipe/docvqa/test_parser.py` — code-fence extraction:
single fence, multiple fences (last wins), no fence (returns `None`),
nested backticks, fence with vs without `python` lang, fence inside
`<think>` (must be ignored — only fences *outside* `<think>` count).

Runs in <10 s.

### Layer 2 — agent loop end-to-end with a dummy LM

`tests/recipe/docvqa/test_agent_loop.py` — instantiate
`DocVQAReplAgentLoop` and run it against a scripted `server_manager`
that returns canned assistant turns from a fixture. Asserts:
- `prompt_ids` / `response_ids` / `response_mask` shapes.
- Round-trip: re-rendering the recorded message list through
  `apply_chat_template` reproduces `prompt_ids ++ response_ids`
  byte-for-byte. **This is the `<think>` preservation check.** If
  it fails on stock Qwen3-8B, swap to `willcb/Qwen3-8B`.
- `SUBMIT` terminates and `extra_fields["submitted_answer"]` is set.
- Iter cap → `submitted_answer is None`.
- Token-budget cap → clean exit.
- Rollout-dump record (a synthetic JSONL line) round-trips through
  `json.loads`.

Fixtures live at `tests/recipe/docvqa/fixtures/*.json`.

### Layer 3 — ANLS reproduction on val

`recipe/docvqa/scripts/eval.py` runs the new agent loop over
DocVQA-2026 val (25 docs / 80 questions) with **Qwen3-8B no-FT** and
the running 27B VLM at `localhost:8928`. Pass criterion:

| Comparison | Bar |
|---|---|
| Reference: `flat_solo` (lean+nothink) Qwen3-8B + Qwen3.6-27B VLM (recorded once) | baseline |
| New scaffold (`enable_thinking=True`) Qwen3-8B + same VLM | within ±5 ANLS-pp of baseline |

Records:
- Per-category ANLS breakdown.
- Trajectory-length histogram.
- SUBMIT rate (target ≥ 95 %).
- Subprocess-error rate (target ≤ 2 %).
- Wall-clock per question.

If gap is > 5pp, debug before training.

### Layer 4 — sanity GRPO run

`recipe/docvqa/scripts/run_smoke_grpo.sh`. ~50–100 GRPO steps, 200
training questions, n=4 rollouts/group. Confirms:
- No OOM.
- `response_mask` is well-formed.
- Reward distribution non-degenerate.
- Loss decreases or at worst is flat — not blowing up.
- val ANLS doesn't crash relative to Layer 3.

After Layer 4 passes, the scaffold is ready for real training (Phase
2 onwards, future spec).

## 10. Rollout dumps

Use verl's built-in mechanism:
`trainer.rollout_data_dir=${hydra:runtime.output_dir}/rollouts`. After
each training step, verl writes `{global_steps}.jsonl` containing
one line per trajectory:

```json
{
  "input": "...",   "output": "...",   "gts": "...",
  "score": 0.95,    "step": 42,
  "messages": [{"role": "system", "content": "..."}, ...],
  "num_turns": 6, "termination": "submit",
  "submitted_answer": "$1.2 billion",
  "vlm_calls": 4, "search_calls": 2, "wall_clock_s": 23.4
}
```

The structured `messages` list is what we actually inspect — `input`
and `output` are decoded with `skip_special_tokens=True` and lose
role boundaries. `messages` is built during the rollout for free
(it's what we already feed to `apply_chat_template`) and surfaced
through `AgentLoopOutput.extra_fields` → `non_tensor_batch` →
reward-fn `extra_info` → `reward_extra_infos_dict` → JSONL.

Eval rollouts dump too. Distinguish train and eval files via verl's
`log_val_generations` (wandb table) and via a separate
`val_rollout_data_dir` if needed.

## 11. Compute layout

3 × A100 80GB on the project box.

| GPU | Role | Notes |
|---|---|---|
| GPU 0, 1 | Training + rollout (colocated via verl hybrid engine) | FSDP-2 across both for the actor; rollout-time vLLM brought up on the same two GPUs each step. LoRA enabled. |
| GPU 2 | 27B VLM, frozen | Already running in the `vllm` tmux session at port 8928. Used as the `batch_look` backend. Not touched by training. |

Subprocess overhead: ~50 MB RSS each; 16 in-flight rollouts ≈ 1 GB
host RAM. Negligible vs GPU memory.

Disk: page images dominate. DocVQA-2026 val/test ≈ a few GB; full
DocVQA-family training set ≈ 50–200 GB. Manageable.

## 12. Phase-1 launch (GRPO only)

`recipe/docvqa/scripts/run_phase1_grpo.sh` (the production launch,
distinct from `run_smoke_grpo.sh`):

```bash
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.lora_rank=16 \
    actor_rollout_ref.actor.lora_alpha=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/docvqa/agent.yaml \
    actor_rollout_ref.rollout.agent.default_agent_loop=docvqa_repl \
    data.train_files=data/train/questions.json \
    data.val_files=data/val/questions.json \
    custom_reward_function.path=recipe/docvqa/reward.py \
    trainer.rollout_data_dir=\${hydra:runtime.output_dir}/rollouts \
    trainer.log_val_generations=20 \
    ...
```

Exact knobs (batch sizes, learning rate, KL coefficient, gradient
accumulation, max_prompt_length, max_response_length) are chosen
during Layer 4 and recorded in the launch script. They are
implementation details, not part of this spec.

## 13. Future phases (not implemented in this spec)

- **Phase 2 — SDPO peer-PI self-distillation + GRPO.** Cherry-pick
  `compute_self_distillation_loss` and
  `_maybe_build_self_distillation_batch` from `~/repos/SDPO/` into
  `docvqa-verl`. Combined per-token signal
  `λ * A_GRPO + (1 - λ) * A_SDPO`, JSD(α=0.5), top-K=100, EMA
  teacher rate 0.05. Multi-turn adaptation: re-score student
  `response_ids` (which already mask out tool/observation tokens)
  under `[original_prompt + peer_submitted_answer + student_response]`.
  Self-teacher reuses the actor's weights — no extra GPU.

- **Phase 3 — external 27B teacher OPD via mainline verl
  `DistillationConfig`.** Adds a teacher-KL term on top of GRPO
  advantages (`use_policy_gradient=True, use_task_rewards=True`).
  Recipe and shell script copied from
  `examples/on_policy_distillation_trainer/`.

- **Optional warmups:** rejection-sampled SFT on the rollouts of the
  current best 8B + 27B teacher; curriculum across DocVQA-family
  datasets.

These are documented for orientation only. Designs come in their own
specs when we get there.

## 14. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stock Qwen3 chat template collapses `<think>` across turns. | Layer-2 round-trip test catches it. Fall back to `willcb/Qwen3-8B`. |
| Long trajectories overflow `max_response_length`. | Token-budget guard inside the loop; default `max_response_length=32768` and `max_response_tokens_per_turn=4096`. |
| Subprocess hangs (e.g., infinite loop in agent code). | Subprocess-side per-execute timeout (default 120 s, same as flat_solo). Loop catches `CodeInterpreterError` and emits `[Error]` to the model. |
| VLM saturates with many concurrent rollouts. | Rollouts slow down rather than fail (HTTP requests queue at the VLM). If wall-clock balloons, reduce rollout concurrency. |
| Large pages (e.g., maps_5 at 246 MP) materialise on `crop` and OOM the subprocess. | `Image.open` is lazy; `Image.MAX_IMAGE_PIXELS = 500_000_000` (already set in flat_solo) bounds it. The subprocess is its own process — OOM kills it cleanly, the loop emits an error and continues. |
| Phase-1 GRPO cold-starts with zero reward variance (no rollouts above 0 ANLS). | If observed in Layer 4, surface in the spec follow-up; mitigations (rejection-sampled SFT warmup, curriculum) live in future specs. |

## 15. Open questions

- Exact `VLM_MODEL_ID` to send in the chat-completions request. Pull
  from the running `vllm` tmux session's startup args.
- Whether to bake LoRA-merge-into-vLLM step into every rollout phase,
  or to keep adapters loaded as adapter weights in vLLM. Decision at
  Layer-4 time depending on adapter-vs-merge throughput.
- Whether to log the system prompt verbatim per rollout in the JSONL
  (it's identical across rollouts of the same step → wasteful) or
  hash it once and reference. Default: log verbatim — disk is cheap.
- Whether `recipe/docvqa/scripts/eval.py` should also drive the
  trained-model eval, or whether to use the eval harness in
  `~/repos/docvqa`. Default: this spec ships its own eval; the docvqa
  harness comparison comes after Layer 3 passes.
