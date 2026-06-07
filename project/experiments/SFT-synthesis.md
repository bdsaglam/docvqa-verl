# SFT stage — synthesis (corruption discovery + clean restart)

> Updated 2026-06-07. **Supersedes all earlier per-experiment result cards** (removed):
> they measured models trained on corrupted teacher data — see below.

## Headline
**Every SFT model from the first sweep (v1, v2, Arm A/B, v4–v6) was trained on
~93%-corrupted teacher data, so their numbers are invalid.** A scaffold bug let the
27B teacher hallucinate tool output; we discovered it, fixed it, and restarted clean.
The real "does SFT beat baseline" question is **open** and now runs on clean footing.

## The bug (root cause)
- The agent emits `<think>` + a ```` ```python ```` fence; captured stdout becomes the
  next observation. The **only** generation stop was `<|im_end|>`.
- When the model didn't emit `<|im_end|>` after its code fence, it **free-ran and
  role-played the next turns** — fabricating `\nuser\n## Output\n...` observations and
  *more* code blocks (up to **10 fences in one turn**).
- `_FENCE_RE` matches plain ```` ``` ```` too, and we parsed the **last** fence → the
  executed code (and thus the recorded observation) came from a *hallucinated* block.
- **93.5% of anls==1.0 trajectories** had ≥1 such multi-fence turn. Unpatchable: the
  recorded observation is paired with the last-fence (hallucinated) block, not the
  model's real first action — stripping desyncs (code, observation).

## The fix (`docvqa/agent_loop.py`, `docvqa/prompts.py`)
1. **Stop sequences** `["<|im_end|>", "\nuser\n", "\n## Turn", "\n## Output"]` — halt
   the instant the model starts fabricating an observation.
2. **`parse_first_fence=True` default** (eval/collection/RL) — run the model's real
   first action even if a stray block slips in.
3. **Defensive strip** of any fabrication tail (SFT-text path).
4. **Simplified prompt format** to shrink the hallucination surface: observation is
   `## Output (Turn n/N)\n{output}` (no ```` ``` ```` fence); first-user message drops
   the `Begin.` sentinel and puts the **question last** (recency).

## Verification (clean collection v4, fixed scaffold)
- **Hallucination markers: 0.0%** (was pervasive). ✓ Fix confirmed.
- Multi-fence: 93.5% → ~20%, and the residual is **benign** (genuine multi-block, no
  fabricated observations; first-fence runs the right block; build-time first-fence
  truncation makes SFT data fully single-block).
- **Teacher solve-yield jumped ~22% → ~98%** — the fix also helps the 27B *solve*: it
  no longer derails itself by executing hallucinated blocks. (Early-doc sample; watch.)

## Status
Clean restart in progress: collect (mmlb 391 Q, fixed scaffold) → build clean SFT data
→ train `seqkd-clean-v4` (LoRA r32 all-linear, LR 2e-4 constant) → eval baseline +
clean model under the fixed scaffold on `docvqa_mini` (29 Q) then full val (80 Q).

## Carry-forward lessons (scaffold-independent, still valid)
- **Eval methodology:** train on *other* datasets (MMLongBench-Doc) → the full DocVQA
  val is leakage-free, so **no splits** — `docvqa_mini.json` (29 Q / 8 docs, 1 median
  doc per category) for quick iteration, `questions.json` (80 Q) for the reportable
  number (vs docvqa baselines: ≤8B 0.1875, 27B 0.375).
- **LoRA-without-regret:** apply to **all-linear** (MLP matters most), LR ≈ 10–15×
  FullFT (→ ~2e-4 for these short <100-step runs), rank not capacity-limiting at this
  data scale; constant schedule for short runs.
- **VLM perception:** the concise-answer+rationale instruction did **not** help the 27B
  (submit-only 0.373 vs bare-query 0.40, n=1/80) — but that's no proxy for whether it
  helps the weaker 4B; the clean retrain tests it.
- **Per-turn 4K cap is binding for the student** (clips ~13% of turns post-SFT vs 1% of
  teacher turns) — consider raising `max_response_tokens_per_turn` to 8K.
- **Dominant loss term = long-doc runaway / wall_cap** (lives in the agent scaffold),
  not answer quality — caps every method's ceiling; reward shaping / per-turn cap are
  the levers.
