# Findings: `<think>`-trace handling in multi-turn SFT of reasoning LLMs

> Answer to `research-brief-qwen-thinking-multiturn-sft.md`. Method: a fan-out
> deep-research pass (18 sources, 25 claims adversarially verified, 3-vote) plus
> two targeted passes on 2025-2026 frontier agentic models, plus direct
> verification against this repo's code and the vendored verl source. Every
> load-bearing claim is cited to a primary source or a file:line.

---

## TL;DR recommendation (for THIS setup)

**Use Regime 1 — keep every assistant turn's `<think>` in the trained sequence.**
The current pipeline already does this correctly, end-to-end, and I verified each
hop in code:

1. **Deploy = Regime 1 by construction.** `agent_loop.py` is a token-level
   append-only (TITO) loop: it generates on `prompt_ids + response_ids`, appends
   the **exact sampled** assistant token ids, and never re-renders history. Prior
   turns' `<think>` survive byte-for-byte. (`docvqa/agent_loop.py:173-184`)
2. **Train = Regime 1 by construction.** verl's `MultiTurnSFTDataset` tokenizes
   **each message independently** (`apply_chat_template(messages=[message])`), so
   every assistant turn is rendered as a *last* turn → the Qwen template
   reconstructs the `<think>\n` open tag and **keeps** that turn's reasoning.
   (`verl/utils/dataset/multiturn_sft_dataset.py:214-223, 303-318`)
3. **They match.** Both sides are Regime 1, so train distribution == deploy
   distribution. That is the whole game (see the rule below).

The only thing to *confirm operationally* (not change): that
`ignore_input_ids_mismatch=True` is set for training — otherwise verl **raises**
on the Qwen-thinking per-turn-vs-whole-conversation divergence and refuses to
run. The error message names this exact case. (`multiturn_sft_dataset.py:443-456`)

---

## The governing rule (state it explicitly)

> **The training token sequence must be drawn from the same distribution the
> deployment rollout produces.** If you strip prior-turn `<think>` at inference,
> strip it in training; if you retain it at inference, retain it in training. The
> "which regime is better in the abstract" question is downstream of this — match
> first, optimize second.

This is the consensus across the strongest sources, and it is the *reason* the
answer for us is Regime 1: our deploy rollout (TITO) retains prior `<think>`, so
training must too. Primary support:

- **PrimeIntellect "renderers"** (Gallouédec & Rasul, 2026): *"the trainer must
  see the exact token ids the sampler saw."* Their `bridge_to_next_turn` builds
  the next-turn prompt as `prev_prompt_ids + prev_completion_ids` byte-for-byte.
  Quantified mismatch cost on Qwen3.5-35B-A3B + mini-swe-agent: full
  `apply_chat_template` re-render = **32 prefix breaks / 64 rollouts** (only 77
  samples usable); `bridge_to_next_turn` = **0 / 64**. They explicitly list
  Qwen3/Qwen3.5 and DeepSeek-R1 as templates that *"remove `<think>` sections …
  which violates the increasing-context requirement for multi-turn training."*
  <https://github.com/PrimeIntellect-ai/renderers> ·
  <https://www.primeintellect.ai/blog/renderers>

**Concrete guard for us:** never let any eval/serving path feed the model via
`apply_chat_template(full_message_list)`. That path *would* impose Regime 2
(strip prior `<think>`) and silently diverge from training. Our `agent_loop` is
safe; a generic OpenAI-chat-completions server pointed at the same checkpoint
would not be. This is the single highest-leverage thing to keep an eye on.

---

## The binary framing is slightly wrong — and it helps us

The brief's "Regime 1 (keep all) vs Regime 2 (strip prior)" is the right *axis*,
but **no current frontier agentic model is at either pole.** The 2025-2026
consensus (gpt-oss, GLM-4.5/4.6, MiniMax-M2, Kimi-K2-Thinking) is a **hybrid**:

> **Strip prior reasoning once a *user turn* closes the exchange (a `final`
> answer was emitted), but KEEP reasoning across the *in-flight tool-call chain*
> — every assistant turn since the last user message.**

Why this matters for us: **a DocVQA episode is exactly one user turn (the
question) followed by a long tool-call chain to `SUBMIT`, with no intervening
user turns.** So the "strip across user turns" branch *never fires* — the whole
episode is one in-flight chain, and every frontier model keeps reasoning within
it. **For single-question agentic episodes, the frontier hybrid collapses to
Regime 1.** Our scaffold already matches documented best practice; it is not an
idiosyncratic choice. (The hybrid would only start to bite if we trained genuine
multi-question conversations.)

### Frontier evidence (newer models — verified this pass)

| Model | Across USER turns | Within in-flight TOOL chain | Mechanism | Source |
|---|---|---|---|---|
| **gpt-oss / Harmony** | drop CoT after a `final` | **keep** (explicit exception) | token-level channel render | OpenAI Harmony guide¹ |
| **Kimi-K2-Thinking** | strip (empty `<think>`) | **keep** | Jinja split at last non-tool-call assistant msg | model card chat_template² |
| **MiniMax-M2** | strip (keep only latest user turn) | **keep** | Jinja `loop.index0 > last_user_index` | card + team clarification³ |
| **GLM-4.5 / 4.6** | strip (empty `<think></think>`) | **keep** | Jinja `loop.index0 > ns.last_user_index` | chat_template.jinja⁴ |
| **Qwen3 / 3.x** | strip (`split('</think>')[-1]`) | **keep** (`> last_query_index`) | Jinja + official guidance | discussion #1398, card⁵ |
| **DeepSeek-V3.1** | strip `<think>` content entirely | n/a (tools = non-thinking) | template re-render | model card⁶ |
| **Magistral (Mistral)** | **developer's choice** (explicit) | unspecified | `[THINK]` tokens | model card⁷ |
| **Llama-Nemotron** | *no documented rule found* | *unknown* | — | (flagged: don't infer) |

The two **independent, fully-documented** cases — OpenAI gpt-oss and Zhipu GLM —
state the identical rule: *keep reasoning across the active tool chain, drop it
once a `final`/user turn closes the exchange.* That is the strongest convergence
in the literature and it is exactly the branch our episodes live in.

Notable directional signal: **Kimi K2.5/K2.6** introduce a `thinking.keep:"all"`
API option that retains `reasoning_content` from **every** historical assistant
turn (true Regime 1), with the warning that otherwise *"long-horizon tool
workflows degrade."* The frontier is, if anything, drifting *toward* keeping
more reasoning, not less.⁸

¹ <https://developers.openai.com/cookbook/articles/openai-harmony> ·
<https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot>
(*"drop any previous CoT … if the assistant ended in a `final` message. The
exception … is tool/function calling … pass the previous chain-of-thought back
in"*)
² <https://huggingface.co/moonshotai/Kimi-K2-Thinking> (chat_template splits
`hist_msgs` / `suffix_msgs` at the last non-tool-call assistant message; history
gets empty `<think></think>`, the active chain keeps real reasoning)
³ <https://huggingface.co/MiniMaxAI/MiniMax-M2> (*"Do not remove the
`<think>...</think>` part, otherwise … performance will be negatively
affected"*) + team clarification, discussion #38 (*"we only retain the thinking
from the latest user turn … to save context"*)
⁴ <https://huggingface.co/zai-org/GLM-4.5> ·
<https://huggingface.co/zai-org/GLM-4.6> (`chat_template.jinja`)
⁵ <https://github.com/QwenLM/Qwen3/discussions/1398> (maintainer jklj077: *"the
thinking content of the previous turns should be removed **except for multi-step
tool calls**"*) · <https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507>
⁶ <https://huggingface.co/deepseek-ai/DeepSeek-V3.1> (*"the thinking token in the
last turn will be dropped but the `</think>` is retained in every turn"*)
⁷ <https://huggingface.co/mistralai/Magistral-Small-2509> (*"choose … between
keeping reasoning traces during multi-turn interactions or keeping only the final
assistant response"*)
⁸ <https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model>

---

## Why the Qwen template strips (and why that *doesn't* mandate stripping for us)

Question 2 of the brief. Two things, both verified:

1. **What it does:** the strip is the Jinja conditional
   `{%- if loop.index0 > ns.last_query_index %}` … else
   `content.split('</think>')[-1]`, where `last_query_index` is the last
   **non-tool** user query. Assistant turns *after* the last real user query (an
   in-flight tool chain) keep `<think>`; earlier turns are reduced to their
   post-`</think>` answer. Verified against the Qwen3 tokenizer_config and
   corroborated by QwenLM/Qwen3 #1826 and #1398.
2. **Why / whose responsibility:** Qwen's own model card scopes the
   strip-prior recommendation to **the chat template** and explicitly hands the
   decision back to you otherwise: *"implemented in the provided chat template in
   Jinja2. However, for frameworks that do not directly use the Jinja2 chat
   template, it is up to the developers to ensure that the best practice is
   followed."* (<https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507>) A TITO
   append-only loop is precisely such a non-Jinja framework. The "best practice"
   (strip) presumes a **re-rendering deploy**; ours doesn't re-render, so the
   matching best practice for us is **keep** (Regime 1).

The documented *intent* is context economy across genuine multi-turn
conversations, plus the explicit tool-call carve-out — **not** a claim that
keeping reasoning hurts quality. (No source we verified claims a quality penalty
from retaining reasoning across an active tool chain; the MiniMax blog claims the
*opposite* — dropping in-chain state measurably hurts: SWE-Bench Verified
69.4→67.2, BrowseComp 44.0→31.4. Treat those specific deltas as vendor-reported,
single-source.)

---

## Loss-masking guidance (verified against verl source)

The brief's masking questions (Q6) — confirmed standard and already correct in
our stack:

- **Mask the generation-prompt scaffolding**, including the injected `<think>\n`
  open tag. verl does exactly this: for an assistant message it sets
  `loss_mask = ones`, then `loss_mask[:len(self.generation_prompt)] = 0`.
  (`multiturn_sft_dataset.py:234-237`) The reconstructed `<think>\n` prefix lives
  in that masked generation-prompt span → **not trained**, correctly.
- **Mask all non-assistant turns** (system, user, and tool observations) →
  `loss_mask = zeros`. (`multiturn_sft_dataset.py:238-239`) Our observations are
  appended as `role:"user"` (`agent_loop.py:314`), so they fall in this masked
  bucket — both at deploy (`response_mask += [0]*len`, `agent_loop.py:328`) and
  in training. **Loss is computed only over assistant-emitted reasoning + code +
  final action.** This is the standard agentic-SFT convention (mask environment
  tokens, train on action+reasoning) and our two paths agree on it.

---

## Efficiency tradeoff (Q5)

Keeping all-turn reasoning inflates context, but for us the question is moot:
TITO *requires* it (you can't append-only and also drop history), and DocVQA
episodes are bounded (this repo's transfer parquet caps at ~12.5K tokens/episode,
well under the 32K budget — see `.claude/CLAUDE.md`). The cost of the *correct*
regime here is already paid by the scaffold design. The expensive option would be
the *wrong* one (a re-rendering server), and it's expensive in correctness, not
just tokens.

---

## Residual risks / things to confirm (not change)

1. **`ignore_input_ids_mismatch=True` must be set** for the SeqKD/SFT run, or
   verl raises on the Qwen-thinking divergence and won't train. The per-turn
   (Regime 1) concatenation is the one you *want*; the flag just tells verl to
   accept that it differs from a whole-conversation render. Confirm it's in the
   training config. (`multiturn_sft_dataset.py:443-456`)
2. **Offline collection and online RL are separate paths — both already
   Regime 1, no fix needed.**
   - **Offline (SFT / SeqKD):** an `eval.py` run *is* the collection. It saves the
     exact token stream by default — `prompt_ids` / `response_ids` /
     `response_mask` with the assistant-only mask (`eval.py:162, 185-190`) — *and*
     the decoded `messages`. The SeqKD path trains from `messages` via
     `MultiTurnSFTDataset` (per-turn → Regime 1, matches deploy); the saved token
     ids are there for SFT-on-exact-tokens / teacher-forced KD. Either way it's
     Regime 1. (Caveat: the eval HTTP client re-encodes generated text → ids
     — `eval.py:46-48, 75` — so the offline token ids are a faithful
     *re-tokenization* of the text, not byte-identical to what the server sampled.
     Immaterial for teacher-forced SFT/KD, which carry no importance ratio.)
   - **Online (RFT / RL / OPD):** these stages do **not** use offline rollouts.
     verl generates rollouts in-process (sglang/vLLM), which is **TITO by
     construction** — the trainer sees the exact sampled token ids, so
     sampler == trainer token identity is automatic and there is no offline
     re-tokenization to reconcile. The TITO/renderers concern is owned by the
     rollout engine, not a collection script. (This is the *generator* half of
     correct on-policy training; the precision-matching guidance in `CLAUDE.md` is
     the *trainer* half.)
   - The old `collect_trajectories.py` (which saved only `messages`) has been
     removed; `eval.py` is the single collection path.
3. **No published head-to-head quality A/B** (accuracy/length/degeneration) of
   Regime 1 vs Regime 2 for a TITO multi-turn agent surfaced. The recommendation
   rests on (a) the distribution-matching argument, (b) the prefix-break evidence,
   and (c) frontier-practice convergence — not on a downstream A/B. That's a
   strong basis but worth stating honestly.

---

## Sources (verified, primary unless noted)

- PrimeIntellect renderers — <https://github.com/PrimeIntellect-ai/renderers>,
  <https://www.primeintellect.ai/blog/renderers>
- Qwen3 discussion #1398 (maintainer guidance) —
  <https://github.com/QwenLM/Qwen3/discussions/1398>
- Qwen3-4B-Thinking-2507 model card —
  <https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507>
- Qwen3 Technical Report (empty-think convention; `/think` `/no_think` flags) —
  <https://arxiv.org/abs/2505.09388>
- OpenAI Harmony / gpt-oss CoT handling —
  <https://developers.openai.com/cookbook/articles/openai-harmony>,
  <https://developers.openai.com/cookbook/articles/gpt-oss/handle-raw-cot>
- GLM-4.5 / 4.6 chat templates — <https://huggingface.co/zai-org/GLM-4.5>,
  <https://huggingface.co/zai-org/GLM-4.6>
- MiniMax-M2 card + discussion #38 + interleaved-thinking blog —
  <https://huggingface.co/MiniMaxAI/MiniMax-M2>,
  <https://huggingface.co/MiniMaxAI/MiniMax-M2/discussions/38>,
  <https://www.minimax.io/news/why-is-interleaved-thinking-important-for-m2>
- Kimi-K2-Thinking card/template + K2.5/2.6 `thinking.keep` guide —
  <https://huggingface.co/moonshotai/Kimi-K2-Thinking>,
  <https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model>
- DeepSeek-V3.1 card — <https://huggingface.co/deepseek-ai/DeepSeek-V3.1>
- Magistral-Small-2509 card — <https://huggingface.co/mistralai/Magistral-Small-2509>
- (code) `docvqa/agent_loop.py`, `docvqa/scripts/eval.py`,
  `docvqa/scripts/make_sft_data.py`, `verl/utils/dataset/multiturn_sft_dataset.py`

### Caveats on source strength
- The quantified mismatch cost (0 vs 32 prefix breaks) is **single-source**
  (PrimeIntellect), though primary, current, and empirically backed.
- Qwen's strip-prior guidance is a genuine official recommendation but is
  **permissively phrased and scoped to the re-rendering deploy** — applying it to
  our TITO case is an inference, well-grounded via the explicit multi-step-tool
  exception and the developer-responsibility clause.
- DeepSeek-R1 (`2501.12948`) is now a historical reference for this question; its
  reasoning was largely single-turn. The newer agentic models above are the
  relevant evidence, which is why this pass weighted them.
- Several *verl-internals* claims floated by the auto-research (delta
  tokenization, fallback base conversation) were **refuted/unconfirmed** by the
  verifier; the verl behavior stated here is from **direct source reading**, not
  those claims.
