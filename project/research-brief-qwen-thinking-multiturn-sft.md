# Research brief: `<think>`-trace handling in multi-turn SFT of reasoning LLMs

**Audience:** an agent tasked with investigating web sources / papers / model technical
reports and returning a best-practice recommendation.
**Goal:** decide how to handle per-turn `<think>…</think>` reasoning traces when
SFT-/distillation-training a reasoning LLM on **multi-turn** trajectories (e.g. agentic
tool-use), and back it with evidence.

---

## 1. The phenomenon (what triggered this)

Qwen3 / Qwen3.5 chat templates **strip `<think>…</think>` blocks from all assistant
turns except the last** when you render a whole multi-turn conversation. So if you
collect a multi-turn trajectory where the model reasoned on every turn and then
re-render the message list, only the final turn keeps its reasoning; earlier turns'
reasoning is silently removed.

There are (at least) two regimes used in practice for training on such data:

- **Regime 1 — keep all-turn thinking:** every assistant turn's reasoning is in the
  trained sequence. The model learns to reason at every step, conditioned on prior
  turns' reasoning.
- **Regime 2 — strip prior thinking (Qwen default):** only the last turn's reasoning is
  kept; earlier turns are reduced to their final answer/action. Rationale offered: at
  inference the prior turns' think blocks are discarded from context, so training the
  model to reproduce/condition-on them is off-distribution.

**The core question:** which regime is correct/better, and **what does it depend on?**

---

## 2. What we already verified empirically (grounding — don't re-derive)

On Qwen3.5-4B, with multi-turn CodeAct agent trajectories (27B teacher, native thinking):

- Whole-conversation `tokenizer.apply_chat_template(messages)` → prior-turn `<think>`
  **stripped**, only last turn's survives (Regime 2 behavior). Confirmed.
- Our stored assistant contents mostly carry `</think>` **without** an opening `<think>`
  (~89% of turns). Reason: at generation the chat template *prefills* `<|im_start|>assistant\n<think>\n`,
  so the model emits only the reasoning + closing `</think>`; the opening tag lives in
  the prompt, not the completion.
- The training stack we use (verl `MultiTurnSFTDataset`) tokenizes **each turn
  independently** (each rendered as the last turn of a `[dummy_user, assistant]` pair),
  which **preserves reasoning on every turn** → effectively **Regime 1**. It even
  reconstructs the opening `<think>\n`.
- Loss masking: the reconstructed `<think>\n` prefix falls in the **masked
  generation-prompt** (loss=0); reasoning + code + final action are trained (loss=1);
  tool observations masked. This is train/inference-consistent **iff** the serving path
  also keeps prior-turn `<think>` in context.
- Train/inference consistency hinges on the **serving/rollout path**: a token-level /
  append-only ("TITO", token-in-token-out) loop **retains** prior-turn `<think>`;
  a chat-completions server that **re-renders** the message list through the template
  **strips** it (Regime 2). These two are *different distributions* — the training
  regime must match the deployment regime.

So the practical crux is: **Regime 1 vs Regime 2 must match between training and
deployment**, and we currently train Regime 1 (per-turn) while some serving paths would
impose Regime 2. We need the literature/best-practice to decide the target.

---

## 3. Questions to investigate (the actual ask)

1. **Which regime do leading reasoning models use for multi-turn SFT/RL data?** Check
   technical reports: DeepSeek-R1 / R1 distillation, Qwen3 (and the rationale behind the
   template's strip behavior), Kimi k1.5/k2, GLM, Llama/Nemotron reasoning variants,
   Magistral, etc. What do they say about keeping vs dropping prior-turn reasoning in
   multi-turn / agentic data?
2. **Why does the Qwen template strip prior `<think>`?** Is the documented/intended
   reason context-economy, train/deploy matching, or avoiding compounding hallucinated
   reasoning? Find the authoritative statement (Qwen team posts, template commit
   history, HF discussions).
3. **Train/deploy distribution matching:** is the consensus that the *training* regime
   must equal the *inference* regime (i.e. if you strip at inference you must strip in
   training, and vice-versa)? Any ablations on the cost of a mismatch?
4. **Evidence on quality:** any ablations comparing all-turn-thinking vs last-turn-only
   for multi-turn agents — on accuracy, length, reasoning quality, degeneration?
5. **Efficiency tradeoff:** keeping all-turn reasoning inflates context length and
   training cost; quantify when it's worth it.
6. **Loss-masking conventions:** is masking the `<think>` open tag (generation-prompt
   scaffolding) standard? Any guidance on masking reasoning vs answer tokens?
7. **Agentic specifics:** for tool-use agents where the env appends tool observations,
   does best practice differ (e.g. keep reasoning but it's interleaved with masked
   observations)?

---

## 4. Pointers / leads

- **Qwen3 chat template**: inspect `tokenizer_config.json` `chat_template` (Jinja) for
  Qwen3/Qwen3.5 — the conditional that drops `<think>` on non-final assistant turns.
- **TITO / token-in-token-out**: Gallouédec & Rasul, "renderers" (Prime Intellect) —
  token-level chat templating for agentic RL; their prefix-preservation audit explicitly
  names **Qwen3**'s empty-`<think>`-on-last-turn quirk as the one failure among 19
  open-weight families. Repo: github.com/PrimeIntellect-ai/renderers.
- **DeepSeek-R1 paper** and the **Qwen3 technical report** for reasoning post-training
  data construction.
- **verl** `MultiTurnSFTDataset` (`ignore_input_ids_mismatch`) as the concrete example of
  per-turn vs whole-conversation tokenization divergence.

## 5. Deliverable wanted back

A short recommendation: **Regime 1 or Regime 2 for our setup** (multi-turn CodeAct agent,
token-level TITO rollouts at deploy), with citations, the train/deploy-matching rule
stated explicitly, and any masking guidance.
