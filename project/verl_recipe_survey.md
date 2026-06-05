# Verl Recipe Survey for DocVQA Training

Survey of `recipe/` (verl's upstream recipes submodule, pinned at `e7f8895`)
done to inform the cold-start mitigation plan from `proposal.md` §6. Goal:
identify which existing recipes we can adopt or crib from for SFT-with-rejection-
sampling warmup, RL (GRPO/DAPO), and on-policy distillation.

## What our current state needs

From the smoke training run (2026-05-01):

- Layer-3 baseline ANLS = **0.167** on the 24-question heldout split.
- Cold start is real: 84% of GRPO rollout groups had identical scores across
  all 4 rollouts → no GRPO advantage signal. One step of GRPO actually
  **dropped ANLS to 0.083** (proposal §6 predicted this).
- Two policy failure modes dominate: 30% "Unknown" capitulation (0 GTs are
  Unknown in the train batch), 13% iter-cap with no submit.
- Per proposal §6 the mitigations are, in priority order:
  1. OPD bootstrap from 27B teacher,
  2. SFT with rejection sampling on filtered teacher/student trajectories,
  3. Curriculum from easier DocVQA-family datasets.

## Most directly applicable recipes

### `recipe/retool/` — SFT-then-RL on a code-execution agent ★★★

Architecturally the closest match. From `recipe/retool/README.md`:

> Cold Start and Supervised Fine Tuning (SFT): the data generation pipeline
> builds a high-quality dataset containing code-enhanced inference trajectories,
> and supervised fine-tuning enables the model to master basic tool call
> (e.g., code execution) and analysis of the execution results.

Validated results on AIME-2025: SFT alone gets `mean@30 = 0.24`; GRPO on top
gets `mean@30 = 0.6`.

Concrete bits to reuse:

- `verl.trainer.fsdp_sft_trainer` with `data.multiturn.enable=true`,
  `data.multiturn.messages_key=messages`, `data.multiturn.tools_key=tools`.
  Native multi-turn SFT. Our `docvqa/scripts/eval.py` run-dir already
  emits chat-format `messages` so the data side is wired up.
- `recipe/retool/retool_sft_preprocess.py` — pattern for converting raw
  text-with-`<code>` blocks into the chat-format expected by the SFT trainer.
  Our agent loop already produces chat messages so the conversion is simpler
  (just filter by `anls == 1.0` and project to `{messages, tools}` rows).
- `recipe/retool/run_qwen2_7b_sft.sh` — 7B SFT launcher (FSDP1, ulysses SP=4,
  `data.max_length=16384`, train batch 32, micro batch per gpu 4, 6 epochs).
  Closest sized template for our Qwen3-8B run.
- `recipe/retool/run_qwen2_7b_dapo.sh` — the GRPO/DAPO config that runs on
  top of the SFT checkpoint. Notable knobs: `n_resp_per_prompt=16`,
  `clip_ratio_low/high=0.2/0.28`, `actor_lr=1e-6`, `max_turns=16`,
  `train_batch_size=64`, `actor_max_token_len_per_gpu=$((prompt+response)*1)`.

### `recipe/deepeyes/` — multi-turn visual tool agent ★★★

DeepEyes = "thinking with images" via RL. The architectural cousin of our
DocVQA agent: the model gets an image, has a zoom-in/crop tool, runs a
multi-turn loop, and is trained with verifier reward. Equivalent of the
`direct_vlm_solver` in the sibling docvqa repo.

Key signals:

- The README warns explicitly about hi-res images causing OOM ("Note on the
  'Chart' Dataset" — matches our 246MP `maps_5` PIL bomb pain).
- They run a 72B judge via SGLang for reward.
- Uses verl's multimodal + multi-turn agent infrastructure (rather than
  rolling its own loop as we do).

If we end up fine-tuning a VLM (direct_vlm path) instead of an LLM
(rvlm/code-agent path), DeepEyes is the closest template. If we stay with
the code-agent (rvlm) path, retool is the closer match.

### `recipe/dapo/test_dapo_7b_math_lora.sh` — LoRA + DAPO knobs ★★

The proposal lists DAPO as a candidate GRPO variant. This is a small, focused
LoRA + DAPO config to crib post-SFT knobs from:

- `n_resp_per_prompt=16` — directly addresses our 84%-zero-variance problem
  by ~4x'ing the rollout group size (we were at 4).
- Asymmetric clipping: `clip_ratio_low=0.2`, `clip_ratio_high=0.28`.
- `reward_model.reward_manager=dapo` + `overlong_buffer_cfg` — length penalty
  that would punish the 13% of rollouts that wander past iter-cap.
- `actor_lr=1e-6` — slower than our 3e-6, DAPO recommendation.
- `actor_rollout_ref.model.lora_rank=8` — much smaller than our 64; shows
  DAPO is fine with low-rank LoRA.

### `recipe/gkd/megatron/` — on-policy distillation (OPD) ★

Exactly the proposal's priority-1 method ("Async On-Policy Knowledge
Distillation Trainer"). Cites the same Thinking Machines blog post in the
README as the proposal §References does.

Mechanism:
1. Student generates rollouts.
2. Teacher returns top-k log-probabilities per valid token position.
3. Student trained with token-wise sparse KL (or RKL, JSD variants) over
   teacher's top-k support.

What's reusable from the recipe:
- Teacher serving infrastructure: `teacher/vllm_engine.py`, `teacher/proxy.py`,
  `teacher/client.py`, ZMQ-based, async-batched. Could front our existing
  27B vLLM at port 8928 to expose a `get_topk_logprobs(token_ids)` endpoint.
- Distillation loss family: `megatron_distill_losses.py` ships KL / RKL /
  KL_RKL convex combo / JSD — well-tested KL variants.

The wart: **Megatron-only**. The losses use Megatron's vocab-parallel autograd
functions (`get_tensor_model_parallel_*`, `VocabUtility`). Porting to FSDP
means rewriting them against the full vocab tensor and integrating into
verl's FSDP actor's train step. Non-trivial but bounded — the math is
straightforward and the teacher infra is the harder half, already done.

### `recipe/open_math_reasoning/run_sft_qwen3_8b.sh` — exact-model-match SFT template ★

Qwen3-8B-Base SFT launcher with FSDP2, ulysses SP, `no_padding`,
`use_remove_padding=True`, `optim.lr=2e-5`, cosine warmup. Closest template
for the SFT-RS warmup stage.

## Other recipes I looked at but discarded

- `swe_agent/` — multi-turn coding agent, but uses an external subprocess +
  ModelProxy HTTP interception. Our direct in-process agent loop is cleaner
  for our use case. Useful only if we ever need to wrap an external tool.
- `langgraph_agent/` — generic ReAct base. Lower abstraction than what we
  already wrote; not worth migrating to.
- `infigui-g1/`, `r1/`, `r1_ascend/`, `dance_grpo/`, `gvpo/`, `flowrl/`,
  `spo/`, `sppo/`, `spin/`, `fapo/`, `specRL/`, `prime/`, `entropy/` —
  GRPO variants targeted at math reasoning or other narrow domains.

## Upstream commits worth picking up in next rebase

`feat/docvqa-scaffold` is 139 commits behind `main`. Picks of note:

- `1927ad33` — `[rollout, vllm] fix: use engine.sleep() instead of
  collective_rpc (#6456)` — directly in the area we hit with our
  `multi_stage_wake_up=True` fix.
- `5ff595ac` — `[rollout, vllm] fix: treat null rollout seed as 0 for engine
  init (#6503)` — vLLM init hardening.
- `7c3118e5` — `[rollout] feat: enable MooncakeStoreConnector with hard-reset
  on weight update (#6373)` — different but adjacent path for LoRA hot-reload.
- `51c84660` — `[megatron, trainer] fix: preserve BSHD top-k distillation
  shape (#6506)` — touches the GKD code path.
- `75be454b` — `[fsdp] fix: add sp and use_remove_padding validate for SFT
  and RL in fsdp engine (#6502)` — SFT path hardening.
- `3ba81812` — `[fsdp, model] feat: support qwen3_5 ulysses sp (#6482)` —
  Qwen3.5 SP support.

## Recommended path forward

1. **Rebase** `feat/docvqa-scaffold` on `main` to pick up the vLLM and FSDP
   fixes above. Worth doing before the next training run; several of those
   commits touch code we already lean on.
2. **SFT-RS warmup using the retool pattern.** An `eval.py` run-dir
   already emits the right format. Add a small preprocess step that
   filters `anls == 1.0` and projects to `{messages, tools}` rows, then run
   `verl.trainer.fsdp_sft_trainer` with `data.multiturn.enable=true`.
3. **Post-SFT GRPO/DAPO** using DAPO knobs from
   `recipe/dapo/test_dapo_7b_math_lora.sh`: `n_resp_per_prompt=16`,
   asymmetric clip, DAPO reward manager with overlong buffer, lr 1e-6.
4. **OPD** later — port `recipe/gkd/megatron` losses to FSDP. The teacher
   server is reusable as-is once our 27B serves top-k logprobs.

Which path to take depends on whether we end up training the
`rvlm_minimal_solver` (LLM-as-agent, what our `agent_loop.py` mirrors) or
the `direct_vlm_solver` (VLM-as-agent, DeepEyes-like). That decision is open.
