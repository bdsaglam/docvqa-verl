"""DocVQA CodeAct REPL agent loop for verl.

Implements the **CodeAct** scaffold: a strictly append-only transcript in
which the model emits ``<think>...</think>`` + a single ``` ```python ... ``` ``
fence per turn, the code runs in a persistent CPython subprocess with
``batch_look`` (27B VLM perception) + ``SUBMIT``, and the captured stdout is
appended verbatim as the next ``user`` turn. The full message list grows
monotonically (an MDP, fully observable) — this is the property RL / SFT /
distillation losses assume. Contrast the deployed ``rvlm`` solver, which uses
LeanRLM's hidden REPL namespace + ``RESET_HISTORY`` (a POMDP) and is *not* a
fine-tuning target. ANLS reward end-of-trajectory.

Per-turn flow: generation → code parsing → subprocess execution → observation
appending → SUBMIT termination, with parse-error / iter-cap / token-cap
fallbacks.

Tool surface is intentionally narrow — just ``batch_look`` plus ``SUBMIT``.
Earlier revisions also wired a BM25 ``search`` over OCR; the deployment-time
scaffold dropped it (recursive VLM perception is the load-bearing mechanism)
and we follow suit so train-time and deploy-time behavior stay aligned.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from docvqa.parser import parse_python_fences
from docvqa.prompts import (
    build_first_user_message,
    build_observation_message,
    build_system_prompt,
)
from docvqa.sandbox import build_sandbox_code
from docvqa.subprocess_interp import (
    CodeInterpreterError,
    FinalOutput,
    SubprocessInterpreter,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)

# ---------------------------------------------------------------------------
# Knobs
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "page_factor": 1.5,
    "max_iterations_base": 20,
    "max_iterations_cap": 30,
    "max_response_tokens_per_turn": 4096,
    "max_obs_chars": 8000,
    "subprocess_timeout_s": 120.0,
    "parse_error_strikes_to_terminate": 3,
    # Select the FIRST python fence per turn (the model's real intended action) instead of
    # the last. Default True everywhere (eval/collection/RL): if the model hallucinates a
    # trailing tool-output fence after its code, last-fence would grab the fabricated block —
    # first-fence always takes the legit one. The format reward penalizes >1-block turns.
    "parse_first_fence": True,
    # Extra generation stops that cut the model's role-played next-turn tail (fabricated
    # "\nuser\n" / "## Turn" / "## Output" observations). ON for SFT DATA COLLECTION (we want
    # clean teacher trajectories) and eval. **RL sets this False** — the policy must *learn*
    # to stop correctly via the format reward, not have the bad behavior masked by a stop.
    "extra_obs_stops": True,
}


def _adaptive_max_iter(num_pages: int, knobs: dict) -> int:
    import math

    bonus = knobs["page_factor"] * math.sqrt(max(0, num_pages - 9))
    return min(knobs["max_iterations_cap"], knobs["max_iterations_base"] + int(bonus))


# Markers that begin a *hallucinated* next-turn observation the model sometimes
# role-plays after its code fence: the plain role label and the observation headers
# emitted by build_observation_message ("## Turn {n}/{m}", "## Output").
_HALLUC_OBS_MARKERS = ("\nuser\n", "\n## Turn", "\n## Output")


def _strip_hallucinated_observation(text: str) -> str:
    """Truncate ``text`` at the first hallucinated-observation marker, if any.

    The legit ``<think>`` + ```` ```python ``` ```` fence always precedes these markers,
    so the real turn content survives; only the fabricated tool output is dropped."""
    cut = len(text)
    for m in _HALLUC_OBS_MARKERS:
        i = text.find(m)
        if i != -1 and i < cut:
            cut = i
    return text[:cut]


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@register("docvqa_repl")
class DocVQAReplAgentLoop(AgentLoopBase):
    """Per-question REPL agent. One persistent subprocess per rollout."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Optional overrides via config.actor_rollout_ref.rollout.agent.docvqa
        agent_cfg = self.rollout_config.agent.get("docvqa", {}) or {}
        # OmegaConf DictConfig -> plain dict for easier .get usage
        try:
            from omegaconf import OmegaConf

            if hasattr(agent_cfg, "_metadata"):
                agent_cfg = OmegaConf.to_container(agent_cfg, resolve=True) or {}
        except Exception:
            pass
        self._knobs: dict[str, Any] = {**_DEFAULTS, **dict(agent_cfg)}

        self._vlm_base_url: str = (
            agent_cfg.get("vlm_base_url") or os.environ.get("DOCVQA_VLM_BASE_URL") or "http://localhost:8928"
        )
        self._vlm_model_id: str = (
            agent_cfg.get("vlm_model_id") or os.environ.get("DOCVQA_VLM_MODEL_ID") or "Qwen/Qwen3.5-27B"
        )

        self._response_length_cap: int = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        wall_start = time.monotonic()

        question = kwargs["question"]
        question_id = kwargs.get("question_id", "")
        doc_dir = kwargs["doc_dir"]
        category = kwargs.get("category", "unknown")

        meta = json.loads((Path(doc_dir) / "metadata.json").read_text())
        num_pages = meta["num_pages"]

        max_iter = _adaptive_max_iter(num_pages, self._knobs)

        sys_prompt = build_system_prompt(category)
        first_user = build_first_user_message(question, category, num_pages)
        messages: list[dict] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": first_user},
        ]
        prompt_ids: list[int] = await self.apply_chat_template(messages)
        response_ids: list[int] = []
        response_mask: list[int] = []

        import httpx

        from docvqa import tools as host_tools

        vlm_client = httpx.AsyncClient(timeout=120)
        interp: SubprocessInterpreter | None = None

        request_id = uuid4().hex
        termination = "iter_cap"
        submitted_answer: str | None = None
        num_turns = 0
        vlm_calls = 0
        parse_error_strikes = 0
        # Per-turn truncation instrumentation: a turn hits the per-turn
        # max_tokens cap (max_response_tokens_per_turn) iff it emitted that many
        # tokens AND never produced the <|im_end|> stop. Such turns clip the
        # ```python``` block mid-stream → broken code / parse_error next turn.
        # Surfaced so we can judge whether the per-turn cap is large enough.
        turns_truncated = 0
        max_turn_tokens = 0
        # Format-violation counters (trajectory-scalar inputs to the RL format penalty):
        #   multi_block_turns  -- turn emitted >1 ```python``` fence (only the selected one ran)
        #   empty_output_turns -- code ran but printed nothing ("forgot to print")
        multi_block_turns = 0
        empty_output_turns = 0

        try:

            async def _batch_look_host(requests: list[dict]) -> list[str]:
                return await host_tools.batch_look(
                    requests,
                    vlm_client,
                    self._vlm_base_url,
                    self._vlm_model_id,
                )

            def _batch_look_sync(requests):
                """Bridge to async — invoked from a host-side thread by IPC."""
                return asyncio.run_coroutine_threadsafe(
                    _batch_look_host(requests),
                    self.loop,
                ).result(timeout=300)

            interp = SubprocessInterpreter(
                sandbox_code=build_sandbox_code(),
                tools={"batch_look": _batch_look_sync},
                output_fields=[{"name": "answer", "type": "str"}],
                timeout=self._knobs["subprocess_timeout_s"],
                extra_env={"DOC_DIR": doc_dir},
            )
            interp.start()

            for turn in range(1, max_iter + 1):
                num_turns += 1

                # <|im_end|> always; the role-play markers only when extra_obs_stops is set
                # (collection/eval). For RL the policy must learn to stop via the reward, so
                # the recipe sets extra_obs_stops=False and only <|im_end|> applies.
                stops = ["<|im_end|>"]
                if self._knobs["extra_obs_stops"]:
                    stops += ["\nuser\n", "\n## Turn", "\n## Output"]
                token_out = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids + response_ids,
                    sampling_params={
                        **sampling_params,
                        "max_tokens": self._knobs["max_response_tokens_per_turn"],
                        "stop": stops,
                    },
                )
                assistant_ids = list(token_out.token_ids)
                response_ids += assistant_ids
                response_mask += [1] * len(assistant_ids)
                assistant_text = self.tokenizer.decode(
                    assistant_ids,
                    skip_special_tokens=False,
                )
                clean_text = assistant_text.split("<|im_end|>")[0]
                # Defensive: drop any hallucinated-observation tail that slipped past the
                # stop sequences (keeps the SFT-text path clean even if a marker varies).
                # Gated with extra_obs_stops: for RL (False) we keep the raw behavior so the
                # format reward can penalize it and the policy learns to stop on its own.
                if self._knobs["extra_obs_stops"]:
                    clean_text = _strip_hallucinated_observation(clean_text)
                messages.append({"role": "assistant", "content": clean_text})

                # Did this turn hit the per-turn token cap (clipped, no stop token)?
                if len(assistant_ids) > max_turn_tokens:
                    max_turn_tokens = len(assistant_ids)
                if (
                    len(assistant_ids) >= self._knobs["max_response_tokens_per_turn"]
                    and "<|im_end|>" not in assistant_text
                ):
                    turns_truncated += 1

                fences = parse_python_fences(clean_text)
                if len(fences) > 1:
                    multi_block_turns += 1
                if not fences:
                    code = None
                elif self._knobs["parse_first_fence"]:
                    code = fences[0]
                else:
                    code = fences[-1]
                if code is None:
                    parse_error_strikes += 1
                    observation = "[Error] No `python` code block found. Write a single ```python ... ``` block."
                    if parse_error_strikes >= self._knobs["parse_error_strikes_to_terminate"]:
                        await self._append_observation(
                            messages,
                            response_ids,
                            response_mask,
                            turn,
                            max_iter,
                            observation,
                        )
                        termination = "parse_error"
                        break
                else:
                    parse_error_strikes = 0
                    try:
                        result = await self.loop.run_in_executor(
                            None,
                            interp.execute,
                            code,
                        )
                    except (CodeInterpreterError, SyntaxError) as e:
                        result = f"[Error] {e}"

                    if isinstance(result, tuple) and isinstance(result[0], FinalOutput):
                        final, captured = result
                        submitted_answer = final.output.get("answer")
                        observation = (captured + "\n" if captured else "") + f"FINAL: {submitted_answer!r}"
                        await self._append_observation(
                            messages,
                            response_ids,
                            response_mask,
                            turn,
                            max_iter,
                            observation,
                        )
                        termination = "submit"
                        break
                    elif isinstance(result, FinalOutput):
                        submitted_answer = result.output.get("answer")
                        observation = f"FINAL: {submitted_answer!r}"
                        await self._append_observation(
                            messages,
                            response_ids,
                            response_mask,
                            turn,
                            max_iter,
                            observation,
                        )
                        termination = "submit"
                        break
                    elif isinstance(result, str) and result.startswith("[Error]"):
                        observation = result
                    elif isinstance(result, list):
                        observation = "\n".join(map(str, result))
                    elif result:
                        observation = str(result)
                    else:
                        observation = "(no output - did you forget to print?)"
                        empty_output_turns += 1

                    if "batch_look(" in code:
                        vlm_calls += code.count("batch_look(")

                    observation = self._truncate(observation)

                await self._append_observation(
                    messages,
                    response_ids,
                    response_mask,
                    turn,
                    max_iter,
                    observation,
                )

                if len(response_ids) >= self._response_length_cap - 256:
                    termination = "token_cap"
                    break
            else:
                termination = "iter_cap"

        finally:
            if interp is not None:
                try:
                    interp.shutdown()
                except Exception:
                    pass
            try:
                await vlm_client.aclose()
            except Exception:
                pass

        extra_fields = {
            "messages": messages,
            "termination": termination,
            "submitted_answer": submitted_answer,
            "num_turns": num_turns,
            "vlm_calls": vlm_calls,
            "turns_truncated": turns_truncated,
            "max_turn_tokens": max_turn_tokens,
            "multi_block_turns": multi_block_turns,
            "empty_output_turns": empty_output_turns,
            "wall_clock_s": time.monotonic() - wall_start,
            "doc_id": meta["doc_id"],
            "question_id": question_id,
            "category": category,
        }

        # Optional debug: append the full trajectory (incl. messages) to a JSONL file when
        # DOCVQA_TRAJ_DUMP is set. For inspecting RL rollouts (why a rollout struck out / what
        # it submitted). Best-effort: never let a dump error abort a rollout.
        dump_path = os.environ.get("DOCVQA_TRAJ_DUMP")
        if dump_path:
            try:
                with open(dump_path, "a") as _f:
                    _f.write(json.dumps(extra_fields, default=str) + "\n")
            except Exception:
                pass

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self._response_length_cap],
            response_mask=response_mask[: self._response_length_cap],
            num_turns=num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields=extra_fields,
        )

    async def _append_observation(
        self,
        messages: list[dict],
        response_ids: list[int],
        response_mask: list[int],
        turn: int,
        max_iter: int,
        output: str,
    ) -> None:
        text = build_observation_message(turn, max_iter, output)
        messages.append({"role": "user", "content": text})
        obs_ids = await self.apply_chat_template(
            [{"role": "user", "content": text}],
            remove_system_prompt=True,
        )
        # vLLM stops at <|im_end|> — the chat template separator newline
        # between the assistant turn and the next message is not part of the
        # model's output, and apply_chat_template(remove_system_prompt=True)
        # for a single user message starts directly at <|im_start|>. We
        # prepend it here so prompt_ids+response_ids is a faithful reconstruction
        # of `messages` rendered through the chat template.
        nl_token_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        obs_ids = list(nl_token_ids) + list(obs_ids)
        response_ids += obs_ids
        response_mask += [0] * len(obs_ids)

    def _truncate(self, text: str) -> str:
        cap = self._knobs["max_obs_chars"]
        if len(text) <= cap:
            return text
        return text[:cap] + "\n... (truncated)"
