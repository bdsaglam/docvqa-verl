"""DocVQA REPL agent loop for verl.

Replicates the deployed `flat_solo` scaffold: persistent CPython subprocess
with batch_look/search/SUBMIT, model emits <think>...</think> + a single
```python ... ``` fence per turn, ANLS reward end-of-trajectory.

This file currently only implements the class skeleton. Tasks 10 and 11
fill in `run()`.
"""
from __future__ import annotations

import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase, AgentLoopOutput, register,
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
}


def _adaptive_max_iter(num_pages: int, knobs: dict) -> int:
    import math
    bonus = knobs["page_factor"] * math.sqrt(max(0, num_pages - 9))
    return min(knobs["max_iterations_cap"],
               knobs["max_iterations_base"] + int(bonus))


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
            agent_cfg.get("vlm_base_url")
            or os.environ.get("DOCVQA_VLM_BASE_URL")
            or "http://localhost:8928"
        )
        self._vlm_model_id: str = (
            agent_cfg.get("vlm_model_id")
            or os.environ.get("DOCVQA_VLM_MODEL_ID")
            or "qwen3.6-27b"
        )

        self._response_length_cap: int = self.rollout_config.response_length

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        # Implemented in Task 10. SUBMIT handling in Task 11.
        raise NotImplementedError("DocVQAReplAgentLoop.run will be filled in Task 10")
