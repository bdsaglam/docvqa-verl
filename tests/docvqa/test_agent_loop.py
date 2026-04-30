"""End-to-end tests for DocVQAReplAgentLoop with scripted server_manager."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.agent_loop.agent_loop import DictConfigWrap

from docvqa.agent_loop import DocVQAReplAgentLoop


class _FakeTokenOutput:
    def __init__(self, token_ids, log_probs=None):
        self.token_ids = token_ids
        self.log_probs = log_probs
        self.num_preempted = 0
        self.extra_fields = {}
        self.routed_experts = None


class _ScriptedServerManager:
    """Returns scripted token id sequences for each generate() call."""
    def __init__(self, scripted_responses):
        self._responses = list(scripted_responses)

    async def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        if not self._responses:
            raise AssertionError("Out of scripted responses")
        return _FakeTokenOutput(self._responses.pop(0))


def _make_trainer_config(**overrides):
    cfg = OmegaConf.create({
        "actor_rollout_ref": {
            "rollout": {
                "prompt_length": 8192,
                "response_length": 8192,
                "agent": {"agent_loop_config_path": None},
                "multi_turn": {},
                "trace": {"project_name": "test", "experiment_name": "test"},
            },
            "model": {},
        },
        "data": {"apply_chat_template_kwargs": {"enable_thinking": True}},
    })
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("willcb/Qwen3-8B")


def test_agent_loop_instantiates(tokenizer, sample_doc_dir):
    cfg = _make_trainer_config()
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([]),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    assert loop.tokenizer is tokenizer
    assert loop._knobs["max_iterations_base"] == 20  # default
    assert loop._knobs["page_factor"] == 1.5
    assert loop._vlm_base_url == "http://localhost:8928"  # default


def test_agent_loop_uses_overrides(tokenizer):
    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 8192, "response_length": 8192,
            "agent": {
                "agent_loop_config_path": None,
                "docvqa": {
                    "max_iterations_base": 5,
                    "vlm_base_url": "http://other:9999",
                    "vlm_model_id": "fake-model",
                },
            },
            "multi_turn": {},
            "trace": {"project_name": "test", "experiment_name": "test"},
        },
        "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([]),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    assert loop._knobs["max_iterations_base"] == 5
    assert loop._vlm_base_url == "http://other:9999"
    assert loop._vlm_model_id == "fake-model"


def test_agent_loop_run_raises_not_implemented(tokenizer):
    """Run is filled in Task 10. For now it must raise."""
    import asyncio
    cfg = _make_trainer_config()
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([]),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    with pytest.raises(NotImplementedError):
        asyncio.get_event_loop().run_until_complete(
            loop.run(sampling_params={}, question="q", doc_dir="/tmp/x", category="x", gold_answer=None, question_id="q0")
        )
