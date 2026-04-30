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


@pytest.mark.asyncio
async def test_run_iter_cap_with_no_submit(tokenizer, sample_doc_dir):
    """Model emits a print but never SUBMITs; loop hits iter cap."""
    text = "<think>Let me check.</think>\n\n```python\nprint(page_texts[0][:50])\n```"
    assistant_ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)
    scripted = [assistant_ids] * 50

    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 3,
                                 "max_iterations_cap": 3}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager(scripted),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )

    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0",
        question="What was Q3 revenue?",
        doc_dir=str(sample_doc_dir),
        gold_answer="$1.2B",
        category="business_report",
    )
    assert out.extra_fields["termination"] == "iter_cap"
    assert out.extra_fields["submitted_answer"] is None
    assert out.extra_fields["num_turns"] == 3
    assert len(out.response_mask) == len(out.response_ids)
    assert sum(out.response_mask) > 0
    assert any(m == 0 for m in out.response_mask)


@pytest.mark.asyncio
async def test_run_records_messages_in_extra_fields(tokenizer, sample_doc_dir):
    text = "<think>OK.</think>\n```python\nprint('hi')\n```"
    assistant_ids = tokenizer.encode(text + "<|im_end|>", add_special_tokens=False)

    cfg = _make_trainer_config(actor_rollout_ref={
        "rollout": {
            "prompt_length": 16384, "response_length": 32768,
            "agent": {"agent_loop_config_path": None,
                      "docvqa": {"max_iterations_base": 2,
                                 "max_iterations_cap": 2}},
            "multi_turn": {},
            "trace": {"project_name": "t", "experiment_name": "t"},
        }, "model": {},
    })
    loop = DocVQAReplAgentLoop(
        trainer_config=DictConfigWrap(cfg),
        server_manager=_ScriptedServerManager([assistant_ids] * 5),
        tokenizer=tokenizer,
        processor=None, dataset_cls=MagicMock,
        data_config=DictConfigWrap(cfg.data),
    )
    out = await loop.run(
        sampling_params={"temperature": 1.0},
        question_id="q0", question="?", doc_dir=str(sample_doc_dir),
        gold_answer="x", category="business_report",
    )
    msgs = out.extra_fields["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and "Question" in msgs[1]["content"]
    assert msgs[2]["role"] == "assistant"
    assert msgs[3]["role"] == "user" and "Turn" in msgs[3]["content"]
