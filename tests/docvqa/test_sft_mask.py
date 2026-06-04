"""Regression test: MultiTurnSFTDataset on our CodeAct trajectory format.

Guards two correctness properties required before SeqKD training (Stage-0 plan):
  1. A trajectory that begins with a `system` turn (our agent_loop format) loads
     under the Qwen3.5 chat template — see the leading-system fallback in
     verl/utils/chat_template.py:apply_chat_template.
  2. The SFT loss mask is assistant-only: it covers the teacher's reasoning +
     ```python``` + SUBMIT turns and excludes system / user / tool-observation
     tokens.

Skips if the Qwen3.5-4B tokenizer is not available locally (no network in CI).
"""
import pandas as pd
import pytest
from omegaconf import OmegaConf

MODEL = "Qwen/Qwen3.5-4B"

# A 2-iteration CodeAct trajectory in our exact deployed format (system-led).
MESSAGES = [
    {"role": "system", "content": "You are a document-VQA agent in a Python REPL. "
     "Use batch_look(requests) and SUBMIT(answer=...)."},
    {"role": "user", "content": "Question: What was the last state visited?\nThe document has 3 pages."},
    {"role": "assistant", "content": "<think>Look at the pages for states.</think>\n"
     "```python\nres = batch_look([(0, 'List US states on this page.')])\nprint(res[0])\n```"},
    {"role": "user", "content": "[stdout]\nPage 0: Ohio, then Indiana. Ends in Indiana."},
    {"role": "assistant", "content": "<think>Last state is Indiana.</think>\n"
     "```python\nSUBMIT(answer='Indiana')\n```"},
]


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from verl.utils import hf_tokenizer
        return hf_tokenizer(MODEL)
    except Exception as e:  # tokenizer not cached / no network
        pytest.skip(f"Qwen3.5-4B tokenizer unavailable: {type(e).__name__}: {e}")


def _load_item(tokenizer, tmp_path, ignore_mismatch=True):
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
    pq = tmp_path / "traj.parquet"
    pd.DataFrame({"messages": [MESSAGES]}).to_parquet(pq)
    cfg = OmegaConf.create({
        "pad_mode": "no_padding",
        "truncation": "error",
        "max_length": 32768,
        "messages_key": "messages",
        "ignore_input_ids_mismatch": ignore_mismatch,
    })
    ds = MultiTurnSFTDataset(str(pq), tokenizer, cfg)
    return ds[0]


def test_system_led_trajectory_loads_and_masks_assistant_only(tokenizer, tmp_path):
    item = _load_item(tokenizer, tmp_path)
    ids, mask = item["input_ids"], item["loss_mask"]
    assert ids.shape == mask.shape
    assert int(mask.sum()) > 0, "loss mask is empty"

    loss_text = tokenizer.decode(ids[mask.bool()].tolist())
    # context (system/user/observation) must NOT be in the loss region
    assert "document-VQA agent" not in loss_text
    assert "Question:" not in loss_text
    assert "[stdout]" not in loss_text
    # the teacher's reasoning, code, and SUBMIT MUST be in the loss region
    assert "batch_look" in loss_text
    assert "SUBMIT(answer='Indiana')" in loss_text


def test_mismatch_flag_required_for_qwen_thinking(tokenizer, tmp_path):
    # Without ignore_input_ids_mismatch, Qwen's <think> per-turn vs whole-convo
    # tokenization differs and the dataset must raise rather than mis-mask.
    with pytest.raises(Exception):
        _load_item(tokenizer, tmp_path, ignore_mismatch=False)
