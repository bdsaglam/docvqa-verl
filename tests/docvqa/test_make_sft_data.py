import json
import pandas as pd
from docvqa.scripts.make_sft_data import filter_and_project


def _rollout(anls, qid, answer, n_msgs=4):
    return {
        "record_id": f"ds:val:{qid}:0", "question_id": qid,
        "messages": [{"role": "system", "content": "sys"},
                     {"role": "user", "content": "q"},
                     {"role": "assistant", "content": "<think>..</think>"},
                     {"role": "user", "content": "obs"}][:n_msgs],
        "submitted_answer": answer, "anls": anls,
        "termination": "submit", "num_turns": 1,
    }


def test_keeps_only_anls_pass():
    rows = [_rollout(1.0, "q1", "a"), _rollout(0.0, "q2", "b")]
    kept = filter_and_project(rows, max_per_question=None)
    assert len(kept) == 1
    assert kept[0]["messages"][0]["role"] == "system"


def test_caps_per_question():
    rows = [_rollout(1.0, "q1", "a") for _ in range(5)]
    kept = filter_and_project(rows, max_per_question=2)
    assert len(kept) == 2


def test_drops_empty_or_non_submit():
    rows = [{"question_id": "q3", "messages": [], "anls": 1.0,
             "termination": "submit", "submitted_answer": "x"}]
    assert filter_and_project(rows, max_per_question=None) == []


def test_output_is_messages_only(tmp_path):
    from docvqa.scripts.make_sft_data import write_parquet
    kept = filter_and_project([_rollout(1.0, "q1", "a")], max_per_question=None)
    out = tmp_path / "train.parquet"
    write_parquet(kept, out)
    df = pd.read_parquet(out)
    assert list(df.columns) == ["messages"]
    assert df.iloc[0]["messages"][0]["role"] == "system"
