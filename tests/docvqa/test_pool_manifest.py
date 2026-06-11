import json
from pathlib import Path

import yaml


def test_manifest_entries_have_required_fields_and_valid_questions():
    man = yaml.safe_load(Path("docvqa/train/pool.yaml").read_text())
    assert man["datasets"], "manifest must list datasets"
    names = [d["name"] for d in man["datasets"]]
    assert len(names) == len(set(names)), "dataset names must be unique"
    for d in man["datasets"]:
        assert {"name", "questions", "category", "tier", "n_sample"} <= set(d), d
        assert d["tier"] in {"easy", "mid", "hard"}, d
        assert isinstance(d["n_sample"], int) and d["n_sample"] > 0, d
        qp = Path(d["questions"])
        if qp.exists():  # prepared sets must be well-formed verl rows
            rows = json.loads(qp.read_text())
            assert rows, f"empty questions.json: {qp}"
            assert {"record_id", "question", "doc_dir", "answer", "reward_model"} <= set(rows[0]), rows[0].keys()


def test_sample_rows_caps_and_is_deterministic():
    from docvqa.scripts.sample_pool import _sample_rows

    rows = list(range(100))
    s1 = _sample_rows(rows, 10, 7)
    s2 = _sample_rows(rows, 10, 7)
    assert len(s1) == 10 and s1 == s2, "same seed must be reproducible"
    assert set(s1) <= set(rows)
    assert _sample_rows(rows, 200, 7) == rows, "n>=len returns all, order preserved"
    assert _sample_rows(rows, None, 7) == rows
