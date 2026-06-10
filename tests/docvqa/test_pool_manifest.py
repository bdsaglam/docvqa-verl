import json
from pathlib import Path

import yaml


def test_manifest_entries_have_required_fields_and_valid_questions():
    man = yaml.safe_load(Path("docvqa/train/pool.yaml").read_text())
    assert man["datasets"], "manifest must list datasets"
    names = [d["name"] for d in man["datasets"]]
    assert len(names) == len(set(names)), "dataset names must be unique"
    for d in man["datasets"]:
        assert {"name", "questions", "category", "tier", "K"} <= set(d), d
        assert d["tier"] in {"easy", "mid", "hard"}, d
        assert isinstance(d["K"], int) and d["K"] > 0, d
        qp = Path(d["questions"])
        if qp.exists():  # prepared sets must be well-formed verl rows
            rows = json.loads(qp.read_text())
            assert rows, f"empty questions.json: {qp}"
            assert {"record_id", "question", "doc_dir", "answer", "reward_model"} <= set(rows[0]), rows[0].keys()
