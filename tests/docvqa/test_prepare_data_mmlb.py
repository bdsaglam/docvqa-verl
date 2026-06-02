from docvqa.scripts.prepare_data import _mmlb_question_id, _mmlb_rows_for_doc


def test_question_id_is_stable_and_indexed():
    assert _mmlb_question_id("COSTCO_2021_10K.pdf", 3) == "COSTCO_2021_10K.pdf::q3"


def test_rows_for_doc_builds_canonical_rows(tmp_path):
    hf_rows = [
        {"doc_id": "A.pdf", "question": "Q1", "answer": "1", "doc_type": "Financial report"},
        {"doc_id": "A.pdf", "question": "Q2", "answer": "Not answerable", "doc_type": "Financial report"},
    ]
    doc_dir = tmp_path / "docs" / "A.pdf"
    rows = _mmlb_rows_for_doc("train", hf_rows, doc_dir)
    assert len(rows) == 2
    assert rows[0]["question_id"] == "A.pdf::q0"
    assert rows[0]["category"] == "Financial report"
    assert rows[0]["dataset"] == "mmlongbench-doc"
    assert rows[1]["answer"] == "Not answerable"
    assert rows[0]["doc_dir"] == str(doc_dir)
    assert rows[0]["record_id"] == "mmlongbench-doc:train:A.pdf:A.pdf::q0"
