import json
from pathlib import Path

from PIL import Image

from docvqa.scripts.prepare_data import _materialize_single_image_doc
from docvqa.scripts.prepare_data import _gold_answer_str


def test_single_image_doc_saves_one_page_and_metadata(tmp_path):
    docs_dir = tmp_path / "docs"
    img = Image.new("RGB", (8, 8), "white")
    doc_dir = _materialize_single_image_doc(
        doc_id="d1", image=img, category="maps",
        dataset="mapqa", split="train", docs_dir=docs_dir,
    )
    assert doc_dir == (docs_dir / "d1").resolve()
    assert (docs_dir / "d1" / "pages" / "page_0.png").exists()
    meta = json.loads((docs_dir / "d1" / "metadata.json").read_text())
    assert meta["num_pages"] == 1
    assert meta["doc_category"] == "maps"


def test_single_image_doc_is_idempotent(tmp_path):
    docs_dir = tmp_path / "docs"
    img = Image.new("RGB", (8, 8), "white")
    _materialize_single_image_doc(doc_id="d1", image=img, category="maps",
                                  dataset="mapqa", split="train", docs_dir=docs_dir)
    _materialize_single_image_doc(doc_id="d1", image=img, category="maps",
                                  dataset="mapqa", split="train", docs_dir=docs_dir)
    assert list((docs_dir / "d1" / "pages").glob("*.png")) == [docs_dir / "d1" / "pages" / "page_0.png"]


def test_gold_answer_single_vs_multi():
    assert _gold_answer_str(["paris"]) == "paris"
    assert _gold_answer_str(["paris", "Paris"]) == repr(["paris", "Paris"])
    assert _gold_answer_str([]) is None
    assert _gold_answer_str("paris") == "paris"


def test_chartqa_label_to_answer():
    # ChartQA label is a list; _gold_answer_str over a 1-element list returns the bare string
    assert _gold_answer_str(["52.0"]) == "52.0"


def test_mapqa_parser_extracts_question_and_answer():
    # mm_mapqa `data` is a list of turn dicts with keys {data, modality, role}:
    # a leading image turn, then interleaved text user(question)/assistant(answer)
    # turns. Multiple Q/A pairs per row -> parser returns ALL pairs.
    from docvqa.scripts.prepare_data import _mapqa_qa_from_data

    sample = [
        {"data": "0", "modality": "image", "role": "user"},
        {"data": "Which state has the highest value?\nShort answer required.",
         "modality": "text", "role": "user"},
        {"data": "Texas.", "modality": "text", "role": "assistant"},
        {"data": "Is Ohio higher than Iowa?\nBe succinct.",
         "modality": "text", "role": "user"},
        {"data": "No.", "modality": "text", "role": "assistant"},
    ]
    pairs = _mapqa_qa_from_data(sample)
    assert pairs == [
        ("Which state has the highest value?\nShort answer required.", "Texas."),
        ("Is Ohio higher than Iowa?\nBe succinct.", "No."),
    ]


def test_page_count_handles_list_and_stringified():
    from docvqa.scripts.prepare_data import _page_count

    assert _page_count(["p0", "p1", "p2"]) == 3
    assert _page_count("['p0', 'p1']") == 2
    assert _page_count(None) == 0


def test_tatdqa_answer_str_handles_list_scalar_and_scale():
    # TAT-DQA answers are heterogeneous: list[str] (span/multi-span), int/float
    # (numeric reasoning), or str. `scale` (e.g. "percent") is folded into gold.
    from docvqa.scripts.prepare_data import _tatdqa_answer_str

    # single-span list -> bare string
    assert _tatdqa_answer_str(["55%"], "") == "55%"
    # multi-span list -> repr(list) for the ast.literal_eval scorer
    assert _tatdqa_answer_str(["55%", "50%"], "") == repr(["55%", "50%"])
    # scalar numeric + scale -> "<value> <scale>"
    assert _tatdqa_answer_str(105, "percent") == "105 percent"
    assert _tatdqa_answer_str(2.5, "thousand") == "2.5 thousand"
    # scalar numeric, no scale
    assert _tatdqa_answer_str(7, "") == "7"
    # plain string passthrough
    assert _tatdqa_answer_str("net income", "") == "net income"
    # scale applied to each list element
    assert _tatdqa_answer_str(["1.0", "2.0"], "million") == repr(["1.0 million", "2.0 million"])
    # empties -> None
    assert _tatdqa_answer_str([], "") is None
    assert _tatdqa_answer_str("", "") is None
    assert _tatdqa_answer_str(None, "") is None
