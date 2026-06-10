import json
from pathlib import Path

from PIL import Image

from docvqa.scripts.prepare_data import _materialize_single_image_doc


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
