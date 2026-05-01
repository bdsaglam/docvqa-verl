"""Tests that the sandbox preloads pages and page_texts correctly."""
from docvqa.sandbox import build_sandbox_code
from docvqa.subprocess_interp import SubprocessInterpreter


def test_sandbox_loads_pages_and_page_texts(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print(len(pages), len(page_texts))")
        assert "2 2" in out
    finally:
        interp.shutdown()


def test_sandbox_page_text_content(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print('Q3' in page_texts[0])")
        assert "True" in out
    finally:
        interp.shutdown()


def test_sandbox_pages_are_pil_images(sample_doc_dir):
    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute("print(pages[0].size)")
        assert "(100, 100)" in out
    finally:
        interp.shutdown()


def test_batch_look_wraps_pil_images_to_paths(sample_doc_dir):
    """The user-facing batch_look should accept (PIL.Image, query) tuples
    and pass {path, query} dicts to the host handler — matching the prompt's
    documented signature and the deployment-time agent."""
    captured: list = []
    from PIL import Image

    def _fake_host_batch_look(requests):
        captured.append(requests)
        # Confirm each entry has a real path on disk we can re-open as an image.
        return [f"saw {Image.open(r['path']).size} for {r['query']}" for r in requests]

    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        tools={"batch_look": _fake_host_batch_look, "search": lambda *_a, **_kw: []},
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute(
            'print(batch_look([(pages[0], "what page?"), (pages[1], "any text?")]))'
        )
    finally:
        interp.shutdown()

    assert len(captured) == 1
    payload = captured[0]
    assert len(payload) == 2
    assert {p["query"] for p in payload} == {"what page?", "any text?"}
    for entry in payload:
        assert "path" in entry and entry["path"].endswith(".png")
    # And the model sees the host's response on stdout.
    assert "saw (100, 100) for what page?" in out
    assert "saw (100, 100) for any text?" in out


def test_sandbox_look_single_image_convenience(sample_doc_dir):
    """`look(image, query)` should return a single string from batch_look."""
    def _fake_host_batch_look(requests):
        return ["one answer"] * len(requests)

    interp = SubprocessInterpreter(
        sandbox_code=build_sandbox_code(),
        tools={"batch_look": _fake_host_batch_look, "search": lambda *_a, **_kw: []},
        extra_env={"DOC_DIR": str(sample_doc_dir)},
    )
    try:
        out = interp.execute('print(look(pages[0], "describe"))')
    finally:
        interp.shutdown()
    assert "one answer" in out
