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
