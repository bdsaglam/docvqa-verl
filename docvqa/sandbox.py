"""Inline Python injected into the subprocess at startup.

Reads DOC_DIR env var, loads `pages` (PIL.Image list) and `page_texts`
(list[str]). The IPC tool proxies (batch_look, search) and SUBMIT are
already in the REPL namespace before this code runs.
"""
from __future__ import annotations

_TEMPLATE = '''
import os
from pathlib import Path
from PIL import Image
Image.MAX_IMAGE_PIXELS = 500_000_000

_doc_dir = Path(os.environ["DOC_DIR"])
pages = [
    Image.open(p)
    for p in sorted((_doc_dir / "pages").glob("page_*.png"),
                    key=lambda p: int(p.stem.split("_")[1]))
]
page_texts = [
    p.read_text()
    for p in sorted((_doc_dir / "ocr").glob("page_*.md"),
                    key=lambda p: int(p.stem.split("_")[1]))
]
'''


def build_sandbox_code() -> str:
    """Return the Python source string to inject at subprocess startup."""
    return _TEMPLATE
