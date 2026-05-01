"""Inline Python injected into the subprocess at startup.

Reads DOC_DIR env var, loads `pages` (PIL.Image list) and `page_texts`
(list[str]). The IPC tool proxies (`batch_look`, `search`) and `SUBMIT`
are already in the REPL namespace before this code runs.

The host-side `batch_look` handler expects ``[{"path": <png path>, "query":
<text>}, ...]`` so it can read the bytes and POST to the VLM. PIL.Image
objects can't survive JSON IPC. So we override the user-facing `batch_look`
with a wrapper that dumps each image to a tempfile and forwards paths,
matching the prompt's documented signature
(`list[tuple[PIL.Image, str]]`) and the deployment-time agent.
"""
from __future__ import annotations

_TEMPLATE = '''
import os
import tempfile
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

# If the host registered a `batch_look` IPC proxy, rename and wrap it with
# PIL → tempfile-path conversion so the model can pass PIL.Images directly
# (matching the prompt and the deployment-time agent). When tests run the
# sandbox without tools, `batch_look` won't exist — leave the namespace alone.
if "batch_look" in dir():
    _batch_look_proxy = batch_look  # noqa: F821 — provided by interpreter

    def batch_look(requests):
        """Send multiple images to the VLM in parallel.
        Input: list of (PIL.Image, query) tuples.
        Returns: list of str answers (same order as input)."""
        payload = []
        for image, query in requests:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp, format="PNG")
            tmp.close()
            payload.append({"path": tmp.name, "query": query})
        return _batch_look_proxy(payload)

    def look(image, query):
        """Single-image VLM call. Convenience wrapper around batch_look."""
        return batch_look([(image, query)])[0]
'''


def build_sandbox_code() -> str:
    """Return the Python source string to inject at subprocess startup."""
    return _TEMPLATE
