"""Inline Python injected into the subprocess at startup.

Reads ``DOC_DIR`` env var, loads ``pages`` (PIL.Image list). Mirrors the
``rvlm_minimal_solver`` sandbox in ``~/repos/docvqa`` — same variable name,
same tool surface (``batch_look`` only). The IPC tool proxy for ``batch_look``
and ``SUBMIT`` are already in the REPL namespace before this code runs.

The host-side ``batch_look`` handler expects ``[{"path": <png path>, "query":
<text>}, ...]`` so it can read the bytes and POST to the VLM. PIL.Image
objects can't survive JSON IPC. So we override the user-facing ``batch_look``
with a wrapper that dumps each image to a tempfile and forwards paths,
matching the prompt's documented signature
(``list[tuple[PIL.Image, str]]``) and the deployment-time agent.

Earlier revisions of this sandbox also exposed ``page_texts`` (OCR) and a
``search`` tool. The deployment-time ``rvlm_minimal`` scaffold dropped both —
recursive perception with the VLM is the load-bearing mechanism, and the
extra tools muddied the prompt. We follow suit to keep train-time and
deploy-time behavior aligned.
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

# If the host registered a `batch_look` IPC proxy, rename and wrap it with
# PIL → tempfile-path conversion so the model can pass PIL.Images directly
# (matching the prompt and the deployment-time agent). When tests run the
# sandbox without tools, `batch_look` won't exist — leave the namespace alone.
if "batch_look" in dir():
    _batch_look_proxy = batch_look  # noqa: F821 — provided by interpreter

    def batch_look(requests):
        """Send multiple images to the VLM in parallel. Much faster than
        sequential calls.
        Input: list of (PIL.Image, query) tuples. Returns: list of str
        answers (same order). Example:
            batch_look([(pages[0], "layout?"),
                        (pages[1].crop((0,0,500,500)), "read text")])"""
        payload = []
        paths = []
        for image, query in requests:
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(tmp, format="PNG")
            tmp.close()
            paths.append(tmp.name)
            payload.append({"path": tmp.name, "query": query})
        try:
            return _batch_look_proxy(payload)
        finally:
            # The proxy is synchronous — by the time it returns the answers the
            # host VLM has read every image, so the temp PNGs are safe to delete.
            # `delete=False` above is required so the host can open them by path;
            # without this cleanup they leak to /tmp (we once accumulated ~985G of
            # `tmp*.png` renders across collection/eval runs).
            for _p in paths:
                try:
                    os.remove(_p)
                except OSError:
                    pass
'''


def build_sandbox_code() -> str:
    """Return the Python source string to inject at subprocess startup."""
    return _TEMPLATE
