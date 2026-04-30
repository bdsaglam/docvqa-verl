"""Tests for the host-side tool handlers."""
import pytest

from docvqa.tools import search, clear_bm25_cache


@pytest.fixture(autouse=True)
def _reset_bm25_cache():
    clear_bm25_cache()
    yield
    clear_bm25_cache()


def test_search_returns_relevant_page(sample_doc_dir):
    results = search(str(sample_doc_dir), "Q3 revenue", k=5)
    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert top["page"] == 0
    assert top["score"] > 0
    assert "Q3" in top["text"] or "revenue" in top["text"].lower()


def test_search_filters_zero_scores(sample_doc_dir):
    # A nonsense query should return either empty or only positive-score hits.
    results = search(str(sample_doc_dir), "zzzzzz nonsense token unlikely", k=5)
    for r in results:
        assert r["score"] > 0


def test_search_caches_index(sample_doc_dir):
    """Second call should not rebuild the retriever (process-local cache)."""
    from docvqa import tools
    search(str(sample_doc_dir), "Q3", k=3)
    assert str(sample_doc_dir) in tools._BM25_CACHE
    n_before = id(tools._BM25_CACHE[str(sample_doc_dir)])
    search(str(sample_doc_dir), "different", k=3)
    n_after = id(tools._BM25_CACHE[str(sample_doc_dir)])
    assert n_before == n_after  # same cached object


# ---------------------------------------------------------------------------
# batch_look tests
# ---------------------------------------------------------------------------
import base64

import httpx
import pytest

from docvqa.tools import batch_look


def _mock_transport(responses: list[str]) -> httpx.MockTransport:
    """Round-robin through canned VLM completion responses."""
    counter = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = counter["i"] % len(responses)
        counter["i"] += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": responses[i]}}]},
        )

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_batch_look_round_trip(tmp_path):
    from PIL import Image
    p0 = tmp_path / "a.png"; Image.new("RGB", (10, 10), "red").save(p0)
    p1 = tmp_path / "b.png"; Image.new("RGB", (10, 10), "blue").save(p1)

    transport = _mock_transport(["red answer", "blue answer"])
    async with httpx.AsyncClient(transport=transport, timeout=5) as client:
        res = await batch_look(
            requests=[{"path": str(p0), "query": "color?"},
                      {"path": str(p1), "query": "color?"}],
            client=client,
            base_url="http://mock",
            model_id="vlm",
        )
    assert res == ["red answer", "blue answer"]


@pytest.mark.asyncio
async def test_batch_look_handles_http_error(tmp_path):
    from PIL import Image
    p = tmp_path / "a.png"; Image.new("RGB", (10, 10), "red").save(p)

    def fail_handler(_req):
        return httpx.Response(500, text="boom")

    transport = httpx.MockTransport(fail_handler)
    async with httpx.AsyncClient(transport=transport, timeout=5) as client:
        res = await batch_look(
            requests=[{"path": str(p), "query": "?"}],
            client=client, base_url="http://mock", model_id="vlm",
        )
    assert len(res) == 1
    assert res[0].startswith("[VLM error")
