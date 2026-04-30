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
