"""Shared pytest fixtures for docvqa tests."""
from pathlib import Path

import pytest


@pytest.fixture
def sample_doc_dir() -> Path:
    """Path to the committed sample doc_dir fixture."""
    return Path(__file__).parent / "fixtures" / "sample_doc"
