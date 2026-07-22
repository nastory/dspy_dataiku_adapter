"""Integration tests for DataikuEmbedder against a live Dataiku DSS instance.

Prerequisites
-------------
1. Run inside (or with credentials for) a Dataiku DSS code environment.
2. Set environment variable DATAIKU_TEST_EMBED_CONNECTION to the embedding
   connection ID you want to test against.

Running
-------
    pytest tests/integration/ -m integration -v
"""
from __future__ import annotations

import os

import pytest

from dspy_dku import DataikuEmbedder


EMBED_CONNECTION = os.environ.get("DATAIKU_TEST_EMBED_CONNECTION", "")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not EMBED_CONNECTION, reason="DATAIKU_TEST_EMBED_CONNECTION not set")
class TestDataikuEmbedderIntegration:

    @pytest.fixture(scope="class")
    def embedder(self):
        return DataikuEmbedder(model=EMBED_CONNECTION)

    def test_single_text_returns_float_vector(self, embedder):
        result = embedder(["test sentence"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) > 0
        assert all(isinstance(x, float) for x in result[0])

    def test_multiple_texts_return_matching_count(self, embedder):
        texts = ["first", "second", "third"]
        result = embedder(texts)
        assert len(result) == len(texts)
        assert all(len(v) == len(result[0]) for v in result)

    def test_empty_input_returns_empty_list(self, embedder):
        result = embedder([])
        assert result == []
