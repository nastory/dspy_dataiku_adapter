"""Unit tests for DataikuEmbedder — all Mesh calls are mocked."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from dspy_dku.embedder import DataikuEmbedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh() -> MagicMock:
    m = MagicMock()
    m.embed.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return m


@pytest.fixture
def embedder(mesh) -> DataikuEmbedder:
    emb = DataikuEmbedder(model="test-embedding-connection")
    emb._mesh = mesh
    return emb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDataikuEmbedder:
    def test_returns_list_of_float_vectors(self, embedder):
        result = embedder(["text one", "text two"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(v, list) for v in result)
        assert all(isinstance(x, float) for x in result[0])

    def test_vector_values_match_mock(self, embedder):
        result = embedder(["text one", "text two"])
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    def test_empty_input_returns_empty_list(self, embedder, mesh):
        result = embedder([])
        assert result == []
        mesh.embed.assert_not_called()

    def test_delegates_to_mesh_embed(self, embedder, mesh):
        embedder(["hello"])
        mesh.embed.assert_called_once_with(["hello"])

    def test_kwargs_accepted_without_error(self, embedder):
        # kwargs are accepted by the signature but not forwarded;
        # ensure no TypeError is raised
        result = embedder(["hello"], batch_size=32)
        assert isinstance(result, list)
