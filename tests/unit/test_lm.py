"""Unit tests for DataikuLM — all Mesh calls are mocked."""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

import pytest

from dspy_dku.lm import DataikuLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lm(mesh_instance: MagicMock, **kwargs) -> DataikuLM:
    """Create a DataikuLM with its _mesh already replaced by a mock."""
    lm = DataikuLM(model="test-connection", cache=False, **kwargs)
    lm._mesh = mesh_instance
    return lm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mesh(mock_mesh_response) -> MagicMock:
    m = MagicMock()
    m.complete.return_value = mock_mesh_response
    return m


@pytest.fixture
def lm(mesh) -> DataikuLM:
    return _make_lm(mesh, temperature=0.0, max_tokens=64)


# ---------------------------------------------------------------------------
# Basic call behaviour
# ---------------------------------------------------------------------------


class TestDataikuLMCall:
    def test_returns_list_of_strings(self, lm, sample_messages):
        result = lm(messages=sample_messages)
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)
        assert result == ["Hello, world!"]

    def test_prompt_normalised_to_user_message(self, lm, mesh):
        lm(prompt="Say hi.")
        call_kwargs = mesh.complete.call_args.kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "Say hi."}]

    def test_messages_take_precedence_over_prompt(self, lm, mesh, sample_messages):
        lm(prompt="ignored", messages=sample_messages)
        call_kwargs = mesh.complete.call_args.kwargs
        assert call_kwargs["messages"] == sample_messages

    def test_raises_value_error_when_no_input(self, lm):
        with pytest.raises(ValueError, match="prompt.*messages"):
            lm()

    # ------------------------------------------------------------------
    # Parameter forwarding
    # ------------------------------------------------------------------

    def test_temperature_forwarded(self, mesh, sample_messages):
        lm = _make_lm(mesh)
        lm(messages=sample_messages, temperature=0.9)
        settings = mesh.complete.call_args.kwargs["settings"]
        assert settings["temperature"] == 0.9

    def test_max_tokens_mapped_to_maxOutputTokens(self, mesh, sample_messages):
        lm = _make_lm(mesh)
        lm(messages=sample_messages, max_tokens=128)
        settings = mesh.complete.call_args.kwargs["settings"]
        assert settings["maxOutputTokens"] == 128

    def test_per_call_overrides_instance_defaults(self, mesh, sample_messages):
        lm = _make_lm(mesh, temperature=0.0, max_tokens=64)
        lm(messages=sample_messages, temperature=0.5, max_tokens=256)
        settings = mesh.complete.call_args.kwargs["settings"]
        assert settings["temperature"] == 0.5
        assert settings["maxOutputTokens"] == 256

    # ------------------------------------------------------------------
    # Streaming fallback
    # ------------------------------------------------------------------

    def test_stream_true_warns_and_returns_completions(self, lm, sample_messages):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = lm(messages=sample_messages, stream=True)
        assert any("streaming is not supported" in str(w.message) for w in caught)
        assert isinstance(result, list)
        assert isinstance(result[0], str)

    def test_stream_not_forwarded_to_mesh(self, lm, mesh, sample_messages):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lm(messages=sample_messages, stream=True)
        settings = mesh.complete.call_args.kwargs["settings"]
        assert "stream" not in settings

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def test_mesh_failure_raises_runtime_error(self, mock_mesh_failure, sample_messages):
        mesh = MagicMock()
        mesh.complete.return_value = mock_mesh_failure
        lm = _make_lm(mesh)
        with pytest.raises(RuntimeError, match="Mesh call failed"):
            lm(messages=sample_messages)

    # ------------------------------------------------------------------
    # Tool calling
    # ------------------------------------------------------------------

    def test_tool_call_response_returns_list_of_dicts(
        self, mock_mesh_tool_response, sample_messages
    ):
        mesh = MagicMock()
        mesh.complete.return_value = mock_mesh_tool_response
        lm = _make_lm(mesh)
        result = lm(messages=sample_messages, tools=[{"type": "function"}])
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert "tool_calls" in result[0]

    def test_tools_forwarded_in_settings(self, mesh, sample_messages):
        lm = _make_lm(mesh)
        tools = [{"type": "function", "function": {"name": "my_fn"}}]
        lm(messages=sample_messages, tools=tools)
        settings = mesh.complete.call_args.kwargs["settings"]
        assert settings["tools"] == tools

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def test_history_appended_after_call(self, lm, sample_messages):
        lm(messages=sample_messages)
        assert len(lm.history) == 1

    def test_history_contains_usage(self, lm, sample_messages):
        lm(messages=sample_messages)
        entry = lm.history[-1]
        assert entry["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_history_accumulates_across_calls(self, lm, sample_messages):
        lm(messages=sample_messages)
        lm(messages=sample_messages)
        assert len(lm.history) == 2

    def test_history_contains_prompt(self, lm, sample_messages):
        lm(messages=sample_messages)
        assert lm.history[-1]["prompt"] == sample_messages
