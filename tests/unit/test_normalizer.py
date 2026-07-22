"""Unit tests for _Normalizer — pure logic, no mocks needed."""
from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import pytest

from dspy_dku._normalizer import _Normalizer


# ---------------------------------------------------------------------------
# to_mesh_settings
# ---------------------------------------------------------------------------


class TestToMeshSettings:
    def test_max_tokens_mapped(self):
        assert _Normalizer.to_mesh_settings({"max_tokens": 512}) == {"maxOutputTokens": 512}

    def test_top_p_mapped(self):
        assert _Normalizer.to_mesh_settings({"top_p": 0.9}) == {"topP": 0.9}

    def test_stop_string_converted_to_list(self):
        assert _Normalizer.to_mesh_settings({"stop": "END"}) == {"stopSequences": ["END"]}

    def test_stop_list_preserved(self):
        assert _Normalizer.to_mesh_settings({"stop": ["END", "STOP"]}) == {
            "stopSequences": ["END", "STOP"]
        }

    def test_temperature_direct(self):
        assert _Normalizer.to_mesh_settings({"temperature": 0.7}) == {"temperature": 0.7}

    def test_n_direct(self):
        assert _Normalizer.to_mesh_settings({"n": 3}) == {"n": 3}

    def test_tools_passed_through(self):
        tools = [{"type": "function", "function": {"name": "fn"}}]
        result = _Normalizer.to_mesh_settings({"tools": tools})
        assert result["tools"] == tools

    def test_tool_choice_passed_through(self):
        result = _Normalizer.to_mesh_settings({"tool_choice": "auto"})
        assert result["tool_choice"] == "auto"

    def test_stream_silently_dropped(self):
        result = _Normalizer.to_mesh_settings({"stream": True, "temperature": 0.5})
        assert "stream" not in result
        assert result["temperature"] == 0.5

    def test_unknown_param_emits_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _Normalizer.to_mesh_settings({"unknown_param": "value"})
        assert any("unknown_param" in str(w.message) for w in caught)

    def test_unknown_param_not_in_result(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _Normalizer.to_mesh_settings({"unknown_param": "value"})
        assert "unknown_param" not in result

    def test_multiple_params_combined(self):
        result = _Normalizer.to_mesh_settings(
            {"temperature": 0.2, "max_tokens": 256, "top_p": 0.95, "stop": "DONE"}
        )
        assert result == {
            "temperature": 0.2,
            "maxOutputTokens": 256,
            "topP": 0.95,
            "stopSequences": ["DONE"],
        }


# ---------------------------------------------------------------------------
# prompt_to_messages
# ---------------------------------------------------------------------------


class TestPromptToMessages:
    def test_wraps_as_single_user_message(self):
        result = _Normalizer.prompt_to_messages("Hello")
        assert result == [{"role": "user", "content": "Hello"}]

    def test_empty_string(self):
        result = _Normalizer.prompt_to_messages("")
        assert result == [{"role": "user", "content": ""}]


# ---------------------------------------------------------------------------
# mesh_response_to_completions
# ---------------------------------------------------------------------------


class TestMeshResponseToCompletions:
    def test_extracts_content_from_choices(self, mock_mesh_response):
        result = _Normalizer.mesh_response_to_completions(mock_mesh_response)
        assert result == ["Hello, world!"]

    def test_multiple_choices(self, mock_mesh_multi_response):
        result = _Normalizer.mesh_response_to_completions(mock_mesh_multi_response)
        assert result == ["A", "B"]

    def test_fallback_to_text_when_no_choices(self):
        resp = MagicMock()
        resp.choices = []
        resp.text = "fallback"
        assert _Normalizer.mesh_response_to_completions(resp) == ["fallback"]

    def test_none_content_becomes_empty_string(self):
        resp = MagicMock()
        resp.choices = [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}]
        result = _Normalizer.mesh_response_to_completions(resp)
        assert result == [""]


# ---------------------------------------------------------------------------
# mesh_response_has_tool_calls
# ---------------------------------------------------------------------------


class TestMeshResponseHasToolCalls:
    def test_detects_tool_calls_field(self, mock_mesh_tool_response):
        assert _Normalizer.mesh_response_has_tool_calls(mock_mesh_tool_response) is True

    def test_no_tool_calls_returns_false(self, mock_mesh_response):
        assert _Normalizer.mesh_response_has_tool_calls(mock_mesh_response) is False

    def test_detects_by_finish_reason(self):
        resp = MagicMock()
        resp.choices = [
            {"message": {"role": "assistant", "content": None}, "finish_reason": "tool_calls"}
        ]
        assert _Normalizer.mesh_response_has_tool_calls(resp) is True

    def test_empty_choices_returns_false(self):
        resp = MagicMock()
        resp.choices = []
        assert _Normalizer.mesh_response_has_tool_calls(resp) is False


# ---------------------------------------------------------------------------
# mesh_response_to_tool_dicts
# ---------------------------------------------------------------------------


class TestMeshResponseToToolDicts:
    def test_returns_message_dicts(self, mock_mesh_tool_response):
        result = _Normalizer.mesh_response_to_tool_dicts(mock_mesh_tool_response)
        assert len(result) == 1
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_empty_choices_returns_empty_list(self):
        resp = MagicMock()
        resp.choices = []
        assert _Normalizer.mesh_response_to_tool_dicts(resp) == []


# ---------------------------------------------------------------------------
# mesh_usage_to_dspy_usage
# ---------------------------------------------------------------------------


class TestMeshUsageToDspyUsage:
    def test_snake_case_keys(self):
        resp = MagicMock()
        resp.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        assert _Normalizer.mesh_usage_to_dspy_usage(resp) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_camel_case_keys(self):
        resp = MagicMock()
        resp.usage = {"promptTokens": 10, "completionTokens": 5, "totalTokens": 15}
        assert _Normalizer.mesh_usage_to_dspy_usage(resp) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_none_usage_returns_zeros(self):
        resp = MagicMock()
        resp.usage = None
        assert _Normalizer.mesh_usage_to_dspy_usage(resp) == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
