"""Root conftest.py — shared fixtures for all tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared Mesh response fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mesh_response() -> MagicMock:
    """Successful standard text-completion response."""
    resp = MagicMock()
    resp.success = True
    resp.text = "Hello, world!"
    resp.choices = [
        {
            "message": {"role": "assistant", "content": "Hello, world!"},
            "finish_reason": "stop",
            "index": 0,
        }
    ]
    resp.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    return resp


@pytest.fixture
def mock_mesh_multi_response() -> MagicMock:
    """Successful response with two choices (n=2)."""
    resp = MagicMock()
    resp.success = True
    resp.text = "A"
    resp.choices = [
        {"message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"},
        {"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"},
    ]
    resp.usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    return resp


@pytest.fixture
def mock_mesh_tool_response() -> MagicMock:
    """Completion response where the model invokes a tool."""
    resp = MagicMock()
    resp.success = True
    resp.text = None
    resp.choices = [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
            "index": 0,
        }
    ]
    resp.usage = {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35}
    return resp


@pytest.fixture
def mock_mesh_failure() -> MagicMock:
    """Failed Mesh response."""
    resp = MagicMock()
    resp.success = False
    resp.error_message = "Model connection not available"
    return resp


@pytest.fixture
def sample_messages() -> list[dict]:
    return [{"role": "user", "content": "What is 2+2?"}]
