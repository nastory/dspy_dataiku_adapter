"""Root conftest.py — shared fixtures for all tests."""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# dataiku stub
# ---------------------------------------------------------------------------
# dspy_dku._mesh imports `dataiku` at module level, so simply importing
# dspy_dku fails without it. The real `dataiku` package is only available
# inside a DSS code environment (or via a local .tar.gz — see requirements.txt)
# and unit tests mock all Mesh calls anyway, so install a minimal stub here
# when the real package isn't present. Integration tests (run inside DSS)
# import the real `dataiku` package first, so this is a no-op there.
try:
    import dataiku  # noqa: F401
except ImportError:
    _dataiku_stub = types.ModuleType("dataiku")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "dataiku is not installed; this stub only exists so dspy_dku can "
            "be imported for unit tests, which mock all Dataiku Mesh calls."
        )

    _dataiku_stub.api_client = _unavailable
    _dataiku_stub.default_project_key = _unavailable
    sys.modules["dataiku"] = _dataiku_stub


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
