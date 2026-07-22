from __future__ import annotations

import warnings
from typing import Any

# ---------------------------------------------------------------------------
# Parameter mappings
# ---------------------------------------------------------------------------

# DSPy / LiteLLM kwarg name  →  Dataiku Mesh settings key
_PARAM_MAP: dict[str, str] = {
    "temperature": "temperature",
    "max_tokens": "maxOutputTokens",
    "top_p": "topP",
    "stop": "stopSequences",
    "n": "n",
    "response_format": "response_format",
    # Passed through unchanged; extracted by _MeshClient, not put in settings
    "tools": "tools",
    "tool_choice": "tool_choice",
}

# Keys that go into completion.settings (not handled separately)
_SETTINGS_KEYS = frozenset({
    "temperature",
    "maxOutputTokens",
    "topP",
    "stopSequences",
    "n",
    "response_format",
})

# Keys extracted by _MeshClient and set as separate completion attributes
_SEPARATE_KEYS = frozenset({"tools", "tool_choice"})


class _Normalizer:
    """Pure-function utilities for translating between DSPy and Dataiku Mesh
    request / response shapes."""

    # ------------------------------------------------------------------
    # Request normalization
    # ------------------------------------------------------------------

    @staticmethod
    def to_mesh_settings(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Map DSPy / LiteLLM-style kwargs to Dataiku Mesh parameter names.

        - ``stream`` is silently dropped (caller handles it before this point).
        - ``tools`` and ``tool_choice`` are kept in the returned dict so that
          ``_MeshClient.complete`` can extract and set them separately.
        - Unknown parameters emit a ``UserWarning`` and are dropped.
        """
        settings: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key == "stream":
                # stream is handled at the DataikuLM.__call__ level
                continue
            if key in _PARAM_MAP:
                target = _PARAM_MAP[key]
                # stop can arrive as a bare string; Mesh expects list[str]
                if target == "stopSequences" and isinstance(value, str):
                    value = [value]
                settings[target] = value
            else:
                warnings.warn(
                    f"DataikuLM: unsupported parameter '{key}' will be ignored.",
                    UserWarning,
                    stacklevel=4,
                )
        return settings

    @staticmethod
    def prompt_to_messages(prompt: str) -> list[dict[str, str]]:
        """Wrap a plain-text prompt in a single user chat message."""
        return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------------
    # Response normalization — completions
    # ------------------------------------------------------------------

    @staticmethod
    def mesh_response_to_completions(resp: Any) -> list[str]:
        """Extract completion strings from all choices in a Mesh response.

        Falls back to ``resp.text`` if ``choices`` is absent or empty.
        """
        choices = getattr(resp, "choices", None)
        if not choices:
            text = getattr(resp, "text", "") or ""
            return [text]

        results: list[str] = []
        for choice in choices:
            if isinstance(choice, dict):
                content = (choice.get("message") or {}).get("content") or ""
            else:
                msg = getattr(choice, "message", {})
                if isinstance(msg, dict):
                    content = msg.get("content") or ""
                else:
                    content = getattr(msg, "content", "") or ""
            results.append(content)
        return results or [""]

    # ------------------------------------------------------------------
    # Response normalization — tool calls
    # ------------------------------------------------------------------

    @staticmethod
    def mesh_response_has_tool_calls(resp: Any) -> bool:
        """Return True when the first choice contains tool_calls."""
        choices = getattr(resp, "choices", None)
        if not choices:
            return False
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message") or {}
            tool_calls = message.get("tool_calls")
            finish_reason = first.get("finish_reason", "") or ""
        else:
            msg = getattr(first, "message", {})
            message = msg if isinstance(msg, dict) else {}
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            finish_reason = getattr(first, "finish_reason", "") or ""
        return bool(tool_calls) or finish_reason == "tool_calls"

    @staticmethod
    def mesh_response_to_tool_dicts(resp: Any) -> list[dict[str, Any]]:
        """Return the raw ``message`` dict for each choice (includes
        ``tool_calls`` when present)."""
        choices = getattr(resp, "choices", None) or []
        result: list[dict[str, Any]] = []
        for choice in choices:
            if isinstance(choice, dict):
                result.append(choice.get("message") or {})
            else:
                msg = getattr(choice, "message", {})
                result.append(msg if isinstance(msg, dict) else {})
        return result

    # ------------------------------------------------------------------
    # Response normalization — usage
    # ------------------------------------------------------------------

    @staticmethod
    def mesh_usage_to_dspy_usage(resp: Any) -> dict[str, int]:
        """Normalize Mesh token-usage data to snake_case DSPy keys.

        Handles both camelCase (``promptTokens``) and snake_case
        (``prompt_tokens``) field names returned by different Dataiku
        DSS versions / providers.
        """
        usage = getattr(resp, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def _get(obj: Any, snake: str, camel: str) -> int:
            if isinstance(obj, dict):
                val = obj.get(snake) or obj.get(camel, 0)
            else:
                val = getattr(obj, snake, None) or getattr(obj, camel, 0)
            return int(val or 0)

        return {
            "prompt_tokens": _get(usage, "prompt_tokens", "promptTokens"),
            "completion_tokens": _get(usage, "completion_tokens", "completionTokens"),
            "total_tokens": _get(usage, "total_tokens", "totalTokens"),
        }
