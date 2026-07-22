from __future__ import annotations

from typing import Any

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore[assignment]


class Message(TypedDict):
    role: str
    content: str


class CompletionSettings(TypedDict, total=False):
    temperature: float
    maxOutputTokens: int
    topP: float
    stopSequences: list[str]
    n: int
    response_format: dict[str, Any]
    # tools and tool_choice are extracted and set separately on the
    # completion object in _MeshClient.complete()
    tools: list[dict[str, Any]]
    tool_choice: Any


class HistoryEntry(TypedDict):
    prompt: list[dict[str, Any]]
    response: Any
    kwargs: dict[str, Any]
    usage: dict[str, int]
