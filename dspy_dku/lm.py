from __future__ import annotations

import asyncio
import warnings
from typing import Any

import dspy

from ._mesh import _MeshClient
from ._normalizer import _Normalizer
from ._types import HistoryEntry


class DataikuLM(dspy.LM):
    """DSPy LM adapter that routes all inference through Dataiku LLM Mesh.

    Extends ``dspy.LM`` so that every DSPy feature that works through the
    standard LM interface (teleprompting, optimizers, agents, tool-calling,
    structured outputs) works unchanged — only the transport layer is
    replaced with Dataiku LLM Mesh.

    Parameters
    ----------
    model:
        The Dataiku LLM connection ID as configured in DSS
        (e.g. ``"openai-gpt4o"``, ``"my-azure-gpt35"``).
    project_key:
        DSS project key. If ``None``, auto-detected via
        ``dataiku.default_project_key()`` at call time.
    temperature:
        Default sampling temperature. Defaults to ``0.0``.
    max_tokens:
        Default maximum output tokens. Defaults to ``1024``.
    cache:
        Whether to enable DSPy's built-in LM call cache. Defaults to ``True``.
    **kwargs:
        Additional default parameters forwarded to every completion call
        (e.g. ``top_p``, ``stop``).

    Examples
    --------
    >>> import dspy
    >>> from dspy_dku import DataikuLM
    >>> lm = DataikuLM(model="openai-gpt4o", temperature=0.1, max_tokens=512)
    >>> dspy.configure(lm=lm)
    >>> # All standard DSPy usage works unchanged from here.
    """

    def __init__(
        self,
        model: str,
        project_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache: bool = True,
        **kwargs: Any,
    ) -> None:
        # Initialise DSPy's LM base to set up history, cache, and kwargs.
        # The parent __call__ is NEVER invoked; we override it completely.
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            **kwargs,
        )
        self._project_key = project_key
        self._mesh = _MeshClient(model=model, project_key=project_key)

    # ------------------------------------------------------------------
    # Primary inference entrypoint (overrides dspy.LM.__call__)
    # ------------------------------------------------------------------

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        """Execute a completion via Dataiku LLM Mesh.

        Parameters
        ----------
        prompt:
            Plain-text prompt. Converted to a single ``user`` message
            if ``messages`` is not provided.
        messages:
            Chat-format message list (OpenAI role/content dicts).
            Takes precedence over ``prompt``.
        **kwargs:
            Per-call parameter overrides. Supported keys:

            ==================  ====================================
            DSPy / LiteLLM key  Dataiku Mesh equivalent
            ==================  ====================================
            temperature         temperature
            max_tokens          maxOutputTokens
            top_p               topP
            stop                stopSequences (str → list auto)
            n                   n
            response_format     response_format
            tools               completion.tools (separate attr)
            tool_choice         completion.tool_choice (separate)
            stream              *not supported; warns + falls back*
            ==================  ====================================

        Returns
        -------
        list[str]
            One completion string per choice for standard generation.
        list[dict]
            Raw ``message`` dicts (including ``tool_calls``) when the
            model invokes a tool/function.

        Raises
        ------
        ValueError
            If neither ``prompt`` nor ``messages`` is provided.
        RuntimeError
            If Dataiku LLM Mesh returns a failure response.
        """
        if prompt is None and messages is None:
            raise ValueError(
                "DataikuLM: either 'prompt' or 'messages' must be provided."
            )

        if messages is None:
            messages = _Normalizer.prompt_to_messages(prompt)  # type: ignore[arg-type]

        # Merge: instance defaults → call overrides
        # self.kwargs already holds temperature, max_tokens, and any extra defaults
        merged: dict[str, Any] = dict(self.kwargs)
        merged.update(kwargs)

        # Streaming: not supported; fall back with a warning
        if merged.pop("stream", False):
            warnings.warn(
                "DataikuLM: streaming is not supported in this version; "
                "falling back to standard (non-streaming) completion.",
                UserWarning,
                stacklevel=2,
            )

        mesh_settings = _Normalizer.to_mesh_settings(merged)
        resp = self._mesh.complete(messages=messages, settings=mesh_settings)

        usage = _Normalizer.mesh_usage_to_dspy_usage(resp)
        self.history.append(
            HistoryEntry(
                prompt=messages,
                response=resp,
                kwargs=merged,
                usage=usage,
            )
        )

        if _Normalizer.mesh_response_has_tool_calls(resp):
            return _Normalizer.mesh_response_to_tool_dicts(resp)

        return _Normalizer.mesh_response_to_completions(resp)

    # ------------------------------------------------------------------
    # Async wrapper (nice-to-have — wraps sync call in a thread)
    # ------------------------------------------------------------------

    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[str] | list[dict[str, Any]]:
        """Async variant of ``__call__``.

        Offloads the synchronous Dataiku Mesh call to a thread pool so it
        can be awaited without blocking the event loop.
        """
        return await asyncio.to_thread(
            self.__call__,
            prompt=prompt,
            messages=messages,
            **kwargs,
        )
