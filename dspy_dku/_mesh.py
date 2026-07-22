from __future__ import annotations

from typing import Any

import dataiku


class _MeshClient:
    """Thin wrapper around the Dataiku LLM Mesh Python client.

    Lazy-initialises the Dataiku project / LLM handle on the first call so
    that the object can be constructed without a live DSS connection (useful
    for testing with mocks).
    """

    _EMBED_CHUNK_SIZE = 100  # max texts per embedding batch

    def __init__(self, model: str, project_key: str | None = None) -> None:
        self._model = model
        self._project_key = project_key
        self._llm: Any = None  # lazily set by _get_llm()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        if self._llm is None:
            client = dataiku.api_client()
            key = self._project_key or dataiku.default_project_key()
            project = client.get_project(key)
            self._llm = project.get_llm(self._model)
        return self._llm

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        settings: dict[str, Any],
    ) -> Any:
        """Execute a chat completion via Dataiku LLM Mesh.

        Parameters
        ----------
        messages:
            Chat message list in OpenAI / LiteLLM role-content format.
        settings:
            Translated Mesh settings dict (from ``_Normalizer.to_mesh_settings``).
            ``tools`` and ``tool_choice`` are extracted and set as separate
            completion attributes; the remainder goes to ``completion.settings``.
        """
        llm = self._get_llm()
        completion = llm.new_completion()
        # DSSLLMCompletionQuery stores messages in completion.cq["messages"];
        # assigning to completion.messages would just set a stray attribute.
        completion.cq["messages"] = list(messages)

        # Work on a copy so we don't mutate the caller's dict
        settings = dict(settings)
        tools = settings.pop("tools", None)
        tool_choice = settings.pop("tool_choice", None)

        if settings:
            completion.settings.update(settings)
        if tools is not None:
            completion.tools = tools
        if tool_choice is not None:
            completion.tool_choice = tool_choice

        resp = completion.execute()
        if not resp.success:
            error = resp._raw.get("errorMessage") or repr(resp._raw)
            raise RuntimeError(
                f"Dataiku LLM Mesh call failed "
                f"(connection='{self._model}'): {error}"
            )
        return resp

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def embed(self, inputs: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching into chunks of
        ``_EMBED_CHUNK_SIZE`` to respect Mesh limits.

        Parameters
        ----------
        inputs:
            List of texts to embed.

        Returns
        -------
        list[list[float]]
            One float vector per input string, in the same order.

        Raises
        ------
        RuntimeError
            If the Mesh call fails.
        NotImplementedError
            If the connection does not support embeddings (no ``embeddings``
            field on the response).
        """
        llm = self._get_llm()
        all_vectors: list[list[float]] = []

        for i in range(0, len(inputs), self._EMBED_CHUNK_SIZE):
            chunk = inputs[i : i + self._EMBED_CHUNK_SIZE]
            embedding = llm.new_embedding()
            # NOTE: Dataiku DSS 14 uses `embedding.inputs` for the text list.
            # Verify this attribute name against your installed DSS version.
            embedding.inputs = chunk
            resp = embedding.execute()

            if not resp.success:
                error = getattr(resp, "error_message", None) or str(resp)
                raise RuntimeError(
                    f"DataikuLM: embedding call failed "
                    f"(connection='{self._model}'): {error}"
                )

            embeddings = getattr(resp, "embeddings", None)
            if embeddings is None:
                raise NotImplementedError(
                    f"DataikuLM: connection '{self._model}' does not appear to "
                    "support embeddings. Ensure the LLM connection is an "
                    "embedding model in Dataiku DSS."
                )

            all_vectors.extend(embeddings)

        return all_vectors
