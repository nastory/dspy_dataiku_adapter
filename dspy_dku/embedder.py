from __future__ import annotations

from typing import Any

from ._mesh import _MeshClient


class DataikuEmbedder:
    """DSPy-compatible embedder backed by Dataiku LLM Mesh.

    Drop-in replacement for ``dspy.Embedder`` in retrieval and RAG workflows.
    Automatically batches large input lists into chunks of 100 to respect
    Dataiku Mesh limits.

    Parameters
    ----------
    model:
        The Dataiku embedding connection ID as configured in DSS
        (e.g. ``"openai-text-embedding-3-small"``).
    project_key:
        DSS project key. If ``None``, auto-detected via
        ``dataiku.default_project_key()`` at call time.

    Examples
    --------
    >>> from dspy_dku import DataikuEmbedder
    >>> embedder = DataikuEmbedder(model="openai-text-embedding-3-small")
    >>> vectors = embedder(["hello world", "dspy is great"])
    >>> len(vectors)  # 2
    2
    """

    def __init__(
        self,
        model: str,
        project_key: str | None = None,
    ) -> None:
        self._model = model
        self._mesh = _MeshClient(model=model, project_key=project_key)

    def __call__(
        self,
        inputs: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Embed a list of texts.

        Parameters
        ----------
        inputs:
            Texts to embed. An empty list returns ``[]`` immediately without
            making a network call.

        Returns
        -------
        list[list[float]]
            One float vector per input string, in the same order.

        Raises
        ------
        RuntimeError
            If the Dataiku Mesh embedding call fails.
        NotImplementedError
            If the configured connection does not support embeddings.
        """
        if not inputs:
            return []
        return self._mesh.embed(inputs)
