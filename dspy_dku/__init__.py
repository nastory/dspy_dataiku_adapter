"""dspy_dku — DSPy adapter for Dataiku LLM Mesh.

Drop this package into your Dataiku DSS project's ``python-lib/`` directory,
then use it like any other DSPy LM:

    import dspy
    from dspy_dku import DataikuLM, DataikuEmbedder

    lm = DataikuLM(model="your-llm-connection-id")
    dspy.configure(lm=lm)
"""

from .lm import DataikuLM
from .embedder import DataikuEmbedder

__all__ = ["DataikuLM", "DataikuEmbedder"]
