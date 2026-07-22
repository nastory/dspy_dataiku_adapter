"""Integration tests for DataikuLM against a live Dataiku DSS instance.

Prerequisites
-------------
1. Run inside (or with credentials for) a Dataiku DSS code environment.
2. Set environment variable DATAIKU_TEST_LLM_CONNECTION to the LLM connection
   ID you want to test against.

Running
-------
    pytest tests/integration/ -m integration -v

Design constraints (token-efficiency)
--------------------------------------
- max_tokens=10 on every LLM call.
- temperature=0 for deterministic, reproducible results.
- At most one live LLM call per test.
"""
from __future__ import annotations

import os

import dspy
import pytest

from dspy_dku import DataikuLM


LLM_CONNECTION = os.environ.get("DATAIKU_TEST_LLM_CONNECTION", "")

pytestmark = pytest.mark.integration


@pytest.mark.skipif(not LLM_CONNECTION, reason="DATAIKU_TEST_LLM_CONNECTION not set")
class TestDataikuLMIntegration:

    @pytest.fixture(scope="class")
    def lm(self):
        return DataikuLM(
            model=LLM_CONNECTION,
            temperature=0.0,
            max_tokens=10,
            cache=False,
        )

    def test_basic_completion_returns_non_empty_string(self, lm):
        result = lm(prompt="Reply with a single word: yes")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_messages_input_returns_non_empty_string(self, lm):
        result = lm(messages=[{"role": "user", "content": "Say: ok"}])
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert len(result[0]) > 0

    def test_history_entry_recorded(self, lm):
        before = len(lm.history)
        lm(prompt="Say: done")
        assert len(lm.history) == before + 1
        entry = lm.history[-1]
        assert "prompt" in entry
        assert "usage" in entry
        assert "prompt_tokens" in entry["usage"]

    def test_dspy_predict_end_to_end(self, lm):
        dspy.configure(lm=lm)

        class Minimal(dspy.Signature):
            """Minimal single-field signature for smoke-testing DSPy."""
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        predict = dspy.Predict(Minimal)
        result = predict(question="1+1=")
        assert hasattr(result, "answer")
        assert isinstance(result.answer, str)
