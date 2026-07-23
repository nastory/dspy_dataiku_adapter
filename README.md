# dspy_dku

A [DSPy](https://github.com/stanfordnlp/dspy) adapter that routes all LLM and embedding calls through [Dataiku LLM Mesh](https://developer.dataiku.com/latest/concepts-and-examples/llm-mesh.html), with no changes required to DSPy or the Dataiku platform.

## Overview

`dspy_dku` provides two classes:

| Class | Purpose |
|---|---|
| `DataikuLM` | Extends `dspy.LM`. Drop-in replacement for any DSPy LM. |
| `DataikuEmbedder` | Callable embedder compatible with DSPy retrieval / RAG workflows. |

Both classes communicate with Dataiku via the `dataiku` Python package that is pre-installed in every DSS code environment. All inference stays within LLM Mesh — no direct calls to OpenAI, Azure, or any other provider.

## Requirements

- Dataiku DSS 14+
- Python 3.10+
- `dspy-ai` (any recent 2.x release)
- The `dataiku` package (available automatically in DSS code environments)

## Installation

Copy the `dspy_dku/` folder into your DSS project's `python-lib/` directory. No `pip install` is needed — DSS adds `python-lib/` to `sys.path` for all recipes, notebooks, and web apps in the project.

```
my-dss-project/
  python-lib/
    dspy_dku/       ← paste here
      __init__.py
      lm.py
      embedder.py
      ...
```

## Quick start

### Text generation

```python
import dspy
from dspy_dku import DataikuLM

# Use the LLM connection ID from DSS Settings > LLM Mesh
lm = DataikuLM(model="openai-gpt4o")
dspy.configure(lm=lm)

# All standard DSPy usage works from here
class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

result = dspy.Predict(QA)(question="What is the capital of France?")
print(result.answer)
```

### Specifying a project

By default `DataikuLM` resolves the project from the current DSS runtime environment. Pass `project_key` to override:

```python
lm = DataikuLM(model="openai-gpt4o", project_key="MY_PROJECT")
```

### Generation parameters

All standard LiteLLM/OpenAI parameter names are accepted and translated to their Dataiku Mesh equivalents automatically:

```python
lm = DataikuLM(
    model="openai-gpt4o",
    temperature=0.3,
    max_tokens=512,
    top_p=0.95,
    stop=["END"],
)
```

| DSPy / LiteLLM key | Dataiku Mesh setting |
|---|---|
| `temperature` | `temperature` |
| `max_tokens` | `maxOutputTokens` |
| `top_p` | `topP` |
| `stop` | `stopSequences` |
| `n` | `n` |
| `response_format` | `response_format` |
| `tools` | `completion.tools` |
| `tool_choice` | `completion.tool_choice` |
| `stream` | *(not supported — falls back with a warning)* |

Per-call overrides work the same way:

```python
result = lm(messages=[{"role": "user", "content": "Hello"}], temperature=0.0, max_tokens=64)
```

### Tool / function calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

result = lm(messages=[{"role": "user", "content": "Weather in NYC?"}], tools=tools)
# result is list[dict] containing the assistant message with tool_calls
```

### Async

```python
import asyncio

result = asyncio.run(lm.acall(prompt="Hello"))
```

### Embeddings

```python
from dspy_dku import DataikuEmbedder

embedder = DataikuEmbedder(model="openai-text-embedding-3-small")
vectors = embedder(["hello world", "dspy is great"])
# vectors: list[list[float]], one vector per input
```

Large input lists are automatically split into batches of 100.

### Inspecting call history

```python
lm(prompt="Hello")
print(lm.history[-1]["usage"])
# {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
```

## Running tests

Unit tests mock all Dataiku API calls and run without a DSS instance. The
real `dataiku` package isn't required either — `conftest.py` installs a
minimal stub automatically if it isn't already importable, purely so that
`dspy_dku` (which imports `dataiku` at module level) can be loaded:

```bash
pytest tests/unit/ -v
```

Integration tests require a live DSS environment with the target connection IDs set as environment variables:

```bash
export DATAIKU_TEST_LLM_CONNECTION=openai-gpt4o
export DATAIKU_TEST_EMBED_CONNECTION=openai-text-embedding-3-small
pytest tests/integration/ -m integration -v
```

## Project structure

```
dspy_dku/
  __init__.py       exports DataikuLM, DataikuEmbedder
  lm.py             DataikuLM(dspy.LM) — primary LM adapter
  embedder.py       DataikuEmbedder — embedding / RAG adapter
  _mesh.py          _MeshClient — lazy Dataiku API wrapper
  _normalizer.py    _Normalizer — pure-function request/response translation
  _types.py         shared TypedDicts (Message, HistoryEntry, …)
tests/
  unit/             no DSS required; all Mesh calls mocked
  integration/      skipped unless env vars set
conftest.py         shared pytest fixtures
```
