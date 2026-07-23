# dspy_dku — Public API Reference

This document covers the two public classes exported by `dspy_dku`:
`DataikuLM` and `DataikuEmbedder`. Everything else in the package
(`_mesh.py`, `_normalizer.py`, `_types.py`) is a private implementation
detail and may change without notice.

```python
from dspy_dku import DataikuLM, DataikuEmbedder
```

---

## `DataikuLM`

```python
class DataikuLM(dspy.LM)
```

Drop-in DSPy `LM` that routes every completion call through Dataiku LLM
Mesh instead of LiteLLM. It extends `dspy.LM` so that DSPy features built
on the standard LM interface — `dspy.Predict`, teleprompting, optimizers,
`dspy.ReAct` and other agent modules, tool calling, structured outputs —
work unchanged. It replaces the transport layer only: DSPy's own
`__init__` runs (for history/kwargs/cache setup), but the parent's
`__call__` is never invoked.

### Constructor

```python
DataikuLM(
    model: str,
    project_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    cache: bool = True,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Dataiku LLM connection ID as configured in DSS (e.g. `"openai-gpt4o"`, `"my-azure-gpt35"`). |
| `project_key` | `str \| None` | `None` | DSS project key to resolve the LLM connection from. If `None`, resolved lazily at call time via `dataiku.default_project_key()`. |
| `temperature` | `float` | `0.0` | Default sampling temperature applied to every call unless overridden per-call. |
| `max_tokens` | `int` | `1024` | Default maximum output tokens applied to every call unless overridden per-call. |
| `cache` | `bool` | `True` | Enables DSPy's built-in LM call cache (inherited from `dspy.LM`). |
| `**kwargs` | — | — | Additional default parameters merged into every call (e.g. `top_p`, `stop`). Same keys as accepted by `__call__` (see below). |

```python
import dspy
from dspy_dku import DataikuLM

lm = DataikuLM(model="openai-gpt4o", temperature=0.1, max_tokens=512)
dspy.configure(lm=lm)
```

### `__call__`

```python
lm(
    prompt: str | None = None,
    messages: list[dict] | None = None,
    **kwargs,
) -> list[str] | list[dict]
```

Executes one completion via Dataiku LLM Mesh. This is what DSPy invokes
internally whenever the configured LM is used; it can also be called
directly.

**Parameters**

- `prompt` — plain-text prompt. Converted to a single `{"role": "user", "content": prompt}` message when `messages` is not given.
- `messages` — chat-format message list (OpenAI-style `role`/`content` dicts). Takes precedence over `prompt` when both are given.
- `**kwargs` — per-call overrides, merged on top of the instance defaults set in the constructor. Recognized keys:

  | DSPy / LiteLLM key | Dataiku Mesh equivalent | Notes |
  |---|---|---|
  | `temperature` | `temperature` | direct pass-through |
  | `max_tokens` | `maxOutputTokens` | |
  | `top_p` | `topP` | |
  | `stop` | `stopSequences` | a bare string is auto-wrapped into a single-element list |
  | `n` | `n` | number of candidate completions; if Mesh returns fewer than requested, the available choices are returned and a warning is emitted |
  | `response_format` | `response_format` | forwarded as-is; JSON parsing/validation remains DSPy's responsibility |
  | `tools` | `completion.tools` | list of OpenAI-style function-tool schemas |
  | `tool_choice` | `completion.tool_choice` | |
  | `stream` | *(not supported)* | popped before mapping; if truthy, emits a `UserWarning` and falls back to a standard non-streaming call |

  Any other key emits a `UserWarning` and is dropped (does not raise).

**Returns**

- `list[str]` — one completion string per choice, for ordinary text generation.
- `list[dict]` — raw assistant `message` dicts (including `tool_calls`) when the model's response includes tool/function calls. Callers that expect strings (e.g. generic DSPy signatures) should not be fed tool-enabled prompts unless they're prepared to handle this return type.

**Raises**

- `ValueError` — neither `prompt` nor `messages` was provided.
- `RuntimeError` — the Dataiku LLM Mesh call failed (`resp.success` was `False`); the message includes the connection ID and Mesh's reported error.

**Side effects**

Every call appends an entry to `lm.history`:

```python
{
    "prompt": messages,          # the resolved message list sent to Mesh
    "response": resp,            # the raw Mesh response object
    "kwargs": merged,            # instance defaults merged with per-call overrides
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
    },
}
```

```python
lm(prompt="Hello")
print(lm.history[-1]["usage"])
# {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
```

`lm.history` follows standard `dspy.LM` conventions, so it can be cleared
with `lm.history.clear()` if memory is a concern during long optimizer or
agent runs.

**Tool calling example**

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
# result: list[dict] — assistant message(s) with "tool_calls" present
```

### `acall`

```python
await lm.acall(
    prompt: str | None = None,
    messages: list[dict] | None = None,
    **kwargs,
) -> list[str] | list[dict]
```

Async variant of `__call__` with an identical signature, return type, and
error behavior. Because the underlying Dataiku Mesh client is synchronous,
this offloads the call to a thread pool via `asyncio.to_thread` rather than
performing true async I/O.

```python
import asyncio

result = asyncio.run(lm.acall(prompt="Hello"))
```

### Inherited from `dspy.LM`

`DataikuLM` is a subclass of `dspy.LM`, so standard `dspy.LM` attributes
and behavior (e.g. `model`, `kwargs`, `history`, DSPy's cache wrapper) are
available. `dump_state` / `load_state` and other generic `dspy.LM` methods
are not overridden and behave as in upstream DSPy.

---

## `DataikuEmbedder`

```python
class DataikuEmbedder
```

Callable embedder backed by Dataiku LLM Mesh, compatible with DSPy
retrieval / RAG workflows that expect a `list[str] -> list[list[float]]`
callable. This class does not subclass any DSPy base class.

### Constructor

```python
DataikuEmbedder(
    model: str,
    project_key: str | None = None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Dataiku embedding connection ID as configured in DSS (e.g. `"openai-text-embedding-3-small"`). |
| `project_key` | `str \| None` | `None` | DSS project key. If `None`, resolved lazily at call time via `dataiku.default_project_key()`. |

```python
from dspy_dku import DataikuEmbedder

embedder = DataikuEmbedder(model="openai-text-embedding-3-small")
```

### `__call__`

```python
embedder(inputs: list[str], **kwargs) -> list[list[float]]
```

Embeds a list of texts and returns one float vector per input, in the
same order.

**Parameters**

- `inputs` — texts to embed. An empty list returns `[]` immediately without making a network call.
- `**kwargs` — accepted for interface compatibility but currently unused.

**Batching**

Inputs are automatically split into chunks of 100 texts per underlying
Mesh call; results are concatenated back into a single list in the
original input order. This is transparent to the caller.

**Returns**

`list[list[float]]` — one embedding vector per input string.

**Raises**

- `RuntimeError` — the Dataiku Mesh embedding call failed.
- `NotImplementedError` — the configured connection does not support embeddings (e.g. it's a chat/completion connection, not an embedding model).

```python
vectors = embedder(["hello world", "dspy is great"])
len(vectors)  # 2
```

---

## Error handling summary

| Exception | Raised by | When |
|---|---|---|
| `ValueError` | `DataikuLM.__call__` | Neither `prompt` nor `messages` given. |
| `RuntimeError` | `DataikuLM.__call__`, `DataikuEmbedder.__call__` | The Dataiku LLM Mesh call itself failed (`resp.success is False`). |
| `NotImplementedError` | `DataikuEmbedder.__call__` | The connection doesn't support embeddings. |
| `UserWarning` | `DataikuLM.__call__` | An unsupported kwarg was passed and dropped, or `stream=True` was requested and fell back to non-streaming. |

## What's out of scope for this document

- Internal helpers (`_MeshClient`, `_Normalizer`, `_types`) — see
  `docs/dataikulm-for-dspy-design.md` for their design and internal data
  shapes.
- Installation and project setup — see the top-level `README.md`.
- Product goals, scope, and acceptance criteria — see
  `docs/dataikulm-for-dspy-prd.md`.
