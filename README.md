# haive-core

[![PyPI version](https://img.shields.io/pypi/v/haive-core.svg)](https://pypi.org/project/haive-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/haive-core.svg)](https://pypi.org/project/haive-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/pr1m8/haive-core/actions/workflows/ci.yml/badge.svg)](https://github.com/pr1m8/haive-core/actions/workflows/ci.yml)
[![Docs](https://github.com/pr1m8/haive-core/actions/workflows/docs.yml/badge.svg)](https://pr1m8.github.io/haive-core/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/haive-core.svg)](https://pypi.org/project/haive-core/)

**Foundation for the Haive AI agent framework** — engines, graph builder, state schemas, persistence, and stores.

`haive-core` is the foundation layer that all other Haive packages build on. It provides the core abstractions for building LLM-powered agents: engine configuration, state management, graph compilation, persistence, and tool routing.

## Installation

```bash
pip install haive-core
```

## Features

- **🔧 AugLLMConfig** — unified LLM configuration with tools, structured output, and routing
- **🧱 BaseGraph** — high-level graph builder that wraps LangGraph's `StateGraph` with engine integration
- **📋 State Schemas** — pre-built state classes (`StateSchema`, `MessagesState`, `ToolState`, `LLMState`, `MultiAgentState`)
- **🛣️ Tool Routing** — automatic routing for `pydantic_model`, `pydantic_tool`, `parse_output`, and `langchain_tool` types
- **💾 Persistence** — `PostgresStoreWrapper` (sync + async), `MemoryStoreWrapper`, factory pattern
- **📚 Embeddings** — lazy-loaded HuggingFace, OpenAI, Cohere embedding wrappers
- **🪝 Hooks** — pre/post execution hooks for agent lifecycle events

## Quick Start

```python
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.schema.prebuilt.llm_state import LLMState

# Configure an LLM engine with tools and system message
engine = AugLLMConfig(
    temperature=0.7,
    system_message="You are a helpful assistant.",
)

# Use a pre-built state schema (includes engines, tools, messages)
state = LLMState(messages=[])
```

## Documentation

📖 **Full documentation:** https://pr1m8.github.io/haive-core/

## Related Packages

| Package | Description |
|---------|-------------|
| [haive-agents](https://pypi.org/project/haive-agents/) | Production agent implementations |
| [haive-games](https://pypi.org/project/haive-games/) | LLM-powered game agents |
| [haive-tools](https://pypi.org/project/haive-tools/) | Tool implementations |
| [haive-mcp](https://pypi.org/project/haive-mcp/) | Model Context Protocol integration |
| [haive-hap](https://pypi.org/project/haive-hap/) | Haive Agent Protocol |
| [haive-dataflow](https://pypi.org/project/haive-dataflow/) | Data processing pipelines |

## License

MIT © [pr1m8](https://github.com/pr1m8)
