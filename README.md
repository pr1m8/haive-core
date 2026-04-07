# haive-core

[![PyPI version](https://img.shields.io/pypi/v/haive-core.svg)](https://pypi.org/project/haive-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/haive-core.svg)](https://pypi.org/project/haive-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/pr1m8/haive-core/actions/workflows/ci.yml/badge.svg)](https://github.com/pr1m8/haive-core/actions/workflows/ci.yml)
[![Docs](https://github.com/pr1m8/haive-core/actions/workflows/docs.yml/badge.svg)](https://pr1m8.github.io/haive-core/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/haive-core.svg)](https://pypi.org/project/haive-core/)

**Foundation for the Haive AI agent framework** — engines, graph builder, state schemas, persistence, and tool routing.

`haive-core` is the foundation layer that powers every Haive package. It provides a unified abstraction over LangGraph, LangChain, and the major LLM providers, with first-class support for tool execution, state management, and persistence. Where LangGraph gives you a low-level state machine, `haive-core` gives you a high-level agent toolkit with batteries included.

---

## Why haive-core?

LangGraph is powerful but low-level. Building production agents directly on it requires you to:

- Hand-roll node functions with manual state extraction
- Manage tool routing across `pydantic_model`, `pydantic_tool`, `parse_output`, `langchain_tool` types
- Reinvent state schemas every time you add a new agent type
- Manually wire up persistence, embeddings, and engine configuration
- Debug cryptic errors when `state.dict()` serializes `BaseMessage` objects to plain dicts

`haive-core` solves all of this. You configure an LLM with `AugLLMConfig`, drop tools in, pick a state schema, and get a working agent. The framework handles the rest.

---

## Core Components

### 🔧 AugLLMConfig — The Engine Abstraction

`AugLLMConfig` is the unified LLM configuration object. Every agent in the Haive ecosystem is built around one or more of these. It handles:

- **LLM provider selection** — OpenAI, Anthropic, Azure OpenAI, Bedrock, Cohere, Ollama, etc.
- **Tool binding** — pass any LangChain tool, Pydantic model, or callable
- **Tool routing** — automatically routes tools to the correct execution path:
  - `langchain_tool` → standard LangChain tool execution
  - `pydantic_model` → BaseModel without `__call__` (validation only)
  - `pydantic_tool` → BaseModel with `__call__` (executable tool)
  - `parse_output` → BaseModel as structured output target
- **System message** — set once, used everywhere
- **Structured output** — Pydantic-validated responses with v1 (function calling) and v2 (parser) modes
- **Token tracking** — automatic usage and cost tracking
- **Caching** — built-in response cache with TTL

```python
from haive.core.engine.aug_llm import AugLLMConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression, {"__builtins__": {}}))

class WeatherReport(BaseModel):
    """Structured weather output."""
    location: str = Field(description="City name")
    temperature_f: float
    conditions: str

# Engine with tools and structured output
engine = AugLLMConfig(
    temperature=0.3,
    system_message="You are a helpful assistant.",
    tools=[calculator],
    structured_output_model=WeatherReport,
    max_tokens=1000,
)

# Engine is fully serializable, has UUID, name, metadata
print(engine.id, engine.name, engine.engine_type)
```

### 🧱 BaseGraph — High-Level Graph Builder

`BaseGraph` wraps LangGraph's `StateGraph` with engine integration. It knows how to compose `GenericEngineNodeConfig`, `ToolNodeConfig`, and `ValidationNodeConfigV2` nodes correctly. You don't have to manually wire up validation routing or tool nodes — `BaseGraph` does it for you.

```python
from haive.core.graph.state_graph.base_graph2 import BaseGraph
from haive.core.graph.node.engine_node_generic import GenericEngineNodeConfig
from haive.core.graph.node.tool_node_config_v2 import ToolNodeConfig
from haive.core.schema.prebuilt.llm_state import LLMState
from langgraph.graph import START, END

# Build a tool-using agent graph
graph = BaseGraph(name="my_agent")
graph.set_state_schema(LLMState)

# Add LLM node (handles tool calls automatically)
graph.add_node("agent", GenericEngineNodeConfig(name="agent", engine=engine))

# Add tool execution node
graph.add_node("tools", ToolNodeConfig(name="tools", tools=[calculator]))

# Wire it up
graph.add_edge(START, "agent")
graph.add_edge("agent", "tools")
graph.add_edge("tools", END)

# Compile
app = graph.to_langgraph().compile()
result = app.invoke({"messages": [HumanMessage(content="What is 15 * 23?")]})
```

### 📋 State Schemas — Pre-built Pydantic Models

Pre-built state schemas with the right fields for every agent type. Inherit from these instead of rolling your own:

```
StateSchema (base)
├── engines: dict[str, Engine]      ← Required for tool execution
│
├── MessagesState
│   └── messages: list[BaseMessage]  ← Conversation history
│
├── ToolState (extends MessagesState)
│   ├── tools: list                  ← Tool registry
│   ├── tool_routes: dict[str, str]  ← Routing map
│   └── tool_metadata: dict
│
├── LLMState (extends ToolState)
│   └── token_usage, output_schemas, parser config
│       ▲
│       └── ReactAgentState — adds iteration tracking
│
└── MultiAgentState (extends ToolState)
    ├── agents: dict[str, Agent]
    ├── agent_states: dict[str, dict]
    ├── agent_outputs: dict[str, Any]
    └── execution_order: list[str]
```

**Rule of thumb:** If your agent has tools, use `LLMState` or a subclass — it includes the `engines` field that tool nodes need at runtime. If you're building a multi-agent system, use `MultiAgentState`.

```python
from haive.core.schema.prebuilt.llm_state import LLMState
from pydantic import Field

# Extend LLMState with custom fields
class MyAgentState(LLMState):
    """State for my custom agent."""
    plan: str = ""
    iteration: int = 0
    confidence: float = 0.0
```

### 🛣️ Tool Routing System

Tools come in many flavors. `haive-core` automatically detects each and routes them correctly:

| Tool Type | Detection | Route | Example |
|-----------|-----------|-------|---------|
| LangChain `BaseTool` | `isinstance(t, BaseTool)` | `langchain_tool` | `@tool` decorated functions |
| BaseModel with `__call__` | `hasattr(model, "__call__")` | `pydantic_tool` | Stateful tools with config |
| BaseModel without `__call__` | Pydantic class | `pydantic_model` | Validation-only models |
| Structured output target | `structured_output_model=` | `parse_output` | Response schemas |
| Plain callable | `callable(t)` | `function` | Lambda or function |

The `ToolRouteMixin` handles detection automatically — you just pass tools in:

```python
from pydantic import BaseModel

class StatefulSearchTool(BaseModel):
    """A configurable, stateful tool."""
    api_key: str
    max_results: int = 5

    def __call__(self, query: str) -> str:
        # ... search logic ...
        return f"results for {query}"

# Each instance has its own state
tool1 = StatefulSearchTool(api_key="key1", max_results=10)
tool2 = StatefulSearchTool(api_key="key2", max_results=3)

engine = AugLLMConfig(tools=[tool1, tool2, calculator])
# Routes: tool1 → pydantic_tool, tool2 → pydantic_tool, calculator → langchain_tool
```

### 💾 Persistence & Stores

Production agents need persistence. `haive-core` provides a serializable wrapper around LangGraph's stores with sync and async support, connection pooling, and embedding integration.

```python
from haive.core.persistence.store.factory import StoreFactory
from haive.core.persistence.store.types import StoreConfig, StoreType

# PostgreSQL with pgvector for semantic search
config = StoreConfig(
    type=StoreType.POSTGRES_ASYNC,  # or POSTGRES_SYNC
    connection_params={
        "connection_string": "postgresql://haive:haive@localhost/haive"
    },
    embedding_provider="openai:text-embedding-3-small",
    embedding_dims=1536,
    setup_on_init=True,
)

store = StoreFactory.create(config)

# Standard LangGraph store API
await store.aput(("user", "alice"), "fact_1", {"content": "loves Python"})
results = await store.asearch(("user", "alice"), query="programming", limit=5)
```

**Why use the wrapper instead of LangGraph's store directly?**
- Critical fix for `prepared statement already exists` errors with prepared statements disabled
- Connection pool sharing via `connection_id`
- Serializable config (can be stored, shared, distributed)
- Automatic embedding integration
- Sync + async support from one configuration

### 📚 Embeddings — Lazy Loading

Embedding libraries are heavy and have nasty import-time side effects. `haive-core` lazily imports them so you don't pay the cost until you actually use one:

```python
from haive.core.models.embeddings import (
    OpenAIEmbeddings,
    get_huggingface_embeddings,
    CohereEmbeddings,
)

# OpenAI is lightweight, imported eagerly
openai_emb = OpenAIEmbeddings(model="text-embedding-3-small")

# HuggingFace is heavy, lazy-loaded
hf_emb = get_huggingface_embeddings("all-MiniLM-L6-v2")
```

This was a real bug fix — `sentence-transformers v5.2.3` hangs on import. Lazy loading works around it.

### 🪝 Hooks System

Pre/post hooks for agent lifecycle events. Add observability, validation, caching, or side effects without subclassing:

```python
from haive.core.common.mixins.hooks_mixin import HookEvent

agent.add_hook(HookEvent.BEFORE_INVOKE, log_request)
agent.add_hook(HookEvent.AFTER_INVOKE, save_to_database)
agent.add_hook(HookEvent.ON_ERROR, alert_oncall)
```

---

## Installation

```bash
pip install haive-core
```

For specific LLM providers, install the extras:

```bash
pip install haive-core[openai]      # OpenAI
pip install haive-core[anthropic]   # Anthropic Claude
pip install haive-core[azure]       # Azure OpenAI
pip install haive-core[cohere]      # Cohere
pip install haive-core[bedrock]     # AWS Bedrock
pip install haive-core[all]         # Everything
```

---

## Architectural Decisions

### Why a wrapper over LangGraph?

LangGraph is a state machine library. To build production agents on it directly, you write a lot of boilerplate: state schemas, node functions, tool routing, validation. `haive-core` is the missing layer between LangGraph (state machines) and `haive-agents` (production agents). It provides the abstractions that every agent needs without prescribing how to compose them.

### Why pre-built state schemas?

We tried auto-composing state schemas from engine introspection. It worked for simple cases but broke down when agents had tools — the auto-composed schema didn't include the `engines` field that `tool_node` needs. The fix: pre-built schemas (`LLMState`, `MultiAgentState`) that always include the right fields, with auto-composition as a fallback for non-tool agents.

### Why a tool routing system?

A tool can be many things: a LangChain `BaseTool`, a Pydantic model, a callable, a structured output target. Each needs different execution. Rather than forcing users to specify the type, the routing system detects it automatically and routes tools to the correct execution path. This means you can mix tool types freely:

```python
engine = AugLLMConfig(tools=[
    langchain_tool,     # → langchain_tool route
    StatefulTool(),     # → pydantic_tool route
    plain_function,     # → function route
])
engine.structured_output_model = ResponseModel  # → parse_output route
```

---

## Documentation

📖 **Full documentation:** https://pr1m8.github.io/haive-core/

Key topics:
- AugLLMConfig reference — every config option
- State schema hierarchy — when to use which
- Tool routing — types and detection
- BaseGraph patterns — common graph layouts
- Persistence — PostgreSQL, Neo4j, sync vs async
- Hooks — lifecycle events and ordering
- Embeddings — providers and lazy loading

---

## Related Packages

`haive-core` is the foundation. Built on top of it:

| Package | Description |
|---------|-------------|
| [haive-agents](https://pypi.org/project/haive-agents/) | 53+ production agent implementations |
| [haive-games](https://pypi.org/project/haive-games/) | LLM-powered game agents |
| [haive-tools](https://pypi.org/project/haive-tools/) | Tool implementations |
| [haive-mcp](https://pypi.org/project/haive-mcp/) | Dynamic MCP server integration |
| [haive-hap](https://pypi.org/project/haive-hap/) | Haive Agent Protocol (workflow orchestration) |
| [haive-dataflow](https://pypi.org/project/haive-dataflow/) | Data pipelines & component registry |
| [haive-prebuilt](https://pypi.org/project/haive-prebuilt/) | Pre-configured agent presets |

---

## License

MIT © [pr1m8](https://github.com/pr1m8)
