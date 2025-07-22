# haive-core

Core foundation for the Haive AI Agent Framework, providing essential building blocks for agent systems, state management, and sophisticated multi-agent workflows.

## Overview

Haive Core is the foundational package that provides the essential building blocks for the entire Haive ecosystem. It contains the core abstractions, base classes, and utilities that power all other Haive packages, with a special focus on advanced multi-agent systems and hierarchical state management.

## Installation

```bash
poetry add haive-core
```

## Quick Start

### Basic Multi-Agent Setup

```python
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState
from haive.agents.simple import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig

# Create agents
planner = SimpleAgent(
    name="planner",
    engine=AugLLMConfig(),
    structured_output_model=PlanningResult
)

executor = SimpleAgent(
    name="executor",
    engine=AugLLMConfig(),
    structured_output_model=ExecutionResult
)

# Initialize state
state = MultiAgentState(agents=[planner, executor])
```

### Sequential Execution

```python
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3

# Create nodes
plan_node = create_agent_node_v3("planner")
exec_node = create_agent_node_v3("executor")

# Execute sequence
result1 = plan_node(state, config)  # Updates planning fields
result2 = exec_node(state, config)  # Reads planning fields directly
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph

# Build graph
graph = StateGraph(MultiAgentState)
graph.add_node("plan", create_agent_node_v3("planner"))
graph.add_node("execute", create_agent_node_v3("executor"))
graph.add_node("review", create_agent_node_v3("reviewer"))

# Define flow
graph.add_edge("plan", "execute")
graph.add_edge("execute", "review")

# Compile and execute
app = graph.compile()
final_state = app.invoke(state)
```

## Key Features

### 🤖 Advanced Multi-Agent Systems

Sophisticated multi-agent workflows with hierarchical state management and direct field updates:

```python
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3

# Create multi-agent state
state = MultiAgentState(agents=[planner, executor, reviewer])

# Execute Self-Discover workflow
plan_node = create_agent_node_v3("planner")
exec_node = create_agent_node_v3("executor")

result1 = plan_node(state, config)  # Updates planning fields
result2 = exec_node(state, config)  # Reads planning fields directly
```

**Key Benefits:**

- **No Schema Flattening**: Each agent maintains its own schema
- **Direct Field Updates**: Agents update container fields directly
- **Self-Discover Workflows**: Sequential agents reading each other's outputs
- **Type Safety**: Full type validation throughout multi-agent execution

### 🔄 Direct Field Updates

Agents with structured outputs update state fields directly (like engine nodes):

```python
# Traditional approach (complex)
plan = state.agent_outputs["planner"]["plan"]

# Haive approach (direct)
plan = state.plan  # Direct field access
```

### 🏗️ Hierarchical State Management

Container-based state projection maintaining agent isolation:

```python
# Each agent gets exactly what it expects
planner_state = state.get_agent_state("planner")
executor_state = state.get_agent_state("executor")

# Shared resources available to all
state.messages  # Shared conversation
state.tools     # Shared tool registry
```

## What Haive Core Does

### 1. Multi-Agent System - Advanced Coordination

The Multi-Agent system enables sophisticated workflows through hierarchical state management:

- **MultiAgentState**: Container state schema for managing multiple agents
- **AgentNodeV3**: Execution nodes with direct field updates
- **Self-Discover Workflows**: Sequential agents building on each other's outputs
- **Dynamic Agent Composition**: Runtime agent addition and recompilation

**Example:**

```python
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState
from haive.agents.simple import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig

# Create agents with structured output
planner = SimpleAgent(
    name="planner",
    engine=AugLLMConfig(),
    structured_output_model=PlanningResult
)

executor = SimpleAgent(
    name="executor",
    engine=AugLLMConfig(),
    structured_output_model=ExecutionResult
)

# Initialize state
state = MultiAgentState(agents=[planner, executor])

# Execute workflow
from langgraph.graph import StateGraph
graph = StateGraph(MultiAgentState)
graph.add_node("plan", create_agent_node_v3("planner"))
graph.add_node("execute", create_agent_node_v3("executor"))
graph.add_edge("plan", "execute")

app = graph.compile()
final_state = app.invoke(state)
```

### 2. Engine System - The Universal Interface

The Engine system provides a standardized way to interact with any AI component:

- **Unified API**: Every component (LLMs, retrievers, tools) shares the same interface
- **Runtime Configuration**: Dynamically adjust parameters without code changes
- **Schema Management**: Automatic input/output validation and type safety
- **Serialization**: Save and restore any engine's complete state

**Key Components:**

- `InvokableEngine`: For components that process input → output (LLMs, retrievers)
- `NonInvokableEngine`: For utility components (embeddings, loaders)
- `AugLLM`: Enhanced LLM with tools, structured output, and advanced prompting

### 2. Graph System - Dynamic Workflow Builder

Build complex AI workflows as graphs that can be modified at runtime:

- **Dynamic Graph Builder**: Compose workflows programmatically
- **Node Factory**: Create processing nodes from any engine
- **Pattern Registry**: Reusable workflow patterns (RAG, ReAct, etc.)
- **State Management**: Sophisticated state schemas with reducers

**Example Use:**

```python
graph = DynamicGraph(components=[llm, retriever])
graph.add_node("search", retriever)
graph.add_node("generate", llm)
graph.add_edge("search", "generate")
```

### 3. Schema System - Intelligent State Management

Advanced state management designed for AI workflows:

- **Field Sharing**: Share state between parent/child graphs
- **Reducers**: Define how state updates are merged
- **Auto-Generation**: Derive schemas from components
- **Serialization**: Full state persistence

**Features:**

- `StateSchema`: Base class with sharing and reducers
- `SchemaComposer`: Build schemas dynamically
- `StateSchemaManager`: Runtime schema manipulation

### 4. Configuration System - Runtime Control

Sophisticated configuration management:

- **Runnable Config**: Standardized runtime parameters
- **Engine Targeting**: Configure specific engines by ID
- **Hierarchical Config**: Cascading configuration with overrides
- **Thread Management**: Session and conversation handling

### 5. Persistence Layer - State Management

Comprehensive persistence capabilities:

- **Multiple Backends**: PostgreSQL, MongoDB, SQLite, Memory
- **Checkpointing**: Save/restore complete agent states
- **Thread Management**: Conversation history and context
- **Automatic Setup**: Database tables created as needed

### 6. Document Processing - Content Pipeline

Complete document handling system:

- **40+ Loaders**: PDF, Word, Excel, HTML, Markdown, Code files, etc.
- **Smart Splitting**: Intelligent document chunking
- **Transformers**: Document processing and enrichment
- **Source Management**: Local files, web content, databases

### 7. Model Abstractions - Provider Agnostic

Unified interfaces for AI models:

- **LLM Models**: Support for OpenAI, Anthropic, Azure, HuggingFace, etc.
- **Embeddings**: Text embedding models from various providers
- **Retrievers**: Vector search, keyword search, hybrid approaches
- **Vector Stores**: FAISS, Chroma, Pinecone, Weaviate, etc.

### 8. Common Utilities - Shared Infrastructure

Essential utilities used across Haive:

- **Mixins**: Reusable functionality (tools, state, serialization)
- **Type System**: Advanced types and protocols
- **Logging**: Sophisticated logging with UI
- **Discovery**: Automatic component discovery

## Key Capabilities

### Dynamic Composition

```python
# Create engines
llm = AugLLMConfig(model="gpt-4", tools=[SearchTool])
retriever = RetrieverConfig(vector_store=faiss_store)

# Compose into workflow
graph = DynamicGraph([llm, retriever])
```

### State Schema Generation

```python
# Auto-generate schema from components
Schema = SchemaComposer.from_components([llm, retriever])

# Add custom fields
composer = SchemaComposer()
composer.add_field("context", List[str], reducer=operator.add)
```

### Persistence

```python
# Setup persistence
from haive.core.persistence import PostgresCheckpointer
checkpointer = PostgresCheckpointer(connection_string)

# Save state
checkpointer.put(thread_id, state)

# Restore state
state = checkpointer.get(thread_id)
```

### Document Processing

```python
# Load various document types
loader = DocumentLoader()
docs = loader.load("report.pdf", "data.xlsx", "code.py")

# Smart splitting
splitter = TextSplitter(chunk_size=1000)
chunks = splitter.split_documents(docs)
```

## Integration Points

Haive Core integrates with:

- **LangChain/LangGraph**: Extended functionality
- **Pydantic v2**: Type safety and validation
- **Major LLM Providers**: OpenAI, Anthropic, etc.
- **Vector Databases**: All major providers
- **Document Formats**: 40+ file types

## Advanced Usage

### Self-Discover Workflow

```python
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Define structured outputs
class SelectedModules(BaseModel):
    selected_modules: List[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)

class AdaptedModules(BaseModel):
    adapted_modules: List[Dict[str, str]]
    task_context: str

class ReasoningStructure(BaseModel):
    reasoning_structure: Dict[str, Any]
    steps: List[str]
    methodology: str

# Create agents with structured outputs
selector = SimpleAgent(
    name="selector",
    engine=AugLLMConfig(),
    structured_output_model=SelectedModules
)

adapter = SimpleAgent(
    name="adapter",
    engine=AugLLMConfig(),
    structured_output_model=AdaptedModules
)

reasoner = SimpleAgent(
    name="reasoner",
    engine=AugLLMConfig(),
    structured_output_model=ReasoningStructure
)

# Setup state with all required fields
class SelfDiscoverState(MultiAgentState):
    # Input fields
    task_description: str = ""
    available_modules: List[str] = Field(default_factory=list)

    # Output fields (directly updated by agents)
    selected_modules: List[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0

    adapted_modules: List[Dict[str, str]] = Field(default_factory=list)
    task_context: str = ""

    reasoning_structure: Dict[str, Any] = Field(default_factory=dict)
    steps: List[str] = Field(default_factory=list)
    methodology: str = ""

# Execute Self-Discover workflow
state = SelfDiscoverState(
    agents=[selector, adapter, reasoner],
    task_description="How can we reduce plastic waste in oceans?",
    available_modules=["systems_thinking", "root_cause_analysis", "solution_design"]
)

# Sequential execution with direct field access
selector_node = create_agent_node_v3("selector")
adapter_node = create_agent_node_v3("adapter")
reasoner_node = create_agent_node_v3("reasoner")

result1 = selector_node(state, config)  # Updates: selected_modules, rationale, confidence
result2 = adapter_node(state, config)   # Reads: selected_modules, Updates: adapted_modules
result3 = reasoner_node(state, config)  # Reads: adapted_modules, Updates: reasoning_structure

# Final state has all results directly accessible
print(f"Selected modules: {state.selected_modules}")
print(f"Reasoning structure: {state.reasoning_structure}")
print(f"Final methodology: {state.methodology}")
```

### Dynamic Agent Composition

```python
# Add agents at runtime
new_agent = SimpleAgent(name="validator", engine=AugLLMConfig())
state.agents["validator"] = new_agent

# Mark for recompilation
state.mark_agent_for_recompile("validator", "Added new agent")

# Check recompilation needs
if state.needs_any_recompile():
    agents_to_recompile = state.get_agents_needing_recompile()
    print(f"Recompiling: {agents_to_recompile}")
```

## API Reference

### Core Classes

- **MultiAgentState**: Container state schema for managing multiple agents
- **AgentNodeV3Config**: Configuration for agent execution nodes
- **AugLLMConfig**: Augmented LLM configuration with tools and structured output
- **ToolState**: Base state with comprehensive tool management
- **MessagesState**: State with message management and token tracking

### Key Functions

- **create_agent_node_v3()**: Create agent execution nodes with direct field updates
- **create_engine_node()**: Create engine execution nodes
- **create_tool_node()**: Create tool execution nodes

### State Management

- **get_agent_state()**: Get agent's isolated state
- **update_agent_state()**: Update agent state
- **mark_agent_for_recompile()**: Mark agent for recompilation
- **needs_any_recompile()**: Check if any agents need recompilation

### Configuration

- **shared_fields**: Fields to share from container to agent (default: ["messages"])
- **output_mode**: How to handle outputs ("merge", "replace", "isolate")
- **project_state**: Whether to project state to agent schema (default: True)
- **track_recompilation**: Whether to track recompilation needs (default: True)

## Testing

Run tests with real components (no mocks):

```bash
# Run all tests
poetry run pytest packages/haive-core/tests/ -v

# Run specific test categories
poetry run pytest packages/haive-core/tests/schema/ -v
poetry run pytest packages/haive-core/tests/graph/ -v
poetry run pytest packages/haive-core/tests/node/ -v

# Run with coverage
poetry run pytest packages/haive-core/tests/ --cov=haive.core --cov-report=html
```

## Architecture

### Core Components

```
haive-core/
├── engine/          # LLM and augmented engines
├── schema/          # State management and schemas
├── graph/           # Node-based workflow execution
├── models/          # Model abstractions
├── persistence/     # State persistence
├── registry/        # Component registration
└── types/          # Type definitions
```

### Multi-Agent Architecture

```
MultiAgentState (Container)
├── agents: Dict[str, Agent]           # Agent instances
├── agent_states: Dict[str, Dict]      # Isolated agent states
├── messages: List[BaseMessage]        # Shared conversation
├── tools: List[Tool]                  # Shared tools
├── engines: Dict[str, Engine]         # Shared engines
└── [dynamic fields from agents]       # Direct field updates
```

## Performance

### Benchmarks

- **Agent Execution**: ~100ms per agent (excluding LLM calls)
- **State Projection**: <1ms for typical state sizes
- **Field Updates**: <1ms for structured outputs
- **Recompilation**: ~10ms for graph rebuilding

### Optimization Tips

1. **Use Structured Outputs**: Enables direct field updates
2. **Minimize Shared Fields**: Only share necessary data
3. **Batch Operations**: Group agent executions when possible
4. **Monitor State Size**: Keep state manageable for performance

## Use Cases

1. **Building Custom Agents**: Use base classes to create specialized agents
2. **Multi-Agent Workflows**: Compose complex sequential and parallel processes
3. **Self-Discover Reasoning**: Sequential agents building on each other's outputs
4. **State Management**: Handle conversation history and context with type safety
5. **Document Processing**: Build RAG systems with any content
6. **Tool Integration**: Add capabilities to any LLM with automatic routing

## Documentation

- [Multi-Agent Systems Guide](docs/multi_agent_systems.md)
- [API Reference](https://haive.readthedocs.io/api/core)
- [Examples](../../examples/multi_agent/)
- [Testing Philosophy](../../project_docs/active/standards/testing/philosophy.md)

## Contributing

1. Follow the [Testing Philosophy](../../project_docs/active/standards/testing/philosophy.md) - NO MOCKS
2. Use [Google-style docstrings](../../project_docs/active/standards/documentation/google_style.md)
3. Test with real components
4. Add comprehensive examples
5. Update documentation

## License

MIT License - see [LICENSE](../../LICENSE) for details.
