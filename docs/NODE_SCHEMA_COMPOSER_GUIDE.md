# NodeSchemaComposer Complete Guide

**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: 2025-01-18

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Real-World Examples](#real-world-examples)
7. [API Reference](#api-reference)
8. [Testing Guide](#testing-guide)
9. [Migration Guide](#migration-guide)
10. [Architecture](#architecture)

## 🎯 Overview

NodeSchemaComposer is a powerful system for creating graph nodes with flexible input/output configurations. It solves the critical problem of rigid field mappings in node-based systems by enabling arbitrary field transformations like "documents → retrieved_documents" or "result → potato".

### Key Benefits

- **Flexible I/O**: Map any input field to any output field
- **Transform Pipelines**: Apply transformations during mapping
- **Callable Wrapping**: Convert any Python function to a node
- **Type Safety**: Optional runtime type validation
- **Clean Syntax**: Decorators and factory functions
- **Production Ready**: 100+ tests, no mocks, real components

### When to Use

Use NodeSchemaComposer when you need to:

- Change field names between nodes (e.g., "documents" → "retrieved_docs")
- Create nodes from existing functions with custom I/O
- Adapt between different component interfaces
- Build complex processing pipelines
- Implement type-safe node operations

## 🚀 Quick Start

### Installation

NodeSchemaComposer is part of haive-core:

```bash
poetry add haive-core
```

### Basic Example

```python
from haive.core.graph.node.composer import (
    NodeSchemaComposer,
    change_output_key,
    as_node
)

# Example 1: Change output key
retriever = get_retriever_node()  # Returns {"documents": [...]}
adapted = change_output_key(retriever, "documents", "retrieved_documents")
# Now returns {"retrieved_documents": [...]}

# Example 2: Create node from function
@as_node(
    output_mappings=[FieldMapping("result", "should_continue")]
)
def check_length(state):
    return {"result": len(state.messages) > 10}

# Example 3: Full composition
composer = NodeSchemaComposer()
custom_node = composer.from_callable(
    func=process_data,
    input_mappings=[
        FieldMapping("user_query", "query"),
        FieldMapping("context_data", "context")
    ],
    output_mappings=[
        FieldMapping("response", "ai_response"),
        FieldMapping("confidence", "score")
    ]
)
```

## 📚 Core Concepts

### Field Mapping

The foundation of NodeSchemaComposer is the `FieldMapping` class:

```python
@dataclass
class FieldMapping:
    source_path: str          # Where to read from
    target_path: str          # Where to write to
    transform: List[str]      # Optional transformations
    default: Any              # Default if source missing
    required: bool            # Whether field is required
```

### Path Resolution

Supports complex path patterns:

```python
# Simple fields
"messages"                    # state.messages

# Nested access
"config.temperature"          # state.config.temperature

# Array access
"messages[0]"                # First message
"messages[-1]"               # Last message

# Combined
"agents[0].config.model"     # First agent's model config
```

### Transform Pipeline

Apply transformations during mapping:

```python
# Built-in transforms
FieldMapping("text", "upper_text", transform=["uppercase"])
FieldMapping("value", "formatted", transform=["str", "strip"])

# Chain transforms
FieldMapping("id", "formatted_id",
    transform=["uppercase", "add_prefix", "add_suffix"]
)

# Register custom transforms
composer.register_transform_function("double", lambda x: x * 2)
```

## 💻 Basic Usage

### 1. Changing Output Keys

The most common use case - adapting output field names:

```python
# Using factory function
adapted = change_output_key(node, "documents", "retrieved_documents")

# Or with composer
composer = NodeSchemaComposer()
adapted = composer.compose_node(
    base_node=node,
    output_mappings=[
        FieldMapping("documents", "retrieved_documents"),
        FieldMapping("count", "document_count")
    ]
)
```

### 2. Changing Input Keys

Adapt input expectations:

```python
# Node expects "query" but state has "user_question"
adapted = change_input_key(node, "query", "user_question")

# Multiple input mappings
adapted = remap_fields(
    node,
    input_mapping={
        "user_question": "query",
        "background": "context"
    }
)
```

### 3. Creating Nodes from Callables

Convert any function to a node:

```python
def process_messages(messages: List[str], threshold: int = 5) -> Dict[str, Any]:
    return {
        "count": len(messages),
        "over_threshold": len(messages) > threshold
    }

node = composer.from_callable(
    func=process_messages,
    input_mappings=[
        FieldMapping("conversation", "messages"),
        FieldMapping("config.max_length", "threshold", default=10)
    ],
    output_mappings=[
        FieldMapping("count", "message_count"),
        FieldMapping("over_threshold", "should_summarize")
    ]
)
```

### 4. Schema Adapters

Convert between different Pydantic models:

```python
# Define schemas
class OldState(BaseModel):
    user_name: str
    message_text: str

class NewState(BaseModel):
    username: str
    content: str

# Create adapter
adapter = composer.create_adapter(
    source_schema=OldState,
    target_schema=NewState,
    field_mappings=[
        FieldMapping("user_name", "username"),
        FieldMapping("message_text", "content")
    ]
)

# Use adapter
new_instance = adapter.adapt(old_instance)
```

## 🚀 Advanced Features

### 1. Automatic Signature Detection

AdvancedNodeComposer handles various function signatures automatically:

```python
from haive.core.graph.node.composer import AdvancedNodeComposer

composer = AdvancedNodeComposer()

# Handles all these signatures:
def func1(state): ...
def func2(state, config): ...
def func3(state: MyState, config: Dict[str, Any]): ...
def func4() -> Dict: ...  # No params
def func5(state) -> Command: ...

# All work with:
node = composer.from_callable_advanced(func)
```

### 2. Extended Extract/Update Logic

Go beyond simple field mapping:

```python
def extract_conversation_context(state, config):
    """Custom extraction logic."""
    messages = state.messages[-5:]  # Last 5
    return {
        "recent": messages,
        "speakers": count_speakers(messages),
        "has_question": any("?" in m for m in messages)
    }

def update_with_analysis(result, state, config):
    """Custom update logic."""
    return {
        "analysis": result,
        "analyzed_at": datetime.now(),
        "next_action": determine_action(result)
    }

node = composer.from_callable_advanced(
    func=analyze,
    extract_logic=extract_conversation_context,
    update_logic=update_with_analysis
)
```

### 3. Type-Safe Nodes

Create nodes with runtime type validation:

```python
def process(state: InputState, config: Config) -> Result:
    return Result(processed=True)

typed_node = composer.create_typed_callable_node(
    func=process,
    state_type=InputState,
    config_type=Config,
    result_type=Result,
    validate_types=True  # Runtime validation
)
```

### 4. Decorator Patterns

Clean, Pythonic node definitions:

```python
# Simple decorator
@as_node()
def my_processor(state):
    return {"done": True}

# With mappings
@as_node(
    input_mappings=[FieldMapping("messages", "conversation")],
    output_mappings=[FieldMapping("result", "should_continue")]
)
def check_conversation(conversation: List[Message]) -> Dict:
    return {"result": len(conversation) > 10}

# With custom logic
@as_node(
    extract_logic=extract_recent_messages,
    update_logic=update_conversation_state
)
def analyze_conversation(messages):
    return {"sentiment": analyze_sentiment(messages)}
```

### 5. Command/Send Handling

Automatic handling of LangGraph commands:

```python
# Auto-wraps dict returns
@as_node()
def process(state):
    return {"done": True}  # Becomes Command(update={"done": True})

# Preserves explicit Commands
@as_node()
def router(state) -> Command:
    if state.needs_help:
        return Command(goto="help")
    return Command(goto="continue")

# Handles Send for parallel execution
@as_node(handle_command=False)
def split_work(state) -> List[Send]:
    return [
        Send("worker1", {"task": "A"}),
        Send("worker2", {"task": "B"})
    ]
```

### 6. Pipeline Nodes

Create nodes with extract → process → update pipelines:

```python
node = node_with_custom_logic(
    name="document_processor",
    extract=lambda state, config: state.documents[:10],
    process=lambda docs: {"summaries": [summarize(d) for d in docs]},
    update=lambda result, state, config: {
        "summaries": result["summaries"],
        "processed_at": datetime.now()
    }
)
```

## 🌍 Real-World Examples

### Example 1: RAG Pipeline Integration

Common scenario - integrating components with mismatched interfaces:

```python
# Components with different interfaces
retriever = get_retriever()  # Outputs: {"documents": [...]}
llm = get_llm()             # Expects: {"context": str}

# Adapt interfaces
composer = NodeSchemaComposer()

# Adapt retriever output
adapted_retriever = change_output_key(retriever, "documents", "retrieved_docs")

# Create context builder
@as_node(
    input_mappings=[FieldMapping("retrieved_docs", "docs")],
    output_mappings=[FieldMapping("result", "context")]
)
def build_context(docs: List[Dict]) -> Dict:
    return {"result": "\n".join(d["content"] for d in docs)}

# Now they work together!
# retriever → context_builder → llm
```

### Example 2: Multi-Agent Coordination

Coordinate agents with different state schemas:

```python
# Agents expect different fields
planner = PlannerAgent()    # Expects: {"objective": str}
executor = ExecutorAgent()  # Expects: {"plan": dict}

# Create coordinator
@as_node()
def coordinate(state):
    # Adapt for planner
    planner_input = {"objective": state["user_goal"]}
    plan = planner.run(planner_input)

    # Adapt for executor
    executor_input = {"plan": plan["steps"]}
    result = executor.run(executor_input)

    return {
        "plan": plan,
        "execution": result,
        "status": "completed"
    }
```

### Example 3: Document Processing Pipeline

Complex document processing with field adaptations:

```python
composer = AdvancedNodeComposer()

# Document filter with custom logic
def extract_valid_docs(state, config):
    min_length = config.get("min_doc_length", 100)
    return [
        doc for doc in state["documents"]
        if len(doc.get("content", "")) >= min_length
    ]

def process_documents(docs):
    return {
        "summaries": [summarize(doc) for doc in docs],
        "keywords": [extract_keywords(doc) for doc in docs]
    }

def update_with_results(result, state, config):
    return {
        "processed_documents": result["summaries"],
        "document_keywords": result["keywords"],
        "processing_complete": True,
        "processed_at": datetime.now()
    }

processor = composer.from_callable_advanced(
    func=process_documents,
    extract_logic=extract_valid_docs,
    update_logic=update_with_results,
    name="document_processor"
)
```

### Example 4: Conversation Flow Control

Manage conversation state transitions:

```python
@as_node(
    output_mappings=[
        FieldMapping("action", "next_action"),
        FieldMapping("reason", "action_reason")
    ]
)
def conversation_controller(state):
    messages = state.get("messages", [])

    # Check various conditions
    if len(messages) > 50:
        return {
            "action": "summarize",
            "reason": "conversation_too_long"
        }

    if has_unanswered_question(messages):
        return {
            "action": "answer",
            "reason": "pending_question"
        }

    if is_conversation_complete(messages):
        return {
            "action": "end",
            "reason": "natural_conclusion"
        }

    return {
        "action": "continue",
        "reason": "ongoing_conversation"
    }
```

## 📖 API Reference

### NodeSchemaComposer

Main composer class for flexible node I/O configuration.

#### Methods

```python
compose_node(base_node, input_mappings=None, output_mappings=None, name=None)
```

Compose a node with custom I/O mappings.

```python
from_callable(func, input_mappings=None, output_mappings=None, name=None, **kwargs)
```

Create a composed node from a callable function.

```python
create_adapter(source_schema, target_schema, field_mappings, name=None)
```

Create an adapter between two Pydantic schemas.

```python
register_transform_function(name: str, func: Callable)
```

Register a custom transform function.

### AdvancedNodeComposer

Extended composer with advanced callable handling.

#### Methods

```python
from_callable_advanced(func, extract_logic=None, update_logic=None,
                      auto_detect_signature=True, handle_command=True, **kwargs)
```

Create node with extended capabilities.

```python
create_typed_callable_node(func, state_type, config_type, result_type,
                          validate_types=True, **kwargs)
```

Create type-safe callable node.

```python
create_extract_update_node(extract_func, process_func, update_func, name)
```

Create node with extract → process → update pipeline.

### Factory Functions

```python
change_output_key(node, old_key, new_key) -> ComposedNode
```

Change a single output field name.

```python
change_input_key(node, old_key, new_key) -> ComposedNode
```

Change a single input field name.

```python
remap_fields(node, input_mapping=None, output_mapping=None) -> ComposedNode
```

Remap multiple input/output fields.

```python
callable_to_node(func, composer=None, **kwargs) -> AdvancedComposedNode
```

Convert any callable to a node.

```python
node_with_custom_logic(name, extract, process, update, composer=None) -> ComposedNode
```

Create node with custom pipeline.

### Decorators

```python
@as_node(input_mappings=None, output_mappings=None, **kwargs)
```

Decorator to convert function to node.

## 🧪 Testing Guide

### Writing Tests

Always test with real components (no mocks):

```python
def test_node_composition():
    # Create real node
    def process(state, config):
        return {"count": len(state.get("items", []))}

    node = CallableNodeConfig(name="test", callable_func=process)

    # Compose with mappings
    composer = NodeSchemaComposer()
    composed = composer.compose_node(
        base_node=node,
        input_mappings=[FieldMapping("products", "items")],
        output_mappings=[FieldMapping("count", "product_count")]
    )

    # Test with real data
    result = composed({"products": ["a", "b", "c"]}, {})

    assert isinstance(result, Command)
    assert result.update["product_count"] == 3
```

### Testing Patterns

1. **Test each mapping individually**
2. **Verify Command structure**
3. **Test error cases**
4. **Test with real state objects**
5. **Verify transform pipelines**

## 🔄 Migration Guide

### From Direct Node Usage

Before:

```python
# Direct node usage with fixed fields
retriever = RetrieverNode()  # Returns {"documents": [...]}
processor = ProcessorNode()  # Expects {"docs": [...]}
# Incompatible!
```

After:

```python
# With NodeSchemaComposer
retriever = RetrieverNode()
adapted_retriever = change_output_key(retriever, "documents", "docs")
# Now compatible with processor!
```

### From Manual Adapters

Before:

```python
# Manual adaptation code
def adapt_retriever_output(result):
    return {
        "docs": result.get("documents", []),
        "count": result.get("num_documents", 0)
    }
```

After:

```python
# Declarative with NodeSchemaComposer
adapted = remap_fields(
    retriever,
    output_mapping={
        "documents": "docs",
        "num_documents": "count"
    }
)
```

## 🏗️ Architecture

### Component Overview

```
NodeSchemaComposer System
├── Core Components
│   ├── FieldMapping          # Field mapping definition
│   ├── PathResolver          # Complex path resolution
│   ├── ExtractFunctions      # Input extraction library
│   └── UpdateFunctions       # Output update library
├── Main Composers
│   ├── NodeSchemaComposer    # Basic composition
│   └── AdvancedNodeComposer  # Extended features
├── Node Types
│   ├── ComposedNode          # Wrapped nodes
│   ├── ComposedCallableNode  # Wrapped callables
│   └── TypedCallableNode     # Type-safe nodes
└── Utilities
    ├── Factory Functions     # Quick helpers
    └── Decorators           # Clean syntax
```

### Design Principles

1. **Composability**: All components work together
2. **Extensibility**: Easy to add new features
3. **Type Safety**: Optional but available
4. **Real Components**: No mocks in testing
5. **Clean API**: Multiple usage patterns

### Performance Considerations

- Field mapping adds minimal overhead (<1ms)
- Path resolution is optimized for common patterns
- Transform functions are cached
- No memory leaks from composed nodes

## 🎯 Summary

NodeSchemaComposer provides a complete solution for flexible node I/O configuration in the Haive framework. With support for arbitrary field mappings, transform pipelines, and various callable patterns, it enables seamless integration of components with different interfaces.

Key takeaways:

- **Flexible**: Map any field to any field
- **Powerful**: Extended extraction and update logic
- **Clean**: Decorator and factory patterns
- **Safe**: Optional type validation
- **Tested**: 100+ tests with real components
- **Ready**: Production-ready implementation

Start with the basic factory functions (`change_output_key`, `change_input_key`) and explore advanced features as needed. The system grows with your requirements!
