# Haive Graph System

This module provides a comprehensive graph implementation for the Haive framework.

## Overview

The graph system is organized into the following components:

### Core Components

- `BaseGraph`: The base graph implementation with all core functionality
- `SchemaGraph`: Extended graph with schema management capabilities
- `Node`: The node component implementation
- `Branch`: The branch component for conditional routing

### Directory Structure

```
haive/core/graph/state_graph/
├── __init__.py                   # Public exports
├── base_graph2.py                # Core graph implementation
├── schema_graph.py               # Schema-aware graph
├── validation_mixin.py           # Graph validation functionality
├── schema_mixin.py               # Schema management
├── graph_visualizer.py           # Graph visualization
├── components/                   # Core components
│   ├── __init__.py
│   ├── node.py                   # Node implementation
│   └── branch.py                 # Branch implementation
├── conversion/                   # Format conversion
│   ├── __init__.py
│   └── langgraph.py              # LangGraph conversion
└── utils/                        # Utilities
    ├── __init__.py
    └── ...                       # Utility functions
```

## Usage

### Basic Graph Creation

```python
from haive.core.graph.state_graph import BaseGraph
from langgraph.graph import START, END

graph = BaseGraph(name="my_graph")

# Add nodes
graph.add_node("node1", lambda state: state)
graph.add_node("node2", lambda state: state)

# Add edges
graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", END)
```

### Schema-aware Graph

```python
from pydantic import BaseModel
from haive.core.graph.state_graph import SchemaGraph

class MyState(BaseModel):
    count: int = 0
    messages: list = []

graph = SchemaGraph(
    name="schema_graph",
    state_schema=MyState
)

# Convert to LangGraph for execution
langgraph = graph.to_langgraph()
```

### Using Components Directly

```python
from haive.core.graph.state_graph.components import Node, Branch
from haive.core.graph.common.types import NodeType

# Create a node
node = Node(
    name="my_node",
    node_type=NodeType.CALLABLE,
    metadata={"callable": lambda state: state}
)

# Create a branch
branch = Branch(
    name="my_branch",
    source_node="source",
    destinations={True: "target_true", False: "target_false"}
)
```

## Development

The Haive graph system is designed to be extensible and modular. Key principles:

1. **Separation of Concerns**: Components are separated into their own files
2. **Clear Interfaces**: Each module has a well-defined public interface
3. **Type Safety**: Generic parameters ensure type safety
4. **Compatibility**: Both backward compatibility and LangGraph integration

When extending the system, follow these guidelines:

- Add new functionality via mixins or derived classes rather than modifying core classes
- Document all public methods and classes with docstrings
- Add type annotations to all methods and functions
- Write tests for new functionality

## Migration Path

The current implementation is in transition from the older `base_graph.py` to the newer `base_graph2.py`. The final step will be to rename `base_graph2.py` to `base_graph.py` once all functionality is migrated and tested.
