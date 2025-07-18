# Unified Schema Architecture for Haive

**Version**: 1.0  
**Purpose**: Define consistent schema architecture integrating all composers  
**Last Updated**: 2025-01-18

## 🎯 Overview

This document defines a unified, consistent architecture for schema management in Haive, integrating:

- **StateSchema**: Base schema class with field sharing and reducers
- **SchemaComposer**: Dynamic schema building
- **AgentSchemaComposer**: Multi-agent schema patterns
- **NodeSchemaComposer**: Flexible node I/O configuration

## 🏗️ Architecture Hierarchy

```
┌─────────────────────────────────────────────────┐
│             StateSchema (Base)                  │
│  - Field sharing, reducers, engine I/O tracking │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────┴────────────────────────────┐
│           SchemaComposer (Builder)              │
│  - Dynamic schema creation from components      │
│  - Field extraction and definition management   │
└────────────────────┬────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────┴────────┐     ┌───────┴────────────────┐
│AgentSchemaComposer│   │IntegratedNodeComposer  │
│ - Multi-agent     │   │ - Node I/O mapping     │
│ - Field strategies│   │ - Schema integration   │
└──────────────────┘   └────────────────────────┘
```

## 📋 Component Responsibilities

### 1. StateSchema (Foundation)

**Purpose**: Base class for all state management in Haive

**Key Features**:

- Field sharing between parent/child graphs
- Reducer functions for state updates
- Engine I/O tracking
- Serialization support

**When to Use**:

- Creating any stateful component
- Defining graph or agent state
- Need field sharing or reducers

```python
class MyState(StateSchema):
    messages: List[BaseMessage] = Field(default_factory=list)
    count: int = Field(default=0)

    __shared_fields__ = ["messages"]  # Shared with subgraphs
    __reducer_fields__ = {
        "count": lambda old, new: old + new  # Accumulator
    }
```

### 2. SchemaComposer (Dynamic Builder)

**Purpose**: Build schemas dynamically from various sources

**Key Features**:

- Extract fields from engines, models, dicts
- Manage field definitions with metadata
- Auto-detect appropriate base classes
- Track engine relationships

**When to Use**:

- Creating schemas at runtime
- Combining fields from multiple sources
- Need dynamic schema generation

```python
composer = SchemaComposer()
composer.add_fields_from_engine(my_engine)
composer.add_field("custom", str, default="")
MyDynamicState = composer.build_state_schema("MyDynamicState")
```

### 3. AgentSchemaComposer (Multi-Agent)

**Purpose**: Specialized composer for multi-agent systems

**Key Features**:

- Field separation strategies (smart, shared, namespaced)
- Multi-agent coordination patterns
- Preserves message fields with custom reducers
- Handles agent state composition

**When to Use**:

- Building multi-agent systems
- Need field isolation between agents
- Complex agent coordination

```python
composer = AgentSchemaComposer(separation_strategy="smart")
composer.add_agent_schema(agent1.state_schema)
composer.add_agent_schema(agent2.state_schema)
MultiAgentState = composer.build()
```

### 4. NodeSchemaComposer (Node I/O)

**Purpose**: Configure flexible I/O for nodes

**Key Features**:

- Arbitrary field mappings ("result → potato")
- Transform pipelines
- Callable wrapping
- Schema adapters

**When to Use**:

- Changing node input/output fields
- Creating nodes from functions
- Adapting between components

```python
# Basic usage
adapted = change_output_key(node, "documents", "retrieved_docs")

# Advanced usage
composer = NodeSchemaComposer()
node = composer.from_callable(
    func=process,
    output_mappings=[FieldMapping("result", "processed")]
)
```

### 5. IntegratedNodeComposer (Unified)

**Purpose**: Bridge NodeSchemaComposer with StateSchema system

**Key Features**:

- Automatic StateSchema generation for nodes
- Preserves field metadata through mappings
- Integrates with reducers and sharing
- Schema-aware node composition

**When to Use**:

- Need nodes that work with StateSchema
- Want field metadata preservation
- Building schema-aware pipelines

```python
composer = IntegratedNodeComposer()
node = composer.compose_node_with_schema(
    base_node=my_node,
    state_schema=MyStateSchema,
    output_mappings=[FieldMapping("result", "processed")]
)
```

## 🔄 Unified Patterns

### Pattern 1: Schema-First Development

```python
# 1. Define your state schema
class WorkflowState(StateSchema):
    query: str
    documents: List[Dict] = Field(default_factory=list)
    response: Optional[str] = None

    __shared_fields__ = ["query"]  # Shared with subgraphs

# 2. Create nodes that work with schema
@with_state_schema(WorkflowState)
def retrieve(state: WorkflowState) -> Dict[str, Any]:
    # Retrieves documents based on query
    return {"documents": fetch_documents(state.query)}

# 3. Adapt existing nodes to schema
retriever = integrate_node_with_schema(
    existing_retriever_node,
    WorkflowState,
    output_mappings=[FieldMapping("results", "documents")]
)
```

### Pattern 2: Dynamic Schema Composition

```python
# 1. Build schema from components
composer = SchemaComposer()
composer.add_fields_from_engine(llm_engine)
composer.add_fields_from_engine(retriever_engine)
composer.add_field("workflow_id", str)

# 2. Generate state schema
WorkflowState = composer.build_state_schema("WorkflowState")

# 3. Create nodes using the schema
node_composer = IntegratedNodeComposer()
processor = node_composer.from_callable_with_schema(
    func=process_data,
    input_schema=WorkflowState
)
```

### Pattern 3: Multi-Agent with Flexible I/O

```python
# 1. Define agent schemas
agent_composer = AgentSchemaComposer()
agent_composer.add_agent_schema(PlannerState)
agent_composer.add_agent_schema(ExecutorState)
MultiAgentState = agent_composer.build()

# 2. Create agents with I/O adaptation
planner = create_schema_aware_node(
    plan_function,
    MultiAgentState,
    output_mappings=[FieldMapping("plan", "execution_plan")]
)

executor = create_schema_aware_node(
    execute_function,
    MultiAgentState,
    input_mappings=[FieldMapping("execution_plan", "plan")]
)
```

## 📐 Design Principles

### 1. Consistency

- All composers follow similar patterns
- Unified naming conventions
- Consistent API design

### 2. Composability

- Components work together seamlessly
- Can mix and match composers
- Build complex from simple

### 3. Type Safety

- StateSchema provides type hints
- Optional runtime validation
- Preserves type information

### 4. Flexibility

- Multiple usage patterns
- Extensible architecture
- Backward compatible

## 🚀 Migration Guide

### From Raw Nodes to Schema-Aware

```python
# Before: Raw node
def process(state):
    return {"result": state["data"]}

# After: Schema-aware node
@with_state_schema(MyStateSchema)
def process(state: MyStateSchema):
    return {"result": state.data}
```

### From Manual Mapping to Composers

```python
# Before: Manual field mapping
result = node(state)
adapted_result = {
    "processed_data": result["data"],
    "status": result["status"]
}

# After: Declarative mapping
node = composer.compose_node(
    base_node,
    output_mappings=[
        FieldMapping("data", "processed_data"),
        FieldMapping("status", "status")
    ]
)
```

### From Separate Schemas to Unified

```python
# Before: Separate schemas
class NodeInput(BaseModel):
    query: str

class NodeOutput(BaseModel):
    response: str

# After: Unified StateSchema
class NodeState(StateSchema):
    query: str
    response: Optional[str] = None

    __engine_io_mappings__ = {
        "my_engine": {
            "inputs": ["query"],
            "outputs": ["response"]
        }
    }
```

## 📊 Decision Matrix

| Need                     | Use This Component     |
| ------------------------ | ---------------------- |
| Basic state management   | StateSchema            |
| Dynamic schema creation  | SchemaComposer         |
| Multi-agent coordination | AgentSchemaComposer    |
| Change node I/O fields   | NodeSchemaComposer     |
| Nodes with StateSchema   | IntegratedNodeComposer |
| Field sharing/reducers   | StateSchema features   |
| Runtime schema building  | SchemaComposer         |
| Agent state isolation    | AgentSchemaComposer    |
| Function to node         | NodeSchemaComposer     |
| Type-safe nodes          | IntegratedNodeComposer |

## 🔧 Implementation Checklist

### Phase 1: Clean Up Architecture

- [ ] Consolidate duplicate schema_composer.py files
- [ ] Update imports to use consistent paths
- [ ] Document intended use for each composer
- [ ] Create migration guide for existing code

### Phase 2: Enhance Integration

- [ ] Make NodeSchemaComposer extend SchemaComposer
- [ ] Add StateSchema support to all composers
- [ ] Create unified factory functions
- [ ] Build comprehensive examples

### Phase 3: Documentation

- [ ] Update all README files
- [ ] Create architecture diagrams
- [ ] Write best practices guide
- [ ] Build migration tools

### Phase 4: Testing

- [ ] Test all integration points
- [ ] Verify backward compatibility
- [ ] Performance benchmarks
- [ ] Real-world scenarios

## 🎯 Benefits of Unified Architecture

1. **Consistency**: One way to do schema management
2. **Power**: Combine features from all composers
3. **Flexibility**: Multiple usage patterns
4. **Type Safety**: StateSchema throughout
5. **Maintainability**: Clear component boundaries
6. **Extensibility**: Easy to add new features

## 📚 Examples Repository

### Basic Examples

- Creating StateSchema
- Using SchemaComposer
- Node I/O mapping
- Multi-agent schemas

### Advanced Examples

- Dynamic schema generation
- Schema-aware pipelines
- Complex field mappings
- Multi-agent coordination

### Real-World Patterns

- RAG with flexible I/O
- Multi-agent workflows
- Dynamic node composition
- Schema migration

## 🚦 Next Steps

1. **Immediate**: Use IntegratedNodeComposer for new nodes
2. **Short-term**: Migrate existing nodes to schema-aware
3. **Medium-term**: Consolidate schema files
4. **Long-term**: Full unified architecture

The unified schema architecture provides a consistent, powerful way to manage state and I/O throughout the Haive framework. Start with the component that best fits your needs and expand as requirements grow.
