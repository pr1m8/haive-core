# NodeSchemaComposer Implementation & Testing Plan

## Implementation Overview

Based on the analysis of 6 node types, we'll implement a flexible I/O system that unifies patterns while preserving sophisticated features.

## Phase 1: Core Infrastructure (Week 1)

### 1.1 Protocol Definitions

```python
# File: haive/core/graph/node/composer/protocols.py

from typing import Any, Dict, Protocol, TypeVar
from pydantic import BaseModel

TState = TypeVar("TState", bound=BaseModel)
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

class ExtractFunction(Protocol[TState, TInput]):
    """Protocol for extract functions."""
    def __call__(self, state: TState, config: Dict[str, Any]) -> TInput: ...

class UpdateFunction(Protocol[TState, TOutput]):
    """Protocol for update functions."""
    def __call__(self, result: TOutput, state: TState, config: Dict[str, Any]) -> Dict[str, Any]: ...

class TransformFunction(Protocol):
    """Protocol for transform functions."""
    def __call__(self, value: Any) -> Any: ...
```

### 1.2 Field Mapping Engine

```python
# File: haive/core/graph/node/composer/field_mapping.py

from dataclasses import dataclass
from typing import Any, List, Optional
import re

@dataclass
class FieldMapping:
    source_path: str
    target_path: str
    transform: Optional[List[str]] = None
    default: Any = None
    required: bool = False

class FieldMappingEngine:
    """Engine for path-based field access and transformation."""

    def extract_value(self, obj: Any, path: str) -> Any:
        """Extract value using path notation."""
        # Implementation for:
        # - Dot notation: "field.subfield"
        # - Array index: "messages[0]", "messages[-1]"
        # - Method calls: "content.strip()"
        # - Wildcards: "messages[*].content"
```

### 1.3 Extract Functions Library

```python
# File: haive/core/graph/node/composer/extract_functions.py

# Implementation of all extract patterns from design doc:
# - extract_simple
# - extract_messages
# - extract_with_fallback
# - extract_with_projection
# - extract_tool_info
# - extract_message_content
```

### 1.4 Update Functions Library

```python
# File: haive/core/graph/node/composer/update_functions.py

# Implementation of all update patterns from design doc:
# - update_messages
# - update_type_aware
# - update_hierarchical
# - update_dynamic_field
# - update_with_safety_net
# - update_mapped_fields
```

## Phase 2: NodeSchemaComposer Core (Week 2)

### 2.1 Main Composer Class

```python
# File: haive/core/graph/node/composer/composer.py

class NodeSchemaComposer:
    """Main composer for flexible node I/O."""

    def __init__(self):
        self._register_defaults()

    def compose_node(self, base_node_class, **config):
        """Create composed node with flexible I/O."""
        # Implementation
```

### 2.2 Integration with BaseNodeConfig

```python
# File: haive/core/graph/node/base_node_config.py (enhancement)

class BaseNodeConfig:
    # Add composer support
    extract_function: Optional[Union[str, ExtractFunction]] = None
    extract_config: Optional[Dict[str, Any]] = None
    update_function: Optional[Union[str, UpdateFunction]] = None
    update_config: Optional[Dict[str, Any]] = None
```

## Phase 3: Node Migration (Week 3)

### 3.1 Migration Order (Low to High Risk)

1. **ValidationNodeV2** - Simple patterns, good test case
2. **OutputParserNode** - Medium complexity
3. **ToolNodeConfig** - Message-focused updates
4. **ParserNodeV2** - Complex extraction, safety net
5. **EngineNode** - Multi-strategy extraction
6. **AgentNodeV3** - Hierarchical projection

### 3.2 Migration Strategy

For each node:

1. Identify extract/update patterns
2. Map to composer functions
3. Create configuration
4. Test with real components
5. Maintain backward compatibility

## Testing Plan (No Mocks)

### Test Infrastructure

```python
# File: tests/conftest.py

import pytest
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.schema.prebuilt.messages_state import MessagesState
from haive.core.schema.prebuilt.tool_state import ToolState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

@pytest.fixture
def real_engine():
    """Provide real AugLLMConfig for testing."""
    return AugLLMConfig(temperature=0.1)  # Low temp for consistency

@pytest.fixture
def real_state():
    """Provide real state with messages."""
    return MessagesState(messages=[
        HumanMessage("Test message"),
        AIMessage("Response", tool_calls=[{
            "id": "test_123",
            "name": "TestTool",
            "args": {"query": "test"}
        }])
    ])

@pytest.fixture
def real_tools():
    """Provide real tool implementations."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> float:
        """Calculate mathematical expression."""
        return eval(expression)

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    return [calculator, search]
```

### Test Suite Structure

#### 1. Field Mapping Engine Tests

```python
# File: tests/test_field_mapping_engine.py

def test_path_extraction_with_real_state():
    """Test path extraction with real state objects."""
    engine = FieldMappingEngine()
    state = MessagesState(messages=[
        HumanMessage("Hello"),
        AIMessage("Hi there")
    ])

    # Test various path patterns
    assert engine.extract_value(state, "messages[0].content") == "Hello"
    assert engine.extract_value(state, "messages[-1].content") == "Hi there"
    assert engine.extract_value(state, "messages[*].content") == ["Hello", "Hi there"]

def test_transform_pipeline():
    """Test transformation functions."""
    engine = FieldMappingEngine()

    # Test built-in transforms
    assert engine.apply_transforms("hello world", ["uppercase"]) == "HELLO WORLD"
    assert engine.apply_transforms("  test  ", ["strip", "uppercase"]) == "TEST"
    assert engine.apply_transforms('{"key": "value"}', ["parse_json"]) == {"key": "value"}
```

#### 2. Extract Function Tests

```python
# File: tests/test_extract_functions.py

def test_extract_messages_from_real_state():
    """Test message extraction with real MessagesState."""
    state = MessagesState(messages=[
        HumanMessage("Test"),
        AIMessage("Response")
    ])

    messages = CommonExtractFunctions.extract_messages(state, {})
    assert len(messages) == 2
    assert messages[0].content == "Test"

def test_extract_tool_info_from_real_messages():
    """Test tool extraction with real AIMessage."""
    state = MessagesState(messages=[
        AIMessage("Processing", tool_calls=[{
            "id": "123",
            "name": "calculator",
            "args": {"expression": "2+2"}
        }])
    ])

    tool_info = CommonExtractFunctions.extract_tool_info(state, {})
    assert tool_info["tool_name"] == "calculator"
    assert tool_info["tool_id"] == "123"
    assert tool_info["tool_args"]["expression"] == "2+2"

def test_projection_extraction_with_hierarchical_state():
    """Test projection with real hierarchical state."""
    from haive.agents.multi.state import MultiAgentState

    state = MultiAgentState(
        messages=[HumanMessage("Global message")],
        agent_states={
            "agent1": {"local_data": "value1"},
            "agent2": {"local_data": "value2"}
        },
        active_agent="agent1"
    )

    projected = CommonExtractFunctions.extract_with_projection(state, {
        "base_field": "agent_states",
        "projection_key": "active_agent",
        "shared_fields": ["messages"]
    })

    assert projected["local_data"] == "value1"
    assert len(projected["messages"]) == 1
```

#### 3. Update Function Tests

```python
# File: tests/test_update_functions.py

def test_message_update_with_real_components():
    """Test message updates with real messages."""
    state = MessagesState(messages=[HumanMessage("Initial")])
    result = AIMessage("Response")

    update = CommonUpdateFunctions.update_messages(result, state, {})
    assert len(update["messages"]) == 2
    assert update["messages"][-1].content == "Response"

def test_type_aware_update_with_real_results():
    """Test type-aware updates with various result types."""
    state = MessagesState()

    # Test with message result
    msg_result = AIMessage("Test")
    update = CommonUpdateFunctions.update_type_aware(msg_result, state, {})
    assert "messages" in update

    # Test with dict result
    dict_result = {"key": "value"}
    update = CommonUpdateFunctions.update_type_aware(dict_result, state, {})
    assert update == dict_result

    # Test with string result
    str_result = "Simple string"
    update = CommonUpdateFunctions.update_type_aware(str_result, state, {})
    assert update["response"] == "Simple string"

def test_safety_net_with_real_tool_messages():
    """Test safety net creation with real components."""
    from pydantic import BaseModel

    class TestResult(BaseModel):
        value: str = "test"

    state = MessagesState(messages=[
        AIMessage("Process", tool_calls=[{
            "id": "123",
            "name": "TestTool",
            "args": {}
        }])
    ])

    result = TestResult()
    update = CommonUpdateFunctions.update_with_safety_net(result, state, {
        "safety_net_enabled": True
    })

    # Should create ToolMessage
    assert "messages" in update
    assert any(isinstance(msg, ToolMessage) for msg in update["messages"])
```

#### 4. Integration Tests

```python
# File: tests/test_node_composer_integration.py

def test_compose_engine_node_with_mapping():
    """Test full node composition with real engine."""
    composer = NodeSchemaComposer()
    engine = AugLLMConfig()

    # Compose node with "result → potato" mapping
    node = composer.compose_node(
        EngineNodeConfig,
        name="test_node",
        engine=engine,
        extract_function="extract_messages",
        update_mappings=[
            FieldMapping(source_path="content", target_path="potato")
        ]
    )

    # Test with real state
    state = MessagesState(messages=[HumanMessage("Calculate 2+2")])
    result = node(state)

    assert "potato" in result.update
    assert result.update["potato"] is not None

def test_hierarchical_node_composition():
    """Test AgentNodeV3 pattern with real multi-agent state."""
    composer = NodeSchemaComposer()

    node = composer.compose_node(
        AgentNodeConfig,
        name="agent1_node",
        extract_function="extract_with_projection",
        extract_config={
            "base_field": "agent_states",
            "projection_key": "active_agent"
        },
        update_function="update_hierarchical",
        update_config={
            "mode": "merge"
        }
    )

    # Test with real hierarchical state
    from haive.agents.multi.state import MultiAgentState
    state = MultiAgentState(
        agent_states={"agent1": {"data": "initial"}},
        active_agent="agent1"
    )

    # Simulate agent execution
    result = {"data": "updated", "new_field": "value"}
    command = node._create_update(result, state)

    assert command.update["agent_states"]["agent1"]["new_field"] == "value"
```

#### 5. Performance Tests

```python
# File: tests/test_performance.py

def test_field_mapping_performance():
    """Test field mapping performance with large states."""
    import time

    # Create large state
    messages = [HumanMessage(f"Message {i}") for i in range(1000)]
    state = MessagesState(messages=messages)

    engine = FieldMappingEngine()

    # Test extraction performance
    start = time.time()
    for _ in range(100):
        engine.extract_value(state, "messages[-1].content")
    duration = time.time() - start

    assert duration < 0.1  # Should be fast

def test_composer_overhead():
    """Test overhead of composed nodes vs direct implementation."""
    # Compare composed node performance with original
    # Ensure minimal overhead
```

#### 6. Backward Compatibility Tests

```python
# File: tests/test_backward_compatibility.py

def test_existing_nodes_continue_working():
    """Ensure existing nodes work without changes."""
    # Test each node type works as before
    # No breaking changes allowed
```

### Test Execution Plan

1. **Unit Tests** - Test each component in isolation

   ```bash
   poetry run pytest tests/test_field_mapping_engine.py -v
   poetry run pytest tests/test_extract_functions.py -v
   poetry run pytest tests/test_update_functions.py -v
   ```

2. **Integration Tests** - Test composed nodes

   ```bash
   poetry run pytest tests/test_node_composer_integration.py -v
   ```

3. **Migration Tests** - Test each migrated node

   ```bash
   poetry run pytest tests/test_migrated_nodes/ -v
   ```

4. **Performance Tests** - Ensure no regression

   ```bash
   poetry run pytest tests/test_performance.py -v
   ```

5. **Full Suite** - All tests must pass
   ```bash
   poetry run pytest --cov=haive.core.graph.node.composer
   ```

## Success Criteria

1. **Functionality**:
   - All 6 node types can be composed
   - "result → potato" mappings work
   - Complex paths like "messages[-1].content" work
   - Transform pipelines function correctly

2. **Performance**:
   - <5% overhead vs direct implementation
   - Field extraction <1ms for typical paths
   - No memory leaks

3. **Compatibility**:
   - All existing nodes continue to work
   - No breaking API changes
   - Migration path is clear

4. **Testing**:
   - 100% test coverage with real components
   - Zero mocks in test suite
   - All edge cases covered

## Risk Mitigation

1. **Performance Risk**: Profile critical paths, optimize hot loops
2. **Compatibility Risk**: Extensive backward compatibility tests
3. **Complexity Risk**: Clear documentation and examples
4. **Migration Risk**: Gradual rollout, one node at a time

## Timeline

- **Week 1**: Core infrastructure + tests
- **Week 2**: NodeSchemaComposer + integration
- **Week 3**: Node migration (2 per day)
- **Week 4**: Documentation + optimization

Total: 4 weeks to full implementation with comprehensive testing.
