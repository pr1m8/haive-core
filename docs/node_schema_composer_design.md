# NodeSchemaComposer Design Document

## Vision

Create a flexible, pluggable system for node I/O that enables:

- Arbitrary field mappings (e.g., "result → potato")
- Pluggable extract/update functions
- Dynamic schema adaptation
- Type-safe transformations
- Backward compatibility with existing nodes

## Core Architecture

### 1. Extract/Update Function System

```python
from typing import Any, Callable, Dict, Protocol, TypeVar
from pydantic import BaseModel

TState = TypeVar("TState", bound=BaseModel)
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")

class ExtractFunction(Protocol[TState, TInput]):
    """Protocol for extract functions."""

    def __call__(
        self,
        state: TState,
        config: Dict[str, Any]
    ) -> TInput:
        """Extract input from state."""
        ...

class UpdateFunction(Protocol[TState, TOutput]):
    """Protocol for update functions."""

    def __call__(
        self,
        result: TOutput,
        state: TState,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create state update from result."""
        ...
```

### 2. Field Mapping DSL

```python
from typing import Union, List
from dataclasses import dataclass

@dataclass
class FieldMapping:
    """Single field mapping configuration."""
    source_path: str  # e.g., "messages[-1].content"
    target_path: str  # e.g., "potato"
    transform: Optional[List[str]] = None  # e.g., ["uppercase", "strip"]
    default: Any = None
    required: bool = False

class FieldMappingEngine:
    """Engine for executing field mappings."""

    def extract_value(self, obj: Any, path: str) -> Any:
        """Extract value from object using path."""
        # Handle:
        # - Dot notation: "field.subfield"
        # - Array index: "messages[0]"
        # - Array slice: "messages[-1]"
        # - Method calls: "content.strip()"

    def apply_transforms(self, value: Any, transforms: List[str]) -> Any:
        """Apply transform pipeline to value."""
        # Built-in transforms:
        # - uppercase, lowercase, strip
        # - parse_json, to_json
        # - first, last, join
        # - Custom transforms via registry
```

### 3. NodeSchemaComposer

```python
from typing import Dict, Optional, Type
from haive.core.graph.node.base_node_config import BaseNodeConfig

class NodeSchemaComposer:
    """Composer for flexible node I/O configuration."""

    def __init__(self):
        self.extract_functions: Dict[str, ExtractFunction] = {}
        self.update_functions: Dict[str, UpdateFunction] = {}
        self.transform_functions: Dict[str, Callable] = {}
        self._register_defaults()

    def compose_node(
        self,
        base_node: Type[BaseNodeConfig],
        name: str,
        *,
        # Extract configuration
        extract_function: Optional[Union[str, ExtractFunction]] = None,
        extract_mappings: Optional[List[FieldMapping]] = None,
        extract_config: Optional[Dict[str, Any]] = None,

        # Update configuration
        update_function: Optional[Union[str, UpdateFunction]] = None,
        update_mappings: Optional[List[FieldMapping]] = None,
        update_config: Optional[Dict[str, Any]] = None,

        # Node-specific config
        **node_kwargs
    ) -> BaseNodeConfig:
        """Compose a node with flexible I/O configuration."""
```

### 4. Common Extract Functions

```python
class CommonExtractFunctions:
    """Library of reusable extract functions."""

    @staticmethod
    def extract_messages(state: Any, config: Dict[str, Any]) -> List[BaseMessage]:
        """Extract messages from state."""
        field_name = config.get("field_name", "messages")
        return getattr(state, field_name, [])

    @staticmethod
    def extract_mapped_fields(state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields using mapping configuration."""
        mappings = config.get("mappings", [])
        engine = FieldMappingEngine()
        result = {}

        for mapping in mappings:
            value = engine.extract_value(state, mapping.source_path)
            if mapping.transform:
                value = engine.apply_transforms(value, mapping.transform)
            result[mapping.target_path] = value

        return result

    @staticmethod
    def extract_last_message_content(state: Any, config: Dict[str, Any]) -> str:
        """Extract content from last message."""
        messages = CommonExtractFunctions.extract_messages(state, config)
        if messages:
            return messages[-1].content
        return config.get("default", "")
```

### 5. Common Update Functions

```python
class CommonUpdateFunctions:
    """Library of reusable update functions."""

    @staticmethod
    def update_messages(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update messages list with result."""
        field_name = config.get("field_name", "messages")
        messages = list(getattr(state, field_name, []))

        if isinstance(result, list):
            messages.extend(result)
        else:
            messages.append(result)

        return {field_name: messages}

    @staticmethod
    def update_mapped_fields(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields using mapping configuration."""
        mappings = config.get("mappings", [])
        engine = FieldMappingEngine()
        update = {}

        for mapping in mappings:
            value = engine.extract_value(result, mapping.source_path)
            if mapping.transform:
                value = engine.apply_transforms(value, mapping.transform)

            # Handle nested updates
            if "." in mapping.target_path:
                # Create nested structure
                nested_update = engine.create_nested_update(mapping.target_path, value)
                update = engine.merge_updates(update, nested_update)
            else:
                update[mapping.target_path] = value

        return update

    @staticmethod
    def update_with_merge(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge result with existing state fields."""
        merge_fields = config.get("merge_fields", [])
        update = {}

        if isinstance(result, dict):
            for field in merge_fields:
                if field in result:
                    current = getattr(state, field, {})
                    if isinstance(current, dict):
                        update[field] = {**current, **result[field]}
                    else:
                        update[field] = result[field]

        return update
```

### 6. Usage Examples

#### Example 1: Simple Field Mapping

```python
composer = NodeSchemaComposer()

# Configure "result → potato" mapping
node = composer.compose_node(
    EngineNodeConfig,
    name="my_engine_node",
    engine=my_engine,

    # Extract: Get messages from state
    extract_function="extract_messages",

    # Update: Map result to potato field
    update_mappings=[
        FieldMapping(
            source_path="result",
            target_path="potato"
        )
    ]
)
```

#### Example 2: Complex Transformations

```python
# Configure complex field transformations
node = composer.compose_node(
    ToolNodeConfig,
    name="tool_processor",

    # Extract with transformations
    extract_mappings=[
        FieldMapping(
            source_path="messages[-1].content",
            target_path="query",
            transform=["strip", "lowercase"]
        ),
        FieldMapping(
            source_path="config.temperature",
            target_path="temp",
            default=0.7
        )
    ],

    # Update with nested paths
    update_mappings=[
        FieldMapping(
            source_path="tool_results[0].output",
            target_path="analysis.result",
            transform=["parse_json"]
        ),
        FieldMapping(
            source_path="tool_results[0].metadata.tokens",
            target_path="usage.tokens"
        )
    ]
)
```

#### Example 3: Custom Functions

```python
# Register custom extract function
def extract_rag_context(state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """Custom extraction for RAG agents."""
    return {
        "query": state.messages[-1].content,
        "context": state.retrieved_documents,
        "history": state.messages[:-1]
    }

composer.register_extract_function("rag_context", extract_rag_context)

# Use in node
node = composer.compose_node(
    AgentNodeConfig,
    name="rag_agent",
    extract_function="rag_context",
    update_function="update_messages"
)
```

### 7. Integration with Existing Nodes

```python
class FlexibleEngineNode(EngineNodeConfig):
    """Engine node with composed I/O."""

    extract_function: Optional[ExtractFunction] = None
    update_function: Optional[UpdateFunction] = None

    def __call__(self, state: StateLike, config: ConfigLike = None) -> Command:
        # Use composed functions if available
        if self.extract_function:
            input_data = self.extract_function(state, self.extract_config)
        else:
            # Fall back to existing extraction
            input_data = self._extract_smart_input(state, self.engine)

        # Execute engine
        result = self._execute_with_config(self.engine, input_data, config)

        # Use composed update if available
        if self.update_function:
            update = self.update_function(result, state, self.update_config)
        else:
            # Fall back to existing update
            update = self._smart_result_mapping(result, state, self.engine.engine_type)

        return Command(update=update, goto=self.command_goto)
```

### 8. Testing Strategy

```python
def test_field_mapping_engine():
    """Test field extraction and transformation."""
    engine = FieldMappingEngine()

    # Test data
    state = {
        "messages": [
            {"content": "Hello"},
            {"content": "World"}
        ],
        "config": {
            "temperature": 0.7
        }
    }

    # Test path extraction
    assert engine.extract_value(state, "messages[0].content") == "Hello"
    assert engine.extract_value(state, "messages[-1].content") == "World"
    assert engine.extract_value(state, "config.temperature") == 0.7

    # Test transforms
    assert engine.apply_transforms("hello", ["uppercase"]) == "HELLO"
    assert engine.apply_transforms("  world  ", ["strip", "uppercase"]) == "WORLD"

def test_node_composition():
    """Test full node composition with real components."""
    composer = NodeSchemaComposer()
    engine = AugLLMConfig()

    # Compose node with mappings
    node = composer.compose_node(
        EngineNodeConfig,
        name="test_node",
        engine=engine,
        update_mappings=[
            FieldMapping("content", "potato")
        ]
    )

    # Test with real state
    state = MessagesState(messages=[HumanMessage("Test")])
    result = node(state)

    assert "potato" in result.update
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

- [ ] Implement ExtractFunction and UpdateFunction protocols
- [ ] Create FieldMapping and FieldMappingEngine
- [ ] Build CommonExtractFunctions library
- [ ] Build CommonUpdateFunctions library

### Phase 2: NodeSchemaComposer (Week 2)

- [ ] Implement NodeSchemaComposer class
- [ ] Add function registration system
- [ ] Create composed node wrapper
- [ ] Add validation and error handling

### Phase 3: Integration (Week 3)

- [ ] Update BaseNodeConfig to support composition
- [ ] Migrate one node type as proof of concept
- [ ] Create migration utilities
- [ ] Write comprehensive tests

### Phase 4: Full Rollout (Week 4)

- [ ] Update all node types
- [ ] Create documentation
- [ ] Add more transform functions
- [ ] Performance optimization

## Benefits

1. **Flexibility**: Any field can map to any other field
2. **Reusability**: Common patterns become reusable functions
3. **Type Safety**: Protocols ensure correct signatures
4. **Backward Compatible**: Existing nodes continue to work
5. **Testability**: Each component is independently testable
6. **Extensibility**: Easy to add new functions and transforms

## Conclusion

The NodeSchemaComposer provides the missing flexibility in the current node system while building on existing patterns. It enables the "result → potato" style mappings requested while maintaining type safety and backward compatibility.
