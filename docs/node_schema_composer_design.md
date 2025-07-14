# NodeSchemaComposer Design Document

## Vision

Create a flexible, pluggable system for node I/O that enables:

- Arbitrary field mappings (e.g., "result → potato")
- Pluggable extract/update functions
- Dynamic schema adaptation
- Type-safe transformations
- Backward compatibility with existing nodes

## Enhanced Design Based on Node Analysis

### Key Insights from 6-Node Analysis

1. **Extraction Complexity Levels**:
   - **Simple**: Direct field access (ValidationNodeV2)
   - **Medium**: Configurable fields with type handling (ToolNodeConfig, OutputParserNode)
   - **Advanced**: Multi-strategy with fallbacks (EngineNode, ParserNodeV2)
   - **Projection**: Hierarchical state transformation (AgentNodeV3)

2. **Update Patterns**:
   - **Message-centric**: Append to messages list (ToolNodeConfig, ValidationNodeV2)
   - **Dynamic field**: Result-based field naming (OutputParserNode, ParserNodeV2)
   - **Type-aware**: Different handling per result type (EngineNode)
   - **Hierarchical**: Nested state updates (AgentNodeV3)

3. **Special Features to Preserve**:
   - Safety net creation (ParserNodeV2)
   - Multi-source extraction with fallbacks (ParserNodeV2)
   - Projection-based extraction (AgentNodeV3)
   - Type-specific strategies (EngineNode)

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

### 4. Enhanced Extract Functions Library

```python
class CommonExtractFunctions:
    """Library of reusable extract functions based on node patterns."""

    # Basic Extraction (ValidationNodeV2 pattern)
    @staticmethod
    def extract_simple(state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simple field extraction with defaults."""
        fields = config.get("fields", ["messages"])
        return {field: getattr(state, field, None) for field in fields}

    # Message Extraction (Multiple nodes)
    @staticmethod
    def extract_messages(state: Any, config: Dict[str, Any]) -> List[BaseMessage]:
        """Extract messages with enhanced MessageList support."""
        field_name = config.get("field_name", "messages")
        messages = getattr(state, field_name, [])

        # Handle enhanced MessageList (ParserNodeV2 pattern)
        if hasattr(messages, "root"):
            return messages.root
        return list(messages) if messages else []

    # Multi-Strategy Extraction (EngineNode pattern)
    @staticmethod
    def extract_with_fallback(state: Any, config: Dict[str, Any]) -> Any:
        """Multi-strategy extraction with fallbacks."""
        strategies = config.get("strategies", [])

        for strategy in strategies:
            try:
                if strategy["type"] == "field":
                    value = getattr(state, strategy["name"], None)
                elif strategy["type"] == "dict_key":
                    value = state.get(strategy["name"]) if hasattr(state, "get") else None
                elif strategy["type"] == "nested":
                    value = FieldMappingEngine().extract_value(state, strategy["path"])
                elif strategy["type"] == "computed":
                    value = strategy["compute_fn"](state)

                if value is not None:
                    return value
            except:
                continue

        return config.get("default")

    # Projection Extraction (AgentNodeV3 pattern)
    @staticmethod
    def extract_with_projection(state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Project hierarchical state to flat structure."""
        # Get base state
        base_field = config.get("base_field", "agent_states")
        projection_key = config.get("projection_key", "active_agent")
        shared_fields = config.get("shared_fields", ["messages"])

        # Get projected state
        base_states = getattr(state, base_field, {})
        active_key = getattr(state, projection_key, None)
        projected = base_states.get(active_key, {}).copy() if active_key else {}

        # Add shared fields
        for field in shared_fields:
            if hasattr(state, field) and field not in projected:
                projected[field] = getattr(state, field)

        return projected

    # Tool Extraction (ParserNodeV2 pattern)
    @staticmethod
    def extract_tool_info(state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool information from messages."""
        messages = CommonExtractFunctions.extract_messages(state, config)

        # Find last AI message with tool calls
        for msg in reversed(messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call = msg.tool_calls[-1]
                return {
                    "tool_name": getattr(tool_call, "name", tool_call.get("name")),
                    "tool_id": getattr(tool_call, "id", tool_call.get("id")),
                    "tool_args": getattr(tool_call, "args", tool_call.get("args", {}))
                }

        return {}

    # Content Extraction (OutputParserNode pattern)
    @staticmethod
    def extract_message_content(state: Any, config: Dict[str, Any]) -> str:
        """Extract content from various message types."""
        messages = CommonExtractFunctions.extract_messages(state, config)
        parse_all = config.get("parse_all", False)

        messages_to_parse = messages if parse_all else [messages[-1]] if messages else []
        contents = []

        for msg in messages_to_parse:
            content = None
            if hasattr(msg, "content"):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get("content", msg.get("text", msg.get("message")))
            elif isinstance(msg, str):
                content = msg

            if content:
                contents.append(content)

        return contents if parse_all else (contents[0] if contents else "")
```

### 5. Enhanced Update Functions Library

```python
class CommonUpdateFunctions:
    """Library of reusable update functions based on node patterns."""

    # Message Updates (ToolNodeConfig, ValidationNodeV2 pattern)
    @staticmethod
    def update_messages(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update messages list with proper handling."""
        field_name = config.get("field_name", "messages")
        messages = list(getattr(state, field_name, []))

        # Convert result to message if needed
        if config.get("convert_to_message", False):
            from langchain_core.messages import AIMessage, ToolMessage
            if isinstance(result, str):
                result = AIMessage(content=result)
            elif isinstance(result, dict) and "tool_call_id" in result:
                result = ToolMessage(**result)

        if isinstance(result, list):
            messages.extend(result)
        else:
            messages.append(result)

        return {field_name: messages}

    # Type-Aware Updates (EngineNode pattern)
    @staticmethod
    def update_type_aware(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update based on result type with smart mapping."""
        type_map = config.get("type_map", {
            "BaseMessage": "messages",
            "dict": "pass_through",
            "str": "response",
            "list": "results",
            "default": "output"
        })

        # Determine result type
        result_type = type(result).__name__
        if hasattr(result, "_type"):  # LangChain messages
            result_type = "BaseMessage"

        # Get update strategy
        strategy = type_map.get(result_type, type_map.get("default", "output"))

        if strategy == "messages":
            return CommonUpdateFunctions.update_messages(result, state, config)
        elif strategy == "pass_through" and isinstance(result, dict):
            return result
        else:
            return {strategy: result}

    # Hierarchical Updates (AgentNodeV3 pattern)
    @staticmethod
    def update_hierarchical(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update hierarchical state structures."""
        base_field = config.get("base_field", "agent_states")
        key_field = config.get("key_field", "active_agent")
        output_field = config.get("output_field", "agent_outputs")
        mode = config.get("mode", "merge")  # merge, replace, isolate

        # Get current states
        base_states = getattr(state, base_field, {})
        active_key = getattr(state, key_field, None)

        if not active_key:
            return {}

        # Process result
        result_dict = result if isinstance(result, dict) else {"result": result}
        current_state = base_states.get(active_key, {})

        # Apply update mode
        if mode == "merge":
            updated_state = {**current_state, **result_dict}
        elif mode == "replace":
            updated_state = result_dict
        else:  # isolate
            updated_state = current_state

        # Create updates
        update = {
            base_field: {**base_states, active_key: updated_state},
            output_field: {**getattr(state, output_field, {}), active_key: result_dict}
        }

        # Update shared fields if configured
        if config.get("update_shared", True) and "messages" in result_dict:
            update["messages"] = result_dict["messages"]

        return update

    # Dynamic Field Updates (OutputParserNode, ParserNodeV2 pattern)
    @staticmethod
    def update_dynamic_field(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update with dynamic field naming."""
        # Determine field name
        field_name = config.get("field_name")

        if not field_name and hasattr(result, "__class__"):
            # Use class name as field name
            from haive.core.schema.field_utils import create_field_name_from_model
            field_name = create_field_name_from_model(result.__class__)

        if not field_name:
            field_name = config.get("default_field", "parsed_output")

        # Add error fields if configured
        update = {field_name: result}

        if config.get("include_error_fields", True):
            update[config.get("error_field", "parse_error")] = None
            update[config.get("raw_field", "raw_content")] = None

        return update

    # Safety Net Updates (ParserNodeV2 pattern)
    @staticmethod
    def update_with_safety_net(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update with safety net for missing data."""
        base_update = CommonUpdateFunctions.update_dynamic_field(result, state, config)

        if config.get("safety_net_enabled", False):
            messages = getattr(state, "messages", [])
            tool_info = CommonExtractFunctions.extract_tool_info(state, {})

            # Check if ToolMessage exists
            has_tool_message = any(
                hasattr(msg, "tool_call_id") and msg.tool_call_id == tool_info.get("tool_id")
                for msg in messages
            )

            if not has_tool_message and tool_info.get("tool_id"):
                # Create safety net message
                from langchain_core.messages import ToolMessage
                import json

                content = json.dumps(result.model_dump()) if hasattr(result, "model_dump") else str(result)
                safety_message = ToolMessage(
                    content=content,
                    tool_call_id=tool_info["tool_id"],
                    name=tool_info.get("tool_name", "unknown"),
                    additional_kwargs={"created_by": "safety_net"}
                )

                # Add to messages
                base_update["messages"] = list(messages) + [safety_message]

        return base_update

    # Mapped Field Updates (Enhanced)
    @staticmethod
    def update_mapped_fields(result: Any, state: Any, config: Dict[str, Any]) -> Dict[str, Any]:
        """Update fields using mapping configuration."""
        mappings = config.get("mappings", [])
        engine = FieldMappingEngine()
        update = {}

        for mapping in mappings:
            # Extract value from result
            value = engine.extract_value(result, mapping.source_path)

            # Apply transforms
            if mapping.transform:
                value = engine.apply_transforms(value, mapping.transform)

            # Handle nested updates
            if "." in mapping.target_path:
                nested_update = engine.create_nested_update(mapping.target_path, value)
                update = engine.merge_updates(update, nested_update)
            else:
                update[mapping.target_path] = value

        # Apply merge strategy if configured
        if config.get("merge_with_state", False):
            for field, value in update.items():
                if hasattr(state, field) and isinstance(getattr(state, field), dict):
                    current = getattr(state, field)
                    update[field] = {**current, **value} if isinstance(value, dict) else value

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
