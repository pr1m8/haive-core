# Node I/O Pattern Analysis Report

## Executive Summary

After analyzing six key node implementations, I've identified several patterns and opportunities for creating a flexible NodeSchemaComposer system. The current architecture has varying levels of flexibility, with some nodes hardcoded to specific fields while others have sophisticated extraction and mapping capabilities.

### Key Findings

1. **Three I/O Pattern Levels**:
   - **Hardcoded**: Direct field access (e.g., `state.messages`)
   - **Semi-Flexible**: Configurable field names (e.g., `messages_field = "messages"`)
   - **Flexible**: Schema-aware with field definitions and mapping

2. **Extract/Update Patterns**:
   - Most nodes have hardcoded extract functions
   - Update functions vary from simple dict creation to complex merging
   - Opportunity for pluggable extract/update functions

3. **Schema Awareness**:
   - Newer nodes (v2/v3) have schema support via `BaseNodeConfig`
   - Field definitions provide type safety
   - Missing: Dynamic field mapping like "result → potato"

## Per-Node Analysis

### 1. EngineNode (engine_node.py)

#### I/O Patterns

- **Input**: Smart extraction with multiple strategies
  - Explicit mapping via `input_fields`
  - Schema-defined inputs via `__engine_io_mappings__`
  - Engine-defined inputs
  - Type-based defaults
- **Output**: Smart result mapping
  - Message handling (updates messages list)
  - Dictionary pass-through
  - Single value mapping to typed fields

#### Extract Function

```python
# Current: Sophisticated multi-strategy extraction
def _extract_smart_input(self, state, engine):
    # 1. Explicit mapping
    # 2. Schema-defined inputs
    # 3. Engine-defined inputs
    # 4. Type-based defaults
```

#### Update Function

```python
# Current: Type-aware result mapping
def _smart_result_mapping(self, result, state, engine_type):
    # Handles messages, dicts, single values
    # Maps to engine-specific fields
```

#### Flexibility Assessment

- **Extract**: High - multiple strategies
- **Update**: Medium - type-based but not fully configurable
- **Schema**: High - uses field definitions

### 2. AgentNodeV3 (agent_node_v3.py)

#### I/O Patterns

- **Input**: Hierarchical state projection
  - Projects container state to agent schema
  - Shared fields from container
  - Agent-specific state isolation
- **Output**: Hierarchical state updates
  - Updates container's agent_states
  - Updates agent_outputs
  - Merge/replace/isolate modes

#### Extract Function

```python
# Current: Sophisticated projection
def _project_state_for_agent(self, state, agent):
    # Extracts agent's isolated state
    # Adds shared fields from container
    # Returns projected dict
```

#### Update Function

```python
# Current: Multi-mode updates
def _process_agent_output(self, result, state, agent):
    # Converts result to dict
    # Updates based on output_mode
    # Handles hierarchical updates
```

#### Flexibility Assessment

- **Extract**: High - projection-based
- **Update**: High - multiple modes
- **Schema**: Medium - aware but not fully flexible

### 3. ToolNodeConfig (tool_node_config_v2.py)

#### I/O Patterns

- **Input**: Tool-specific extraction
  - Messages from configurable field
  - Tool routes from state/engine
  - Engine lookup for tools
- **Output**: Message updates
  - Appends ToolMessages
  - Error handling fields

#### Extract Function

```python
# Current: Field-based extraction
def _get_messages_from_state(self, state):
    # Gets from configured field name
    # Handles dict/object states
```

#### Update Function

```python
# Current: Message-focused updates
def _create_tool_response(self, messages, tool_results):
    # Updates messages list
    # Adds error fields if needed
```

#### Flexibility Assessment

- **Extract**: Medium - field names configurable
- **Update**: Low - hardcoded to messages
- **Schema**: High - uses field definitions

### 4. OutputParserNodeConfig (output_parsing_v2.py)

#### I/O Patterns

- **Input**: Message content extraction
  - Configurable messages field
  - Multiple message support
- **Output**: Parsed results
  - Dynamic field naming
  - Error fields
  - Raw content on failure

#### Extract Function

```python
# Current: Message content extraction
def _extract_content_from_message(self, message):
    # Handles multiple message types
    # Extracts content intelligently
```

#### Update Function

```python
# Current: Result-based updates
def _create_success_response(self, parsed_result):
    # Maps to configurable output field
    # Includes error state
```

#### Flexibility Assessment

- **Extract**: Medium - content extraction logic
- **Update**: Medium - field name configurable
- **Schema**: High - dynamic field creation

### 5. ValidationNodeConfigV2 (validation_node_config_v2.py)

#### I/O Patterns

- **Input**: Tool call validation
  - Messages extraction (configurable field)
  - Tool routes from state/engine
  - Engine lookup for Pydantic models
- **Output**: Routing decisions
  - Creates ToolMessages for Pydantic validation
  - Routes to tool_node or parser_node
  - Updates messages with validation results

#### Extract Function

```python
# Current: Tool call extraction from AIMessage
def __call__(self, state):
    messages = state.get("messages", [])
    tool_calls = last_message.tool_calls
    tool_routes = state.get("tool_routes", {})
```

#### Update Function

```python
# Current: Message and routing updates
def __call__(self, state):
    new_messages = []  # ToolMessages for validation
    destinations = set()  # Routing destinations
    return Command(update={"messages": new_messages}, goto=destination)
```

#### Flexibility Assessment

- **Extract**: Low - hardcoded field access
- **Update**: Low - fixed to messages field
- **Schema**: Low - minimal schema awareness
- **Routing**: Medium - configurable node names

### 6. ParserNodeConfigV2 (parser_node_config_v2.py)

#### I/O Patterns

- **Input**: Complex tool message extraction
  - Messages from configurable field
  - Tool information from AIMessage
  - Engine lookup for tool/schema classes
  - ToolMessage content extraction
- **Output**: Parsed structured data
  - Dynamic field naming based on model
  - Safety net ToolMessage creation
  - Error handling with detailed messages

#### Extract Function

```python
# Current: Multi-step extraction
def _extract_tool_from_messages(self, messages):
    # 1. Find last AIMessage with tool calls
    # 2. Extract tool name and ID
    # 3. Find corresponding ToolMessage
    # 4. Extract content from various sources
```

#### Update Function

```python
# Current: Dynamic field updates with safety net
def __call__(self, state):
    # Parse content into model
    # Determine field name dynamically
    # Create safety net ToolMessage if missing
    return Command(update={field_name: parsed_result})
```

#### Flexibility Assessment

- **Extract**: High - sophisticated multi-source extraction
- **Update**: Medium - dynamic field naming
- **Schema**: High - uses field utilities
- **Safety**: High - safety net feature for missing messages

## Cross-Node Patterns

### Common Extract Patterns

1. **Field Access Pattern**:

   ```python
   # Direct access
   value = state.field_name

   # Configurable access
   value = getattr(state, self.field_name)

   # Safe access with fallback
   value = self._get_state_value(state, field_name, default)
   ```

2. **Multi-Source Pattern**:

   ```python
   # Try multiple sources in order
   # 1. State attribute
   # 2. Dict key
   # 3. Nested access
   # 4. Default value
   ```

3. **Type-Aware Pattern**:
   ```python
   # Different extraction based on type
   if engine_type == EngineType.RETRIEVER:
       return extract_retriever_fields()
   elif engine_type == EngineType.LLM:
       return extract_llm_fields()
   ```

### Common Update Patterns

1. **Message Update Pattern**:

   ```python
   # Get existing, append new
   messages = list(state.messages)
   messages.append(new_message)
   return {"messages": messages}
   ```

2. **Merge Pattern**:

   ```python
   # Merge with existing state
   current = getattr(state, "field", {})
   updated = {**current, **new_data}
   return {"field": updated}
   ```

3. **Conditional Update Pattern**:
   ```python
   # Update based on result type
   if isinstance(result, BaseMessage):
       return update_messages()
   elif isinstance(result, dict):
       return result
   else:
       return {default_field: result}
   ```

## Schema Integration Assessment

### Current Schema Awareness

1. **BaseNodeConfig Integration**:
   - Provides `input_schema` and `output_schema`
   - Field definitions for type safety
   - Default field generation

2. **Field Definition Usage**:
   - StandardFields registry for common patterns
   - Type-safe field creation
   - Metadata and validation

3. **Missing Capabilities**:
   - No runtime field mapping
   - No pluggable transform functions
   - Limited cross-field dependencies

### NodeSchemaComposer Integration Points

1. **Extract Function Hooks**:

   ```python
   # Instead of hardcoded extraction
   input_data = self.extract_fn(state, self.extract_config)
   ```

2. **Update Function Hooks**:

   ```python
   # Instead of hardcoded updates
   update_dict = self.update_fn(result, state, self.update_config)
   ```

3. **Field Mapping Configuration**:
   ```python
   # Enable "result → potato" mappings
   field_mappings = {
       "engine_output.result": "potato",
       "messages[-1].content": "last_message"
   }
   ```

## Recommendations

### 1. Priority Improvements

1. **Create Pluggable Extract/Update System**:
   - Define extract/update function interfaces
   - Create library of common functions
   - Allow custom function injection

2. **Implement Field Mapping DSL**:
   - Path-based field access (e.g., "messages[-1].content")
   - Transform functions (e.g., "uppercase", "parse_json")
   - Conditional mappings

3. **Enhance Schema Composition**:
   - Runtime schema modification
   - Field dependency resolution
   - Validation integration

### 2. NodeSchemaComposer Design

```python
class NodeSchemaComposer:
    """Compose node schemas with flexible I/O."""

    def __init__(self):
        self.extract_functions = {}
        self.update_functions = {}
        self.field_mappings = {}

    def register_extract_function(self, name: str, fn: ExtractFn):
        """Register reusable extract function."""

    def register_update_function(self, name: str, fn: UpdateFn):
        """Register reusable update function."""

    def compose_node_schema(
        self,
        node_config: BaseNodeConfig,
        extract_config: ExtractConfig,
        update_config: UpdateConfig,
        field_mappings: Dict[str, str]
    ) -> ComposedNodeConfig:
        """Compose node with flexible I/O."""
```

### 3. Implementation Roadmap

1. **Phase 1**: Extract/Update Function Library
   - Common extract patterns
   - Common update patterns
   - Function composition

2. **Phase 2**: Field Mapping System
   - Path-based access
   - Transform pipeline
   - Validation

3. **Phase 3**: Full Integration
   - Update existing nodes
   - Migration utilities
   - Testing framework

## Testing Requirements

### Real Component Testing

Each node needs tests with:

- Real engines (AugLLMConfig)
- Real state schemas
- Real tool implementations
- No mocks

### Test Scenarios

1. **Extract Function Tests**:
   - Various state types
   - Missing fields
   - Type conversions

2. **Update Function Tests**:
   - Result type variations
   - Merge scenarios
   - Error handling

3. **Field Mapping Tests**:
   - Complex paths
   - Transform chains
   - Error cases

## Conclusion

The current node system has sophisticated I/O handling in some areas but lacks the flexibility for arbitrary field mappings and pluggable functions. The NodeSchemaComposer can build on existing patterns while adding the missing flexibility for "result → potato" style mappings and dynamic schema adaptation.

The key is to preserve the good patterns (type awareness, multi-strategy extraction) while adding configurability where it's currently hardcoded.
