# Field Mapping in haive-core

**Package**: haive-core  
**Component**: Graph Node System  
**Status**: Available Now

## 🎯 Overview

Field mapping in haive-core allows you to transform data between different field names at the node level. This is implemented in `EngineNodeConfig` and provides the foundation for agent field coordination.

## 🚀 Quick Start

### Basic Field Mapping

```python
from haive.core.graph.node.engine_node import EngineNodeConfig
from haive.core.engine.aug_llm import AugLLMConfig

# Create engine node with field mapping
engine = AugLLMConfig(name="processor")
node = EngineNodeConfig(
    name="processor_node",
    engine=engine,
    output_fields={"result": "potato"}  # Maps engine.result → state.potato
)
```

### Multiple Field Mapping

```python
# Map multiple fields
node = EngineNodeConfig(
    name="analyzer",
    engine=engine,
    input_fields={
        "user_query": "query",      # state.user_query → engine.query
        "context_data": "context"   # state.context_data → engine.context
    },
    output_fields={
        "result": "analysis",       # engine.result → state.analysis
        "confidence": "score"       # engine.confidence → state.score
    }
)
```

## 📋 Field Mapping Formats

### List Format (Field Selection)

```python
# Select specific fields without renaming
input_fields = ["messages", "query"]      # Only extract these fields
output_fields = ["result", "metadata"]    # Only output these fields
```

### Dict Format (Field Renaming)

```python
# Rename fields during mapping
input_fields = {"old_name": "new_name"}   # old_name → new_name
output_fields = {"engine_field": "state_field"}
```

## 🔧 Implementation Details

### EngineNodeConfig Implementation

Located in: `haive/core/graph/node/engine_node.py`

```python
class EngineNodeConfig(NodeConfig):
    """Engine-based node with field mapping support."""

    # Field mappings (lines 31-32)
    input_fields: list[str] | dict[str, str] | None = Field(default=None)
    output_fields: list[str] | dict[str, str] | None = Field(default=None)

    def __call__(self, state: StateLike, config: ConfigLike = None):
        """Execute with field mapping."""
        # Input extraction with mapping (line 182-191)
        if self.input_fields:
            input_data = self._extract_mapped_input(state)
        else:
            input_data = self._extract_smart_input(state, engine)

        # Engine execution
        result = self._execute_with_config(engine, input_data, config)

        # Output mapping (line 213-221)
        if self.output_fields:
            wrapped = self._create_mapped_output(result)
        else:
            wrapped = self._wrap_smart_result(result, state, engine)
```

### Field Mapping Logic

#### Input Mapping

```python
def _extract_mapped_input(self, state):
    """Extract input with field mapping."""
    if isinstance(self.input_fields, dict):
        # Dict format: {"state_field": "engine_field"}
        mapped_input = {}
        for state_field, engine_field in self.input_fields.items():
            if hasattr(state, state_field):
                mapped_input[engine_field] = getattr(state, state_field)
        return mapped_input

    elif isinstance(self.input_fields, list):
        # List format: ["field1", "field2"]
        return {field: getattr(state, field) for field in self.input_fields
                if hasattr(state, field)}
```

#### Output Mapping

```python
def _create_mapped_output(self, result):
    """Create output with field mapping."""
    if isinstance(self.output_fields, dict):
        # Dict format: {"engine_field": "state_field"}
        mapped_output = {}
        for engine_field, state_field in self.output_fields.items():
            if hasattr(result, engine_field):
                mapped_output[state_field] = getattr(result, engine_field)
            elif isinstance(result, dict) and engine_field in result:
                mapped_output[state_field] = result[engine_field]
        return mapped_output

    elif isinstance(self.output_fields, list):
        # List format: ["field1", "field2"]
        return {field: getattr(result, field) for field in self.output_fields
                if hasattr(result, field)}
```

## 🎯 Usage Patterns

### Pattern 1: Agent Node with Field Mapping

```python
from haive.core.graph.node.agent_node import AgentNodeConfig

# Create agent node with field mapping
agent_node = AgentNodeConfig(
    name="processor",
    agent=my_agent,
    output_fields={"response": "formatted_output"}
)
```

### Pattern 2: Custom Node with Field Mapping

```python
class CustomNodeWithMapping(EngineNodeConfig):
    """Custom node that always maps specific fields."""

    def __init__(self, **kwargs):
        # Set default field mappings
        kwargs.setdefault("output_fields", {
            "result": "processed_data",
            "metadata": "processing_info"
        })
        super().__init__(**kwargs)
```

### Pattern 3: Dynamic Field Mapping

```python
def create_mapped_node(engine, mapping_config):
    """Create node with dynamic field mapping."""
    return EngineNodeConfig(
        name=f"{engine.name}_mapped",
        engine=engine,
        input_fields=mapping_config.get("input", None),
        output_fields=mapping_config.get("output", None)
    )

# Usage
mapping = {
    "input": {"query": "question"},
    "output": {"result": "answer"}
}
node = create_mapped_node(my_engine, mapping)
```

## 🔍 Debugging Field Mapping

### Enable Debug Mode

```python
# Enable debug logging in EngineNodeConfig
node = EngineNodeConfig(
    name="debug_node",
    engine=engine,
    output_fields={"result": "potato"},
    debug=True  # Shows field mapping details
)
```

### Debug Output Example

```
ENGINE NODE EXECUTION: debug_node
Step 2: Extracting Input
Using schema-based input extraction: ['query', 'context']
Extracted input_data: {'query': 'test', 'context': 'background'}

Step 4: Creating Update
Using schema-based output creation: ['potato']
Final update: {'potato': 'processed result'}
```

## ⚠️ Current Limitations

### What Works Now

- ✅ Simple field renaming (`"result" → "potato"`)
- ✅ Multiple field mapping
- ✅ Input and output mapping
- ✅ List and dict formats
- ✅ Type preservation

### What Doesn't Work Yet

- ❌ Nested field extraction (`"data.nested.field"`)
- ❌ Array indexing (`"messages[0].content"`)
- ❌ Transform functions (`uppercase`, `strip`)
- ❌ Conditional mapping
- ❌ Cross-node field coordination (automatic)

## 🚀 Future Enhancements

### Planned: NodeSchemaComposer

```python
# Future capability
from haive.core.schema.node_schema_composer import NodeSchemaComposer, FieldMapping

composer = NodeSchemaComposer(
    field_mappings=[
        FieldMapping(
            source_path="messages[-1].content",
            target_path="potato",
            transform=["strip", "uppercase"]
        )
    ]
)

node = EngineNodeConfig(
    name="advanced_node",
    engine=engine,
    schema_composer=composer  # Advanced field mapping
)
```

## 📚 Related Components

### In haive-core

- `haive/core/graph/node/base_config.py` - Base node configuration
- `haive/core/graph/node/agent_node.py` - Agent node implementation
- `haive/core/schema/field_definition.py` - Field definitions

### In haive-agents

- `haive/agents/base/agent.py` - Base agent class
- `haive/agents/multi/enhanced_multi_agent_v3.py` - Multi-agent coordination
- `haive/agents/simple/agent.py` - Simple agent implementation

## 🔗 Examples

See the `examples/` directory for complete examples:

- `examples/field_mapping_basic.py` - Basic field mapping
- `examples/field_mapping_multi_agent.py` - Multi-agent field coordination
- `examples/custom_node_mapping.py` - Custom node with field mapping

---

**Field mapping is available now in haive-core!** Use `EngineNodeConfig` with `output_fields={"result": "potato"}` to start mapping fields immediately.
