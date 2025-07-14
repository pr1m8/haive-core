# NodeSchemaComposer: Analysis Requirements for Flexible I/O Architecture

**Document Version**: 1.0  
**Purpose**: Comprehensive analysis requirements for designing NodeSchemaComposer with flexible extract/update functions  
**Target**: Agent delegation for node file analysis  
**Date**: 2025-01-13

## 🎯 Objective

Analyze existing v2/v3 node implementations to understand their I/O patterns and schema requirements for designing a **NodeSchemaComposer** that enables flexible field mapping and pluggable extract/update functions.

This analysis will support the vision of nodes that can:

- Map outputs to arbitrary fields (`result → potato`)
- Use configurable extract/update functions
- Adapt to different state schemas dynamically
- Support Agent vs AgentLike schema composition patterns

## 📁 Files to Analyze

### **Primary Node Files**

1. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/output_parsing_v2.py`
2. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/engine_node.py`
3. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/agent_node_v3.py`
4. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/agent_node_v2.py`
5. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/validation_node_config_v2.py`
6. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/tool_node_config_v2.py`
7. `/home/will/Projects/haive/backend/haive/packages/haive-core/src/haive/core/graph/node/parser_node_config_v2.py`

### **Supporting Files** (if helpful for context)

- Base node configurations
- Schema composition utilities
- Message transformation utilities

## 🔍 Analysis Framework

### **1. Node I/O Pattern Analysis**

For each node file, document:

#### **A. Input Requirements**

- **Required Fields**: What fields must exist in state for node to function?
- **Optional Fields**: What fields are used if available?
- **Field Types**: Expected data types for each field
- **Nested Access**: How does node access nested data structures?
- **Default Handling**: What happens when expected fields are missing?

#### **B. Output Generation**

- **Output Fields**: What new fields does node add to state?
- **Modified Fields**: What existing fields does node update?
- **Output Types**: Data types of generated outputs
- **Conditional Outputs**: When are certain outputs generated?
- **Merge Strategy**: How are outputs combined with existing state?

#### **C. Field Dependencies**

- **Input Dependencies**: Which input fields depend on each other?
- **Output Dependencies**: Which outputs are generated based on inputs?
- **Cross-Node Dependencies**: Fields this node expects from other nodes?

### **2. Extract Function Analysis**

#### **A. Current Extract Patterns**

- **Hardcoded Extraction**: Where are field names hardcoded?
- **Dynamic Extraction**: Where is extraction configurable?
- **Extraction Logic**: How complex is the data extraction?
- **Type Conversion**: What input type conversions happen?
- **Error Handling**: How are missing/invalid inputs handled?

#### **B. Extract Function Flexibility**

```python
# Document patterns like:
def extract_example(state):
    # Current pattern
    return state.messages  # Hardcoded

    # vs potential pattern
    return self.extract_fn(state)  # Configurable
```

#### **C. Extract Configuration Opportunities**

- Where could extract functions be made pluggable?
- What parameters would extract functions need?
- How could field mapping be configured?

### **3. Update Function Analysis**

#### **A. Current Update Patterns**

- **Direct Assignment**: `state.field = value`
- **List Operations**: `state.messages.append()`
- **Dictionary Updates**: `state.context.update()`
- **Complex Merging**: Custom merge logic
- **Conditional Updates**: When updates are applied

#### **B. Update Function Flexibility**

```python
# Document patterns like:
def update_example(result, state):
    # Current pattern
    state.messages.append(result)  # Hardcoded field

    # vs potential pattern
    return self.update_fn(result, state)  # Configurable
```

#### **C. Update Configuration Opportunities**

- Where could update functions be made pluggable?
- What field mapping capabilities exist?
- How could "result → potato" mappings be supported?

### **4. Schema Awareness Analysis**

#### **A. Schema Detection**

- How does node determine available fields?
- Does node adapt behavior based on schema?
- How does node handle schema changes?

#### **B. Schema Requirements**

- What schema assumptions does node make?
- How are schema conflicts handled?
- What happens with unknown schemas?

#### **C. Schema Integration Points**

- Where could NodeSchemaComposer hook in?
- What schema information does node need?
- How could schema composition be improved?

### **5. Transformation Pipeline Analysis**

#### **A. Current Transformations**

- **Type Conversions**: `str → AIMessage`, `dict → ToolMessage`
- **Data Restructuring**: Flattening, nesting, reformatting
- **Content Processing**: Parsing, validation, enrichment
- **Format Standardization**: Common output formats

#### **B. Transformation Patterns**

```python
# Document transformation chains like:
input → validation → type_conversion → processing → output_formatting → state_update
```

#### **C. Transformation Flexibility**

- Which transformations are configurable?
- Where are transformations hardcoded?
- What transformation pipelines could be extracted?

### **6. Configuration and Flexibility Assessment**

#### **A. Current Configuration Options**

- What parameters control node behavior?
- How is I/O behavior configured?
- What runtime configuration is possible?

#### **B. Flexibility Gaps**

- Where is behavior hardcoded that should be configurable?
- What configuration options are missing?
- How could flexibility be improved?

#### **C. Plugin Architecture Potential**

- Where could pluggable components be added?
- What interfaces would plugins need?
- How could backward compatibility be maintained?

## 📊 Required Deliverables

### **1. Node I/O Matrix**

```markdown
| Node Type  | Input Fields      | Output Fields      | Extract Pattern | Update Pattern | Flexibility Level |
| ---------- | ----------------- | ------------------ | --------------- | -------------- | ----------------- |
| EngineNode | messages, context | messages, metadata | Hardcoded       | Hardcoded      | Low               |
| ToolNode   | tool_calls, tools | tool_messages      | Semi-flexible   | Semi-flexible  | Medium            |
| ...        | ...               | ...                | ...             | ...            | ...               |
```

### **2. Extract/Update Function Catalog**

For each node:

- Current extract function (pseudocode)
- Current update function (pseudocode)
- Flexibility assessment
- Plugin opportunities
- Configuration needs

### **3. Transformation Pipeline Map**

- Current transformation chains per node
- Reusable transformation functions
- Missing transformation types
- Pipeline composition opportunities

### **4. Schema Integration Assessment**

- Current schema awareness level
- Schema integration points
- NodeSchemaComposer hook opportunities
- Schema composition requirements

### **5. Configuration Enhancement Roadmap**

- Priority configuration improvements
- Plugin architecture opportunities
- Field mapping enhancement potential
- Backward compatibility considerations

### **6. Gap Analysis & Recommendations**

- Critical missing functionality
- Architecture improvement opportunities
- Implementation complexity assessment
- Migration strategy considerations

### **7. Testing Analysis & Enhancement Plan**

- Current test coverage assessment (with no-mocks compliance)
- Real component testing requirements per node
- Test gap identification and remediation plan
- Integration testing scenarios for NodeSchemaComposer
- Performance testing requirements for flexible I/O

## 🎯 Key Questions to Answer

### **Schema Composition Questions**

1. How do nodes currently determine their schema requirements?
2. Where could NodeSchemaComposer integrate most effectively?
3. What schema information do nodes need at runtime?
4. How could schema changes be propagated to nodes?

### **Extract/Update Flexibility Questions**

1. Which nodes have the most flexible I/O patterns?
2. Where are the biggest opportunities for pluggable functions?
3. How could field mapping be implemented cleanly?
4. What backward compatibility constraints exist?

### **Field Mapping Questions**

1. How could "result → potato" mappings be supported?
2. What field mapping DSL would be most usable?
3. How could complex transformations be configured?
4. What validation would field mappings need?

### **Architecture Questions**

1. How should extract/update functions be parameterized?
2. What interfaces should pluggable components implement?
3. How could configuration be made discoverable?
4. What performance implications exist?

### **Testing Questions**

1. Which nodes currently use mocks that should use real components?
2. What real engine/tool dependencies are needed for comprehensive testing?
3. How can extract/update function flexibility be tested with real scenarios?
4. What integration test scenarios are needed for NodeSchemaComposer?
5. How should performance be tested for flexible I/O transformations?

## 🧪 Testing Requirements & Completion Status

### **No-Mocks Testing Philosophy**

All analysis must consider **real component testing** requirements:

#### **A. Test Pattern Analysis**

- How are nodes currently tested? (mocks vs real components)
- What real component dependencies exist?
- How could nodes be tested with real engines/tools/state?

#### **B. Testing Gaps**

- Which nodes lack comprehensive tests?
- Where are mocks used that should be real components?
- What test scenarios are missing?

#### **C. Real Component Test Requirements**

For each node, identify:

- **Real engines** needed for testing
- **Real state schemas** required
- **Real tool/message dependencies**
- **Integration test scenarios**

#### **D. Test Completion Criteria**

Document for each node:

```markdown
| Node Type  | Test Coverage | Uses Mocks | Real Components          | Test Gaps       |
| ---------- | ------------- | ---------- | ------------------------ | --------------- |
| EngineNode | 85%           | No         | AugLLMConfig, real state | Edge cases      |
| ToolNode   | 70%           | Some       | Real tools, mock LLM     | LLM integration |
```

### **Testing Enhancement Requirements**

1. **Eliminate remaining mocks** in node tests
2. **Add real component integration** tests
3. **Test extract/update flexibility** with real scenarios
4. **Validate schema composition** with real agents

## 🚀 Success Criteria

The analysis should enable:

1. **Design NodeSchemaComposer** that understands all node types
2. **Create pluggable extract/update architecture** for flexible I/O
3. **Support arbitrary field mapping** (result → any field name)
4. **Enable dynamic node adaptation** to different schemas
5. **Maintain backward compatibility** with existing node implementations
6. **Enable comprehensive real-component testing** of all node functionality

## 📋 Output Format

Please provide analysis in this structure:

```markdown
# Node Analysis Report

## Executive Summary

- Key findings
- Architecture recommendations
- Implementation priorities

## Per-Node Analysis

### [Node Name]

- I/O patterns
- Extract/update functions
- Flexibility assessment
- Plugin opportunities

## Cross-Node Patterns

- Common patterns
- Reusable components
- Architecture insights

## Schema Integration Assessment

- Current schema awareness
- Integration opportunities
- NodeSchemaComposer requirements

## Recommendations

- Priority improvements
- Architecture changes
- Implementation roadmap
```

## 🔗 Context & Background

This analysis supports the broader vision of:

- **Agent vs AgentLike** architectural separation
- **Dynamic agent systems** with runtime adaptation
- **Multi-agent coordination** with flexible state management
- **SimpleAgentV2** as a clean foundation pattern
- **Schema composition hierarchy** (Node → Agent → Multi-Agent)

The goal is to enable a future where nodes are **schema-aware, field-flexible, and dynamically configurable** rather than rigid, hardcoded components.

---

**Ready for delegation to analysis agent with full context and clear deliverable expectations.**
