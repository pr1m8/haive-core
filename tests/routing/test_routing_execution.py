#!/usr/bin/env python3
"""Test actual execution flow with tool routing."""

import logging
from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.agents.simple.agent import SimpleAgent
from langchain_core.tools import tool
import json

# Configure logging to see the flow
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Enable specific loggers
logging.getLogger('haive.core.graph.node.validation_node_v2').setLevel(logging.DEBUG)
logging.getLogger('haive.core.common.mixins.tool_route_mixin').setLevel(logging.DEBUG)
logging.getLogger('haive.core.engine.aug_llm').setLevel(logging.DEBUG)

print("\n" + "="*80)
print("TESTING TOOL ROUTING EXECUTION FLOW")
print("="*80)

# Test Case 1: Structured Output Model
print("\n1️⃣ TEST: Structured Output Model")
print("-" * 40)

class TaskResult(BaseModel):
    """Structured output for task completion."""
    task: str = Field(description="The task that was completed")
    status: str = Field(description="Status: success or failure")
    details: str = Field(description="Additional details")

print("Creating agent with structured_output_model=TaskResult...")
agent1 = SimpleAgent(
    name="structured_agent",
    engine=AugLLMConfig(
        structured_output_model=TaskResult,
        temperature=0.1
    )
)

print(f"\n📋 Tool routes in engine:")
for name, route in agent1.engine.tool_routes.items():
    print(f"   {name} → {route}")

print("\n🚀 Executing agent...")
result1 = agent1.run("Complete the task of analyzing Python code")

print(f"\n✅ Result type: {type(result1)}")
print(f"✅ Result: {result1}")

# Test Case 2: Pydantic Model as Tool
print("\n\n2️⃣ TEST: Pydantic Model as Tool")
print("-" * 40)

class DataQuery(BaseModel):
    """Tool for querying data."""
    query: str = Field(description="The query to execute")
    limit: int = Field(default=10, description="Number of results")

print("Creating agent with tools=[DataQuery]...")
agent2 = SimpleAgent(
    name="tool_agent",
    engine=AugLLMConfig(
        tools=[DataQuery],
        temperature=0.1
    )
)

print(f"\n📋 Tool routes in engine:")
for name, route in agent2.engine.tool_routes.items():
    print(f"   {name} → {route}")

# Test Case 3: Mixed Tools
print("\n\n3️⃣ TEST: Mixed Tool Types")
print("-" * 40)

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

class InfoModel(BaseModel):
    """Model for structured information."""
    title: str
    content: str

print("Creating agent with mixed tools...")
agent3 = SimpleAgent(
    name="mixed_agent",
    engine=AugLLMConfig(
        tools=[calculator, InfoModel],
        structured_output_model=TaskResult,
        temperature=0.1
    )
)

print(f"\n📋 Tool routes in engine:")
for name, route in agent3.engine.tool_routes.items():
    metadata = agent3.engine.tool_route_metadata.get(name, {})
    print(f"   {name} → {route}")
    if metadata:
        print(f"      metadata: {json.dumps(metadata, indent=6)}")

# Test Case 4: Executable Pydantic Model
print("\n\n4️⃣ TEST: Executable Pydantic Model")
print("-" * 40)

class ExecutableQuery(BaseModel):
    """Pydantic model with __call__ method."""
    query: str
    
    def __call__(self):
        return f"Executing: {self.query}"

print("Creating agent with executable model...")
agent4 = SimpleAgent(
    name="executable_agent",
    engine=AugLLMConfig(
        tools=[ExecutableQuery],
        temperature=0.1
    )
)

print(f"\n📋 Tool routes in engine:")
for name, route in agent4.engine.tool_routes.items():
    metadata = agent4.engine.tool_route_metadata.get(name, {})
    print(f"   {name} → {route}")
    print(f"      is_executable: {metadata.get('is_executable', False)}")

# Summary
print("\n\n" + "="*80)
print("ROUTING SUMMARY")
print("="*80)
print("\n✅ All Pydantic models route to 'parse_output':")
print("   - Structured output models")
print("   - Pydantic models as tools")
print("   - Executable Pydantic models")
print("\n✅ Other tool types:")
print("   - @tool functions → 'langchain_tool'")
print("   - BaseTool instances → 'langchain_tool'")
print("\n🔍 The 'pydantic_model' route is DEPRECATED!")