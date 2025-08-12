#!/usr/bin/env python3
"""Comprehensive test for tool routing with different tool types and structured output models."""

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field, validator

from haive.agents.simple.agent import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Patch ValidationNodeV2 to add extra logging
import haive.core.graph.node.validation_node_v2 as vn_module

original_call = vn_module.ValidationNodeV2.__call__

def patched_call(self, state, config=None):
    logger.info("🔍 ValidationNodeV2 INTERCEPTED - Analyzing tool routes:")

    # Get tool routes from engine
    engine = self._get_engine_from_state(state)
    if engine and hasattr(engine, "tool_routes"):
        logger.info(f"📋 Tool routes in engine: {engine.tool_routes}")
        for tool_name, route in engine.tool_routes.items():
            logger.info(f"  • {tool_name} → {route}")

    # Get last message
    messages = getattr(state, self.messages_key, [])
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "tool_calls"):
            tool_calls = getattr(last_message, "tool_calls", [])
            logger.info(f"🔧 Tool calls in message: {len(tool_calls)}")
            for tc in tool_calls:
                tool_name = tc.get("name", "unknown")
                route = engine.tool_routes.get(tool_name, "unknown") if engine else "no_engine"
                logger.info(f"  • Tool call: {tool_name} → route: {route}")

    return original_call(self, state, config)

vn_module.ValidationNodeV2.__call__ = patched_call

# Test 1: Different Pydantic model types
print("\n" + "="*80)
print("TEST 1: PYDANTIC MODELS AS TOOLS VS STRUCTURED OUTPUT")
print("="*80)

class SimpleModel(BaseModel):
    """Basic Pydantic model."""
    name: str
    value: int

class ComplexModel(BaseModel):
    """Complex Pydantic model with nested fields."""
    title: str
    items: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None

# Model with __call__ method (executable)
class ExecutableModel(BaseModel):
    """Pydantic model with __call__ method."""
    query: str

    def __call__(self):
        return f"Executed: {self.query}"

# Test 1a: Pydantic model as tool
print("\n--- Test 1a: Pydantic models as tools ---")
config_1a = AugLLMConfig(
    tools=[SimpleModel, ComplexModel, ExecutableModel],
    temperature=0.1
)
agent_1a = SimpleAgent(
    name="test_pydantic_tools",
    engine=config_1a,
)

print("\n🔍 Tool routes after setup:")
for name, route in config_1a.tool_routes.items():
    print(f"  • {name} → {route}")

# Test 1b: Pydantic model as structured output
print("\n--- Test 1b: Pydantic model as structured output ---")
config_1b = AugLLMConfig(
    structured_output_model=SimpleModel,
    temperature=0.1,
)
agent_1b = SimpleAgent(
    name="test_structured_output",
    engine=config_1b,
)

print("\n🔍 Tool routes after setup:")
for name, route in config_1b.tool_routes.items():
    print(f"  • {name} → {route}")

# Test 1c: Both tools and structured output
print("\n--- Test 1c: Both tools AND structured output ---")
config_1c = AugLLMConfig(
    tools=[ComplexModel],
    structured_output_model=SimpleModel,
    temperature=0.1,
)
agent_1c = SimpleAgent(
    name="test_both",
    engine=config_1c,
)

print("\n🔍 Tool routes after setup:")
for name, route in config_1c.tool_routes.items():
    print(f"  • {name} → {route}")

# Test 2: Different tool types
print("\n" + "="*80)
print("TEST 2: DIFFERENT TOOL TYPES")
print("="*80)

# Function tool
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return str(eval(expression))

# StructuredTool
structured_calc = StructuredTool.from_function(
    func=lambda x, y: x + y,
    name="add_numbers",
    description="Add two numbers"
)

# Custom BaseTool
class SearchTool(BaseTool):
    name: str = "search"
    description: str = "Search for information"

    def _run(self, query: str) -> str:
        return f"Search results for: {query}"

    async def _arun(self, query: str) -> str:
        return self._run(query)

search_tool = SearchTool()

print("\n--- Test 2: Mixed tool types ---")
config_2 = AugLLMConfig(
    tools=[calculator, structured_calc, search_tool, SimpleModel],
    structured_output_model=ComplexModel,
    temperature=0.1,
)
agent_2 = SimpleAgent(
    name="test_mixed_tools",
    engine=config_2,
)

print("\n🔍 Tool routes after setup:")
for name, route in config_2.tool_routes.items():
    metadata = config_2.tool_route_metadata.get(name, {})
    print(f"  • {name} → {route}")
    print(f"    metadata: {metadata}")

# Test 3: Runtime execution and routing
print("\n" + "="*80)
print("TEST 3: RUNTIME EXECUTION AND ROUTING")
print("="*80)

async def test_execution():
    """Test actual execution to see routing in action."""

    # Test with structured output
    print("\n--- Executing agent with structured output ---")
    agent = SimpleAgent(
        name="runtime_test",
        engine=AugLLMConfig(
            structured_output_model=SimpleModel,
            temperature=0.1,
                ),
        )

    # This should trigger the structured output flow
    result = await agent.arun("Create a SimpleModel with name='test' and value=42")
    print(f"\n✅ Result type: {type(result)}")
    print(f"✅ Result: {result}")

    # Test with Pydantic model as tool
    print("\n--- Executing agent with Pydantic tool ---")
    agent_tool = SimpleAgent(
        name="tool_test",
        engine=AugLLMConfig(
            tools=[SimpleModel],
            temperature=0.1,
                ),
        )

    # This should trigger tool validation
    result_tool = await agent_tool.arun("Use SimpleModel tool with name='tool_test' and value=99")
    print(f"\n✅ Result: {result_tool}")

# Run async test
print("\n🚀 Running execution tests...")
asyncio.run(test_execution())

# Test 4: Edge cases
print("\n" + "="*80)
print("TEST 4: EDGE CASES")
print("="*80)

# Empty model
class EmptyModel(BaseModel):
    """Model with no fields."""

# Model with complex validation
class ValidatedModel(BaseModel):
    """Model with validators."""
    email: str
    age: int = Field(ge=0, le=150)

    @validator("email")
    def validate_email(cls, v):
        if "@" not in v:
            raise ValueError("Invalid email")
        return v

print("\n--- Test 4: Edge case models ---")
config_4 = AugLLMConfig(
    tools=[EmptyModel, ValidatedModel],
    temperature=0.1,
)
agent_4 = SimpleAgent(
    name="test_edge_cases",
    engine=config_4,
)

print("\n🔍 Tool routes after setup:")
for name, route in config_4.tool_routes.items():
    print(f"  • {name} → {route}")

# Final summary
print("\n" + "="*80)
print("SUMMARY: ROUTING BEHAVIOR")
print("="*80)
print("\n✅ All Pydantic BaseModel subclasses should route to 'parse_output'")
print("✅ This includes models used as tools AND structured output models")
print("✅ The deprecated 'pydantic_model' route should show warnings if used")
print("\n🔍 Check the logs above to verify all routes are correct!")
