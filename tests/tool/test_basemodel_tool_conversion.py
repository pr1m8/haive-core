#!/usr/bin/env python3
"""Test how BaseModel tools are actually handled."""

import logging

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.engine.tool import ToolEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("TESTING BASEMODEL TOOL CONVERSION")
print("="*80)

# Test 1: Non-executable BaseModel
class NonExecutableModel(BaseModel):
    """Model without __call__ method."""
    name: str
    value: int

print("\n1️⃣ Non-executable BaseModel as tool")
print("-"*40)

# Need to pass the class, not an instance
engine1 = ToolEngine()
engine1.tools = [NonExecutableModel]  # Set after initialization
print(f"Original tools: {engine1.tools}")
print(f"Processed tools: {engine1.get_tools()}")
print(f"Tool count: {len(engine1.get_tools())}")

# Test 2: Executable BaseModel
class ExecutableModel(BaseModel):
    """Model with __call__ method."""
    query: str

    def __call__(self):
        return f"Executed: {self.query}"

print("\n\n2️⃣ Executable BaseModel as tool")
print("-"*40)

engine2 = ToolEngine()
engine2.tools = [ExecutableModel]
print(f"Original tools: {engine2.tools}")
processed_tools = engine2.get_tools()
print(f"Processed tools: {processed_tools}")
print(f"Tool count: {len(processed_tools)}")

if processed_tools:
    tool = processed_tools[0]
    print(f"\nTool type: {type(tool)}")
    print(f"Is StructuredTool: {isinstance(tool, StructuredTool)}")
    print(f"Tool name: {getattr(tool, 'name', 'N/A')}")

# Test 3: Check routes in AugLLMConfig
print("\n\n3️⃣ Routes in AugLLMConfig")
print("-"*40)

# With non-executable model
config1 = AugLLMConfig(tools=[NonExecutableModel])
print(f"\nNon-executable model routes: {config1.tool_routes}")

# With executable model
config2 = AugLLMConfig(tools=[ExecutableModel])
print(f"Executable model routes: {config2.tool_routes}")
print(f"Processed tools in config: {config2.get_tools()}")

# With structured output
config3 = AugLLMConfig(structured_output_model=NonExecutableModel)
print(f"\nStructured output routes: {config3.tool_routes}")

print("\n" + "="*80)
print("CONCLUSIONS:")
print("="*80)
print("1. Non-executable BaseModel → Cannot be used as tool (warning)")
print("2. Executable BaseModel → Converted to StructuredTool")
print("3. StructuredTool should get 'langchain_tool' route")
print("4. structured_output_model → Gets 'parse_output' route")
