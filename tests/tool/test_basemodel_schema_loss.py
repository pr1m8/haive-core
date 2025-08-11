#!/usr/bin/env python3
"""Test schema loss when BaseModel becomes a tool."""

from pydantic import BaseModel, Field
from langchain_core.tools import tool, StructuredTool
from typing import Type
import json

class SearchConfig(BaseModel):
    """Configuration for search tool."""
    api_key: str = Field(description="API key")
    endpoint: str = Field(default="https://api.example.com", description="API endpoint")
    max_retries: int = Field(default=3, description="Max retries")
    
    def __call__(self, query: str) -> str:
        """Execute search with this configuration."""
        return f"Searching '{query}' at {self.endpoint} with key {self.api_key[:4]}..."

print("="*60)
print("BASEMODEL SCHEMA LOSS ANALYSIS")
print("="*60)

# 1. Original BaseModel schema
print("\n1️⃣ Original BaseModel Schema")
print("-"*40)
print("Fields:", list(SearchConfig.model_fields.keys()))
print("Required:", [k for k, v in SearchConfig.model_fields.items() if v.is_required()])

# 2. Create instance with configuration
instance = SearchConfig(api_key="sk-123456789")
print(f"\n2️⃣ Instance Configuration")
print(f"api_key: {instance.api_key}")
print(f"endpoint: {instance.endpoint}")
print(f"max_retries: {instance.max_retries}")

# 3. Different ways to make it a tool
print("\n3️⃣ Tool Creation Methods")
print("-"*40)

# Method A: Direct @tool on instance's __call__
print("\nMethod A: Direct callable wrapping")
tool_a = tool(instance.__call__)
print(f"Tool A schema: {list(tool_a.args_schema.model_fields.keys())}")
print(f"Execution: {tool_a.invoke({'query': 'test'})}")

# Method B: Wrapper that preserves state
print("\n\nMethod B: State-preserving wrapper")
@tool
def search_with_config(query: str) -> str:
    """Search using pre-configured instance."""
    return instance(query)

print(f"Tool B schema: {list(search_with_config.args_schema.model_fields.keys())}")
print(f"Execution: {search_with_config.invoke({'query': 'test'})}")

# Method C: Full schema exposure
print("\n\nMethod C: Full schema in tool")
@tool
def search_full_config(api_key: str, endpoint: str, max_retries: int, query: str) -> str:
    """Search with full configuration."""
    config = SearchConfig(api_key=api_key, endpoint=endpoint, max_retries=max_retries)
    return config(query)

print(f"Tool C schema: {list(search_full_config.args_schema.model_fields.keys())}")

# Method D: What if we want BaseModel validation?
print("\n\nMethod D: BaseModel as tool input schema")
class SearchInput(BaseModel):
    """Input for search including config and query."""
    api_key: str
    endpoint: str = "https://api.example.com"
    query: str

@tool(args_schema=SearchInput)
def search_with_validation(api_key: str, endpoint: str, query: str) -> str:
    """Search with validated inputs."""
    config = SearchConfig(api_key=api_key, endpoint=endpoint)
    return config(query)

print(f"Tool D schema: {list(search_with_validation.args_schema.model_fields.keys())}")

# 4. The routing question
print("\n\n4️⃣ Routing Implications")
print("-"*40)
print("If we route BaseModel with __call__ as:")
print("- 'pydantic_tool' → Loses BaseModel schema, only __call__ params")
print("- 'pydantic_model' → Could preserve full schema?")
print("- 'function' → Treats as plain callable")

print("\n5️⃣ User Control Options")
print("-"*40)
print("1. Explicit route override: tools=[(SearchConfig, 'pydantic_model')]")
print("2. Metadata flag: preserve_schema=True")
print("3. Different tool types for different behaviors")
print("4. Wrapper pattern for full control")

print("\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print("BaseModel with __call__ creates tension between:")
print("- Configuration schema (BaseModel fields)")
print("- Execution schema (__call__ parameters)")
print("\nAutomatic conversion loses configuration schema!")
print("Manual wrapping gives full control over what's exposed.")