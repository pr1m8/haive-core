#!/usr/bin/env python3
"""Test smart BaseModel routing - fixed version."""


from pydantic import BaseModel, Field

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.engine.aug_llm import AugLLMConfig


class SearchTool(BaseModel):
    """A tool that needs configuration."""
    api_key: str = Field(description="API key for the service")
    endpoint: str = Field(default="https://api.example.com", description="API endpoint")
    max_results: int = Field(default=10, description="Maximum results")

    def __call__(self, query: str, filters: str = "") -> str:
        """Execute search with the configured settings."""
        return f"Searching '{query}' at {self.endpoint} (key: {self.api_key[:8]}..., max: {self.max_results})"

print("="*80)
print("SMART BASEMODEL ROUTING ANALYSIS")
print("="*80)

# Test 1: Route Detection Differences
print("\n1️⃣ Route Detection: ToolRouteMixin vs AugLLMConfig")
print("-"*60)

# ToolRouteMixin analysis (more sophisticated)
mixin = ToolRouteMixin()
class_route, class_meta = mixin._analyze_tool(SearchTool)
print("ToolRouteMixin - SearchTool CLASS:")
print(f"  Route: '{class_route}'")
print(f"  is_executable: {class_meta.get('is_executable')}")

instance = SearchTool(api_key="sk-1234567890")
instance_route, instance_meta = mixin._analyze_tool(instance)
print("\nToolRouteMixin - SearchTool INSTANCE:")
print(f"  Route: '{instance_route}'")
print(f"  callable_type: {instance_meta.get('callable_type')}")

# AugLLMConfig behavior (simpler)
config1 = AugLLMConfig(tools=[SearchTool])
print("\nAugLLMConfig - SearchTool CLASS:")
print(f"  Route: {config1.tool_routes.get('SearchTool')}")

# Test 2: The Gap - ToolRouteMixin sees it correctly
print("\n\n2️⃣ The Gap Between Detection and Usage")
print("-"*60)
print("ToolRouteMixin detects:")
print(f"  SearchTool (class) → '{class_route}' (executable: {class_meta.get('is_executable')})")
print(f"  instance → '{instance_route}'")
print("\nAugLLMConfig uses:")
print("  SearchTool (class) → 'pydantic_model'")
print("  instance → Causes error")

print("\nThe disconnect: ToolRouteMixin is smarter but AugLLMConfig overrides it!")

# Test 3: What Should Happen
print("\n\n3️⃣ Proposed Smart Behavior")
print("-"*60)

print("DESIRED BEHAVIOR:")
print("1. SearchTool (class) should create tool with FULL schema:")
print("   - Fields: api_key, endpoint, max_results, query, filters")
print("   - Creates new instance each call")

print("\n2. instance should create tool with CALL schema:")
print("   - Fields: query, filters")
print("   - Uses pre-configured instance")

# Test 4: Manual Implementation of Desired Behavior
print("\n\n4️⃣ Manual Implementation Test")
print("-"*60)

from langchain_core.tools import tool


# Simulate class-based tool (full schema)
@tool
def search_full_schema(api_key: str, endpoint: str, max_results: int, query: str, filters: str = "") -> str:
    """Search with full configuration."""
    tool = SearchTool(api_key=api_key, endpoint=endpoint, max_results=max_results)
    return tool(query, filters)

print("Class-based tool schema (full):")
full_schema = search_full_schema.args_schema.model_json_schema()
print(f"  Fields: {list(full_schema['properties'].keys())}")

# Simulate instance-based tool (call schema)
@tool
def search_call_schema(query: str, filters: str = "") -> str:
    """Search with pre-configured instance."""
    return instance(query, filters)

print("\nInstance-based tool schema (call only):")
call_schema = search_call_schema.args_schema.model_json_schema()
print(f"  Fields: {list(call_schema['properties'].keys())}")

# Test 5: Override Mechanism
print("\n\n5️⃣ Override Mechanism Test")
print("-"*60)

config2 = AugLLMConfig(tools=[SearchTool])
print(f"Before override: {config2.tool_routes.get('SearchTool')}")

# Override to use full schema
config2.set_tool_route("SearchTool", "pydantic_model_full", {"mode": "full_schema"})
print(f"After override: {config2.tool_routes.get('SearchTool')}")

print("\n" + "="*80)
print("IMPLEMENTATION PLAN:")
print("="*80)
print("1. ✅ ToolRouteMixin correctly detects class vs instance")
print("2. ❌ AugLLMConfig should use ToolRouteMixin analysis")
print("3. ❌ Need 'pydantic_model_full' route for full schema tools")
print("4. ❌ ValidationNode needs to handle new route")
print("5. ❌ ToolEngine needs full schema conversion mode")
print("6. ✅ Override mechanism exists")

print("\nNext: Wire up the detection → routing → execution pipeline!")
