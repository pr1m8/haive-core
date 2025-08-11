#!/usr/bin/env python3
"""Test smart BaseModel routing with defaults and overrides."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.engine.tool import ToolEngine
from langchain_core.tools import StructuredTool
import json

class SearchTool(BaseModel):
    """A tool that needs configuration."""
    api_key: str = Field(description="API key for the service")
    endpoint: str = Field(default="https://api.example.com", description="API endpoint")
    max_results: int = Field(default=10, description="Maximum results")
    
    def __call__(self, query: str, filters: str = "") -> str:
        """Execute search with the configured settings."""
        return f"Searching '{query}' at {self.endpoint} (key: {self.api_key[:8]}..., max: {self.max_results})"

print("="*80)
print("SMART BASEMODEL ROUTING END-TO-END TEST")
print("="*80)

# Test 1: Smart Defaults - Class vs Instance Detection
print("\n1️⃣ Smart Defaults Test")
print("-"*50)

# Test class routing
mixin = ToolRouteMixin()
class_route, class_meta = mixin._analyze_tool(SearchTool)
print(f"SearchTool CLASS route: '{class_route}'")
print(f"  is_executable: {class_meta.get('is_executable', 'N/A')}")

# Test instance routing  
instance = SearchTool(api_key="sk-1234567890")
instance_route, instance_meta = mixin._analyze_tool(instance)
print(f"SearchTool INSTANCE route: '{instance_route}'")
print(f"  callable_type: {instance_meta.get('callable_type', 'N/A')}")

# Test 2: AugLLMConfig Smart Defaults
print("\n\n2️⃣ AugLLMConfig Smart Defaults")
print("-"*50)

# With class (should preserve BaseModel schema in future)
config1 = AugLLMConfig(tools=[SearchTool])
print(f"Class in AugLLMConfig:")
print(f"  Route: {config1.tool_routes.get('SearchTool', 'Not found')}")
print(f"  Tool count: {len(config1.get_tools())}")

# Try to get the actual tool
tools1 = config1.get_tools()
if tools1:
    tool1 = tools1[0]
    print(f"  Tool type: {type(tool1).__name__}")
    print(f"  Tool name: {tool1.name}")

# With instance (should use __call__ schema only)
try:
    config2 = AugLLMConfig(tools=[instance])
    print(f"\nInstance in AugLLMConfig:")
    print(f"  Route: {config2.tool_routes}")
except Exception as e:
    print(f"\nInstance in AugLLMConfig: ERROR - {e}")

# Test 3: Manual Override Capability
print("\n\n3️⃣ Manual Override Test")
print("-"*50)

config3 = AugLLMConfig(tools=[SearchTool])
print(f"Before override: {config3.tool_routes.get('SearchTool')}")

# Override the route
config3.set_tool_route("SearchTool", "pydantic_tool", {"override": True})
print(f"After override: {config3.tool_routes.get('SearchTool')}")
print(f"Metadata: {config3.get_tool_metadata('SearchTool')}")

# Test 4: ToolEngine Behavior
print("\n\n4️⃣ ToolEngine Conversion Test")
print("-"*50)

engine = ToolEngine()

# Test with class (can't be converted without instance)
print("Testing class conversion:")
class_result = engine._convert_model_to_tool(SearchTool)
print(f"  Class result: {class_result}")

# Test with instance
print("\nTesting instance conversion:")
instance_result = engine._convert_model_to_tool(instance)
if instance_result:
    print(f"  Instance result: {type(instance_result).__name__}")
    print(f"  Tool name: {instance_result.name}")
    print(f"  Tool description: {instance_result.description}")
    
    # Check what schema we get
    if hasattr(instance_result, 'args_schema'):
        schema = instance_result.args_schema.model_json_schema()
        print(f"  Schema fields: {list(schema['properties'].keys())}")
else:
    print("  Instance conversion failed")

# Test 5: End-to-End Schema Comparison
print("\n\n5️⃣ Schema Comparison")
print("-"*50)

print("BaseModel schema (what we could expose):")
base_schema = SearchTool.model_json_schema()
print(f"  Fields: {list(base_schema['properties'].keys())}")

print("\n__call__ schema (what we currently expose):")
import inspect
sig = inspect.signature(SearchTool.__call__)
call_params = [p for p in sig.parameters.keys() if p != 'self']
print(f"  Parameters: {call_params}")

# Test 6: Proposed Behavior
print("\n\n6️⃣ Proposed Smart Behavior")
print("-"*50)
print("SHOULD work like this:")
print("1. tools=[SearchTool] → Full schema (api_key + endpoint + max_results + query + filters)")
print("2. tools=[instance] → Call schema (query + filters)")
print("3. Override available via set_tool_route()")

print("\nCURRENTLY:")
print("1. tools=[SearchTool] → 'pydantic_model' route but no execution")
print("2. tools=[instance] → Conversion error or 'function' route")
print("3. Override works but limited effect")

print("\n" + "="*80)
print("IMPLEMENTATION NEEDED:")
print("="*80)
print("✅ ToolRouteMixin detects class vs instance correctly")
print("❌ AugLLMConfig needs to handle instances gracefully") 
print("❌ Need route that preserves BaseModel schema")
print("❌ ToolEngine conversion needs both class and instance modes")
print("✅ Override mechanism exists")