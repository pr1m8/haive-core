#!/usr/bin/env python3
"""Debug BaseModel instance handling in AugLLMConfig."""

from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
import traceback

class ExecutableModel(BaseModel):
    """BaseModel with __call__."""
    multiplier: int = Field(default=2)
    
    def __call__(self, value: int) -> int:
        return value * self.multiplier

print("="*60)
print("INSTANCE HANDLING DEBUG")
print("="*60)

# Create instance
instance = ExecutableModel(multiplier=5)
print(f"Instance created: {instance}")
print(f"Instance type: {type(instance)}")
print(f"Instance callable: {callable(instance)}")
print(f"Instance has get method: {hasattr(instance, 'get')}")

# Test 1: What happens during AugLLMConfig creation?
print("\n1️⃣ AugLLMConfig Creation Debug")
print("-"*40)

try:
    print("Creating AugLLMConfig with instance...")
    config = AugLLMConfig(tools=[instance])
    print("✅ SUCCESS!")
    print(f"Tool routes: {config.tool_routes}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

# Test 2: Let's check the tool validation
print("\n\n2️⃣ Tool Validation Debug")
print("-"*40)

print("Testing tool validation directly...")
try:
    # This is what gets called during field validation
    validated = AugLLMConfig.validate_tools([instance])
    print(f"✅ Tool validation passed: {validated}")
except Exception as e:
    print(f"❌ Tool validation failed: {e}")
    traceback.print_exc()

# Test 3: Let's check _analyze_tool directly
print("\n\n3️⃣ _analyze_tool Debug")
print("-"*40)

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
mixin = ToolRouteMixin()

try:
    route, metadata = mixin._analyze_tool(instance)
    print(f"✅ Route analysis: '{route}'")
    print(f"Metadata: {metadata}")
except Exception as e:
    print(f"❌ Route analysis failed: {e}")
    traceback.print_exc()

# Test 4: Let's check what specific method is failing
print("\n\n4️⃣ Method Call Debug")  
print("-"*40)

print("Testing specific method calls on instance:")
print(f"hasattr(instance, 'get'): {hasattr(instance, 'get')}")
print(f"type(instance).__name__: {type(instance).__name__}")

# See if it's in the __init__ or validation
print("\nTesting minimal config creation...")
try:
    config = AugLLMConfig()  # Empty config
    print("✅ Empty config created")
    
    # Try adding tool manually
    config.tools = [instance]
    print("✅ Tools set directly")
    
    # Try sync
    config._sync_tool_routes()
    print("✅ Sync completed")
    print(f"Routes: {config.tool_routes}")
    
except Exception as e:
    print(f"❌ Manual addition failed: {e}")
    traceback.print_exc()

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)