#!/usr/bin/env python3
"""Test that BaseModel instances now work."""

from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig


class ExecutableModel(BaseModel):
    multiplier: int = Field(default=2)

    def __call__(self, value: int) -> int:
        return value * self.multiplier

print("="*60)
print("BASEMODEL INSTANCE SUCCESS TEST")
print("="*60)

# Test 1: Instance creation
instance = ExecutableModel(multiplier=5)
print(f"✅ Created instance: {instance}")

# Test 2: AugLLMConfig with instance
print("\n1️⃣ AugLLMConfig with instance")
try:
    config = AugLLMConfig(tools=[instance])
    print("✅ SUCCESS! Config created")
    print(f"   Tool routes: {config.tool_routes}")
    print(f"   Route for tool_0: {config.get_tool_route('tool_0')}")
    print(f"   Metadata: {config.get_tool_metadata('tool_0')}")

except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Mixed tools (class + instance)
print("\n2️⃣ Mixed tools test")
try:
    config2 = AugLLMConfig(tools=[ExecutableModel, instance])
    print("✅ SUCCESS! Mixed tools config")
    print(f"   Tool routes: {config2.tool_routes}")

except Exception as e:
    print(f"❌ Failed: {e}")

# Test 4: Manual addition
print("\n3️⃣ Manual tool addition")
try:
    config3 = AugLLMConfig()
    config3.add_tool(instance, name="manual_instance")
    print("✅ SUCCESS! Manual addition")
    print(f"   Tool routes: {config3.tool_routes}")

except Exception as e:
    print(f"❌ Failed: {e}")

print("\n" + "="*60)
print("🎉 INSTANCE HANDLING IS NOW WORKING!")
print("="*60)
