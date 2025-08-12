#!/usr/bin/env python3
"""Minimal test to isolate BaseModel instance issue."""

import traceback

from pydantic import BaseModel

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin


class SimpleModel(BaseModel):
    value: int = 1

    def __call__(self, x: int) -> int:
        return x * self.value

print("="*50)
print("MINIMAL INSTANCE DEBUG")
print("="*50)

instance = SimpleModel(value=5)
print(f"Instance: {instance}")

# Test 1: ToolRouteMixin directly
print("\n1️⃣ ToolRouteMixin Direct")
try:
    mixin = ToolRouteMixin(tools=[instance])
    print(f"✅ ToolRouteMixin works: {mixin.tools}")
except Exception as e:
    print(f"❌ ToolRouteMixin failed: {e}")
    traceback.print_exc()

# Test 2: Import chain to find where the issue comes from
print("\n2️⃣ Import Chain Debug")

try:
    from haive.core.common.mixins.structured_output_mixin import StructuredOutputMixin
    print("✅ StructuredOutputMixin imported")

    struct_mixin = StructuredOutputMixin(tools=[instance])
    print("✅ StructuredOutputMixin works")

except Exception as e:
    print(f"❌ StructuredOutputMixin failed: {e}")
    traceback.print_exc()

print("\n3️⃣ InvokableEngine Test")
try:
    print("✅ InvokableEngine imported")
except Exception as e:
    print(f"❌ InvokableEngine failed: {e}")

# Test 4: Try to find the culprit validator
print("\n4️⃣ Field Validator Check")

# Let's see if the issue is in a Pydantic validator
try:
    from haive.core.engine.aug_llm.config import AugLLMConfig

    # Try bypassing validation
    config = AugLLMConfig.__new__(AugLLMConfig)
    config.tools = [instance]
    print("✅ Direct assignment works")

except Exception as e:
    print(f"❌ Direct assignment failed: {e}")
    traceback.print_exc()

print("\n" + "="*50)
