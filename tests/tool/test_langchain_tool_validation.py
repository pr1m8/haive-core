#!/usr/bin/env python3
"""Test to see what's triggering LangChain's tool validation."""

from pydantic import BaseModel, Field
import traceback

class SimpleModel(BaseModel):
    value: int = 1
    
    def __call__(self, x: int) -> int:
        return x * self.value

# Test: Let's see what happens if we try to use our model where LangChain expects a tool
print("="*60)
print("LANGCHAIN TOOL VALIDATION DEBUG")  
print("="*60)

instance = SimpleModel(value=5)

# Test 1: Direct LangChain tool validation
print("\n1️⃣ Direct LangChain Validation")
try:
    from langchain_core.tools import BaseTool
    
    # Try to see if our instance somehow gets treated as a BaseTool
    print(f"isinstance(instance, BaseTool): {isinstance(instance, BaseTool)}")
    
    # Check what happens with LangChain's field validation
    from langchain_core.tools.base import BaseToolkit
    print(f"isinstance(instance, BaseToolkit): {isinstance(instance, BaseToolkit)}")
    
except Exception as e:
    print(f"Error: {e}")

# Test 2: Check if it's the Field type annotation
print("\n2️⃣ Field Type Annotation Issue")

# Check what types LangChain expects
from typing import get_type_hints

try:
    from langchain_core.tools import BaseTool, StructuredTool
    from haive.core.engine.aug_llm.config import AugLLMConfig
    
    # Get type hints for the tools field  
    hints = get_type_hints(AugLLMConfig)
    if 'tools' in hints:
        print(f"tools field type: {hints['tools']}")
    else:
        print("No tools field in AugLLMConfig hints")
        
    # Check parent classes
    for base in AugLLMConfig.__mro__:
        if hasattr(base, '__annotations__') and 'tools' in base.__annotations__:
            print(f"{base.__name__} has tools: {base.__annotations__['tools']}")
            
except Exception as e:
    print(f"Type hint error: {e}")
    
# Test 3: The actual error reproduction
print("\n3️⃣ Error Reproduction")
try:
    from haive.core.engine.aug_llm.config import AugLLMConfig
    
    # Try minimal creation to see exactly where it fails
    print("Creating config with minimal data...")
    config = AugLLMConfig(llm_config={"model": "gpt-4"}, tools=[instance])
    
except Exception as e:
    print(f"Config creation failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    
print("\n" + "="*60)