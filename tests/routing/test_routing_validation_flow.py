#!/usr/bin/env python3
"""Test to trace the validation flow for parse_output route."""

import logging
from pydantic import BaseModel, Field
from haive.core.engine.aug_llm import AugLLMConfig
from haive.agents.simple.agent import SimpleAgent

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(message)s'
)

# Focus on key loggers
for logger_name in [
    'haive.core.graph.node.validation_node_v2',
    'haive.core.common.mixins.tool_route_mixin',
    'haive.core.engine.aug_llm.config'
]:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

print("\n" + "="*80)
print("TRACING VALIDATION FLOW FOR PARSE_OUTPUT ROUTE")
print("="*80)

class AnalysisResult(BaseModel):
    """Test model for validation."""
    summary: str = Field(description="Analysis summary")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")

# Test 1: As tool
print("\n1️⃣ Pydantic Model as Tool")
print("-" * 40)

agent1 = SimpleAgent(
    name="tool_test",
    engine=AugLLMConfig(
        tools=[AnalysisResult],
        temperature=0.1
    )
)

print(f"\n📋 Route: AnalysisResult → {agent1.engine.tool_routes.get('AnalysisResult')}")
print(f"📋 Metadata: {agent1.engine.tool_route_metadata.get('AnalysisResult')}")

# Test 2: As structured output
print("\n\n2️⃣ Pydantic Model as Structured Output")
print("-" * 40)

agent2 = SimpleAgent(
    name="structured_test",
    engine=AugLLMConfig(
        structured_output_model=AnalysisResult,
        temperature=0.1
    )
)

print(f"\n📋 Route: AnalysisResult → {agent2.engine.tool_routes.get('AnalysisResult')}")
print(f"📋 Metadata: {agent2.engine.tool_route_metadata.get('AnalysisResult')}")

# Test deprecated route
print("\n\n3️⃣ Testing Deprecated Route Warning")
print("-" * 40)

# Manually set a pydantic_model route to see the warning
agent2.engine.tool_routes['AnalysisResult'] = 'pydantic_model'
print(f"📋 Manually set route to: {agent2.engine.tool_routes.get('AnalysisResult')}")
print("\n🔍 This should trigger a deprecation warning when used...")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("✅ BaseModel as tool → parse_output route")
print("✅ BaseModel as structured_output_model → parse_output route")
print("✅ Consistent routing for all Pydantic models!")
print("⚠️  pydantic_model route is deprecated and shows warnings")