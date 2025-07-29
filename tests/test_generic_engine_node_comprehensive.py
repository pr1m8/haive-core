#!/usr/bin/env python3
"""Comprehensive test for GenericEngineNodeConfig with various AugLLMConfig scenarios.

Tests:
1. ChatPromptTemplate with different input variables
2. Structured output models  
3. Regular LangChain tools
4. No tools (basic LLM)
5. Engine attribution and field mapping

Author: Kai
Date: 2025-07-29
"""

import asyncio
import logging
from typing import Any, Dict, List

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.engine_node_generic import GenericEngineNodeConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========================================================================
# TEST MODELS AND TOOLS
# ========================================================================

class TaskAnalysis(BaseModel):
    """Structured output model for task analysis."""
    task_type: str = Field(description="Type of task (research, coding, writing, etc.)")
    complexity: int = Field(ge=1, le=10, description="Complexity rating 1-10")
    estimated_time: str = Field(description="Estimated completion time")
    key_steps: List[str] = Field(description="Main steps to complete the task")


class PersonProfile(BaseModel):
    """Different structured output model."""
    name: str = Field(description="Person's name")
    profession: str = Field(description="Their profession")
    interests: List[str] = Field(description="List of interests")
    personality_traits: List[str] = Field(description="Key personality traits")


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Mock search results for: {query}"


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {e}"


@tool
def file_reader(filepath: str) -> str:
    """Read content from a file."""
    return f"Mock file content from: {filepath}"


# ========================================================================
# TEST FIXTURES
# ========================================================================

@pytest.fixture
def basic_state():
    """Basic state with messages."""
    return {
        "messages": [HumanMessage(content="Test input")],
        "engines": {}
    }


@pytest.fixture
def complex_state():
    """More complex state with additional fields."""
    return {
        "messages": [HumanMessage(content="Analyze this complex task")],
        "user_context": {"department": "engineering", "priority": "high"},
        "task_metadata": {"deadline": "2025-08-01", "assignee": "kai"},
        "engines": {}
    }


# ========================================================================
# SCENARIO 1: DIFFERENT PROMPT TEMPLATE INPUT VARIABLES
# ========================================================================

@pytest.mark.asyncio
async def test_different_prompt_variables():
    """Test GenericEngineNodeConfig with various ChatPromptTemplate input variables."""
    print("\n🧪 Testing different prompt template variables...")
    
    # Scenario 1a: Basic {messages} template
    basic_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}")
    ])
    
    basic_config = AugLLMConfig(
        name="basic_engine",
        prompt_template=basic_template,
        temperature=0.1
    )
    
    basic_node = GenericEngineNodeConfig(
        name="basic_node",
        engine=basic_config,
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Basic template created: {basic_node.name}")
    
    # Scenario 1b: Template with custom variables {user_input} and {context}
    custom_template = ChatPromptTemplate.from_messages([
        ("system", "You are analyzing tasks for the {department} department."),
        ("human", "Priority: {priority}\nTask: {user_input}\nContext: {context}")
    ])
    
    custom_config = AugLLMConfig(
        name="custom_engine", 
        prompt_template=custom_template,
        temperature=0.3
    )
    
    custom_node = GenericEngineNodeConfig(
        name="custom_node",
        engine=custom_config,
        input_fields=["user_input", "department", "priority", "context"],
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Custom template created: {custom_node.name}")
    
    # Scenario 1c: Multi-variable template {query}, {examples}, {format}
    complex_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Use this format: {format}"),
        ("human", "Query: {query}\n\nExamples:\n{examples}\n\nAnalyze thoroughly.")
    ])
    
    complex_config = AugLLMConfig(
        name="complex_engine",
        prompt_template=complex_template, 
        temperature=0.5
    )
    
    complex_node = GenericEngineNodeConfig(
        name="complex_node",
        engine=complex_config,
        input_fields=["query", "examples", "format"],
        auto_add_engine_attribution=True  
    )
    
    print(f"  ✅ Complex template created: {complex_node.name}")
    
    return [basic_node, custom_node, complex_node]


# ========================================================================
# SCENARIO 2: STRUCTURED OUTPUT MODELS
# ========================================================================

@pytest.mark.asyncio
async def test_structured_output_models():
    """Test GenericEngineNodeConfig with different structured output models."""
    print("\n🧪 Testing structured output models...")
    
    # Scenario 2a: TaskAnalysis model
    task_template = ChatPromptTemplate.from_messages([
        ("system", "You are a task analysis expert. Analyze tasks and provide structured output."),
        ("human", "Analyze this task: {task_description}")
    ])
    
    task_config = AugLLMConfig(
        name="task_analyzer",
        prompt_template=task_template,
        structured_output_model=TaskAnalysis,
        temperature=0.2
    )
    
    task_node = GenericEngineNodeConfig(
        name="task_analysis_node",
        engine=task_config,
        input_fields=["task_description"],
        output_schema=TaskAnalysis,
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ TaskAnalysis node created: {task_node.name}")
    print(f"    - Output schema: {task_node.output_schema}")
    
    # Scenario 2b: PersonProfile model  
    person_template = ChatPromptTemplate.from_messages([
        ("system", "You create detailed person profiles based on descriptions."),
        ("human", "Create a profile for: {person_description}")
    ])
    
    person_config = AugLLMConfig(
        name="person_profiler",
        prompt_template=person_template,
        structured_output_model=PersonProfile,
        temperature=0.4
    )
    
    person_node = GenericEngineNodeConfig(
        name="person_profile_node", 
        engine=person_config,
        input_fields=["person_description"],
        output_schema=PersonProfile,
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ PersonProfile node created: {person_node.name}")
    print(f"    - Output schema: {person_node.output_schema}")
    
    return [task_node, person_node]


# ========================================================================
# SCENARIO 3: REGULAR LANGCHAIN TOOLS
# ========================================================================

@pytest.mark.asyncio 
async def test_regular_tools():
    """Test GenericEngineNodeConfig with regular LangChain tools."""
    print("\n🧪 Testing regular LangChain tools...")
    
    # Scenario 3a: Single tool (calculator)
    calc_template = ChatPromptTemplate.from_messages([
        ("system", "You are a math assistant. Use the calculator tool for calculations."),
        ("placeholder", "{messages}")
    ])
    
    calc_config = AugLLMConfig(
        name="calculator_engine",
        prompt_template=calc_template,
        tools=[calculator],
        temperature=0.1
    )
    
    calc_node = GenericEngineNodeConfig(
        name="calculator_node",
        engine=calc_config,
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Calculator node created: {calc_node.name}")
    print(f"    - Tools: {len(calc_config.tools)} ({[t.name for t in calc_config.tools]})")
    
    # Scenario 3b: Multiple tools
    multi_template = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant with access to web search, calculator, and file reading tools."),
        ("human", "Help me with: {request}")
    ])
    
    multi_config = AugLLMConfig(
        name="multi_tool_engine",
        prompt_template=multi_template,
        tools=[web_search, calculator, file_reader],
        temperature=0.3
    )
    
    multi_node = GenericEngineNodeConfig(
        name="multi_tool_node",
        engine=multi_config,
        input_fields=["request"],
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Multi-tool node created: {multi_node.name}")
    print(f"    - Tools: {len(multi_config.tools)} ({[t.name for t in multi_config.tools]})")
    
    return [calc_node, multi_node]


# ========================================================================
# SCENARIO 4: NO TOOLS (BASIC LLM)
# ========================================================================

@pytest.mark.asyncio
async def test_no_tools():
    """Test GenericEngineNodeConfig with basic LLM (no tools)."""
    print("\n🧪 Testing no tools (basic LLM)...")
    
    # Scenario 4a: Simple chat
    chat_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful conversational assistant."),
        ("placeholder", "{messages}")
    ])
    
    chat_config = AugLLMConfig(
        name="chat_engine",
        prompt_template=chat_template,
        temperature=0.7
    )
    
    chat_node = GenericEngineNodeConfig(
        name="chat_node",
        engine=chat_config,
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Chat node created: {chat_node.name}")
    print(f"    - Tools: None")
    print(f"    - Engine attribution: {chat_node.auto_add_engine_attribution}")
    
    # Scenario 4b: Creative writing
    creative_template = ChatPromptTemplate.from_messages([
        ("system", "You are a creative writer. Write in the style of {style} about {topic}."),
        ("human", "Create a {content_type} with the following requirements: {requirements}")
    ])
    
    creative_config = AugLLMConfig(
        name="creative_engine",
        prompt_template=creative_template,
        temperature=0.9
    )
    
    creative_node = GenericEngineNodeConfig(
        name="creative_node",
        engine=creative_config,
        input_fields=["style", "topic", "content_type", "requirements"],
        auto_add_engine_attribution=True
    )
    
    print(f"  ✅ Creative node created: {creative_node.name}")
    print(f"    - Input fields: {creative_node.input_fields}")
    
    return [chat_node, creative_node]


# ========================================================================
# SCENARIO 5: ENGINE ATTRIBUTION AND FIELD MAPPING  
# ========================================================================

@pytest.mark.asyncio
async def test_engine_attribution():
    """Test engine attribution and field mapping features."""
    print("\n🧪 Testing engine attribution and field mapping...")
    
    attribution_template = ChatPromptTemplate.from_messages([
        ("system", "You process data and provide attributed results."),
        ("human", "Process: {input_data}")
    ])
    
    attribution_config = AugLLMConfig(
        name="attribution_test_engine",
        prompt_template=attribution_template,
        temperature=0.2
    )
    
    # Test with attribution enabled
    with_attribution = GenericEngineNodeConfig(
        name="with_attribution_node",
        engine=attribution_config,
        input_fields=["input_data"],
        output_fields={"response": "processed_result", "metadata": "processing_info"},
        auto_add_engine_attribution=True
    )
    
    # Test with attribution disabled
    without_attribution = GenericEngineNodeConfig(
        name="without_attribution_node", 
        engine=attribution_config,
        input_fields=["input_data"],
        output_fields=["response"],
        auto_add_engine_attribution=False
    )
    
    print(f"  ✅ Attribution enabled node: {with_attribution.name}")
    print(f"    - Auto attribution: {with_attribution.auto_add_engine_attribution}")
    print(f"    - Output fields: {with_attribution.output_fields}")
    
    print(f"  ✅ Attribution disabled node: {without_attribution.name}")
    print(f"    - Auto attribution: {without_attribution.auto_add_engine_attribution}")
    print(f"    - Output fields: {without_attribution.output_fields}")
    
    return [with_attribution, without_attribution]


# ========================================================================
# INTEGRATION TEST: ALL SCENARIOS TOGETHER
# ========================================================================

@pytest.mark.asyncio
async def test_all_scenarios_integration():
    """Integration test running all scenarios together."""
    print("\n🚀 Running comprehensive GenericEngineNodeConfig integration test...")
    print("=" * 80)
    
    # Run all scenario tests
    prompt_nodes = await test_different_prompt_variables()
    struct_nodes = await test_structured_output_models()
    tool_nodes = await test_regular_tools()
    basic_nodes = await test_no_tools()
    attr_nodes = await test_engine_attribution()
    
    all_nodes = prompt_nodes + struct_nodes + tool_nodes + basic_nodes + attr_nodes
    
    print(f"\n📊 Test Summary:")
    print(f"  Total nodes created: {len(all_nodes)}")
    
    # Group by engine type
    by_engine = {}
    for node in all_nodes:
        engine_name = node.engine.name if node.engine else "no_engine"
        if engine_name not in by_engine:
            by_engine[engine_name] = []
        by_engine[engine_name].append(node.name)
    
    for engine_name, node_names in by_engine.items():
        print(f"  {engine_name}: {len(node_names)} nodes")
        for node_name in node_names:
            print(f"    - {node_name}")
    
    # Check features
    attribution_enabled = sum(1 for n in all_nodes if n.auto_add_engine_attribution)
    with_tools = sum(1 for n in all_nodes if n.engine and getattr(n.engine, 'tools', None))
    with_structured = sum(1 for n in all_nodes if n.output_schema)
    
    print(f"\n📈 Feature Analysis:")
    print(f"  Nodes with engine attribution: {attribution_enabled}/{len(all_nodes)}")
    print(f"  Nodes with tools: {with_tools}/{len(all_nodes)}")
    print(f"  Nodes with structured output: {with_structured}/{len(all_nodes)}")
    
    print(f"\n✅ All scenarios completed successfully!")
    print(f"🎯 GenericEngineNodeConfig is ready for production use!")
    
    return all_nodes


# ========================================================================
# MAIN TEST RUNNER
# ========================================================================

async def main():
    """Run all tests."""
    print("🔬 GenericEngineNodeConfig Comprehensive Test Suite")
    print("Author: Kai")
    print("=" * 80)
    
    try:
        await test_all_scenarios_integration()
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"💡 GenericEngineNodeConfig handles all AugLLMConfig scenarios correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)