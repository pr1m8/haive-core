#!/usr/bin/env python3
"""
Step 1: Integration Tests - ToolRouteMixin + ToolEngine

This module tests the integration between ToolRouteMixin and ToolEngine,
ensuring they work together seamlessly for tool management and analysis.
"""

import pytest
from typing import Annotated
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt.tool_node import InjectedState
from pydantic import BaseModel, Field

from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
from haive.core.engine.tool import ToolEngine
from haive.core.engine.tool.types import ToolCapability, ToolCategory
from haive.core.tools.store_manager import StoreManager


class ToolContainer(ToolRouteMixin):
    """Tool container that uses ToolRouteMixin."""
    
    def __init__(self, **data):
        super().__init__(**data)


# Test tools
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def state_aware_tool(
    message: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Tool that accesses injected state."""
    msg_count = len(state.get("messages", []))
    return f"Processed '{message}' with {msg_count} messages in state"


class SearchResult(BaseModel):
    """Search result model."""
    query: str = Field(description="The search query")
    results: list[str] = Field(description="Search results")
    count: int = Field(description="Number of results")


class TestToolRouteMixinToolEngineIntegration:
    """Test integration between ToolRouteMixin and ToolEngine."""
    
    def test_basic_tool_route_mixin_functionality(self):
        """Test basic ToolRouteMixin functionality works."""
        container = ToolContainer()
        
        # Add tools
        container.add_tool(calculator)
        container.add_tool(state_aware_tool)
        
        # Verify tools were added
        assert len(container.tools) == 2
        assert len(container.tool_routes) == 2
        assert len(container.tool_instances) == 2
        
        # Verify tool retrieval
        calc_tool = container.get_tool("calculator")
        assert calc_tool is not None
        assert calc_tool.name == "calculator"
        
        state_tool = container.get_tool("state_aware_tool")
        assert state_tool is not None
        assert state_tool.name == "state_aware_tool"
    
    def test_tool_engine_analysis_of_route_mixin_tools(self):
        """Test ToolEngine analysis of tools managed by ToolRouteMixin."""
        container = ToolContainer()
        
        # Add tools via ToolRouteMixin
        container.add_tool(calculator)
        container.add_tool(state_aware_tool)
        
        # Create ToolEngine with the same tools
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        
        # Verify ToolEngine analysis
        calc_props = engine.get_tool_properties("calculator")
        assert calc_props is not None
        assert calc_props.tool_type.value == "structured_tool"
        assert calc_props.category == ToolCategory.COMPUTATION
        
        state_props = engine.get_tool_properties("state_aware_tool")
        assert state_props is not None
        assert state_props.is_state_tool
        assert ToolCapability.INJECTED_STATE in state_props.capabilities
    
    def test_tool_engine_enhanced_tools_in_route_mixin(self):
        """Test ToolEngine-enhanced tools work with ToolRouteMixin."""
        # Create enhanced tools using ToolEngine
        state_tool = ToolEngine.create_state_tool(
            calculator,
            reads_state=True,
            state_keys=["messages"]
        )
        
        interruptible_tool = ToolEngine.create_interruptible_tool(
            lambda x: f"Interruptible: {x}",
            name="interruptible_test"
        )
        
        # Add to ToolRouteMixin container
        container = ToolContainer()
        container.add_tool(state_tool)
        container.add_tool(interruptible_tool)
        
        # Verify tools were added correctly
        assert len(container.tools) == 2
        assert "calculator" in container.tool_routes
        assert "interruptible_test" in container.tool_routes
        
        # Verify enhanced properties are preserved
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        
        calc_props = engine.get_tool_properties("calculator")
        assert calc_props.is_state_tool
        assert ToolCapability.READS_STATE in calc_props.capabilities
        
        int_props = engine.get_tool_properties("interruptible_test")
        assert int_props.is_interruptible
        assert ToolCapability.INTERRUPTIBLE in int_props.capabilities
    
    def test_store_tools_integration(self):
        """Test store tools created by ToolEngine work with ToolRouteMixin."""
        # Create store tools
        store_manager = StoreManager()
        store_tools = ToolEngine.create_store_tools_suite(
            store_manager=store_manager,
            namespace=("test", "integration"),
            include_tools=["store", "search"]
        )
        
        # Add to ToolRouteMixin container
        container = ToolContainer()
        for tool in store_tools:
            container.add_tool(tool)
        
        # Verify store tools were added
        assert len(container.tools) == 2
        
        # Find store and search tools
        store_tool_names = [name for name in container.tool_routes 
                           if "store" in name and "search" not in name]
        search_tool_names = [name for name in container.tool_routes 
                            if "search" in name]
        
        assert len(store_tool_names) > 0
        assert len(search_tool_names) > 0
        
        # Verify ToolEngine analysis of store tools
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        
        store_props = engine.get_tool_properties(store_tool_names[0])
        assert store_props.category == ToolCategory.MEMORY
        assert ToolCapability.STORE in store_props.capabilities
        
        search_props = engine.get_tool_properties(search_tool_names[0])
        assert search_props.category == ToolCategory.MEMORY
        assert ToolCapability.RETRIEVER in search_props.capabilities
    
    def test_structured_output_tools_integration(self):
        """Test structured output tools integration."""
        # Create structured output tool
        def search_function(query: str) -> SearchResult:
            """Search for information."""
            return SearchResult(
                query=query,
                results=[f"Result for {query}"],
                count=1
            )
        
        structured_tool = ToolEngine.create_structured_output_tool(
            func=search_function,
            name="search_tool",
            description="Search with structured output",
            output_model=SearchResult
        )
        
        # Add to ToolRouteMixin container
        container = ToolContainer()
        container.add_tool(structured_tool)
        
        # Verify tool was added
        assert len(container.tools) == 1
        assert "search_tool" in container.tool_routes
        
        # Verify ToolEngine analysis
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        props = engine.get_tool_properties("search_tool")
        
        assert props.structured_output_model == SearchResult
        assert props.is_structured_output_model
        assert ToolCapability.STRUCTURED_OUTPUT in props.capabilities
        assert ToolCapability.VALIDATED_OUTPUT in props.capabilities
    
    def test_tool_routing_strategies(self):
        """Test tool routing strategies work with mixed tool sources."""
        container = ToolContainer()
        
        # Add various types of tools
        container.add_tool(calculator)  # Regular tool
        container.add_tool(state_aware_tool)  # InjectedState tool
        
        # Add ToolEngine-enhanced tools
        interruptible = ToolEngine.create_interruptible_tool(
            lambda x: f"Int: {x}",
            name="interruptible"
        )
        container.add_tool(interruptible)
        
        # Create ToolEngine with routing strategies
        engine = ToolEngine(
            tools=container.tools,
            enable_analysis=True,
            routing_strategy="capability"
        )
        
        # Test capability-based queries
        state_tools = engine.get_state_tools()
        assert "state_aware_tool" in state_tools
        
        interruptible_tools = engine.get_interruptible_tools()
        assert "interruptible" in interruptible_tools
        
        # Test category-based queries  
        computation_tools = engine.get_tools_by_category(ToolCategory.COMPUTATION)
        assert "calculator" in computation_tools
    
    def test_tool_metadata_synchronization(self):
        """Test metadata synchronization between ToolRouteMixin and ToolEngine."""
        container = ToolContainer()
        
        # Add tool with explicit metadata
        container.add_tool(
            calculator, 
            metadata={"priority": "high", "cost": "low"}
        )
        
        # Verify ToolRouteMixin metadata
        calc_metadata = container.get_tool_metadata("calculator")
        assert calc_metadata is not None
        assert calc_metadata["priority"] == "high"
        assert calc_metadata["cost"] == "low"
        
        # Verify ToolEngine can analyze the same tool
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        props = engine.get_tool_properties("calculator")
        
        # ToolEngine should have its own analysis
        assert props is not None
        assert props.name == "calculator"
        assert props.category == ToolCategory.COMPUTATION
    
    def test_mixed_tool_sources_compatibility(self):
        """Test compatibility when tools come from different sources."""
        # Create separate tool instances
        @tool
        def regular_calc(expression: str) -> str:
            """Calculate mathematical expressions - regular version."""
            return f"Regular: {eval(expression)}"
        
        enhanced_tool = ToolEngine.create_state_tool(
            lambda x: f"Enhanced: {x}",
            name="enhanced",
            reads_state=True
        )
        
        # Add to ToolRouteMixin container
        container = ToolContainer()
        container.add_tool(regular_calc)
        container.add_tool(enhanced_tool)
        
        # Both ToolRouteMixin and ToolEngine should work
        assert len(container.tools) == 2
        assert len(container.tool_routes) == 2
        
        engine = ToolEngine(tools=container.tools, enable_analysis=True)
        
        # Both tools should be analyzable
        regular_props = engine.get_tool_properties("regular_calc")
        enhanced_props = engine.get_tool_properties("enhanced")
        
        assert regular_props is not None
        assert enhanced_props is not None
        assert not regular_props.is_state_tool  # Regular tool
        assert enhanced_props.is_state_tool     # Enhanced tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])