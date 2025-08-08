#!/usr/bin/env python3
"""
Comprehensive ToolEngine Integration Test Suite

This tests the complete integration of:
- ToolEngine + ToolRouteMixin + AugLLMConfig
- Tool routing refactor (structured_output_model → parse_output)
- Engine conversion methods 
- Core tools integration
- Schema integration
- ValidationNodeConfigV2 integration
"""

import pytest
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from langchain_core.tools import tool

from haive.core.engine.tool import ToolEngine
from haive.core.engine.tool.types import ToolCapability
from haive.core.engine.aug_llm import AugLLMConfig


class ComprehensiveTestModel(BaseModel):
    """Comprehensive test model for all scenarios."""
    query: str = Field(description="Test query")
    results: List[str] = Field(description="Test results")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegularValidationModel(BaseModel):
    """Regular Pydantic model (not structured output)."""
    name: str = Field(description="Name field")
    value: int = Field(ge=0, description="Value field")


def calculator_func(expression: str) -> str:
    """Test calculator tool."""
    return f"Calculated: {expression} = 42"

# Create tool from function to avoid pytest collection issue
calculator_tool = tool("test_calculator")(calculator_func)


class TestComprehensiveToolIntegration:
    """Comprehensive integration test suite."""

    def test_routing_refactor_success(self):
        """Test the main routing refactor: structured_output_model → parse_output."""
        # Create AugLLMConfig with structured output model
        config = AugLLMConfig(structured_output_model=ComprehensiveTestModel)
        
        # Verify the fix
        model_route = config.tool_routes.get('ComprehensiveTestModel')
        assert model_route == 'parse_output', f"Expected 'parse_output', got '{model_route}'"
        
        # Verify metadata
        metadata = config.get_tool_metadata('ComprehensiveTestModel')
        assert metadata is not None
        assert metadata.get('is_structured_output') is True

    def test_regular_pydantic_model_unchanged(self):
        """Test that regular Pydantic models still get pydantic_model route."""
        config = AugLLMConfig(tools=[RegularValidationModel])
        
        regular_route = config.tool_routes.get('RegularValidationModel')
        assert regular_route == 'pydantic_model', f"Expected 'pydantic_model', got '{regular_route}'"

    def test_tool_engine_conversions_available(self):
        """Test that all ToolEngine conversion methods are available."""
        conversion_methods = [
            'from_aug_llm_config',
            'from_retriever_config',
            'from_vectorstore_config',
            'from_document_engine',
            'from_multiple_engines'
        ]
        
        for method in conversion_methods:
            assert hasattr(ToolEngine, method), f"Missing conversion method: {method}"
            assert callable(getattr(ToolEngine, method))

    def test_tool_engine_is_invokable_engine(self):
        """Test ToolEngine inheritance."""
        from haive.core.engine.base import InvokableEngine
        
        assert issubclass(ToolEngine, InvokableEngine)
        
        # Test instance
        engine = ToolEngine()
        assert isinstance(engine, InvokableEngine)
        assert hasattr(engine, 'engine_type')

    def test_structured_output_tools_get_capabilities(self):
        """Test that structured output tools get proper capabilities."""
        # Create structured output tool
        structured_tool = ToolEngine.create_structured_output_tool(
            func=lambda query: ComprehensiveTestModel(
                query=query,
                results=["result1", "result2"]
            ),
            name="test_structured",
            description="Test structured output tool",
            output_model=ComprehensiveTestModel
        )
        
        # Verify capabilities
        capabilities = getattr(structured_tool, '__tool_capabilities__', set())
        assert ToolCapability.STRUCTURED_OUTPUT in capabilities
        assert ToolCapability.VALIDATED_OUTPUT in capabilities

    def test_tool_capabilities_routing_logic(self):
        """Test that tools with STRUCTURED_OUTPUT capability get proper treatment."""
        # Create a tool with capabilities
        enhanced_tool = ToolEngine.augment_tool(
            calculator_tool,
            structured_output_model=ComprehensiveTestModel,
            name="enhanced_calculator"
        )
        
        # Verify it has capabilities
        capabilities = getattr(enhanced_tool, '__tool_capabilities__', set())
        assert ToolCapability.STRUCTURED_OUTPUT in capabilities

    def test_core_tools_integration(self):
        """Test core.tools integration."""
        # Test method exists
        assert hasattr(ToolEngine, 'create_store_tools_suite')
        assert callable(ToolEngine.create_store_tools_suite)

    def test_schema_integration(self):
        """Test schema integration."""
        from haive.core.schema.prebuilt.validation_routing_example import _get_target_node_for_route
        
        # Test updated route mapping
        parse_output_target = _get_target_node_for_route('parse_output')
        pydantic_model_target = _get_target_node_for_route('pydantic_model')
        
        assert parse_output_target == 'parser_tools'
        assert pydantic_model_target == 'pydantic_tools'

    def test_mixed_tool_types_routing(self):
        """Test comprehensive routing with mixed tool types."""
        # Create config with multiple tool types
        config = AugLLMConfig(
            structured_output_model=ComprehensiveTestModel,
            tools=[calculator_tool, RegularValidationModel]
        )
        
        # Verify each tool gets correct route
        expected_routes = {
            'ComprehensiveTestModel': 'parse_output',
            'RegularValidationModel': 'pydantic_model',
            'test_calculator': 'langchain_tool'
        }
        
        # Verify we have the expected tool routes
        
        for tool_name, expected_route in expected_routes.items():
            actual_route = config.tool_routes.get(tool_name)
            assert actual_route == expected_route, f"Tool {tool_name}: expected {expected_route}, got {actual_route}"

    def test_tool_metadata_preservation(self):
        """Test that tool metadata is properly preserved."""
        config = AugLLMConfig(structured_output_model=ComprehensiveTestModel)
        
        metadata = config.get_tool_metadata('ComprehensiveTestModel')
        assert metadata is not None
        assert metadata.get('is_structured_output') is True
        assert metadata.get('tool_type') == 'structured_output_model'
        assert metadata.get('purpose') == 'structured_output'

    def test_backward_compatibility(self):
        """Test that existing functionality still works."""
        # Test regular LangChain tools
        config = AugLLMConfig(tools=[calculator_tool])
        # The tool should be registered with its correct name
        assert 'test_calculator' in config.tool_routes
        assert config.tool_routes['test_calculator'] == 'langchain_tool'
        
        # Test regular Pydantic models
        config2 = AugLLMConfig(tools=[RegularValidationModel])
        assert 'RegularValidationModel' in config2.tool_routes
        assert config2.tool_routes['RegularValidationModel'] == 'pydantic_model'

    def test_tool_engine_factory_methods(self):
        """Test ToolEngine factory methods."""
        factory_methods = [
            'create_structured_output_tool',
            'create_retriever_tool', 
            'create_state_tool',
            'create_interruptible_tool',
            'create_store_tools_suite',
            'augment_tool'
        ]
        
        for method in factory_methods:
            assert hasattr(ToolEngine, method), f"Missing factory method: {method}"
            method_obj = getattr(ToolEngine, method)
            assert callable(method_obj)
            # Should be class method
            assert hasattr(method_obj, '__self__')

    def test_end_to_end_structured_output_flow(self):
        """Test complete end-to-end structured output flow."""
        # 1. Create structured output model
        model = ComprehensiveTestModel
        
        # 2. Create AugLLMConfig with structured output
        config = AugLLMConfig(structured_output_model=model)
        
        # 3. Verify routing
        assert config.tool_routes['ComprehensiveTestModel'] == 'parse_output'
        
        # 4. Verify metadata
        metadata = config.get_tool_metadata('ComprehensiveTestModel')
        assert metadata['is_structured_output'] is True
        
        # 5. Verify schema integration would route to parser_tools
        from haive.core.schema.prebuilt.validation_routing_example import _get_target_node_for_route
        target = _get_target_node_for_route('parse_output')
        assert target == 'parser_tools'

    def test_integration_completeness(self):
        """Test that all integration pieces are in place."""
        # 1. ToolEngine is InvokableEngine ✓
        from haive.core.engine.base import InvokableEngine
        assert issubclass(ToolEngine, InvokableEngine)
        
        # 2. Routing refactor works ✓
        config = AugLLMConfig(structured_output_model=ComprehensiveTestModel)
        assert config.tool_routes['ComprehensiveTestModel'] == 'parse_output'
        
        # 3. Conversion methods exist ✓
        assert hasattr(ToolEngine, 'from_aug_llm_config')
        
        # 4. Core tools integration ✓
        assert hasattr(ToolEngine, 'create_store_tools_suite')
        
        # 5. Schema integration ✓
        from haive.core.schema.prebuilt.validation_routing_example import _get_target_node_for_route
        assert _get_target_node_for_route('parse_output') == 'parser_tools'


@pytest.mark.integration  
class TestRealComponentIntegration:
    """Integration tests with real components (marked for integration test runs)."""

    @pytest.mark.skip(reason="Requires real LLM setup")
    def test_real_aug_llm_integration(self):
        """Test with real AugLLMConfig and LLM."""
        # Would test actual LLM calls with structured output
        pass

    @pytest.mark.skip(reason="Requires real store setup")
    def test_real_store_tools_integration(self):
        """Test with real store tools."""
        # Would test actual store tools creation and usage
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])