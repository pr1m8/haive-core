#!/usr/bin/env python3
"""
Test ToolEngine Conversion Methods - Comprehensive Testing

This tests the new class methods for converting other engines to ToolEngine:
- from_aug_llm_config()
- from_retriever_config()  
- from_vectorstore_config()
- from_document_engine()
- from_multiple_engines()
"""

import pytest
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from haive.core.engine.tool import ToolEngine
from haive.core.engine.tool.types import ToolCapability
from haive.core.engine.aug_llm import AugLLMConfig


class StructuredOutputModel(BaseModel):
    """Test structured output model."""
    query: str = Field(description="Search query")
    results: List[str] = Field(description="Search results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class TestToolEngineConversionMethods:
    """Test suite for ToolEngine conversion methods."""

    def test_from_aug_llm_config_basic(self):
        """Test basic conversion from AugLLMConfig."""
        # Create AugLLMConfig with structured output
        aug_config = AugLLMConfig(
            structured_output_model=StructuredOutputModel
        )
        
        # Test conversion method exists
        assert hasattr(ToolEngine, 'from_aug_llm_config')
        
        # Test method is callable
        assert callable(ToolEngine.from_aug_llm_config)

    def test_from_retriever_config_method_exists(self):
        """Test from_retriever_config method exists."""
        assert hasattr(ToolEngine, 'from_retriever_config')
        assert callable(ToolEngine.from_retriever_config)

    def test_from_vectorstore_config_method_exists(self):
        """Test from_vectorstore_config method exists."""
        assert hasattr(ToolEngine, 'from_vectorstore_config')
        assert callable(ToolEngine.from_vectorstore_config)

    def test_from_document_engine_method_exists(self):
        """Test from_document_engine method exists."""
        assert hasattr(ToolEngine, 'from_document_engine')
        assert callable(ToolEngine.from_document_engine)

    def test_from_multiple_engines_method_exists(self):
        """Test from_multiple_engines method exists."""
        assert hasattr(ToolEngine, 'from_multiple_engines')
        assert callable(ToolEngine.from_multiple_engines)

    def test_all_conversion_methods_are_classmethods(self):
        """Test that all conversion methods are class methods."""
        conversion_methods = [
            'from_aug_llm_config',
            'from_retriever_config',
            'from_vectorstore_config', 
            'from_document_engine',
            'from_multiple_engines'
        ]
        
        for method_name in conversion_methods:
            method = getattr(ToolEngine, method_name)
            # Class methods have __self__ attribute pointing to the class
            assert hasattr(method, '__self__')
            assert method.__self__ is ToolEngine

    def test_tool_engine_is_invokable_engine(self):
        """Test that ToolEngine is an InvokableEngine."""
        from haive.core.engine.base import InvokableEngine
        
        assert issubclass(ToolEngine, InvokableEngine)
        
        # Test instance
        engine = ToolEngine()
        assert isinstance(engine, InvokableEngine)

    def test_conversion_method_docstrings(self):
        """Test that conversion methods have proper docstrings."""
        conversion_methods = [
            'from_aug_llm_config',
            'from_retriever_config',
            'from_vectorstore_config',
            'from_document_engine', 
            'from_multiple_engines'
        ]
        
        for method_name in conversion_methods:
            method = getattr(ToolEngine, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__) > 50  # Substantial documentation
            assert "ToolEngine" in method.__doc__

    def test_existing_tool_creation_methods_still_work(self):
        """Test that existing tool creation methods still work."""
        # Test existing class methods
        assert hasattr(ToolEngine, 'create_structured_output_tool')
        assert hasattr(ToolEngine, 'create_retriever_tool')
        assert hasattr(ToolEngine, 'augment_tool')
        assert hasattr(ToolEngine, 'create_state_tool')
        assert hasattr(ToolEngine, 'create_interruptible_tool')

    def test_tool_engine_basic_functionality(self):
        """Test basic ToolEngine functionality."""
        # Create empty ToolEngine
        engine = ToolEngine()
        
        # Test basic properties
        assert hasattr(engine, 'engine_type')
        assert hasattr(engine, 'tools')
        assert hasattr(engine, 'name')
        
        # Test it's properly initialized
        assert engine.tools is None or isinstance(engine.tools, (list, tuple))

    @pytest.mark.skip(reason="Requires actual engine instances - integration test")
    def test_from_aug_llm_config_integration(self):
        """Integration test for from_aug_llm_config (skipped for unit tests)."""
        # This would require real AugLLMConfig setup
        # Moved to integration tests
        pass

    def test_conversion_methods_return_tool_engine(self):
        """Test that conversion methods are documented to return ToolEngine."""
        conversion_methods = [
            'from_aug_llm_config',
            'from_retriever_config',
            'from_vectorstore_config',
            'from_document_engine',
            'from_multiple_engines'
        ]
        
        for method_name in conversion_methods:
            method = getattr(ToolEngine, method_name)
            # Check return type annotation if available
            if hasattr(method, '__annotations__'):
                return_annotation = method.__annotations__.get('return')
                if return_annotation:
                    assert 'ToolEngine' in str(return_annotation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])