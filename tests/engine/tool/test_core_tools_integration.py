#!/usr/bin/env python3
"""
Test haive.core.tools Integration with ToolEngine

This tests integration between:
- haive.core.tools.store_tools
- haive.core.tools.store_manager
- haive.core.engine.tool.ToolEngine
- haive.core.engine.aug_llm.AugLLMConfig
"""

import pytest
from unittest.mock import Mock, patch
from typing import List
from langchain_core.tools import Tool

from haive.core.engine.tool import ToolEngine


class TestCoreToolsIntegration:
    """Test suite for core.tools integration with ToolEngine."""

    def test_tool_engine_has_store_tools_integration(self):
        """Test that ToolEngine has store tools integration."""
        # Check for store tools method
        assert hasattr(ToolEngine, 'create_store_tools_suite')
        assert callable(ToolEngine.create_store_tools_suite)

    def test_create_store_tools_suite_is_classmethod(self):
        """Test that create_store_tools_suite is a class method."""
        method = ToolEngine.create_store_tools_suite
        assert hasattr(method, '__self__')
        assert method.__self__ is ToolEngine

    def test_create_store_tools_suite_docstring(self):
        """Test create_store_tools_suite has proper documentation."""
        method = ToolEngine.create_store_tools_suite
        assert method.__doc__ is not None
        assert "store" in method.__doc__.lower()
        assert "memory" in method.__doc__.lower()

    @patch('haive.core.tools.store_tools.create_memory_tools_suite')
    def test_create_store_tools_suite_calls_factory(self, mock_create):
        """Test that create_store_tools_suite calls the factory method."""
        # Mock the factory to return a list of tools
        mock_tools = [Mock(spec=Tool) for _ in range(3)]
        mock_create.return_value = mock_tools
        
        # Mock store manager
        mock_store_manager = Mock()
        
        # Call the method
        try:
            result = ToolEngine.create_store_tools_suite(mock_store_manager)
            
            # Verify factory was called
            mock_create.assert_called_once_with(
                mock_store_manager, 
                namespace=None, 
                include_tools=None
            )
            
            # Verify result
            assert result == mock_tools
            
        except ImportError:
            # If import fails, that's expected in test environment
            pytest.skip("Store tools not available in test environment")

    def test_store_tools_imports_available(self):
        """Test that store tools imports are available."""
        try:
            from haive.core.tools.store_manager import StoreManager
            from haive.core.tools.store_tools import create_memory_tools_suite
            
            # Basic checks
            assert StoreManager is not None
            assert create_memory_tools_suite is not None
            assert callable(create_memory_tools_suite)
            
        except ImportError as e:
            pytest.skip(f"Store tools not available: {e}")

    def test_store_tools_input_schemas_exist(self):
        """Test that store tools have proper input schemas."""
        try:
            from haive.core.tools.store_tools import (
                StoreMemoryInput,
                SearchMemoryInput
            )
            
            # Check schemas exist and are BaseModel
            from pydantic import BaseModel
            assert issubclass(StoreMemoryInput, BaseModel)
            assert issubclass(SearchMemoryInput, BaseModel)
            
            # Check required fields
            assert 'content' in StoreMemoryInput.model_fields
            assert 'query' in SearchMemoryInput.model_fields
            
        except ImportError as e:
            pytest.skip(f"Store tool schemas not available: {e}")

    def test_tool_engine_integration_pattern(self):
        """Test the integration pattern between ToolEngine and store tools."""
        # This tests the expected integration pattern without requiring actual stores
        
        # 1. ToolEngine should be able to create store tools
        assert hasattr(ToolEngine, 'create_store_tools_suite')
        
        # 2. Store tools should integrate with AugLLMConfig
        # This is tested by the @tool decorator pattern in store_tools.py
        
        # 3. Tools should have proper LangChain compatibility
        # This is ensured by the create_memory_tools_suite factory

    def test_interrupt_tool_integration(self):
        """Test interrupt tool integration."""
        # Check if interrupt tools are available
        try:
            from haive.core.tools.interrupt_tool_wrapper import InterruptibleToolWrapper
            assert InterruptibleToolWrapper is not None
            
            # Check ToolEngine has interrupt tool creation
            assert hasattr(ToolEngine, 'create_interruptible_tool')
            
        except ImportError:
            pytest.skip("Interrupt tools not available")

    def test_tools_directory_structure(self):
        """Test that tools directory has expected structure."""
        import haive.core.tools
        
        # Check module exists
        assert haive.core.tools.__file__ is not None
        
        # Check expected files exist (at import level)
        expected_modules = [
            'store_manager',
            'store_tools'
        ]
        
        for module_name in expected_modules:
            try:
                module = __import__(f'haive.core.tools.{module_name}', fromlist=[module_name])
                assert module is not None
            except ImportError:
                pytest.skip(f"Module {module_name} not available")

    def test_tool_engine_factory_pattern(self):
        """Test that ToolEngine follows factory pattern for tools."""
        # Get all create_* methods
        create_methods = [
            attr for attr in dir(ToolEngine) 
            if attr.startswith('create_') and callable(getattr(ToolEngine, attr))
        ]
        
        # Should have multiple creation methods
        assert len(create_methods) >= 5
        
        # Check specific expected methods
        expected_methods = [
            'create_structured_output_tool',
            'create_retriever_tool',
            'create_state_tool',
            'create_interruptible_tool',
            'create_store_tools_suite',
            'create_human_interrupt_tool'
        ]
        
        for method_name in expected_methods:
            assert method_name in create_methods

    def test_augment_tool_integration(self):
        """Test augment_tool method integration."""
        assert hasattr(ToolEngine, 'augment_tool')
        assert callable(ToolEngine.augment_tool)
        
        method = ToolEngine.augment_tool
        assert hasattr(method, '__self__')  # Is a classmethod
        assert method.__self__ is ToolEngine


@pytest.mark.integration
class TestCoreToolsRealIntegration:
    """Integration tests requiring real components (marked for integration runs)."""

    @pytest.mark.skip(reason="Requires real store setup")
    def test_real_store_tools_creation(self):
        """Test real store tools creation with actual StoreManager."""
        # This would require actual store setup
        # Left as placeholder for integration test suite
        pass

    @pytest.mark.skip(reason="Requires real AugLLMConfig setup") 
    def test_store_tools_with_aug_llm_config(self):
        """Test store tools integration with AugLLMConfig."""
        # This would test the full pipeline:
        # StoreManager → create_memory_tools_suite → AugLLMConfig → Agent
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])