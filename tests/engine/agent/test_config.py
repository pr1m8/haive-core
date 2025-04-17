# tests/core/engine/agent/config/test_base.py

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from haive_core.engine.agent.config import AgentConfig
from haive_core.engine.base import EngineType, Engine
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.schema.state_schema import StateSchema

class TestAgentConfig:
    """Tests for the AgentConfig base class."""
    
    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = AgentConfig()
        
        assert config.engine_type == EngineType.AGENT
        assert config.name.startswith("agent_")
        assert isinstance(config.engine, AugLLMConfig)
        assert config.engines == {}
        assert config.visualize is True
        assert config.auto_end is True
        
    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        llm = AugLLMConfig(name="test_llm")
        config = AgentConfig(
            name="test_agent",
            engine=llm,
            visualize=False,
            auto_end=False,
            debug=True
        )
        
        assert config.name == "test_agent"
        assert config.engine == llm
        assert config.visualize is False
        assert config.auto_end is False
        assert config.debug is True
        
    def test_derive_schema(self):
        """Test schema derivation."""
        llm = AugLLMConfig(name="test_llm")
        config = AgentConfig(
            name="test_agent",
            engine=llm
        )
        
        schema = config.derive_schema()
        
        assert issubclass(schema, StateSchema)
        #assert hasattr(schema, "messages")
        #assert hasattr(schema, "__runnable_config__")
        
    def test_resolve_engine_from_instance(self):
        """Test resolving engine from direct instance."""
        llm = AugLLMConfig(name="test_llm")
        config = AgentConfig(engine=llm)
        
        resolved = config.resolve_engine()
        
        assert resolved == llm
        
    @patch("src.haive.core.engine.base.EngineRegistry")
    def test_resolve_engine_from_registry(self, mock_registry):
        # Setup mock
        mock_instance = MagicMock()
        mock_registry.get_instance.return_value = mock_instance
        
        # Create a mock LLM config with the name "test_llm"
        mock_llm = AugLLMConfig(name="test_llm")
        
        # Configure the mock registry to return the mock LLM for ALL engine types
        def mock_get(engine_type, name):
            if name == "test_llm":
                return mock_llm
            return None
        
        mock_instance.get = mock_get
        mock_instance.engines = {
            EngineType.LLM: {"test_llm": mock_llm},
            # Add other engine types as needed
        }
        
        config = AgentConfig(engine="test_llm")
        
        # Call the actual method 
        resolved = config.resolve_engine()
        
        # Assert the correct interactions
        mock_registry.get_instance.assert_called_once()
        assert resolved == mock_llm
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        llm = AugLLMConfig(name="test_llm")
        config = AgentConfig(
            name="test_agent",
            engine=llm,
            debug=True
        )
        
        data = config.to_dict()
        
        assert data["name"] == "test_agent"
        assert data["engine_type"] == EngineType.AGENT
        assert data["debug"] is True
        assert "engine" in data
        
    def test_apply_runnable_config(self):
        """Test extracting parameters from runnable_config."""
        config = AgentConfig(name="test_agent")
        
        runnable_config = {
            "configurable": {
                "thread_id": "test_thread",
                "debug": True,
                "engine_configs": {
                    "test_agent": {"temperature": 0.7}
                }
            }
        }
        
        params = config.apply_runnable_config(runnable_config)
        
        assert params["thread_id"] == "test_thread"
        assert params["debug"] is True
        assert params["temperature"] == 0.7