# tests/core/engine/agent/test_agent.py

import pytest
from unittest.mock import patch, MagicMock
import os
import json
from typing import Dict, Any

from haive_core.engine.agent.agent import Agent
from haive_core.engine.agent.config import AgentConfig 
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.schema.state_schema import StateSchema
from haive_core.graph.dynamic_graph_builder import ComponentRef

# Create a concrete implementation for testing
class TestAgentImplementation(Agent[AgentConfig]):
    """Concrete Agent implementation for testing."""
    
    def setup_workflow(self):
        """Set up a simple test workflow."""
        from langgraph.graph import START, END
        
        # Create a simple callable function node that doesn't rely on LLM
        def test_node_function(state):
            return {"output": "test response"}
        
        # Add the function node to the graph
        self.graph.add_node(
            name="test_node",
            config=test_node_function,
            command_goto=END
        )
        
        self.graph.add_edge(START, "test_node")
    
    def _create_graph_builder(self):
        """Override graph builder to handle component references."""
        from haive_core.graph.dynamic_graph_builder import DynamicGraph
        
        # Create graph with empty components list to avoid validation issues
        self.graph = DynamicGraph(
            name=self.config.name,
            description=getattr(self.config, 'description', None),
            components=[],  # Empty components list to avoid validation issues
            state_schema=self.state_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            default_runnable_config=self.runnable_config,
            visualize=self.config.visualize
        )

class TestAgent:
    """Tests for the Agent base class."""
    
    @pytest.fixture
    def config(self, tmp_path):
        """Create a test agent config."""
        output_dir = str(tmp_path / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create an AugLLMConfig for testing
        llm_config = AugLLMConfig(
            name="test_llm",
            system_message="You are a helpful assistant."
        )
        
        return AgentConfig(
            name="test_agent",
            engine=llm_config,
            output_dir=output_dir
        )
    
    # Rest of the test methods remain the same...
    
    @pytest.fixture
    def mock_checkpointer(self):
        """Create a mock checkpointer."""
        return MagicMock()
    
    @patch("src.haive.core.engine.agent.agent.setup_checkpointer")
    def test_init(self, mock_setup, config):
        """Test initialization of agent."""
        # Setup mock
        mock_checkpointer = MagicMock()
        mock_setup.return_value = mock_checkpointer
        
        # Create the agent
        agent = TestAgentImplementation(config)
        
        # Verify initialization
        assert agent.config == config
        assert agent.checkpointer == mock_checkpointer
        assert agent.engines["main"] == agent.engine
        assert hasattr(agent, "state_schema")
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")
        assert hasattr(agent, "graph")
        assert hasattr(agent, "app")
    
    @patch("src.haive.core.engine.agent.agent.setup_checkpointer")
    @patch("src.haive.core.engine.agent.agent.ensure_pool_open")
    @patch("src.haive.core.engine.agent.agent.register_thread_if_needed")
    def test_run(self, mock_register, mock_ensure_pool, mock_setup, config):
        """Test running the agent."""
        # Setup mocks
        mock_checkpointer = MagicMock()
        mock_setup.return_value = mock_checkpointer
        mock_pool = MagicMock()
        mock_ensure_pool.return_value = mock_pool
        
        # Create the agent
        agent = TestAgentImplementation(config)
        
        # Mock the app and its methods
        agent.app = MagicMock()
        agent.app.get_state.return_value = None
        agent.app.invoke.return_value = {"output": "test response"}
        
        # Run the agent
        result = agent.run("test input", thread_id="test_thread")
        
        # Verify the execution
        mock_ensure_pool.assert_called_once_with(mock_checkpointer)
        mock_register.assert_called_once()
        agent.app.invoke.assert_called_once()
        assert result == {"output": "test response"}
    
    @patch("src.haive.core.engine.agent.agent.setup_checkpointer")
    def test_save_state_history(self, mock_setup, config, tmp_path):
        """Test saving state history."""
        # Setup mock
        mock_checkpointer = MagicMock()
        mock_setup.return_value = mock_checkpointer
        
        # Create the agent
        agent = TestAgentImplementation(config)
        
        # Mock the app and its methods
        agent.app = MagicMock()
        mock_state = {"messages": [{"role": "user", "content": "test"}]}
        agent.app.get_state.return_value = mock_state
        
        # Create a real state filename
        state_dir = tmp_path / "state_history"
        os.makedirs(state_dir, exist_ok=True)
        state_file = state_dir / "test_state.json"
        agent.state_filename = str(state_file)
        
        # Save state history
        agent.save_state_history()
        
        # Verify file was written
        assert os.path.exists(state_file)
        with open(state_file, 'r') as f:
            saved_state = json.load(f)
            assert saved_state == mock_state
    
    @patch("src.haive.core.engine.agent.agent.setup_checkpointer")
    def test_prepare_runnable_config(self, mock_setup, config):
        """Test preparing runnable config."""
        # Setup mock
        mock_checkpointer = MagicMock()
        mock_setup.return_value = mock_checkpointer
        
        # Create the agent
        agent = TestAgentImplementation(config)
        
        # Test with thread_id and debug parameter
        runnable_config = agent._prepare_runnable_config(thread_id="test_thread", debug=True)
        
        # Debug should be in configurable section
        assert runnable_config["configurable"]["thread_id"] == "test_thread"
        assert runnable_config["configurable"]["debug"] is True
        
        # Test with config override and engine parameter (top_p)
        base_config = {"configurable": {"temperature": 0.7}}
        runnable_config = agent._prepare_runnable_config(config=base_config, top_p=0.9)
        
        # Temperature should be in configurable section (from base_config)
        assert runnable_config["configurable"]["temperature"] == 0.7
        
        # Engine parameter top_p should be in engine_configs section
        assert "engine_configs" in runnable_config["configurable"]
        assert "llm_config" in runnable_config["configurable"]["engine_configs"]
        assert runnable_config["configurable"]["engine_configs"]["llm_config"]["top_p"] == 0.9
        
        # Also should be in configurable section for backward compatibility
        assert runnable_config["configurable"]["top_p"] == 0.9
    
    @patch("src.haive.core.engine.agent.agent.setup_checkpointer")
    @patch("src.haive.core.engine.agent.agent.ensure_pool_open")
    @patch("src.haive.core.engine.agent.agent.close_pool_if_needed")
    def test_stream(self, mock_close, mock_ensure_pool, mock_setup, config):
        """Test streaming from the agent."""
        # Setup mocks
        mock_checkpointer = MagicMock()
        mock_setup.return_value = mock_checkpointer
        mock_pool = MagicMock()
        mock_ensure_pool.return_value = mock_pool
        
        # Create the agent
        agent = TestAgentImplementation(config)
        
        # Mock the app and its methods
        agent.app = MagicMock()
        agent.app.get_state.return_value = None
        agent.app.stream.return_value = [
            {"output": "partial 1"},
            {"output": "partial 2"},
            {"output": "final"}
        ]
        
        # Stream from the agent
        results = list(agent.stream("test input", thread_id="test_thread"))
        
        # Verify the execution
        mock_ensure_pool.assert_called_once_with(mock_checkpointer)
        agent.app.stream.assert_called_once()
        mock_close.assert_called_once_with(mock_checkpointer, mock_pool)
        assert len(results) == 3
        assert results[2]["output"] == "final"