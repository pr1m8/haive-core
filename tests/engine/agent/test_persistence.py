# tests/test_agent_persistence.py

import pytest
import os
import uuid
import json
import tempfile

from haive_core.engine.agent.config import AgentConfig
from haive_core.engine.agent.agent import SimpleAgent, SimpleAgentConfig
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.engine.agent.persistence.types import CheckpointerType
from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.handlers import setup_checkpointer
from langgraph.checkpoint.memory import MemorySaver

# Test default persistence configuration
def test_default_persistence_config():
    config = AgentConfig()
    assert config.persistence is not None
    assert config.persistence.type == CheckpointerType.postgres
    
    # Setup checkpointer (should fall back to memory in test)
    checkpointer = setup_checkpointer(config)
    assert checkpointer is not None

# Test explicit memory persistence
def test_memory_persistence():
    # Create config with memory persistence
    config = SimpleAgentConfig(
        name="memory_agent",
        persistence=CheckpointerConfig(
            type=CheckpointerType.memory
        )
    )
    
    # Setup checkpointer
    checkpointer = setup_checkpointer(config)
    assert isinstance(checkpointer, MemorySaver)

# Test state persistence between runs (using memory checkpointer)
def test_state_persistence_between_runs():
    # Create a unique thread ID for this test
    thread_id = f"test-thread-{uuid.uuid4()}"
    
    # Create config with memory persistence
    config = SimpleAgentConfig(
        name="persistence_test_agent",
        persistence=CheckpointerConfig(
            type=CheckpointerType.memory
        ),
        engine=AugLLMConfig(name="test_llm")
    )
    
    # Mock the engine's invoke method
    def mock_invoke(self, input_data, runnable_config=None):
        messages = input_data.get("messages", [])
        count = input_data.get("count", 0)
        return {
            "messages": messages,
            "count": count + 1,
            "last_input": str(input_data)
        }
    
    # Apply the mock
    import types
    config.engine.invoke = types.MethodType(mock_invoke, config.engine)
    
    # Build agent
    agent = config.build_agent()
    
    # First run with initial state
    result1 = agent.run({"messages": []}, thread_id=thread_id)
    assert result1["count"] == 1
    
    # Second run with same thread should preserve state
    result2 = agent.run("New input", thread_id=thread_id)
    assert result2["count"] == 2
    assert "New input" in result2["last_input"]

# Test saving state history
def test_save_state_history():
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config with the temp dir
        config = SimpleAgentConfig(
            name="history_test_agent",
            output_dir=temp_dir,
            save_history=True
        )
        
        # Build agent
        agent = config.build_agent()
        
        # Check that state_filename is in the temp directory
        assert temp_dir in agent.state_filename
        
        # Run agent
        agent.run("Test input")
        
        # Check if state history file was created
        assert os.path.exists(agent.state_filename)
        
        # Verify content
        with open(agent.state_filename, "r") as f:
            state_data = json.load(f)
            assert isinstance(state_data, dict)