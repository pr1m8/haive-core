"""Test the refactored MetaStateSchema with graph/recompilation focus.

This test validates that the refactored MetaStateSchema:
1. Focuses on graph composition rather than tool routing
2. Properly integrates RecompileMixin functionality
3. Manages agent lifecycle and execution
4. Tracks recompilation needs
"""

from datetime import datetime
from unittest.mock import Mock

import pytest
from haive.agents.simple.agent import SimpleAgent
from langchain_core.messages import AIMessage, HumanMessage

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.schema.prebuilt.meta_state import MetaStateSchema


class TestMetaStateSchemaRefactored:
    """Test the refactored MetaStateSchema with graph/recompilation focus."""

    def test_meta_state_creation_without_agent(self):
        """Test creating MetaStateSchema without an agent."""
        meta_state = MetaStateSchema()

        assert meta_state.agent is None
        assert meta_state.agent_name is None
        assert meta_state.agent_type is None
        assert meta_state.execution_status == "ready"
        assert meta_state.graph_context == {}
        assert meta_state.composition_metadata == {}
        assert meta_state.needs_recompile is False

    def test_meta_state_creation_with_agent(self):
        """Test creating MetaStateSchema with an agent."""
        # Create a simple agent
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(agent=agent)

        assert meta_state.agent is agent
        assert meta_state.agent_name == "test_agent"
        assert meta_state.agent_type == "SimpleAgent"
        assert meta_state.execution_status == "ready"
        assert "created_at" in meta_state.graph_context
        assert "composition_type" in meta_state.graph_context
        assert meta_state.composition_metadata["agent_name"] == "test_agent"
        assert meta_state.composition_metadata["agent_type"] == "SimpleAgent"

    def test_recompile_mixin_integration(self):
        """Test that RecompileMixin functionality is properly integrated."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Test marking for recompilation
        meta_state.mark_for_recompile("Test reason")

        assert meta_state.needs_recompile is True
        assert "Test reason" in meta_state.recompile_reasons
        assert len(meta_state.recompile_history) == 1

        # Test recompilation status
        status = meta_state.get_recompile_status()
        assert status["needs_recompile"] is True
        assert status["reason_count"] == 1
        assert "Test reason" in status["pending_reasons"]

    def test_update_agent_triggers_recompilation(self):
        """Test that updating agent triggers recompilation."""
        # Create first agent
        engine1 = AugLLMConfig(name="engine1")
        agent1 = SimpleAgent(name="agent1", engine=engine1)
        meta_state = MetaStateSchema(agent=agent1)

        # Create second agent
        engine2 = AugLLMConfig(name="engine2")
        agent2 = SimpleAgent(name="agent2", engine=engine2)

        # Update agent should trigger recompilation
        meta_state.update_agent(agent2)

        assert meta_state.agent is agent2
        assert meta_state.agent_name == "agent2"
        assert meta_state.agent_type == "SimpleAgent"
        assert meta_state.needs_recompile is True
        assert "Agent changed from agent1 to agent2" in meta_state.recompile_reasons

    def test_graph_focused_execution_preparation(self):
        """Test that input preparation focuses on graph context."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(
            agent=agent,
            messages=[HumanMessage(content="Test message")],
            graph_context={"test_key": "test_value"},
            agent_state={"state_key": "state_value"},
        )

        # Test input preparation
        input_data = meta_state.prepare_agent_input(additional_input={"extra": "data"})

        assert "messages" in input_data
        assert "graph_context" in input_data
        assert "agent_state" in input_data
        assert "extra" in input_data
        assert input_data["graph_context"]["test_key"] == "test_value"
        assert input_data["agent_state"]["state_key"] == "state_value"

    def test_execution_summary_focuses_on_graph_status(self):
        """Test that execution summary focuses on graph and recompilation status."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Add some graph context
        meta_state.graph_context["execution_count"] = 3
        meta_state.graph_context["last_execution"] = "2024-01-01T12:00:00"
        meta_state.mark_for_recompile("Test recompilation")

        summary = meta_state.get_execution_summary()

        assert summary["agent_name"] == "test_agent"
        assert summary["agent_type"] == "SimpleAgent"
        assert summary["execution_count"] == 3
        assert summary["last_execution"] == "2024-01-01T12:00:00"
        assert summary["needs_recompilation"] is True
        assert "graph_context" in summary
        assert "composition_metadata" in summary
        assert "recompilation_status" in summary

    def test_from_agent_classmethod_with_graph_context(self):
        """Test creating MetaStateSchema from agent with graph context."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        initial_state = {"initialized": True}
        graph_context = {"workflow": "test_workflow"}

        meta_state = MetaStateSchema.from_agent(
            agent=agent, initial_state=initial_state, graph_context=graph_context
        )

        assert meta_state.agent is agent
        assert meta_state.agent_state == initial_state
        assert meta_state.graph_context["workflow"] == "test_workflow"

    def test_clone_with_agent_preserves_graph_focus(self):
        """Test that cloning with new agent preserves graph focus."""
        # Create original agent and meta state
        engine1 = AugLLMConfig(name="engine1")
        agent1 = SimpleAgent(name="agent1", engine=engine1)
        meta_state = MetaStateSchema(agent=agent1)

        # Create new agent
        engine2 = AugLLMConfig(name="engine2")
        agent2 = SimpleAgent(name="agent2", engine=engine2)

        # Clone with new agent
        cloned_state = meta_state.clone_with_agent(agent2)

        assert cloned_state.agent is agent2
        assert cloned_state.agent_name == "agent2"
        assert cloned_state.execution_status == "ready"
        assert cloned_state.execution_result is None
        assert cloned_state.agent_state == {}
        assert cloned_state.graph_context["composition_type"] == "cloned"

    def test_string_representation_includes_recompilation_status(self):
        """Test that string representation includes recompilation status."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Mark for recompilation
        meta_state.mark_for_recompile("Test reason")

        str_repr = str(meta_state)
        assert "test_agent" in str_repr
        assert "SimpleAgent" in str_repr
        assert "needs_recompile=True" in str_repr

        repr_str = repr(meta_state)
        assert "test_agent" in repr_str
        assert "SimpleAgent" in repr_str
        assert "needs_recompile=True" in repr_str

    def test_reset_execution_state_clears_recompilation(self):
        """Test that reset_execution_state clears recompilation state."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Mark for recompilation
        meta_state.mark_for_recompile("Test reason")
        assert meta_state.needs_recompile is True

        # Reset execution state
        meta_state.reset_execution_state()

        assert meta_state.execution_status == "ready"
        assert meta_state.execution_result is None
        assert meta_state.agent_state == {}
        assert meta_state.needs_recompile is False  # Should be cleared

    def test_check_agent_recompilation_detects_agent_needs(self):
        """Test that check_agent_recompilation detects when agent needs recompilation."""
        # Create mock agent that needs recompilation
        mock_agent = Mock()
        mock_agent.needs_recompile = True

        meta_state = MetaStateSchema(agent=mock_agent)

        assert meta_state.check_agent_recompilation() is True

        # Test with agent that doesn't need recompilation
        mock_agent.needs_recompile = False
        assert meta_state.check_agent_recompilation() is False

    def test_get_agent_engine_simplified(self):
        """Test that get_agent_engine is simplified for graph composition."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Test getting main engine
        retrieved_engine = meta_state.get_agent_engine("main")
        assert retrieved_engine is agent.engine

        # Test getting non-existent engine
        assert meta_state.get_agent_engine("nonexistent") is None
