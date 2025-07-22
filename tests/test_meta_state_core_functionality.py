"""Test the core MetaStateSchema functionality for graph composition and recompilation.

This test validates the core meta-agent functionality:
1. Agent embedding and execution
2. Graph composition and context management
3. Recompilation tracking and management
4. Agent state projection and management
5. Agent lifecycle management

NO MESSAGES - pure graph composition focus.
"""

from unittest.mock import Mock

from haive.agents.simple.agent import SimpleAgent

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.schema.prebuilt.meta_state import MetaStateSchema


class TestMetaStateCoreFunctionality:
    """Test MetaStateSchema core functionality for graph composition."""

    def test_pure_graph_composition_without_messages(self):
        """Test MetaStateSchema focuses on graph composition, not messages."""
        # Create meta state - no messages field
        meta_state = MetaStateSchema()

        # Should not have messages field
        assert not hasattr(meta_state, "messages")

        # Should have graph composition fields
        assert hasattr(meta_state, "graph_context")
        assert hasattr(meta_state, "agent_state")
        assert hasattr(meta_state, "composition_metadata")
        assert hasattr(meta_state, "execution_result")

        # Should have recompilation tracking
        assert hasattr(meta_state, "needs_recompile")
        assert hasattr(meta_state, "recompile_reasons")

    def test_agent_embedding_and_lifecycle(self):
        """Test core agent embedding and lifecycle management."""
        # Create agent
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        # Embed agent in meta state
        meta_state = MetaStateSchema(
            agent=agent,
            agent_state={"task": "test_task", "progress": 0.5},
            graph_context={"workflow": "test_workflow"},
        )

        # Verify agent embedding
        assert meta_state.agent is agent
        assert meta_state.agent_name == "test_agent"
        assert meta_state.agent_type == "SimpleAgent"

        # Verify state management
        assert meta_state.agent_state["task"] == "test_task"
        assert meta_state.agent_state["progress"] == 0.5
        assert meta_state.graph_context["workflow"] == "test_workflow"

        # Verify lifecycle metadata
        assert meta_state.composition_metadata["agent_name"] == "test_agent"
        assert meta_state.composition_metadata["agent_type"] == "SimpleAgent"

    def test_agent_state_projection(self):
        """Test how meta state projects state to the contained agent."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(
            agent=agent,
            agent_state={"input": "test_input", "context": "test_context"},
            graph_context={"workflow": "test_workflow", "step": 1},
        )

        # Test input preparation for agent execution
        prepared_input = meta_state.prepare_agent_input(
            additional_input={"extra": "data"}
        )

        # Should include agent state
        assert prepared_input["input"] == "test_input"
        assert prepared_input["context"] == "test_context"

        # Should include graph context
        assert prepared_input["graph_context"]["workflow"] == "test_workflow"
        assert prepared_input["graph_context"]["step"] == 1

        # Should include additional input
        assert prepared_input["extra"] == "data"

    def test_recompilation_tracking_integration(self):
        """Test that recompilation tracking works correctly."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)
        meta_state = MetaStateSchema(agent=agent)

        # Initial state - no recompilation needed
        assert meta_state.needs_recompile is False
        assert len(meta_state.recompile_reasons) == 0

        # Mark for recompilation
        meta_state.mark_for_recompile("Graph structure changed")

        # Should be marked for recompilation
        assert meta_state.needs_recompile is True
        assert "Graph structure changed" in meta_state.recompile_reasons

        # Should have recompilation history
        assert len(meta_state.recompile_history) == 1
        assert meta_state.recompile_history[0]["reason"] == "Graph structure changed"
        assert meta_state.recompile_history[0]["action"] == "marked_for_recompile"

    def test_agent_execution_with_graph_context(self):
        """Test agent execution with graph context."""
        # Create mock agent that can be executed
        mock_agent = Mock()
        mock_agent.name = "mock_agent"
        mock_agent.__class__.__name__ = "MockAgent"
        mock_agent.run = Mock(
            return_value={"output": "test_output", "state": {"updated": True}}
        )

        meta_state = MetaStateSchema(
            agent=mock_agent,
            agent_state={"input": "test_input"},
            graph_context={"workflow": "test_workflow"},
        )

        # Execute the agent
        result = meta_state.execute_agent()

        # Verify execution
        assert result["status"] == "success"
        assert result["output"]["output"] == "test_output"

        # Verify state was updated
        assert meta_state.agent_state["updated"] is True
        assert meta_state.execution_status == "completed"
        assert meta_state.execution_result is not None

        # Verify graph context was updated
        assert meta_state.graph_context["execution_count"] == 1
        assert "last_execution" in meta_state.graph_context

    def test_agent_update_triggers_recompilation(self):
        """Test that updating the agent triggers recompilation."""
        # Create first agent
        engine1 = AugLLMConfig(name="engine1")
        agent1 = SimpleAgent(name="agent1", engine=engine1)

        # Create second agent
        engine2 = AugLLMConfig(name="engine2")
        agent2 = SimpleAgent(name="agent2", engine=engine2)

        meta_state = MetaStateSchema(agent=agent1)

        # Initial state - no recompilation needed
        assert meta_state.needs_recompile is False

        # Update agent
        meta_state.update_agent(agent2)

        # Should trigger recompilation
        assert meta_state.needs_recompile is True
        assert "Agent changed from agent1 to agent2" in meta_state.recompile_reasons

        # Should update agent metadata
        assert meta_state.agent is agent2
        assert meta_state.agent_name == "agent2"
        assert meta_state.composition_metadata["agent_name"] == "agent2"

    def test_execution_summary_graph_focused(self):
        """Test that execution summary focuses on graph composition."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(agent=agent)
        meta_state.graph_context["execution_count"] = 5
        meta_state.graph_context["last_execution"] = "2024-01-01T12:00:00"
        meta_state.mark_for_recompile("Test reason")

        summary = meta_state.get_execution_summary()

        # Should focus on graph composition, not messages
        assert "agent_name" in summary
        assert "agent_type" in summary
        assert "execution_count" in summary
        assert "graph_context" in summary
        assert "composition_metadata" in summary
        assert "recompilation_status" in summary
        assert "needs_recompilation" in summary

        # Should not have message-related fields
        assert "messages" not in summary
        assert "message_count" not in summary

        # Should have correct values
        assert summary["execution_count"] == 5
        # The MetaStateSchema itself was marked for recompilation
        assert summary["recompilation_status"]["needs_recompile"] is True
        assert "Test reason" in summary["recompilation_status"]["pending_reasons"]

    def test_agent_recompilation_detection(self):
        """Test detection of agent recompilation needs."""
        # Create mock agent with recompilation needs
        mock_agent = Mock()
        mock_agent.name = "mock_agent"
        mock_agent.__class__.__name__ = "MockAgent"
        mock_agent.needs_recompile = True

        meta_state = MetaStateSchema(agent=mock_agent)

        # Should detect agent needs recompilation
        assert meta_state.check_agent_recompilation() is True

        # Change agent state
        mock_agent.needs_recompile = False
        assert meta_state.check_agent_recompilation() is False

    def test_cloning_preserves_graph_composition(self):
        """Test that cloning preserves graph composition focus."""
        # Create original setup
        engine1 = AugLLMConfig(name="engine1")
        agent1 = SimpleAgent(name="agent1", engine=engine1)

        original_state = MetaStateSchema(
            agent=agent1,
            agent_state={"original": "state"},
            graph_context={"workflow": "original"},
        )

        # Create new agent
        engine2 = AugLLMConfig(name="engine2")
        agent2 = SimpleAgent(name="agent2", engine=engine2)

        # Clone with new agent
        cloned_state = original_state.clone_with_agent(agent2)

        # Should have new agent
        assert cloned_state.agent is agent2
        assert cloned_state.agent_name == "agent2"

        # Should reset graph-focused fields
        assert cloned_state.execution_result is None
        assert cloned_state.execution_status == "ready"
        assert cloned_state.agent_state == {}
        assert cloned_state.graph_context["composition_type"] == "cloned"

    def test_from_agent_factory_method(self):
        """Test creating MetaStateSchema from agent with graph context."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        initial_state = {"initialized": True, "task": "test"}
        graph_context = {"workflow": "test_workflow", "step": 1}

        meta_state = MetaStateSchema.from_agent(
            agent=agent, initial_state=initial_state, graph_context=graph_context
        )

        # Should have correct agent
        assert meta_state.agent is agent
        assert meta_state.agent_name == "test_agent"

        # Should have correct state
        assert meta_state.agent_state == initial_state
        assert meta_state.graph_context == graph_context

    def test_reset_execution_state_clears_graph_state(self):
        """Test that reset clears graph execution state."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(agent=agent)
        meta_state.execution_status = "error"
        meta_state.execution_result = {"error": "test_error"}
        meta_state.agent_state = {"corrupted": "state"}
        meta_state.mark_for_recompile("Test reason")

        # Reset execution state
        meta_state.reset_execution_state()

        # Should clear graph execution state
        assert meta_state.execution_status == "ready"
        assert meta_state.execution_result is None
        assert meta_state.agent_state == {}
        assert meta_state.needs_recompile is False

    def test_shared_fields_are_graph_focused(self):
        """Test that shared fields are focused on graph composition."""
        meta_state = MetaStateSchema()

        # Should have graph-focused shared fields
        expected_shared = ["execution_result", "execution_status", "graph_context"]
        assert meta_state.__shared_fields__ == expected_shared

        # Should not have message-related shared fields
        assert "messages" not in meta_state.__shared_fields__

        # Should have graph-focused reducers
        expected_reducers = [
            "execution_result",
            "graph_context",
            "composition_metadata",
        ]
        assert all(
            field in meta_state.__reducer_fields__ for field in expected_reducers
        )
        assert "messages" not in meta_state.__reducer_fields__

    def test_string_representation_shows_graph_state(self):
        """Test that string representation shows graph state."""
        engine = AugLLMConfig(name="test_engine")
        agent = SimpleAgent(name="test_agent", engine=engine)

        meta_state = MetaStateSchema(agent=agent)
        meta_state.graph_context["execution_count"] = 3
        meta_state.agent_state = {"task": "test", "progress": 0.5}
        meta_state.mark_for_recompile("Test reason")

        str_repr = str(meta_state)
        repr_str = repr(meta_state)

        # Should show graph state, not messages
        assert "test_agent" in str_repr
        assert "SimpleAgent" in str_repr
        assert "needs_recompile=True" in str_repr

        assert "test_agent" in repr_str
        assert "SimpleAgent" in repr_str
        assert "needs_recompile=True" in repr_str
        assert "agent_state_keys=['task', 'progress']" in repr_str

        # Should not mention messages
        assert "messages" not in str_repr
        assert "messages" not in repr_str
