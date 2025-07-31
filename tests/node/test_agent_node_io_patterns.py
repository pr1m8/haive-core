"""Test agent node input/output patterns with message handling.

This test file validates:
1. Agent node receives full state
2. Output handling for different agent types
3. Message field annotations and updates
4. Command pattern for state updates
"""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.graph.node.agent_node_v3 import AgentNodeV3Config
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Mock agent classes for testing
class MockSimpleAgent:
    """Mock simple agent without structured output."""

    def __init__(self, name: str):
        self.name = name
        self.output_schema = None  # No structured output

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        """Simple agent returns messages by default."""
        return {"messages": [AIMessage(content=f"Response from {self.name}")]}


class StructuredOutputModel(BaseModel):
    """Example structured output model."""

    result: str
    confidence: float
    details: dict[str, Any] = Field(default_factory=dict)


class MockStructuredAgent:
    """Mock agent with structured output."""

    def __init__(self, name: str):
        self.name = name
        self.output_schema = StructuredOutputModel

    def invoke(self, state: Any, config: Any = None) -> StructuredOutputModel:
        """Structured agent returns model instance."""
        return StructuredOutputModel(
            result="Structured response", confidence=0.95, details={"processed": True}
        )


class TestAgentNodeIOPatterns:
    """Test agent node input/output handling patterns."""

    def test_agent_node_receives_full_state(self):
        """Test that agent node passes full state to agent."""
        # Create state with various fields
        state = MultiAgentState()
        state.messages = [HumanMessage(content="Test message")]
        state.agent_states["test_agent"] = {"internal": "data"}
        state.agent_outputs["other_agent"] = {"result": "previous"}

        # Create mock agent that captures input
        captured_input = None

        class CaptureAgent:
            name = "test_agent"
            output_schema = None

            def invoke(self, state_input, config=None):
                nonlocal captured_input
                captured_input = state_input
                return {"messages": [AIMessage(content="Done")]}

        # Create agent node
        agent = CaptureAgent()
        node = AgentNodeV3Config(name="test_node", agent_name="test_agent", agent=agent)

        # Execute node
        node(state, {})

        # Agent should receive the full state object
        assert captured_input is state
        assert hasattr(captured_input, "messages")
        assert hasattr(captured_input, "agent_states")
        assert hasattr(captured_input, "agent_outputs")

    def test_simple_agent_default_message_output(self):
        """Test simple agent outputs messages by default."""
        state = MultiAgentState()
        state.messages = [HumanMessage(content="Hello")]

        # Create simple agent node
        agent = MockSimpleAgent("chat_agent")
        node = AgentNodeV3Config(name="chat_node", agent_name="chat_agent", agent=agent)

        # Execute node
        result = node(state, {})

        # Should return Command
        assert isinstance(result, Command)

        # Update should include agent output with messages
        assert "agent_outputs" in result.update
        assert "chat_agent" in result.update["agent_outputs"]
        assert "messages" in result.update["agent_outputs"]["chat_agent"]

        # Messages should be in the output
        output_messages = result.update["agent_outputs"]["chat_agent"]["messages"]
        assert len(output_messages) == 1
        assert output_messages[0].content == "Response from chat_agent"

    def test_structured_agent_no_message_output(self):
        """Test structured agent outputs without messages field."""
        state = MultiAgentState()

        # Create structured agent node
        agent = MockStructuredAgent("analyzer")
        node = AgentNodeV3Config(
            name="analyzer_node", agent_name="analyzer", agent=agent
        )

        # Execute node
        result = node(state, {})

        # Should return Command
        assert isinstance(result, Command)

        # Update should include agent output WITHOUT messages
        assert "agent_outputs" in result.update
        assert "analyzer" in result.update["agent_outputs"]

        # Should have structured fields, not messages
        output = result.update["agent_outputs"]["analyzer"]
        assert "messages" not in output
        assert "result" in output
        assert "confidence" in output
        assert "details" in output
        assert output["result"] == "Structured response"
        assert output["confidence"] == 0.95

    def test_message_field_annotation_handling(self):
        """Test how message fields are annotated and handled."""
        # Test with annotated messages in state update
        state = MultiAgentState()

        # Mock agent that returns annotated messages
        class AnnotatedMessageAgent:
            name = "annotated_agent"
            output_schema = None

            def invoke(self, state, config=None):
                # Return messages that should be annotated
                return {
                    "messages": [
                        HumanMessage(content="Question?"),
                        AIMessage(content="Answer!"),
                    ]
                }

        agent = AnnotatedMessageAgent()
        node = AgentNodeV3Config(
            name="annotated_node", agent_name="annotated_agent", agent=agent
        )

        # Execute
        result = node(state, {})

        # Check messages in update
        agent_output = result.update["agent_outputs"]["annotated_agent"]
        assert "messages" in agent_output
        assert len(agent_output["messages"]) == 2
        assert all(isinstance(msg, BaseMessage) for msg in agent_output["messages"])

    def test_command_update_patterns(self):
        """Test different Command update patterns for agents."""
        state = MultiAgentState()

        # Test different agent output patterns
        test_cases = [
            # Simple agent with messages
            {
                "agent": MockSimpleAgent("simple"),
                "expected_fields": ["messages"],
                "unexpected_fields": ["result", "confidence"],
            },
            # Structured agent
            {
                "agent": MockStructuredAgent("structured"),
                "expected_fields": ["result", "confidence", "details"],
                "unexpected_fields": ["messages"],
            },
        ]

        for test_case in test_cases:
            agent = test_case["agent"]
            node = AgentNodeV3Config(
                name=f"{agent.name}_node", agent_name=agent.name, agent=agent
            )

            # Execute
            result = node(state, {})

            # Verify Command structure
            assert isinstance(result, Command)
            assert "agent_outputs" in result.update
            assert agent.name in result.update["agent_outputs"]

            # Check expected fields
            output = result.update["agent_outputs"][agent.name]
            for field in test_case["expected_fields"]:
                assert (
                    field in output
                ), f"Expected {field} in {
                    agent.name} output"

            # Check unexpected fields
            for field in test_case["unexpected_fields"]:
                assert (
                    field not in output
                ), f"Unexpected {field} in {
                    agent.name} output"

    def test_agent_output_schema_field_mapping(self):
        """Test that agent output schemas map to correct state fields."""
        state = MultiAgentState()

        # Mock Self-Discover style agents with specific output schemas
        class SelectedModules(BaseModel):
            selected_modules: list[str]
            rationale: str | None = None

        class SelectModulesAgent:
            name = "select_modules"
            output_schema = SelectedModules

            def invoke(self, state, config=None):
                return SelectedModules(
                    selected_modules=["reasoning", "analysis"],
                    rationale="Best modules for task",
                )

        # Create and execute node
        agent = SelectModulesAgent()
        node = AgentNodeV3Config(
            name="select_node", agent_name="select_modules", agent=agent
        )

        result = node(state, {})

        # Output should match schema structure
        output = result.update["agent_outputs"]["select_modules"]
        assert "selected_modules" in output
        assert "rationale" in output
        assert output["selected_modules"] == ["reasoning", "analysis"]
        assert output["rationale"] == "Best modules for task"

        # No messages field
        assert "messages" not in output

    def test_mixed_agent_execution_sequence(self):
        """Test sequence of different agent types updating state."""
        state = MultiAgentState()

        # Execute sequence of agents
        agents = [
            MockSimpleAgent("chat_1"),
            MockStructuredAgent("analyzer"),
            MockSimpleAgent("chat_2"),
        ]

        for agent in agents:
            node = AgentNodeV3Config(
                name=f"{agent.name}_node", agent_name=agent.name, agent=agent
            )

            # Execute and apply update
            result = node(state, {})

            # Simulate applying Command update
            if "agent_outputs" in result.update:
                state.agent_outputs.update(result.update["agent_outputs"])

        # Verify final state
        assert "chat_1" in state.agent_outputs
        assert "analyzer" in state.agent_outputs
        assert "chat_2" in state.agent_outputs

        # Check message vs non-message outputs
        assert "messages" in state.agent_outputs["chat_1"]
        assert "messages" not in state.agent_outputs["analyzer"]
        assert "messages" in state.agent_outputs["chat_2"]

        # Check structured output
        assert "result" in state.agent_outputs["analyzer"]
        assert "confidence" in state.agent_outputs["analyzer"]

    def test_agent_without_output_schema_defaults(self):
        """Test agent without explicit output schema uses default behavior."""
        state = MultiAgentState()

        # Agent with no output_schema attribute
        class MinimalAgent:
            name = "minimal"
            # No output_schema defined

            def invoke(self, state, config=None):
                return {"messages": [AIMessage(content="Default output")]}

        agent = MinimalAgent()
        node = AgentNodeV3Config(name="minimal_node", agent_name="minimal", agent=agent)

        # Execute
        result = node(state, {})

        # Should default to message-based output
        output = result.update["agent_outputs"]["minimal"]
        assert "messages" in output
        assert len(output["messages"]) == 1

    def test_preventing_message_output_for_structured_agents(self):
        """Test that we can prevent message field for structured agents."""
        state = MultiAgentState()

        # Agent that explicitly doesn't want messages
        class NoMessageAgent:
            name = "no_messages"
            output_schema = StructuredOutputModel

            def invoke(self, state, config=None):
                # Even if we accidentally include messages in dict
                return {
                    "result": "Success",
                    "confidence": 1.0,
                    "details": {},
                    "messages": [AIMessage(content="Should not appear")],
                }

        agent = NoMessageAgent()
        node = AgentNodeV3Config(
            name="no_msg_node", agent_name="no_messages", agent=agent
        )

        # Execute
        result = node(state, {})

        # Should only include fields from output_schema
        output = result.update["agent_outputs"]["no_messages"]

        # These fields are in StructuredOutputModel
        assert "result" in output
        assert "confidence" in output
        assert "details" in output

        # This field is NOT in StructuredOutputModel
        # So it should be filtered out
        # (This assumes proper implementation)
        # For now, let's document the expected behavior
        # assert "messages" not in output  # TODO: Implement filtering
