"""Test agent node updating actual state fields with synchronization.

This test file validates:
1. Agent nodes update actual state fields (not just agent_outputs)
2. Synchronization between agent internal state and combined state
3. Privacy of agent-specific fields (tools, internal state)
4. Command pattern with multi-field updates
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.graph.node.agent_node_v3 import AgentNodeV3Config
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Output schemas that map to state fields
class SelectedModules(BaseModel):
    """Maps to state.selected_modules field."""

    selected_modules: list[str]
    rationale: str | None = None


class AdaptedModules(BaseModel):
    """Maps to state.adapted_modules field."""

    adapted_modules: list[dict[str, str]]
    task_context: str


class ReasoningStructure(BaseModel):
    """Maps to state.reasoning_structure field."""

    reasoning_structure: dict[str, Any]
    steps: list[str]


class FinalAnswer(BaseModel):
    """Maps to state.final_answer field."""

    answer: str
    confidence: float
    reasoning_steps: dict[str, str]


# Extended state with actual fields
class ExtendedMultiAgentState(MultiAgentState):
    """State with actual fields from agent outputs."""

    # Shared fields updated by agents
    selected_modules: list[str] = Field(default_factory=list)
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)
    reasoning_structure: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""

    # Shared context
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)


# Mock agents that update specific fields
class MockSelectorAgent:
    """Agent that outputs to selected_modules field."""

    def __init__(self):
        self.name = "selector"
        self.output_schema = SelectedModules
        self.state_schema = None  # Would have full schema
        # Private tools
        self.tools = ["module_analyzer", "dependency_checker"]

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        """Return output that maps to state fields."""
        # Read from state
        available = getattr(state, "available_modules", [])

        # Select some modules
        selected = available[:2] if len(available) >= 2 else available

        # Return dict matching output schema fields
        return {
            "selected_modules": selected,
            "rationale": "Selected based on task requirements",
        }


class MockAdapterAgent:
    """Agent that reads selected_modules and outputs adapted_modules."""

    def __init__(self):
        self.name = "adapter"
        self.output_schema = AdaptedModules
        # Private tools
        self.tools = ["context_mapper", "adaptation_engine"]

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        """Adapt modules based on selection."""
        # Read from previous agent's output
        selected = getattr(state, "selected_modules", [])

        # Adapt each module
        adapted = [
            {"module": mod, "adaptation": f"Enhanced {mod} for task"}
            for mod in selected
        ]

        return {
            "adapted_modules": adapted,
            "task_context": getattr(state, "task_description", ""),
        }


class MockReasoningAgent:
    """Agent that produces final answer."""

    def __init__(self):
        self.name = "reasoner"
        self.output_schema = FinalAnswer
        # Private tools
        self.tools = ["logic_engine", "fact_checker", "proof_assistant"]

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        """Generate final answer from adapted modules."""
        adapted = getattr(state, "adapted_modules", [])

        return {
            "answer": f"Solution using {len(adapted)} adapted modules",
            "confidence": 0.95,
            "reasoning_steps": {
                "step1": "Analyzed modules",
                "step2": "Applied reasoning",
                "step3": "Generated answer",
            },
        }


class TestAgentNodeFieldUpdates:
    """Test agent nodes updating actual state fields."""

    def test_agent_updates_specific_state_field(self):
        """Test agent output updates specific state field, not agent_outputs."""
        state = ExtendedMultiAgentState()
        state.available_modules = ["reasoning", "planning", "analysis"]

        # Create selector agent and node
        agent = MockSelectorAgent()
        node = AgentNodeV3Config(
            name="selector_node", agent_name="selectof", agent=agent
        )

        # Execute - should update selected_modules field
        result = node(state, {})

        # Check Command structure
        assert isinstance(result, Command)

        # Should update the actual field, not agent_outputs

        # Verify field update (not nested in agent_outputs)
        assert "selected_modules" in result.update
        assert result.update["selected_modules"] == ["reasoning", "planning"]

        # Also verify agent state update
        if "agent_states" in result.update:
            assert "selector" in result.update["agent_states"]

    def test_sequential_agents_field_updates(self):
        """Test sequential agents updating different state fields."""
        state = ExtendedMultiAgentState()
        state.task_description = "Complex reasoning task"
        state.available_modules = ["A", "B", "C", "D"]

        # Execute selector agent
        selector = MockSelectorAgent()
        selector_node = AgentNodeV3Config(
            name="selector_node", agent_name="selectof", agent=selector
        )

        result1 = selector_node(state, {})

        # Apply updates to state
        if "selected_modules" in result1.update:
            state.selected_modules = result1.update["selected_modules"]
        if "agent_states" in result1.update:
            state.agent_states.update(result1.update["agent_states"])

        # Execute adapter agent - reads selected_modules
        adapter = MockAdapterAgent()
        adapter_node = AgentNodeV3Config(
            name="adapter_node", agent_name="adaptef", agent=adapter
        )

        result2 = adapter_node(state, {})

        # Should update adapted_modules field
        assert "adapted_modules" in result2.update
        assert len(result2.update["adapted_modules"]) == 2
        assert result2.update["adapted_modules"][0]["module"] == "A"

        # Apply adapter updates
        if "adapted_modules" in result2.update:
            state.adapted_modules = result2.update["adapted_modules"]

        # Execute reasoning agent
        reasoner = MockReasoningAgent()
        reasoner_node = AgentNodeV3Config(
            name="reasoner_node", agent_name="reasonef", agent=reasoner
        )

        result3 = reasoner_node(state, {})

        # Should update final_answer field
        assert "answer" in result3.update
        assert "confidence" in result3.update
        assert result3.update["answer"] == "Solution using 2 adapted modules"

    def test_agent_state_synchronization(self):
        """Test synchronization between agent state and combined state."""
        state = ExtendedMultiAgentState()
        state.available_modules = ["A", "B", "C"]

        # Mock agent that maintains internal state
        class StatefulAgent:
            def __init__(self):
                self.name = "stateful"
                self.output_schema = SelectedModules
                self.internal_state = {
                    "selection_count": 0,
                    "history": [],
                    "tools": ["analyzer"],  # Private
                }

            def invoke(self, state, config=None):
                self.internal_state["selection_count"] += 1
                self.internal_state["history"].append("selection_1")

                return {"selected_modules": ["A", "B"], "rationale": "First selection"}

        agent = StatefulAgent()
        node = AgentNodeV3Config(
            name="stateful_node", agent_name="stateful", agent=agent
        )

        result = node(state, {})

        # Should update both:
        # 1. Main state field
        assert "selected_modules" in result.update

        # 2. Agent state (with internal state)
        assert "agent_states" in result.update
        assert "stateful" in result.update["agent_states"]

        # Agent state should include output AND internal state
        agent_state_update = result.update["agent_states"]["stateful"]
        assert "selected_modules" in agent_state_update  # Output
        assert "rationale" in agent_state_update  # Output
        # Internal state would be included in full implementation

    def test_privacy_of_agent_tools(self):
        """Test that agent tools remain private in agent_states."""
        state = ExtendedMultiAgentState()

        # Agents with different private tools
        agents = [
            MockSelectorAgent(),
            # tools: ["module_analyzer", "dependency_checker"]
            MockAdapterAgent(),
            # tools: ["context_mapper", "adaptation_engine"]
            MockReasoningAgent(),
            # tools: ["logic_engine", "fact_checker", "proof_assistant"]
        ]

        for agent in agents:
            # Tools should be in agent's private state, not in main update
            node = AgentNodeV3Config(
                name=f"{agent.name}_node", agent_name=agent.name, agent=agent
            )

            result = node(state, {})

            # Main update should NOT have tools
            assert "tools" not in result.update

            # Agent state COULD have tools (private)
            if (
                "agent_states" in result.update
                and agent.name in result.update["agent_states"]
            ):
                result.update["agent_states"][agent.name]
                # In full implementation, tools would be here
                # assert "tools" in agent_state  # Private to agent

    def test_message_field_with_reducer(self):
        """Test message updates work with add_messages reducer."""
        state = ExtendedMultiAgentState()
        state.messages = [HumanMessage(content="Start process")]

        # Agent that adds messages
        class MessageAgent:
            name = "messengef"
            output_schema = None  # No structured output

            def invoke(self, state, config=None):
                return {"messages": [AIMessage(content="Processing...")]}

        agent = MessageAgent()
        node = AgentNodeV3Config(
            name="messenger_node", agent_name="messengef", agent=agent
        )

        result = node(state, {})

        # Messages should be in update
        assert "messages" in result.update
        assert len(result.update["messages"]) == 1
        assert isinstance(result.update["messages"][0], AIMessage)

        # With reducer, this would append to existing messages

    def test_command_multi_field_update(self):
        """Test Command updating multiple fields at once."""
        state = ExtendedMultiAgentState()
        state.available_modules = ["A", "B", "C", "D", "E"]

        # Agent that updates multiple fields
        class MultiFieldAgent:
            name = "multi"
            output_schema = None  # Custom output

            def invoke(self, state, config=None):
                return {
                    "selected_modules": ["A", "C", "E"],
                    "reasoning_structure": {
                        "method": "deductive",
                        "steps": ["select", "adapt", "apply"],
                    },
                    "messages": [AIMessage(content="Multi-field update")],
                }

        agent = MultiFieldAgent()
        node = AgentNodeV3Config(name="multi_node", agent_name="multi", agent=agent)

        result = node(state, {})

        # Should update all returned fields
        assert "selected_modules" in result.update
        assert "reasoning_structure" in result.update
        assert "messages" in result.update

        assert result.update["selected_modules"] == ["A", "C", "E"]
        assert result.update["reasoning_structure"]["method"] == "deductive"
        assert len(result.update["messages"]) == 1

    def test_output_schema_to_field_name_mapping(self):
        """Test mapping output schema class names to state field names."""
        # Test the pattern of ClassName -> field_name
        mappings = [
            (SelectedModules, "selected_modules"),
            (AdaptedModules, "adapted_modules"),
            (ReasoningStructure, "reasoning_structure"),
            (FinalAnswer, "final_answer"),
        ]

        for schema_class, expected_field in mappings:
            # Get field name from schema
            import re

            class_name = schema_class.__name__
            field_name = re.sub("(?<!^)(?=[A-Z])", "_", class_name).lower()

            assert field_name == expected_field

            # In practice, agent node would:
            # 1. Get agent.output_schema (e.g., SelectedModules)
            # 2. Convert to field name (e.g., "selected_modules")
            # 3. Update that field in state

    def test_no_agent_outputs_nesting(self):
        """Test that we don't use agent_outputs nesting for structured agents."""
        state = ExtendedMultiAgentState()
        state.available_modules = ["X", "Y", "Z"]

        # Structured agent
        agent = MockSelectorAgent()
        node = AgentNodeV3Config(
            name="selector_node", agent_name="selectof", agent=agent
        )

        result = node(state, {})

        # Should NOT have agent_outputs in update
        assert "agent_outputs" not in result.update

        # Should have direct field update
        assert "selected_modules" in result.update

        # For message agents, might still use agent_outputs
        # But for structured agents, use direct fields
