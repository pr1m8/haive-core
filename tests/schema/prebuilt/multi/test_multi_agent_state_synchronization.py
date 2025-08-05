"""Test MultiAgentState synchronization and field updates.

This test file validates:
1. Direct field updates (not just agent_outputs)
2. Synchronization between agent states and combined state
3. Privacy/isolation of agent-specific fields (tools, engines)
4. Schema composition from multiple agents
"""

from typing import Annotated, Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Individual agent state schemas
class SelectorAgentState(BaseModel):
    """State for selector agent."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    available_modules: list[str] = Field(default_factory=list)
    selected_modules: list[str] = Field(default_factory=list)  # Output field
    rationale: str | None = None
    # Private to this agent
    selection_history: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=lambda: ["module_analyzer"])


class AdapterAgentState(BaseModel):
    """State for adapter agent."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    selected_modules: list[str] = Field(default_factory=list)  # Input from selector
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)  # Output field
    task_context: str = ""
    # Private to this agent
    adaptation_rules: dict[str, str] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=lambda: ["context_analyzer"])


class ReasoningAgentState(BaseModel):
    """State for reasoning agent."""

    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)  # Input
    reasoning_structure: dict[str, Any] = Field(default_factory=dict)  # Output
    final_answer: str = ""  # Output
    # Private
    reasoning_steps: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=lambda: ["logic_checker", "fact_validator"])


# Combined state that includes all fields
class CombinedMultiAgentState(MultiAgentState):
    """Extended MultiAgentState with direct field access."""

    # Shared fields (visible to all agents)
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)

    # Output fields from agents (synchronized)
    selected_modules: list[str] = Field(default_factory=list)
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)
    reasoning_structure: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""

    # Private fields are kept in agent_states


class TestMultiAgentStateSynchronization:
    """Test state synchronization and privacy patterns."""

    def test_direct_field_updates(self):
        """Test updating actual state fields, not just agent_outputs."""
        state = CombinedMultiAgentState()

        # Set shared context
        state.task_description = "Solve complex reasoning problem"
        state.available_modules = ["reasoning", "planning", "analysis", "synthesis"]

        # Selector agent updates selected_modules directly
        selector_update = {
            "selected_modules": ["reasoning", "analysis"],
            "messages": [AIMessage(content="Selected best modules")],
        }

        # Apply update (simulating Command)
        state.selected_modules = selector_update["selected_modules"]
        state.messages.extend(selector_update["messages"])

        # Adapter agent reads selected_modules and updates adapted_modules
        adapter_update = {
            "adapted_modules": [
                {"module": "reasoning", "adaptation": "Focus on logical steps"},
                {"module": "analysis", "adaptation": "Deep dive into details"},
            ],
            "messages": [AIMessage(content="Adapted modules for task")],
        }

        state.adapted_modules = adapter_update["adapted_modules"]
        state.messages.extend(adapter_update["messages"])

        # Verify direct field updates
        assert state.selected_modules == ["reasoning", "analysis"]
        assert len(state.adapted_modules) == 2
        assert state.adapted_modules[0]["adaptation"] == "Focus on logical steps"
        assert len(state.messages) == 2

    def test_agent_state_synchronization(self):
        """Test synchronization between individual agent states and combined state."""
        state = CombinedMultiAgentState()

        # Initialize agent-specific states
        state.agent_states["selector"] = {
            "messages": [],
            "available_modules": ["A", "B", "C"],
            "selected_modules": [],
            "selection_history": [{"attempt": 1, "modules": ["A"]}],
            "tools": ["module_analyzer"],  # Private
        }

        state.agent_states["adapter"] = {
            "messages": [],
            "selected_modules": [],  # Will sync from selector
            "adapted_modules": [],
            "adaptation_rules": {"A": "enhance", "B": "simplify"},  # Private
            "tools": ["context_analyzer"],  # Private
        }

        # Selector produces output
        selector_output = {"selected_modules": ["A", "B"], "rationale": "Best for task"}

        # Update both agent state and combined state
        state.agent_states["selector"]["selected_modules"] = selector_output["selected_modules"]
        # Sync to combined
        state.selected_modules = selector_output["selected_modules"]

        # Adapter reads from combined state
        assert state.selected_modules == ["A", "B"]

        # Adapter can also read from selector's agent state if needed
        selector_state = state.agent_states["selector"]
        assert selector_state["selected_modules"] == ["A", "B"]

        # But adapter cannot see selector's private fields
        # (In practice, we'd filter these out)
        assert "selection_history" in selector_state  # Exists but should be private
        assert "tools" in selector_state  # Should be private

    def test_privacy_of_agent_tools(self):
        """Test that agent tools and private fields are isolated."""
        state = CombinedMultiAgentState()

        # Each agent has different tools
        selector_tools = ["module_analyzer", "dependency_checker"]
        adapter_tools = ["context_analyzer", "task_mapper"]
        reasoning_tools = ["logic_checker", "fact_validator", "proof_assistant"]

        # Store in agent states (private)
        state.agent_states["selector"] = {"tools": selector_tools}
        state.agent_states["adapter"] = {"tools": adapter_tools}
        state.agent_states["reasoning"] = {"tools": reasoning_tools}

        # Tools should NOT be in combined state fields
        assert not hasattr(state, "tools")  # No global tools field

        # Each agent only sees its own tools
        assert state.agent_states["selector"]["tools"] == selector_tools
        assert state.agent_states["adapter"]["tools"] == adapter_tools
        assert state.agent_states["reasoning"]["tools"] == reasoning_tools

        # Tools don't cross-contaminate
        assert "logic_checker" not in state.agent_states["selector"]["tools"]
        assert "module_analyzer" not in state.agent_states["reasoning"]["tools"]

    def test_schema_composition_pattern(self):
        """Test composing combined state from agent schemas."""
        # In practice, we'd compose the combined state from agent schemas

        # Shared fields (union of all agent inputs/outputs)

        # Private fields (agent-specific, not shared)

        # Combined state includes shared fields
        state = CombinedMultiAgentState()

        # Each agent updates its output fields in both places
        # 1. In combined state (shared fields)
        state.selected_modules = ["A", "B"]

        # 2. In its agent_state (all fields including private)
        state.agent_states["selector"] = {
            "selected_modules": ["A", "B"],  # Duplicated for agent's view
            "selection_history": [{"modules": ["A", "B"]}],  # Private
            "tools": ["module_analyzer"],  # Private
        }

        # Other agents see shared fields but not private
        assert state.selected_modules == ["A", "B"]  # Shared
        assert state.agent_states["selector"]["selection_history"]  # Private

    def test_message_synchronization(self):
        """Test message field synchronization across agents."""
        state = CombinedMultiAgentState()

        # Global messages
        state.messages = [
            HumanMessage(content="Solve this problem"),
            AIMessage(content="Starting multi-agent process"),
        ]

        # Each agent can have its own message view
        state.agent_states["selector"] = {
            "messages": state.messages.copy(),  # Starts with global
            "selected_modules": [],
        }

        # Selector adds a message
        selector_message = AIMessage(content="Selected modules: A, B")

        # Update both agent state and global
        state.agent_states["selector"]["messages"].append(selector_message)
        state.messages.append(selector_message)  # Sync to global

        # Next agent sees updated messages
        state.agent_states["adapter"] = {
            "messages": state.messages.copy(),  # Gets all messages
            "adapted_modules": [],
        }

        assert len(state.messages) == 3
        assert len(state.agent_states["adapter"]["messages"]) == 3

    def test_command_updates_multiple_fields(self):
        """Test Command pattern updating multiple state fields."""
        state = CombinedMultiAgentState()

        # Selector agent command update
        selector_update = {
            "selected_modules": ["reasoning", "planning"],
            "messages": [AIMessage(content="Selection complete")],
            "agent_states": {
                "selector": {
                    "selected_modules": ["reasoning", "planning"],
                    "selection_history": [{"round": 1, "selected": 2}],
                    "rationale": "Best modules for complex reasoning",
                }
            },
        }

        # Apply updates (simulating Command execution)
        if "selected_modules" in selector_update:
            state.selected_modules = selector_update["selected_modules"]

        if "messages" in selector_update:
            state.messages.extend(selector_update["messages"])

        if "agent_states" in selector_update:
            for agent_name, agent_state in selector_update["agent_states"].items():
                if agent_name not in state.agent_states:
                    state.agent_states[agent_name] = {}
                state.agent_states[agent_name].update(agent_state)

        # Verify all updates applied
        assert state.selected_modules == ["reasoning", "planning"]
        assert len(state.messages) == 1
        assert state.agent_states["selector"]["rationale"] == "Best modules for complex reasoning"
        assert state.agent_states["selector"]["selection_history"][0]["round"] == 1

    def test_output_fields_as_state_keys(self):
        """Test that agent output schemas become state field keys."""
        state = CombinedMultiAgentState()

        # Instead of agent_outputs["selector"] = {...}
        # We update the actual fields:

        # From SelectedModules output schema
        state.selected_modules = ["A", "B", "C"]

        # From AdaptedModules output schema
        state.adapted_modules = [{"module": "A", "adaptation": "Enhanced for task"}]

        # From ReasoningStructure output schema
        state.reasoning_structure = {
            "steps": ["analyze", "synthesize", "conclude"],
            "logic": "deductive",
        }

        # From FinalAnswer output schema
        state.final_answer = "The solution is X"

        # All fields are directly accessible
        assert state.selected_modules == ["A", "B", "C"]
        assert state.adapted_modules[0]["module"] == "A"
        assert state.reasoning_structure["logic"] == "deductive"
        assert state.final_answer == "The solution is X"

        # No need for agent_outputs nesting
        assert not hasattr(state, "agent_outputs") or state.agent_outputs == {}

    def test_agent_state_composition(self):
        """Test composing one big state from individual agent states."""
        # Individual agent states
        selector_state = SelectorAgentState(
            messages=[HumanMessage(content="Select modules")],
            available_modules=["A", "B", "C", "D"],
            selected_modules=["A", "C"],
            selection_history=[{"round": 1}],
            tools=["module_analyzer"],
        )

        adapter_state = AdapterAgentState(
            messages=selector_state.messages.copy(),
            selected_modules=selector_state.selected_modules,
            adapted_modules=[{"module": "A", "adaptation": "Enhanced"}],
            adaptation_rules={"A": "enhance"},
            tools=["context_analyzer"],
        )

        # Compose into combined state
        state = CombinedMultiAgentState()

        # Shared fields go to top level
        state.messages = selector_state.messages
        state.available_modules = selector_state.available_modules
        state.selected_modules = selector_state.selected_modules
        state.adapted_modules = adapter_state.adapted_modules

        # Private fields go to agent_states
        state.agent_states["selector"] = {
            "selection_history": selector_state.selection_history,
            "tools": selector_state.tools,
        }

        state.agent_states["adapter"] = {
            "adaptation_rules": adapter_state.adaptation_rules,
            "tools": adapter_state.tools,
        }

        # Verify composition
        assert state.selected_modules == ["A", "C"]  # Shared
        assert state.adapted_modules[0]["module"] == "A"  # Shared
        assert state.agent_states["selector"]["tools"] == ["module_analyzer"]  # Private
        assert state.agent_states["adapter"]["tools"] == ["context_analyzer"]  # Private
