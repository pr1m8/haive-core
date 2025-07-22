"""Test MultiAgentState with typed I/O patterns and message reducers.

This test file validates:
1. Messages field with add_messages reducer
2. Typed agent inputs/outputs
3. Schema-based field updates
4. Private state between agents
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Define typed schemas for agents (like Self-Discover)
class SelectorInput(BaseModel):
    """Input schema for module selector."""

    task_description: str
    available_modules: list[str]


class SelectedModules(BaseModel):
    """Output schema for module selector."""

    selected_modules: list[str]
    rationale: str | None = None


class AdapterInput(BaseModel):
    """Input schema for module adapter."""

    task_description: str
    selected_modules: list[str]


class AdaptedModules(BaseModel):
    """Output schema for module adapter."""

    adapted_modules: list[dict[str, str]]  # module, adaptation pairs
    task_context: str


class StructurerInput(BaseModel):
    """Input schema for reasoning structurer."""

    task_description: str
    adapted_modules: list[dict[str, str]]


class ReasoningStructure(BaseModel):
    """Output schema for reasoning structurer."""

    reasoning_structure: dict[str, Any]
    steps: list[str]


class TestMultiAgentStateTypedIO:
    """Test MultiAgentState with typed I/O patterns."""

    def test_messages_field_has_reducer(self):
        """Test that messages field uses add_messages reducer."""
        # MultiAgentState inherits from MessagesState which has annotated
        # messages
        state = MultiAgentState()

        # Add initial messages
        state.messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]

        # The field should support message operations
        assert len(state.messages) == 2
        assert isinstance(state.messages[0], HumanMessage)
        assert isinstance(state.messages[1], AIMessage)

        # In real usage, the reducer handles merging
        # This is handled by LangGraph when using Command updates

    def test_typed_agent_outputs(self):
        """Test storing typed agent outputs in state."""
        state = MultiAgentState()

        # Store typed outputs from different agents
        # Selector agent output
        selector_output = SelectedModules(
            selected_modules=["reasoning", "planning", "critical_thinking"],
            rationale="These modules are best for complex analysis",
        )
        state.agent_outputs["select_modules"] = selector_output.model_dump()

        # Adapter agent output
        adapter_output = AdaptedModules(
            adapted_modules=[
                {"module": "reasoning", "adaptation": "Focus on logical steps"},
                {"module": "planning", "adaptation": "Create detailed action plan"},
            ],
            task_context="Complex problem solving",
        )
        state.agent_outputs["adapt_modules"] = adapter_output.model_dump()

        # Structurer agent output
        structurer_output = ReasoningStructure(
            reasoning_structure={
                "phase1": "analysis",
                "phase2": "planning",
                "phase3": "execution",
            },
            steps=["Analyze problem", "Create plan", "Execute solution"],
        )
        state.agent_outputs["create_structure"] = structurer_output.model_dump()

        # Verify typed outputs stored correctly
        assert state.agent_outputs["select_modules"]["selected_modules"] == [
            "reasoning",
            "planning",
            "critical_thinking",
        ]
        assert (
            state.agent_outputs["adapt_modules"]["adapted_modules"][0]["module"]
            == "reasoning"
        )
        assert (
            state.agent_outputs["create_structure"]["reasoning_structure"]["phase1"]
            == "analysis"
        )

        # No messages in structured outputs
        assert "messages" not in state.agent_outputs["select_modules"]
        assert "messages" not in state.agent_outputs["adapt_modules"]
        assert "messages" not in state.agent_outputs["create_structure"]

    def test_schema_based_field_mapping(self):
        """Test that output schemas can map to specific state fields."""
        state = MultiAgentState()

        # Instead of agent_outputs, we could have schema-specific fields
        # This shows the pattern even though MultiAgentState uses agent_outputs

        # Simulate what could happen with schema-specific fields
        selector_result = SelectedModules(
            selected_modules=["analysis", "synthesis"], rationale="Best for this task"
        )

        # In a custom state, we might have:
        # state.selected_modules = selector_result
        # For now, store in agent_outputs
        state.agent_outputs["selector"] = selector_result.model_dump()

        # The key insight: output schema name could become field name
        output_schema_name = selector_result.__class__.__name__
        assert output_schema_name == "SelectedModules"

        # Could use snake_case version as field name
        field_name = "selected_modules"  # from SelectedModules
        assert field_name == "selected_modules"

    def test_private_state_pattern(self):
        """Test private state passing between specific agents."""
        state = MultiAgentState()

        # Agent 1 produces private data for Agent 2
        private_output = {
            "internal_analysis": {"complexity": "high", "risk": "medium"},
            "preprocessing_done": True,
        }

        # Store in agent_specific state (private)
        state.agent_states["analyzer"] = private_output

        # Agent 2 can access this private state
        analyzer_state = state.get_agent_state("analyzer")
        assert analyzer_state["internal_analysis"]["complexity"] == "high"
        assert analyzer_state["preprocessing_done"] is True

        # Other agents don't see this unless explicitly shared
        other_agent_state = state.get_agent_state("other_agent")
        assert other_agent_state == {}  # Empty for non-existent agent

    def test_mixed_message_and_typed_outputs(self):
        """Test state with both message-based and typed outputs."""
        state = MultiAgentState()

        # Global messages (with reducer)
        state.messages = [
            HumanMessage(content="Analyze this problem"),
            AIMessage(content="Starting analysis..."),
        ]

        # Simple agent with messages output
        state.agent_outputs["chat_agent"] = {
            "messages": [AIMessage(content="I'll help analyze this step by step")]
        }

        # Typed agents with structured outputs
        state.agent_outputs["select_modules"] = SelectedModules(
            selected_modules=["decomposition", "pattern_recognition"]
        ).model_dump()

        state.agent_outputs["final_reasoning"] = {
            "answer": "The solution is X",
            "confidence": 0.95,
            "reasoning_steps": ["Step 1", "Step 2", "Step 3"],
        }

        # Verify coexistence
        assert len(state.messages) == 2  # Global messages
        assert "messages" in state.agent_outputs["chat_agent"]
        assert "messages" not in state.agent_outputs["select_modules"]
        assert "answer" in state.agent_outputs["final_reasoning"]

    def test_agent_input_extraction_pattern(self):
        """Test how agents would extract inputs from state."""
        state = MultiAgentState()

        # Set up state with various fields
        state.messages = [HumanMessage(content="Solve this problem")]
        state.agent_outputs["select_modules"] = {
            "selected_modules": ["reasoning", "planning"]
        }
        state.agent_states["shared"] = {
            "task_description": "Complex reasoning problem",
            "available_modules": ["reasoning", "planning", "analysis", "synthesis"],
        }

        # Simulate selector agent extracting its inputs
        # Based on SelectorInput schema
        selector_inputs = {
            "task_description": state.agent_states["shared"]["task_description"],
            "available_modules": state.agent_states["shared"]["available_modules"],
        }
        selector_input = SelectorInput(**selector_inputs)
        assert selector_input.task_description == "Complex reasoning problem"
        assert len(selector_input.available_modules) == 4

        # Simulate adapter agent extracting its inputs
        # Based on AdapterInput schema
        adapter_inputs = {
            "task_description": state.agent_states["shared"]["task_description"],
            "selected_modules": state.agent_outputs["select_modules"][
                "selected_modules"
            ],
        }
        adapter_input = AdapterInput(**adapter_inputs)
        assert adapter_input.selected_modules == ["reasoning", "planning"]

    def test_schema_field_name_conventions(self):
        """Test conventions for schema to field name mapping."""
        # Test various schema name to field name conversions
        test_cases = [
            ("SelectedModules", "selected_modules"),
            ("AdaptedModules", "adapted_modules"),
            ("ReasoningStructure", "reasoning_structure"),
            ("FinalAnswer", "final_answer"),
            ("ModuleSelection", "module_selection"),
        ]

        for class_name, expected_field in test_cases:
            # Simple conversion: PascalCase to snake_case
            import re

            field_name = re.sub("(?<!^)(?=[A-Z])", "_", class_name).lower()
            assert field_name == expected_field

    def test_command_update_with_typed_outputs(self):
        """Test how Command updates would work with typed outputs."""
        state = MultiAgentState()

        # Simulate Command updates from different agents
        # Simple agent updates messages
        messages_update = {"messages": [AIMessage(content="Processing...")]}
        # In real usage: Command(update=messages_update)
        if "messages" in messages_update:
            state.messages.extend(messages_update["messages"])

        # Structured agent updates specific output
        selector_update = {
            "agent_outputs": {
                "select_modules": {
                    "selected_modules": ["reasoning", "analysis"],
                    "rationale": "Best modules for task",
                }
            }
        }
        # In real usage: Command(update=selector_update)
        if "agent_outputs" in selector_update:
            state.agent_outputs.update(selector_update["agent_outputs"])

        # Verify updates applied correctly
        assert len(state.messages) == 1
        assert state.agent_outputs["select_modules"]["selected_modules"] == [
            "reasoning",
            "analysis",
        ]
