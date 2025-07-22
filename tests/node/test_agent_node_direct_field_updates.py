"""Test agent node updating state fields directly based on output schemas.

This demonstrates the desired behavior where agents with typed output schemas
update the corresponding state fields directly, not through agent_outputs.
"""

from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Define output schemas that should map to state fields
class SelectedModules(BaseModel):
    """Should update state.selected_modules directly."""

    selected_modules: list[str]
    rationale: str | None = None


class AdaptedModules(BaseModel):
    """Should update state.adapted_modules directly."""

    adapted_modules: list[dict[str, str]]
    task_context: str


# State with actual fields matching output schemas
class WorkflowState(MultiAgentState):
    """State with fields that match agent output schemas."""

    # Input context
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)

    # Fields that agents will update directly
    selected_modules: list[str] = Field(default_factory=list)
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)
    selection_rationale: str | None = None
    adaptation_context: str | None = None


class TestDirectFieldUpdates:
    """Test the desired behavior of direct field updates."""

    def test_desired_agent_output_behavior(self):
        """Test how we want agent outputs to work."""
        state = WorkflowState()
        state.available_modules = ["reasoning", "planning", "analysis"]

        # Simulate what an agent with SelectedModules output schema should do
        agent_output = SelectedModules(
            selected_modules=["reasoning", "planning"],
            rationale="Best modules for the task",
        )

        # The desired Command update structure
        desired_update = {
            # Direct field updates - schema fields map to state fields
            "selected_modules": agent_output.selected_modules,
            "selection_rationale": agent_output.rationale,  # Optional mapping
            # Messages if any
            "messages": [
                AIMessage(content="Selected modules based on task requirements")
            ],
            # Agent state tracking (but not the primary output mechanism)
            "agent_states": {
                "selector": {
                    "selected_modules": agent_output.selected_modules,
                    "rationale": agent_output.rationale,
                    "execution_count": 1,
                }
            },
        }

        # This is what the Command should look like
        Command(update=desired_update)

        # Apply to state
        state.selected_modules = desired_update["selected_modules"]
        state.selection_rationale = desired_update["selection_rationale"]
        state.messages.extend(desired_update["messages"])
        state.agent_states.update(desired_update["agent_states"])

        # Verify state updated correctly
        assert state.selected_modules == ["reasoning", "planning"]
        assert state.selection_rationale == "Best modules for the task"
        assert len(state.messages) == 1

    def test_schema_to_field_mapping_pattern(self):
        """Test the pattern for mapping output schema to state fields."""
        # Pattern 1: Direct name match
        # SelectedModules.selected_modules -> state.selected_modules

        # Pattern 2: Schema class name to field
        # SelectedModules -> state.selected_modules (snake_case)

        # Pattern 3: Explicit mapping

        # Example implementation in agent node
        def get_field_updates(
            output_schema_instance: BaseModel,
            field_mappings: dict[str, str] | None = None,
        ):
            """Get state field updates from output schema instance."""
            updates = {}
            field_mappings = field_mappings or {}

            for field_name, field_value in output_schema_instance.model_dump().items():
                # Check explicit mapping first
                if field_name in field_mappings:
                    state_field = field_mappings[field_name]
                else:
                    # Default: use same field name
                    state_field = field_name

                updates[state_field] = field_value

            return updates

        # Test the mapping
        output = SelectedModules(
            selected_modules=["A", "B"], rationale="Test rationale"
        )

        updates = get_field_updates(output, {"rationale": "selection_rationale"})

        assert updates == {
            "selected_modules": ["A", "B"],
            "selection_rationale": "Test rationale",
        }

    def test_agent_sequence_with_direct_updates(self):
        """Test sequence of agents updating fields directly."""
        state = WorkflowState()
        state.task_description = "Build a reasoning system"
        state.available_modules = ["A", "B", "C", "D"]

        # Agent 1: Selector - outputs SelectedModules
        selector_output = SelectedModules(
            selected_modules=["A", "C"], rationale="Most relevant for reasoning"
        )

        # Command from selector agent
        selector_command = Command(
            update={
                "selected_modules": selector_output.selected_modules,
                "selection_rationale": selector_output.rationale,
                "agent_states": {"selector": selector_output.model_dump()},
            }
        )

        # Apply updates
        for key, value in selector_command.update.items():
            if key == "agent_states":
                state.agent_states.update(value)
            elif hasattr(state, key):
                setattr(state, key, value)

        # Agent 2: Adapter - reads selected_modules, outputs AdaptedModules
        # It can read directly from state.selected_modules!
        assert state.selected_modules == ["A", "C"]

        adapter_output = AdaptedModules(
            adapted_modules=[
                {"module": "A", "adaptation": "Enhanced for reasoning"},
                {"module": "C", "adaptation": "Optimized for logic"},
            ],
            task_context=state.task_description,
        )

        # Command from adapter agent
        adapter_command = Command(
            update={
                "adapted_modules": adapter_output.adapted_modules,
                "adaptation_context": adapter_output.task_context,
                "agent_states": {"adapter": adapter_output.model_dump()},
            }
        )

        # Apply updates
        for key, value in adapter_command.update.items():
            if key == "agent_states":
                state.agent_states.update(value)
            elif hasattr(state, key):
                setattr(state, key, value)

        # Verify full state
        assert state.selected_modules == ["A", "C"]
        assert state.selection_rationale == "Most relevant for reasoning"
        assert len(state.adapted_modules) == 2
        assert state.adapted_modules[0]["module"] == "A"
        assert state.adaptation_context == "Build a reasoning system"

    def test_mixed_output_types(self):
        """Test agents with different output types."""
        WorkflowState()

        # Agent 1: Message-only output (no schema)
        {
            "messages": [AIMessage(content="Starting process")],
            "agent_outputs": {
                "starter": {"messages": [AIMessage(content="Starting process")]}
            },
        }

        # Agent 2: Structured output (with schema)
        {
            "selected_modules": ["X", "Y"],
            "messages": [AIMessage(content="Selected modules")],
            "agent_states": {
                "selector": {"selected_modules": ["X", "Y"], "execution_time": 1.23}
            },
        }

        # Agent 3: Complex structured output

        # The key insight: agents with output_schema update state fields directly
        # Agents without output_schema use agent_outputs pattern
