"""Test agent node field mapping from output schemas to state fields.

This test demonstrates how agent nodes should map agent output schemas
to actual state fields for direct updates.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Output schemas
class ModuleSelection(BaseModel):
    """Output schema for module selection."""

    selected_modules: list[str]
    confidence_scores: dict[str, float]
    selection_reason: str


class TaskAnalysis(BaseModel):
    """Output schema for task analysis."""

    task_complexity: str
    required_capabilities: list[str]
    estimated_duration: int


# State with corresponding fields
class AnalysisWorkflowState(MultiAgentState):
    """State with fields matching output schemas."""

    # Inputs
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)

    # Fields from ModuleSelection schema
    selected_modules: list[str] = Field(default_factory=list)
    confidence_scores: dict[str, float] = Field(default_factory=dict)
    selection_reason: str = ""

    # Fields from TaskAnalysis schema
    task_complexity: str = ""
    required_capabilities: list[str] = Field(default_factory=list)
    estimated_duration: int = 0


class MockSchemaAgent:
    """Mock agent that outputs typed schema."""

    def __init__(self, name: str, output_schema: type[BaseModel]):
        self.name = name
        self.output_schema = output_schema
        self.state_schema = AnalysisWorkflowState  # The full state schema

    def invoke(self, state: Any, config: Any = None) -> BaseModel:
        """Return instance of output schema."""
        if self.output_schema == ModuleSelection:
            return ModuleSelection(
                selected_modules=["reasoning", "planning"],
                confidence_scores={"reasoning": 0.95, "planning": 0.87},
                selection_reason="Best modules for complex reasoning tasks",
            )
        if self.output_schema == TaskAnalysis:
            return TaskAnalysis(
                task_complexity="high",
                required_capabilities=[
                    "logical_reasoning", "planning", "synthesis"],
                estimated_duration=120,
            )
        raise ValueError(f"Unknown output schema: {self.output_schema}")


class TestAgentNodeFieldMapping:
    """Test agent node mapping output schemas to state fields."""

    def test_agent_output_schema_field_mapping(self):
        """Test that agent output schema fields map to state fields."""
        state = AnalysisWorkflowState()
        state.available_modules = [
            "reasoning", "planning", "analysis", "synthesis"]

        # Create agent with ModuleSelection output schema
        agent = MockSchemaAgent("selector", ModuleSelection)

        # What we want: agent node detects output schema and maps fields
        # Current behavior: stores in agent_outputs
        # Desired behavior: update state fields directly

        # The agent returns ModuleSelection instance
        output = agent.invoke(state)
        assert isinstance(output, ModuleSelection)

        # Desired Command structure from agent node
        Command(
            update={
                # Direct field updates from output schema
                "selected_modules": output.selected_modules,
                "confidence_scores": output.confidence_scores,
                "selection_reason": output.selection_reason,
                # Also track in agent_states for history
                "agent_states": {"selector": output.model_dump()},
            }
        )

        # This is what we want the agent node to produce

    def test_sequential_schema_agents(self):
        """Test sequential agents with schema outputs updating state."""
        state = AnalysisWorkflowState()
        state.task_description = "Design a complex reasoning system"

        # Agent 1: Task Analyzer
        analyzer = MockSchemaAgent("analyzer", TaskAnalysis)
        analysis_output = analyzer.invoke(state)

        # Simulate desired agent node behavior
        # Update state fields from TaskAnalysis schema
        state.task_complexity = analysis_output.task_complexity
        state.required_capabilities = analysis_output.required_capabilities
        state.estimated_duration = analysis_output.estimated_duration

        # Agent 2: Module Selector (can read task analysis results)
        selector = MockSchemaAgent("selector", ModuleSelection)

        # Selector can access analyzer's outputs directly from state!
        assert state.task_complexity == "high"
        assert state.required_capabilities == [
            "logical_reasoning",
            "planning",
            "synthesis",
        ]

        selection_output = selector.invoke(state)

        # Update state fields from ModuleSelection schema
        state.selected_modules = selection_output.selected_modules
        state.confidence_scores = selection_output.confidence_scores
        state.selection_reason = selection_output.selection_reason

        # Verify complete state
        assert state.task_complexity == "high"
        assert state.selected_modules == ["reasoning", "planning"]
        assert state.confidence_scores["reasoning"] == 0.95

    def test_schema_field_detection_pattern(self):
        """Test pattern for detecting which fields to update from schema."""

        def get_schema_field_mapping(
            output_schema: type[BaseModel], state_schema: type[BaseModel]
        ) -> dict[str, str]:
            """Get mapping of output schema fields to state fields."""
            mapping = {}

            output_fields = output_schema.model_fields
            state_fields = state_schema.model_fields

            for field_name in output_fields:
                if field_name in state_fields:
                    # Direct match
                    mapping[field_name] = field_name
                else:
                    # Could add custom mapping logic here
                    # For now, skip fields not in state
                    pass

            return mapping

        # Test the detection
        mapping = get_schema_field_mapping(
            ModuleSelection, AnalysisWorkflowState)

        assert mapping == {
            "selected_modules": "selected_modules",
            "confidence_scores": "confidence_scores",
            "selection_reason": "selection_reason",
        }

        # All fields from ModuleSelection exist in AnalysisWorkflowState
        assert len(mapping) == len(ModuleSelection.model_fields)

    def test_mixed_schema_and_message_agents(self):
        """Test mixing agents with schemas and without."""
        state = AnalysisWorkflowState()

        # Agent 1: Message-based (no output schema)
        # Should use agent_outputs pattern
        Command(
            update={
                "messages": [AIMessage(content="Starting analysis")],
                "agent_outputs": {
                    "starter": {"messages": [AIMessage(content="Starting analysis")]}
                },
            }
        )

        # Agent 2: Schema-based (with output schema)
        # Should update fields directly
        analyzer = MockSchemaAgent("analyzer", TaskAnalysis)
        output = analyzer.invoke(state)

        Command(
            update={
                # Direct field updates
                "task_complexity": output.task_complexity,
                "required_capabilities": output.required_capabilities,
                "estimated_duration": output.estimated_duration,
                # Plus tracking
                "agent_states": {"analyzer": output.model_dump()},
            }
        )

        # Key difference:
        # - No output_schema → use agent_outputs
        # - Has output_schema → update state fields directly

    def test_agent_node_output_processing(self):
        """Test how agent node should process different output types."""

        # Pattern to implement in agent node:
        def process_agent_output(
                agent: Any, output: Any, state: Any) -> dict[str, Any]:
            """Process agent output based on output schema."""
            update = {}

            # Check if agent has output_schema
            if hasattr(agent, "output_schema") and agent.output_schema:
                # Agent has typed output - map to state fields
                output_dict = (
                    output.model_dump() if isinstance(output, BaseModel) else output
                )

                # Get state fields
                state_fields = set()
                if hasattr(state, "model_fields"):
                    state_fields = set(state.model_fields.keys())

                # Update matching fields
                for field_name, field_value in output_dict.items():
                    if field_name in state_fields:
                        update[field_name] = field_value

                # Also store in agent_states for history
                update.setdefault("agent_states", {})[agent.name] = output_dict

            else:
                # No output schema - use agent_outputs pattern
                update.setdefault("agent_outputs", {})[agent.name] = output

            return update

        # Test with schema agent
        agent = MockSchemaAgent("test", ModuleSelection)
        output = agent.invoke(None)
        state = AnalysisWorkflowState()

        update = process_agent_output(agent, output, state)

        # Should have direct field updates
        assert "selected_modules" in update
        assert "confidence_scores" in update
        assert "selection_reason" in update

        # Should also track in agent_states
        assert "agent_states" in update
        assert "test" in update["agent_states"]
