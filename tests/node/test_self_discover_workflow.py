"""Test Self-Discover workflow with fixed agent node output.

This demonstrates how the fixed agent node enables Self-Discover style
workflows where agents read each other's outputs directly from state fields.
"""

from typing import Any

from haive.agents.simple import SimpleAgent
from pydantic import BaseModel, Field

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Self-Discover output schemas
class SelectedModules(BaseModel):
    """Output from module selection agent."""

    selected_modules: list[str]
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)


class AdaptedModules(BaseModel):
    """Output from module adaptation agent."""

    adapted_modules: list[dict[str, str]]
    task_context: str
    adaptation_notes: str | None = None


class ReasoningStructure(BaseModel):
    """Output from reasoning structure agent."""

    reasoning_structure: dict[str, Any]
    steps: list[str]
    methodology: str


class FinalAnswer(BaseModel):
    """Output from final reasoning agent."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_path: dict[str, str]
    supporting_evidence: list[str]


# State with all the fields
class SelfDiscoverState(MultiAgentState):
    """State for Self-Discover workflow with direct field access."""

    # Input fields
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)

    # Output fields from agents (directly accessible!)
    selected_modules: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0

    adapted_modules: list[dict[str, str]] = Field(default_factory=list)
    task_context: str = ""
    adaptation_notes: str | None = None

    reasoning_structure: dict[str, Any] = Field(default_factory=dict)
    steps: list[str] = Field(default_factory=list)
    methodology: str = ""

    answer: str = ""
    reasoning_path: dict[str, str] = Field(default_factory=dict)
    supporting_evidence: list[str] = Field(default_factory=list)


def test_self_discover_workflow():
    """Test complete Self-Discover workflow with direct field updates."""
    # Initialize state
    state = SelfDiscoverState()
    state.task_description = "How can we reduce plastic waste in oceans?"
    state.available_modules = [
        "systems_thinking",
        "root_cause_analysis",
        "stakeholder_analysis",
        "solution_design",
        "impact_assessment",
        "implementation_planning",
        "cost_benefit_analysis",
        "environmental_modeling",
    ]

    # Mock LLM config for testing
    llm_config = AugLLMConfig(temperature=0.1)

    # Agent 1: Module Selection

    selector_agent = SimpleAgent(
        name="module_selector",
        engine=llm_config,
        structured_output_model=SelectedModules,
        system_message="""You are an expert at selecting reasoning modules.
        Analyze the task and select 3-4 most relevant modules.""",
    )

    # Add to state (simulating proper setup)
    state.agents["module_selector"] = selector_agent

    # Create and execute node
    selector_node = create_agent_node_v3(
        agent_name="module_selector", name="selector_node"
    )

    # Mock the agent's response for testing
    class MockSelectorAgent:
        name = "module_selector"
        output_schema = SelectedModules

        def invoke(self, state_dict, config=None):
            # Agent reads available modules from state
            return SelectedModules(
                selected_modules=[
                    "root_cause_analysis",
                    "solution_design",
                    "impact_assessment",
                ],
                rationale="These modules address problem identification, solution creation, and validation",
                confidence=0.9,
            )

    # Replace with mock for demo
    state.agents["module_selector"] = MockSelectorAgent()

    result1 = selector_node(state, {"debug": False})

    # Apply updates to state
    for key, value in result1.update.items():
        if hasattr(state, key) and key != "agent_states":
            setattr(state, key, value)

    # Agent 2: Module Adaptation

    # This agent can read selected_modules DIRECTLY from state!

    class MockAdapterAgent:
        name = "module_adapter"
        output_schema = AdaptedModules

        def invoke(self, state_dict, config=None):
            # Can read previous agent's output directly!
            return AdaptedModules(
                adapted_modules=[
                    {
                        "module": "root_cause_analysis",
                        "adaptation": "Focus on sources of ocean plastic",
                    },
                    {
                        "module": "solution_design",
                        "adaptation": "Design prevention and cleanup solutions",
                    },
                    {
                        "module": "impact_assessment",
                        "adaptation": "Measure environmental and economic impacts",
                    },
                ],
                task_context=state.task_description,
                adaptation_notes="Modules adapted for marine environment focus",
            )

    state.agents["module_adapter"] = MockAdapterAgent()

    adapter_node = create_agent_node_v3(
        agent_name="module_adapter", name="adapter_node"
    )

    result2 = adapter_node(state, {"debug": False})

    # Apply updates
    for key, value in result2.update.items():
        if hasattr(state, key) and key != "agent_states":
            setattr(state, key, value)

    # Agent 3: Reasoning Structure

    class MockReasoningAgent:
        name = "reasoning_builder"
        output_schema = ReasoningStructure

        def invoke(self, state_dict, config=None):
            return ReasoningStructure(
                reasoning_structure={
                    "analyze": "Identify sources and pathways of plastic waste",
                    "design": "Create prevention and cleanup solutions",
                    "assess": "Evaluate environmental and economic impacts",
                },
                steps=[
                    "Map plastic waste sources",
                    "Analyze ocean currents and accumulation",
                    "Design intervention strategies",
                    "Assess feasibility and impact",
                ],
                methodology="systematic_problem_solving",
            )

    state.agents["reasoning_builder"] = MockReasoningAgent()

    reasoning_node = create_agent_node_v3(
        agent_name="reasoning_builder", name="reasoning_node"
    )

    result3 = reasoning_node(state, {"debug": False})

    # Apply updates
    for key, value in result3.update.items():
        if hasattr(state, key) and key != "agent_states":
            setattr(state, key, value)

    # Final verification


if __name__ == "__main__":
    test_self_discover_workflow()
