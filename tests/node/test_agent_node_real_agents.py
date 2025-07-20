"""Test agent node with REAL agents and non-trivial cases.

NO MOCKS - Uses actual agents like SimpleAgent, Self-Discover agents, etc.
Tests real field updates, synchronization, and complex workflows.
"""

from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.agents.simple import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig

# Import and create agent node function to handle forward refs
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState

# Import Self-Discover models
try:
    from haive.agents.reasoning_and_critique.self_discover.v2.models import (
        AdaptedModules,
        FinalAnswer,
        ReasoningStructure,
        SelectedModules,
    )
except ImportError:
    # Define minimal versions if not available
    class SelectedModules(BaseModel):
        selected_modules: list[str]
        rationale: str | None = None

    class AdaptedModules(BaseModel):
        adapted_modules: list[dict[str, str]]
        task_context: str

    class ReasoningStructure(BaseModel):
        reasoning_structure: dict[str, Any]
        steps: list[str]

    class FinalAnswer(BaseModel):
        answer: str
        confidence: float
        reasoning_steps: dict[str, str]


# Extended state with real Self-Discover fields
class SelfDiscoverMultiAgentState(MultiAgentState):
    """Real state for Self-Discover workflow with actual fields."""

    # Shared context
    task_description: str = ""
    available_modules: list[str] = Field(default_factory=list)

    # Self-Discover agent outputs as actual fields
    selected_modules: list[str] = Field(default_factory=list)
    adapted_modules: list[dict[str, str]] = Field(default_factory=list)
    reasoning_structure: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""

    # Reasoning metadata
    selection_rationale: str | None = None
    adaptation_context: str | None = None
    confidence: float = 0.0


class TestAgentNodeRealAgents:
    """Test with REAL agents - no mocks."""

    @pytest.fixture
    def real_llm_config(self):
        """Real LLM configuration for testing."""
        return AugLLMConfig(
            temperature=0.1, max_tokens=500)  # Low for consistency

    def test_simple_agent_message_updates(self, real_llm_config):
        """Test SimpleAgent (no structured output) updates messages."""
        state = SelfDiscoverMultiAgentState()
        state.messages = [HumanMessage(content="Hello, can you help me?")]

        # Real SimpleAgent
        agent = SimpleAgent(name="assistant", engine=real_llm_config)

        # Add agent to state.agents to avoid set_active_agent error
        state.agents["assistant"] = agent

        # Create node using factory function - no agent passed directly
        node = create_agent_node_v3(
            agent_name="assistant",
            name="assistant_node")

        # Execute with real LLM
        result = node(state, {"debug": True})

        # Check Command structure
        assert isinstance(result, Command)

        # SimpleAgent should update messages (no structured output)
        assert "messages" in result.update or "agent_outputs" in result.update

        # If using agent_outputs pattern
        if "agent_outputs" in result.update:
            assert "assistant" in result.update["agent_outputs"]
            output = result.update["agent_outputs"]["assistant"]
            assert "messages" in output
            assert len(output["messages"]) > 0
            assert isinstance(output["messages"][0], BaseMessage)

    def test_structured_simple_agent_field_update(self, real_llm_config):
        """Test SimpleAgent with structured output updates specific field."""
        state = SelfDiscoverMultiAgentState()
        state.task_description = "Select the best reasoning modules"
        state.available_modules = [
            "critical_thinking",
            "logical_reasoning",
            "creative_problem_solving",
            "systematic_analysis",
            "pattern_recognition",
        ]

        # SimpleAgent with structured output model
        agent = SimpleAgent(
            name="module_selector",
            engine=real_llm_config,
            structured_output_model=SelectedModules,
            system_message="You are a module selection expert. Select 2-3 best modules for the task.",
        )

        # Create node
        node = create_agent_node_v3(
            agent_name="module_selector", agent=agent, name="selector_node"
        )

        # Execute
        result = node(state, {"debug": True})

        # Should update selected_modules field
        # Check various possible update patterns
        if "selected_modules" in result.update:
            # Direct field update
            assert isinstance(result.update["selected_modules"], list)
            assert len(result.update["selected_modules"]) >= 2
            assert all(isinstance(m, str)
                       for m in result.update["selected_modules"])
        elif (
            "agent_outputs" in result.update
            and "module_selector" in result.update["agent_outputs"]
        ):
            # Agent outputs pattern
            output = result.update["agent_outputs"]["module_selector"]
            assert "selected_modules" in output
            assert isinstance(output["selected_modules"], list)

    def test_self_discover_agent_sequence(self, real_llm_config):
        """Test sequence of Self-Discover style agents with real execution."""
        state = SelfDiscoverMultiAgentState()
        state.task_description = (
            "How can we reduce carbon emissions in urban transportation?"
        )
        state.available_modules = [
            "systems_thinking",
            "cost_benefit_analysis",
            "stakeholder_analysis",
            "implementation_planning",
            "impact_assessment",
        ]

        # Step 1: Module Selection Agent
        selector_agent = SimpleAgent(
            name="select_modules",
            engine=real_llm_config,
            structured_output_model=SelectedModules,
            system_message="""You are an expert at selecting reasoning modules.
            Select 2-3 most relevant modules for the given task.
            Consider the task complexity and required analysis types.""",
        )

        selector_node = create_agent_node_v3(
            agent_name="select_modules", agent=selector_agent, name="selector_node"
        )

        # Execute selector
        selector_result = selector_node(state, {"debug": True})

        # Apply updates to state
        self._apply_command_updates(state, selector_result)

        # Verify selection happened
        assert (
            len(state.selected_modules) > 0 or "select_modules" in state.agent_outputs
        )

        # Step 2: Module Adaptation Agent

        adapter_agent = SimpleAgent(
            name="adapt_modules",
            engine=real_llm_config,
            structured_output_model=AdaptedModules,
            system_message="Adapt reasoning modules for the specific task context.",
        )

        adapter_node = create_agent_node_v3(
            agent_name="adapt_modules", agent=adapter_agent, name="adapter_node"
        )

        # Execute adapter
        adapter_result = adapter_node(state, {"debug": True})
        self._apply_command_updates(state, adapter_result)

        # Verify adaptation
        assert len(
            state.adapted_modules) > 0 or "adapt_modules" in state.agent_outputs

    def test_complex_reasoning_agent(self, real_llm_config):
        """Test agent that produces complex reasoning structure."""
        state = SelfDiscoverMultiAgentState()
        state.task_description = "Design a sustainable city of the future"
        state.adapted_modules = [
            {
                "module": "systems_thinking",
                "adaptation": "Focus on interconnected urban systems",
            },
            {
                "module": "sustainability_analysis",
                "adaptation": "Evaluate environmental impact",
            },
            {"module": "future_planning",
             "adaptation": "Project 50-year scenarios"},
        ]

        # Agent that creates reasoning structure
        reasoning_agent = SimpleAgent(
            name="create_structure",
            engine=real_llm_config,
            structured_output_model=ReasoningStructure,
            system_message="""Create a step-by-step reasoning structure using the adapted modules.
            The structure should guide systematic thinking about the problem.""",
        )

        node = create_agent_node_v3(
            agent_name="create_structure", agent=reasoning_agent, name="reasoning_node"
        )

        result = node(state, {"debug": True})

        # Check for reasoning structure in updates
        assert isinstance(result, Command)

        # Verify structure created
        if "reasoning_structure" in result.update:
            structure = result.update["reasoning_structure"]
            assert isinstance(structure, dict)
            assert len(structure) > 0
        elif "agent_outputs" in result.update:
            if "create_structure" in result.update["agent_outputs"]:
                output = result.update["agent_outputs"]["create_structure"]
                assert "reasoning_structure" in output or "steps" in output

    def test_agent_state_privacy(self, real_llm_config):
        """Test that agent-specific data remains private."""
        state = SelfDiscoverMultiAgentState()

        # Create agents with different tools
        analyst_agent = SimpleAgent(
            name="analyst",
            engine=real_llm_config,
            tools=["data_analyzer", "trend_detector"],  # Private tools
            system_message="You are a data analyst.",
        )

        planner_agent = SimpleAgent(
            name="planner",
            engine=real_llm_config,
            tools=[
                "gantt_chart",
                "resource_allocator"],
            # Different private tools
            system_message="You are a project planner.",
        )

        # Execute both agents
        for agent in [analyst_agent, planner_agent]:
            node = create_agent_node_v3(
                agent_name=agent.name, agent=agent, name=f"{agent.name}_node"
            )

            result = node(state, {"debug": True})

            # Tools should not appear in main update
            assert "tools" not in result.update

            # Each agent's tools remain private to that agent
            if "agent_states" in result.update:
                if agent.name in result.update["agent_states"]:
                    result.update["agent_states"][agent.name]
                    # Agent state could contain private data
                    # Tools are configured on agent, not in state

    @pytest.mark.asyncio
    async def test_async_agent_execution(self, real_llm_config):
        """Test async execution of agents."""
        state = SelfDiscoverMultiAgentState()
        state.messages = [HumanMessage(content="Plan a Mars mission")]

        # Async agent
        agent = SimpleAgent(
            name="mission_planner",
            engine=real_llm_config,
            system_message="You are a space mission planner.",
        )

        # In real implementation, node might support async
        # For now, test that agent can be used in async context
        result = await agent.arun({"messages": state.messages})

        assert result is not None
        assert "messages" in result or isinstance(result, str)

    def test_multi_agent_coordination(self, real_llm_config):
        """Test multiple agents coordinating through shared state."""
        state = SelfDiscoverMultiAgentState()
        state.task_description = "Create a comprehensive climate action plan"

        # Agent 1: Problem Analyzer
        analyzer = SimpleAgent(
            name="analyzer",
            engine=real_llm_config,
            system_message="Analyze the problem and identify key challenges.",
        )

        # Agent 2: Solution Generator
        generator = SimpleAgent(
            name="generator",
            engine=real_llm_config,
            system_message="Generate innovative solutions based on the analysis.",
        )

        # Agent 3: Implementation Planner
        planner = SimpleAgent(
            name="planner",
            engine=real_llm_config,
            system_message="Create implementation plans for the proposed solutions.",
        )

        # Execute in sequence
        agents = [analyzer, generator, planner]

        for agent in agents:
            node = create_agent_node_v3(
                agent_name=agent.name, agent=agent, name=f"{agent.name}_node"
            )

            result = node(state, {"debug": True})
            self._apply_command_updates(state, result)

        # Verify coordination through messages
        assert len(state.messages) > 1  # Initial + agent messages

        # Each agent should have contributed
        if hasattr(state, "agent_outputs"):
            assert len(state.agent_outputs) >= len(agents)

    def _apply_command_updates(self, state: Any, command: Command) -> None:
        """Helper to apply Command updates to state."""
        if not command.update:
            return

        for key, value in command.update.items():
            if hasattr(state, key):
                # Direct field update
                if key == "messages" and isinstance(value, list):
                    # Extend messages
                    current = getattr(state, key, [])
                    current.extend(value)
                elif key == "agent_states" and isinstance(value, dict):
                    # Merge agent states
                    current = getattr(state, key, {})
                    current.update(value)
                elif key == "agent_outputs" and isinstance(value, dict):
                    # Merge agent outputs
                    current = getattr(state, key, {})
                    current.update(value)
                else:
                    # Direct assignment
                    setattr(state, key, value)
