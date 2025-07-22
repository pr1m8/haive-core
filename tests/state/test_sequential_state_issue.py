#!/usr/bin/env python3
"""Test script to identify the core state management issues in multi-agent systems.

This script demonstrates the problems with:
1. Global state vs typed state schemas
2. Agent-specific input/output schema mismatches
3. State projection/mapping issues
"""

import contextlib
import sys
from typing import Any

from haive.agents.simple.agent import SimpleAgent
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import Field

from haive.core.engine.llm.factories.litellm_factory import LiteLLMFactory
from haive.core.schema.agent_schema_composer import AgentSchemaComposer, BuildMode
from haive.core.schema.state_schema import StateSchema

sys.path.insert(0, "/home/will/Projects/haive/backend/haive")


# Core imports


class PlannerState(StateSchema):
    """State schema for a planner agent."""

    messages: list[BaseMessage] = Field(default_factory=list)
    task: str = Field(default="")
    plan: str | None = Field(default=None)
    plan_steps: list[str] = Field(default_factory=list)


class ExecutorState(StateSchema):
    """State schema for an executor agent."""

    messages: list[BaseMessage] = Field(default_factory=list)
    plan: str = Field(default="")
    execution_result: str | None = Field(default=None)
    completed_steps: list[str] = Field(default_factory=list)


class GlobalMultiAgentState(StateSchema):
    """Global state that attempts to combine all agent states."""

    # Shared fields
    messages: list[BaseMessage] = Field(default_factory=list)

    # Planner fields
    task: str = Field(default="")
    plan: str | None = Field(default=None)
    plan_steps: list[str] = Field(default_factory=list)

    # Executor fields
    execution_result: str | None = Field(default=None)
    completed_steps: list[str] = Field(default_factory=list)

    # Coordination fields
    current_agent: str = Field(default="planner")
    agent_outputs: dict[str, Any] = Field(default_factory=dict)


def test_schema_composition_issues():
    """Test the schema composition issues."""
    # Create simple agents with different state schemas
    factory = LiteLLMFactory()
    llm = factory.create_engine(model="gpt-4o-mini")

    # Create planner agent
    planner = SimpleAgent(name="planner", engine=llm, state_schema=PlannerState)

    # Create executor agent
    executor = SimpleAgent(name="executor", engine=llm, state_schema=ExecutorState)

    agents = [planner, executor]

    # Test AgentSchemaComposer
    composer = AgentSchemaComposer(agents)

    # Test SEQUENCE mode
    with contextlib.suppress(Exception):
        composer.build_schema(BuildMode.SEQUENCE)

    # Test PARALLEL mode
    with contextlib.suppress(Exception):
        composer.build_schema(BuildMode.PARALLEL)


def test_state_projection_issues():
    """Test state projection and agent execution issues."""
    # Create a global state instance
    global_state = GlobalMultiAgentState(
        messages=[HumanMessage(content="Create a plan for building a web app")],
        task="Build a web app",
        current_agent="planner",
    )

    # Create agents
    factory = LiteLLMFactory()
    llm = factory.create_engine(model="gpt-4o-mini")

    planner = SimpleAgent(name="planner", engine=llm, state_schema=PlannerState)

    # Problem 1: Agent expects PlannerState but gets GlobalMultiAgentState
    try:
        # This should fail because the agent expects PlannerState
        # but global_state is GlobalMultiAgentState
        planner_input = {"messages": global_state.messages, "task": global_state.task}

        # This will likely fail due to schema mismatch
        planner.invoke(planner_input)

    except Exception:
        pass


def test_proposed_solution():
    """Test a proposed solution with proper state projection."""

    # Solution: Create state projection functions
    def project_global_to_planner(global_state: GlobalMultiAgentState) -> PlannerState:
        """Project global state to planner state."""
        return PlannerState(
            messages=global_state.messages,
            task=global_state.task,
            plan=global_state.plan,
            plan_steps=global_state.plan_steps,
        )

    def merge_planner_to_global(
        planner_result: PlannerState, global_state: GlobalMultiAgentState
    ) -> GlobalMultiAgentState:
        """Merge planner result back to global state."""
        return GlobalMultiAgentState(
            messages=planner_result.messages,
            task=planner_result.task,
            plan=planner_result.plan,
            plan_steps=planner_result.plan_steps,
            # Preserve other fields
            execution_result=global_state.execution_result,
            completed_steps=global_state.completed_steps,
            current_agent="executor",  # Move to next agent
            agent_outputs={
                **global_state.agent_outputs,
                "planner": planner_result.model_dump(),
            },
        )

    # Test the projection approach
    global_state = GlobalMultiAgentState(
        messages=[HumanMessage(content="Create a plan for building a web app")],
        task="Build a web app",
        current_agent="planner",
    )

    # Create agent
    factory = LiteLLMFactory()
    llm = factory.create_engine(model="gpt-4o-mini")

    planner = SimpleAgent(name="planner", engine=llm, state_schema=PlannerState)

    try:
        # Step 1: Project global state to agent state
        planner_state = project_global_to_planner(global_state)

        # Step 2: Execute agent with proper state
        planner_input = planner_state.model_dump()

        result = planner.invoke(planner_input)

        # Step 3: Merge result back to global state
        planner_result = PlannerState(**result) if isinstance(result, dict) else result

        merge_planner_to_global(planner_result, global_state)

    except Exception:
        import traceback

        traceback.print_exc()


if __name__ == "__main__":

    test_schema_composition_issues()
    test_state_projection_issues()
    test_proposed_solution()
