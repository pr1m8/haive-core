"""Test AgentNodeV3 in isolation to understand its behavior.

This test demonstrates:
1. How AgentNodeV3 projects state from MultiAgentState to agent-specific schemas
2. How structured output agents update fields directly
3. How message-based agents use agent_outputs pattern
4. The state isolation and sharing mechanisms
"""

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from haive.agents.simple.agent_v3 import SimpleAgentV3
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.graph.node.agent_node_v3 import create_agent_node_v3
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Define structured output models
class ResearchFindings(BaseModel):
    """Research agent output."""

    findings: list[str] = Field(description="Key findings from research")
    sources: list[str] = Field(description="Sources consulted")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")


class AnalysisResult(BaseModel):
    """Analysis agent output."""

    summary: str = Field(description="Summary of analysis")
    insights: list[str] = Field(description="Key insights discovered")
    recommendations: list[str] = Field(description="Action recommendations")
    risk_level: str = Field(description="Risk assessment: low/medium/high")


def test_agent_node_v3_isolated():
    """Test AgentNodeV3 behavior in isolation."""

    # Create agents with different output patterns
    research_agent = SimpleAgentV3(
        name="researcher",
        engine=AugLLMConfig(
            temperature=0.7,
            system_message="You are a research agent. Provide findings about the topic.",
            structured_output_model=ResearchFindings,
        ),
    )

    analysis_agent = SimpleAgentV3(
        name="analyzer",
        engine=AugLLMConfig(
            temperature=0.3,
            system_message="You are an analysis agent. Analyze the research findings.",
            structured_output_model=AnalysisResult,
        ),
    )

    # Message-based agent (no structured output)
    summary_agent = SimpleAgentV3(
        name="summarizer",
        engine=AugLLMConfig(
            temperature=0.5,
            system_message="You are a summary agent. Summarize the conversation.",
        ),
    )

    # Initialize MultiAgentState
    state = MultiAgentState(
        agents={
            "researcher": research_agent,
            "analyzer": analysis_agent,
            "summarizer": summary_agent,
        }
    )
    state.messages = [HumanMessage(content="Research AI safety and provide an analysis")]

    # Test 1: Research Agent with Structured Output

    research_node = create_agent_node_v3("researcher")
    result1 = research_node(state, {"debug": True})

    for key, value in result1.update.items():
        if key == "agent_states" or isinstance(value, list):
            pass
        else:
            pass

    # Apply updates to state
    for key, value in result1.update.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # Test 2: Analysis Agent Reading Research Output

    # Show what fields are available in state now
    for field in ["findings", "sources", "confidence"]:
        if hasattr(state, field):
            value = getattr(state, field)

    analysis_node = create_agent_node_v3("analyzer")
    result2 = analysis_node(state, {"debug": True})

    for key, value in result2.update.items():
        if key == "agent_states" or isinstance(value, list):
            pass
        else:
            pass

    # Apply updates
    for key, value in result2.update.items():
        if hasattr(state, key):
            setattr(state, key, value)

    # Test 3: Message-based Agent

    summary_node = create_agent_node_v3("summarizer")
    result3 = summary_node(state, {"debug": True})

    for key, value in result3.update.items():
        if key in {"agent_outputs", "agent_states"} or (
            key == "messages" and isinstance(value, list)
        ):
            pass

    # Final State Summary

    for field in [
        "findings",
        "sources",
        "confidence",
        "summary",
        "insights",
        "recommendations",
        "risk_level",
    ]:
        if hasattr(state, field):
            value = getattr(state, field)
            if isinstance(value, list):
                pass
            else:
                pass

    for _agent_name, _agent_state in state.agent_states.items():
        pass

    if hasattr(state, "agent_outputs"):
        for _agent_name, _output in state.agent_outputs.items():
            pass


def test_state_projection_details():
    """Test the state projection mechanism in detail."""

    # Create agent with specific state requirements
    agent = SimpleAgentV3(
        name="test_agent",
        engine=AugLLMConfig(
            temperature=0.5,
            system_message="You are a test agent.",
        ),
    )

    # Create state with various fields
    state = MultiAgentState(agents={"test_agent": agent})
    state.messages = [HumanMessage(content="Test message")]

    # Add custom agent state
    state.agent_states["test_agent"] = {
        "private_field": "agent-specific data",
        "counter": 42,
        "status": "ready",
    }

    # Add some shared fields
    state.shared_context = {"global_config": "value"}

    # Create node with custom shared fields
    node = create_agent_node_v3(
        "test_agent",
        shared_fields=["messages", "shared_context"],  # Custom shared fields
        project_state=True,
    )

    # Manually test projection
    projected = node._project_state_for_agent(state, agent)

    for _key, value in projected.items():
        if isinstance(value, list):
            pass
        else:
            pass


if __name__ == "__main__":
    # Run synchronous test
    test_agent_node_v3_isolated()

    # Run projection test
    test_state_projection_details()
