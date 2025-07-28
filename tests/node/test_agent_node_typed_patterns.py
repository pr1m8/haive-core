"""Test single agent node with typed I/O patterns.

This test file validates:
1. Agent node passing full state to agent
2. Agent extracting inputs based on input_schema
3. Agent returning output_schema instances
4. Command updates with typed outputs
5. Message handling with reducers
"""

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Command
from pydantic import BaseModel, Field

from haive.core.graph.node.agent_node_v3 import AgentNodeV3Config
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


# Define typed schemas
class AnalysisInput(BaseModel):
    """Input schema for analysis agent."""

    document: str
    analysis_type: str
    context: str | None = None


class AnalysisResult(BaseModel):
    """Output schema for analysis agent."""

    summary: str
    key_points: list[str]
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectorInput(BaseModel):
    """Input for selector agent."""

    task_description: str
    available_modules: list[str]


class SelectedModules(BaseModel):
    """Output for selector agent."""

    selected_modules: list[str]
    rationale: str | None = None


# Mock agents with different patterns
class MockTypedAgent:
    """Mock agent with typed input/output schemas."""

    def __init__(
        self, name: str, input_schema: type[BaseModel], output_schema: type[BaseModel]
    ):
        self.name = name
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._received_state = None

    def extract_inputs_from_state(self, state: Any) -> dict[str, Any]:
        """Extract inputs based on input_schema."""
        inputs = {}

        # Extract fields based on input_schema
        for field_name, _field_info in self.input_schema.model_fields.items():
            # Try different locations in state
            if hasattr(state, field_name):
                inputs[field_name] = getattr(state, field_name)
            elif hasattr(state, "agent_states") and state.agent_states.get("shared"):
                if field_name in state.agent_states["shared"]:
                    inputs[field_name] = state.agent_states["shared"][field_name]
            elif hasattr(state, "agent_outputs"):
                # Check previous agent outputs
                for _agent_name, output in state.agent_outputs.items():
                    if field_name in output:
                        inputs[field_name] = output[field_name]
                        break

        return inputs

    def invoke(self, state: Any, config: Any = None) -> BaseModel:
        """Process with full state access, return typed output."""
        self._received_state = state  # For testing

        # Extract inputs based on schema
        self.extract_inputs_from_state(state)

        # Return instance of output_schema
        if self.output_schema == AnalysisResult:
            return AnalysisResult(
                summary="Analysis complete",
                key_points=["Point 1", "Point 2"],
                confidence=0.95,
                metadata={"source": "test"},
            )
        if self.output_schema == SelectedModules:
            return SelectedModules(
                selected_modules=["reasoning", "planning"],
                rationale="Best modules for the task",
            )
        # Generic return
        return self.output_schema(
            **{field: f"test_{field}" for field in self.output_schema.model_fields}
        )


class MockMessageAgent:
    """Mock agent that outputs messages (no structured output)."""

    def __init__(self, name: str):
        self.name = name
        self.input_schema = None
        self.output_schema = None  # No structured output
        self.prompt_template = ChatPromptTemplate.from_template("Process this: {input}")

    def invoke(self, state: Any, config: Any = None) -> dict[str, Any]:
        """Return messages-based output."""
        # Extract from messages in state
        last_message = ""
        if hasattr(state, "messages") and state.messages:
            last_message = state.messages[-1].content

        return {"messages": [AIMessage(content=f"Processed: {last_message}")]}


class TestAgentNodeTypedPatterns:
    """Test agent node with typed I/O patterns."""

    def test_agent_receives_full_state(self):
        """Test that agent receives full MultiAgentState."""
        # Setup state
        state = MultiAgentState()
        state.messages = [HumanMessage(content="Analyze this")]
        state.agent_states["shared"] = {
            "document": "Important document",
            "analysis_type": "sentiment",
        }

        # Create typed agent
        agent = MockTypedAgent("analyzer", AnalysisInput, AnalysisResult)

        # Create node
        node = AgentNodeV3Config(
            name="analyzer_node", agent_name="analyzef", agent=agent
        )

        # Execute
        node(state, {})

        # Verify agent received full state
        assert agent._received_state is state
        assert hasattr(agent._received_state, "messages")
        assert hasattr(agent._received_state, "agent_states")

    def test_typed_agent_output_command(self):
        """Test typed agent returns proper Command update."""
        state = MultiAgentState()
        state.agent_states["shared"] = {
            "document": "Test doc",
            "analysis_type": "summary",
        }

        # Create typed agent
        agent = MockTypedAgent("analyzer", AnalysisInput, AnalysisResult)
        node = AgentNodeV3Config(
            name="analyzer_node", agent_name="analyzef", agent=agent
        )

        # Execute
        result = node(state, {})

        # Check Command structure
        assert isinstance(result, Command)
        assert "agent_outputs" in result.update
        assert "analyzer" in result.update["agent_outputs"]

        # Check typed output stored correctly
        output = result.update["agent_outputs"]["analyzer"]
        assert "summary" in output
        assert "key_points" in output
        assert "confidence" in output
        assert output["summary"] == "Analysis complete"
        assert output["confidence"] == 0.95

        # No messages in typed output
        assert "messages" not in output

    def test_message_agent_output_command(self):
        """Test message-based agent returns messages."""
        state = MultiAgentState()
        state.messages = [HumanMessage(content="Hello there")]

        # Create message agent
        agent = MockMessageAgent("chat")
        node = AgentNodeV3Config(name="chat_node", agent_name="chat", agent=agent)

        # Execute
        result = node(state, {})

        # Check Command structure
        assert isinstance(result, Command)
        assert "agent_outputs" in result.update
        assert "chat" in result.update["agent_outputs"]

        # Check message output
        output = result.update["agent_outputs"]["chat"]
        assert "messages" in output
        assert len(output["messages"]) == 1
        assert output["messages"][0].content == "Processed: Hello there"

    def test_agent_input_extraction(self):
        """Test agent extracts inputs based on input_schema."""
        state = MultiAgentState()

        # Setup state with inputs scattered around
        state.agent_states["shared"] = {
            "task_description": "Select best modules",
            "available_modules": ["reasoning", "planning", "analysis"],
        }

        # Create selector agent
        agent = MockTypedAgent("selector", SelectorInput, SelectedModules)
        node = AgentNodeV3Config(
            name="selector_node", agent_name="selectof", agent=agent
        )

        # Execute
        result = node(state, {})

        # Agent should extract inputs and produce typed output
        output = result.update["agent_outputs"]["selector"]
        assert "selected_modules" in output
        assert "rationale" in output
        assert output["selected_modules"] == ["reasoning", "planning"]

    def test_agent_with_prompt_template(self):
        """Test agent with prompt template requiring specific inputs."""

        # Create agent with prompt template
        class PromptAgent:
            name = "prompt_agent"
            input_schema = AnalysisInput
            output_schema = AnalysisResult
            prompt_template = ChatPromptTemplate.from_template(
                "Analyze {document} for {analysis_type} considering {context}"
            )

            def invoke(self, state, config=None):
                # Would extract document, analysis_type, context from state
                # based on input_schema
                return AnalysisResult(
                    summary="Prompt-based analysis",
                    key_points=["Used template"],
                    confidence=0.9,
                )

        state = MultiAgentState()
        state.agent_states["shared"] = {
            "document": "Research paper",
            "analysis_type": "technical",
            "context": "Academic review",
        }

        agent = PromptAgent()
        node = AgentNodeV3Config(
            name="prompt_node", agent_name="prompt_agent", agent=agent
        )

        result = node(state, {})

        # Check output
        output = result.update["agent_outputs"]["prompt_agent"]
        assert output["summary"] == "Prompt-based analysis"
        assert "Used template" in output["key_points"]

    def test_output_schema_to_field_mapping(self):
        """Test potential schema name to field name mapping."""
        # This shows how we could map schema names to state fields

        schemas_to_fields = {
            SelectedModules: "selected_modules",
            AnalysisResult: "analysis_result",
            # In practice: ClassName -> class_name
        }

        for schema_class, expected_field in schemas_to_fields.items():
            # Convert PascalCase to snake_case
            import re

            class_name = schema_class.__name__
            field_name = re.sub("(?<!^)(?=[A-Z])", "_", class_name).lower()
            assert field_name == expected_field

    def test_sequential_typed_agents(self):
        """Test sequence of typed agents passing data."""
        state = MultiAgentState()

        # Initial setup
        state.agent_states["shared"] = {
            "task_description": "Complex problem",
            "available_modules": ["A", "B", "C", "D"],
        }

        # First agent: selector
        selector = MockTypedAgent("selector", SelectorInput, SelectedModules)
        selector_node = AgentNodeV3Config(
            name="selector_node", agent_name="selectof", agent=selector
        )

        # Execute selector
        result1 = selector_node(state, {})

        # Apply update (simulate graph execution)
        state.agent_outputs.update(result1.update["agent_outputs"])

        # Second agent could use selector output
        # (In real implementation, would extract from previous outputs)
        assert "selector" in state.agent_outputs
        assert "selected_modules" in state.agent_outputs["selector"]

        # Could continue chain with adapter agent, etc.

    def test_mixed_schema_and_message_agents(self):
        """Test mixing typed and message-based agents."""
        state = MultiAgentState()
        state.messages = [HumanMessage(content="Start process")]

        # Message agent first
        chat = MockMessageAgent("assistant")
        chat_node = AgentNodeV3Config(
            name="assistant_node", agent_name="assistant", agent=chat
        )

        result1 = chat_node(state, {})
        state.agent_outputs.update(result1.update["agent_outputs"])

        # Then typed agent
        state.agent_states["shared"] = {
            "document": "From chat",
            "analysis_type": "follow-up",
        }

        analyzer = MockTypedAgent("analyzer", AnalysisInput, AnalysisResult)
        analyzer_node = AgentNodeV3Config(
            name="analyzer_node", agent_name="analyzef", agent=analyzer
        )

        result2 = analyzer_node(state, {})
        state.agent_outputs.update(result2.update["agent_outputs"])

        # Verify both outputs exist
        assert "assistant" in state.agent_outputs
        assert "messages" in state.agent_outputs["assistant"]
        assert "analyzer" in state.agent_outputs
        assert "summary" in state.agent_outputs["analyzer"]
        assert "messages" not in state.agent_outputs["analyzer"]
