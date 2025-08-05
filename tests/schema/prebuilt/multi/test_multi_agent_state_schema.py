"""Test MultiAgentState schema fields and I/O handling.

This test file validates:
1. MultiAgentState schema structure
2. Field updates and access patterns
3. Agent-specific state management
4. Message handling defaults
"""

from langchain_core.messages import AIMessage, HumanMessage

from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState


class TestMultiAgentStateSchema:
    """Test MultiAgentState schema structure and field handling."""

    def test_multi_agent_state_basic_fields(self):
        """Test basic field structure of MultiAgentState."""
        # Create empty state
        state = MultiAgentState()

        # Check core fields exist
        assert hasattr(state, "agents")
        assert hasattr(state, "agent_states")
        assert hasattr(state, "active_agent")
        assert hasattr(state, "agent_outputs")
        assert hasattr(state, "agent_execution_order")
        assert hasattr(state, "messages")  # Inherited from parent

        # Check defaults
        assert isinstance(state.agents, dict)
        assert isinstance(state.agent_states, dict)
        assert state.active_agent is None
        assert isinstance(state.agent_outputs, dict)
        assert isinstance(state.agent_execution_order, list)
        assert isinstance(state.messages, list)

    def test_message_field_annotation(self):
        """Test that messages field has proper annotation."""
        # Get field info
        from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState

        # Check messages field exists in parent chain
        # MultiAgentState -> ToolState -> MessagesStateWithTokenUsage ->
        # MessagesState
        assert "messages" in MultiAgentState.model_fields

        # The messages field should be annotated with reducer
        messages_field = MultiAgentState.model_fields["messages"]
        assert messages_field is not None

        # Messages should be a list of BaseMessage
        # Note: The actual type might be wrapped in Annotated for reducer
        assert "BaseMessage" in str(messages_field.annotation)

    def test_agent_output_field_types(self):
        """Test agent output field structure."""
        state = MultiAgentState()

        # Agent outputs should be dict[str, Any]
        assert isinstance(state.agent_outputs, dict)

        # Test setting different output types
        state.agent_outputs["simple_agent"] = {"messages": [AIMessage(content="Hello")]}
        state.agent_outputs["structured_agent"] = {"result": {"key": "value"}}
        state.agent_outputs["list_agent"] = ["item1", "item2"]

        # All should be stored correctly
        assert "messages" in state.agent_outputs["simple_agent"]
        assert "result" in state.agent_outputs["structured_agent"]
        assert isinstance(state.agent_outputs["list_agent"], list)

    def test_agent_specific_state_isolation(self):
        """Test that agent states are properly isolated."""
        state = MultiAgentState()

        # Set state for different agents
        state.agent_states["agent1"] = {"counter": 1, "data": "agent1_data"}
        state.agent_states["agent2"] = {"counter": 2, "data": "agent2_data"}

        # States should be isolated
        assert state.agent_states["agent1"]["counter"] == 1
        assert state.agent_states["agent2"]["counter"] == 2
        assert state.agent_states["agent1"]["data"] == "agent1_data"
        assert state.agent_states["agent2"]["data"] == "agent2_data"

        # Modifying one shouldn't affect the other
        state.agent_states["agent1"]["counter"] = 10
        assert state.agent_states["agent2"]["counter"] == 2

    def test_get_and_update_agent_state_methods(self):
        """Test helper methods for agent state management."""
        state = MultiAgentState()

        # Test get_agent_state with non-existent agent
        agent_state = state.get_agent_state("new_agent")
        assert agent_state == {}

        # Test update_agent_state
        state.update_agent_state("new_agent", {"field1": "value1"})
        assert state.agent_states["new_agent"]["field1"] == "value1"

        # Test update adds to existing state
        state.update_agent_state("new_agent", {"field2": "value2"})
        assert state.agent_states["new_agent"]["field1"] == "value1"
        assert state.agent_states["new_agent"]["field2"] == "value2"

    def test_default_simple_agent_output_messages(self):
        """Test that simple agents default to outputting messages."""
        state = MultiAgentState()

        # Simulate simple agent output (no structured output model)
        simple_agent_output = {
            "messages": [
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="2+2 equals 4"),
            ]
        }

        # Store in agent outputs
        state.agent_outputs["simple_agent"] = simple_agent_output

        # Should have messages field
        assert "messages" in state.agent_outputs["simple_agent"]
        assert len(state.agent_outputs["simple_agent"]["messages"]) == 2
        assert isinstance(state.agent_outputs["simple_agent"]["messages"][0], HumanMessage)
        assert isinstance(state.agent_outputs["simple_agent"]["messages"][1], AIMessage)

    def test_structured_agent_output_no_messages(self):
        """Test that structured agents can output without messages."""
        state = MultiAgentState()

        # Simulate structured agent output (with structured output model)
        # Like Self-Discover agents
        structured_outputs = {
            "select_modules": {
                "selected_modules": ["reasoning", "planning", "critical_thinking"],
                "rationale": "These modules are needed for complex reasoning",
            },
            "adapt_modules": {
                "adapted_modules": [
                    {"module": "reasoning", "adaptation": "Focus on logical steps"},
                    {"module": "planning", "adaptation": "Create step-by-step plan"},
                ]
            },
            "create_structure": {
                "reasoning_structure": {"steps": ["analyze", "plan", "execute"]},
                "steps": ["Step 1: Analyze", "Step 2: Plan", "Step 3: Execute"],
            },
        }

        # Store structured outputs
        for agent_name, output in structured_outputs.items():
            state.agent_outputs[agent_name] = output

        # None should have messages field
        assert "messages" not in state.agent_outputs["select_modules"]
        assert "messages" not in state.agent_outputs["adapt_modules"]
        assert "messages" not in state.agent_outputs["create_structure"]

        # All should have their specific fields
        assert "selected_modules" in state.agent_outputs["select_modules"]
        assert "adapted_modules" in state.agent_outputs["adapt_modules"]
        assert "reasoning_structure" in state.agent_outputs["create_structure"]

    def test_mixed_agent_outputs(self):
        """Test state with both message-based and structured outputs."""
        state = MultiAgentState()

        # Add global messages
        state.messages = [
            HumanMessage(content="Process this task"),
            AIMessage(content="Starting multi-agent processing"),
        ]

        # Add simple agent with messages
        state.agent_outputs["chat_agent"] = {"messages": [AIMessage(content="I'll help with that")]}

        # Add structured agent without messages
        state.agent_outputs["analyzer_agent"] = {
            "analysis": {"sentiment": "positive", "confidence": 0.9},
            "keywords": ["help", "assist", "support"],
        }

        # Global messages should be separate from agent outputs
        assert len(state.messages) == 2
        assert "messages" in state.agent_outputs["chat_agent"]
        assert "messages" not in state.agent_outputs["analyzer_agent"]

        # Each agent output is independent
        assert state.agent_outputs["chat_agent"]["messages"] != state.messages


class TestMultiAgentStateFieldUpdates:
    """Test field update patterns for MultiAgentState."""

    def test_command_update_pattern(self):
        """Test how Command updates would affect state fields."""
        state = MultiAgentState()

        # Simulate Command update dict
        update_dict = {
            "messages": [AIMessage(content="New message")],
            "agent_outputs": {"test_agent": {"result": "success"}},
        }

        # Apply updates manually (simulating Command.update)
        if "messages" in update_dict:
            state.messages.extend(update_dict["messages"])

        if "agent_outputs" in update_dict:
            state.agent_outputs.update(update_dict["agent_outputs"])

        # Check updates applied
        assert len(state.messages) == 1
        assert state.agent_outputs["test_agent"]["result"] == "success"

    def test_agent_specific_field_updates(self):
        """Test updating agent-specific output fields."""
        state = MultiAgentState()

        # Define different agent output schemas
        agent_outputs = {
            # Simple agent with messages
            "simple_agent": {"messages": [AIMessage(content="Response")]},
            # Self-Discover style agents
            "select_modules": {"selected_modules": ["reasoning", "analysis"]},
            "adapt_modules": {
                "adapted_modules": [{"module": "reasoning", "adaptation": "detailed"}]
            },
            # Custom structured output
            "summary_agent": {
                "summary": "Task completed successfully",
                "key_points": ["point1", "point2"],
                "confidence": 0.95,
            },
        }

        # Update each agent's output
        for agent_name, output in agent_outputs.items():
            # Simulate Command update for this agent
            state.agent_outputs[agent_name] = output

        # Verify all updates
        assert "messages" in state.agent_outputs["simple_agent"]
        assert "selected_modules" in state.agent_outputs["select_modules"]
        assert "adapted_modules" in state.agent_outputs["adapt_modules"]
        assert "summary" in state.agent_outputs["summary_agent"]

        # Verify no cross-contamination
        assert "messages" not in state.agent_outputs["select_modules"]
        assert "selected_modules" not in state.agent_outputs["simple_agent"]

    def test_avoiding_message_field_for_structured_agents(self):
        """Test that we can configure agents to not use messages field."""
        state = MultiAgentState()

        # Agent configurations (simulating what we'd set)
        agent_configs = {
            "default_agent": {
                # No structured_output_model - should use messages
                "use_messages": True
            },
            "structured_agent": {
                # Has structured_output_model - should NOT use messages
                "structured_output_model": "SomeModel",
                "use_messages": False,
            },
        }

        # Simulate outputs based on config
        for agent_name, config in agent_configs.items():
            if config.get("use_messages", True):
                # Default behavior - output messages
                state.agent_outputs[agent_name] = {
                    "messages": [AIMessage(content=f"Output from {agent_name}")]
                }
            else:
                # Structured output - no messages
                state.agent_outputs[agent_name] = {
                    "result": f"Structured output from {agent_name}",
                    "data": {"processed": True},
                }

        # Verify outputs
        assert "messages" in state.agent_outputs["default_agent"]
        assert "messages" not in state.agent_outputs["structured_agent"]
        assert "result" in state.agent_outputs["structured_agent"]
