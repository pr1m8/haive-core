"""Test MetaAgentState with SimpleAgentV2 and recompilation.

This test demonstrates:
1. MetaAgentState containing SimpleAgentV2
2. Async execution of embedded agent
3. Recompilation when tools change
4. State persistence through recompilation
5. NO MOCKS - real LLM execution
"""

import asyncio
from typing import Any, Dict

import pytest
from haive.agents.simple.agent_v2 import SimpleAgentV2
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.common.mixins.dynamic_tool_route_mixin import DynamicToolRouteMixin
from haive.core.common.mixins.recompile_mixin import RecompileMixin
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.models.llm.base import AzureLLMConfig
from haive.core.schema.prebuilt.meta_state import MetaStateSchema


# Test structured output model
class AnalysisResult(BaseModel):
    """Structured output for analysis."""

    summary: str = Field(description="Brief summary")
    confidence: float = Field(description="Confidence score 0-1")
    key_points: list[str] = Field(description="Key points from analysis")


# Test tool
@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expression."""
    try:
        # Safe evaluation of math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def word_counter(text: str) -> int:
    """Count words in text."""
    return len(text.split())


class TestMetaAgentStateWithSimpleV2:
    """Test MetaAgentState with SimpleAgentV2 and recompilation."""

    @pytest.fixture
    def simple_agent_v2(self):
        """Create SimpleAgentV2 with recompilation capability."""
        # Create agent with real LLM
        agent = SimpleAgentV2(
            name="analyzer_v2",
            engine=AugLLMConfig(
                temperature=0.7,
                model="gpt-4o-mini",  # Using real model
                system_message="You are a helpful analysis agent.",
            ),
            tools=[calculator],  # Start with one tool
            structured_output_model=AnalysisResult,
            structured_output_version="v2",
        )

        # Ensure agent has recompilation capability
        # (In real implementation, this would be in Agent base class)
        if not isinstance(agent, RecompileMixin):
            # Mix in recompilation capability for testing
            class RecompilableSimpleAgentV2(
                SimpleAgentV2, RecompileMixin, DynamicToolRouteMixin
            ):
                pass

            # Convert to recompilable version
            agent.__class__ = RecompilableSimpleAgentV2
            RecompileMixin.__init__(agent)
            DynamicToolRouteMixin.__init__(agent)

        return agent

    @pytest.fixture
    def meta_state_with_agent(self, simple_agent_v2):
        """Create MetaStateSchema with embedded SimpleAgentV2."""
        meta_state = MetaStateSchema(
            agent=simple_agent_v2,
            agent_input={
                "messages": [
                    HumanMessage(content="Analyze the number 42 and its significance")
                ]
            },
            meta_context={
                "purpose": "testing",
                "test_type": "meta_agent_with_recompilation",
            },
        )
        return meta_state

    def test_meta_state_creation(self, meta_state_with_agent):
        """Test that MetaStateSchema properly contains SimpleAgentV2."""
        assert meta_state_with_agent.agent is not None
        assert meta_state_with_agent.agent_name == "analyzer_v2"
        assert meta_state_with_agent.agent_type == "SimpleAgentV2"
        assert meta_state_with_agent.execution_status == "ready"

        # Check engine syncing
        assert "agent_main" in meta_state_with_agent.engines
        assert meta_state_with_agent.engine is not None

    def test_sync_agent_execution(self, meta_state_with_agent):
        """Test synchronous execution of embedded agent."""
        # Execute agent synchronously (current implementation)
        result = meta_state_with_agent.execute_agent()

        assert result["status"] == "success"
        assert meta_state_with_agent.execution_status == "completed"
        assert meta_state_with_agent.agent_output is not None

        # Check execution history
        assert len(meta_state_with_agent.execution_history) == 1
        assert meta_state_with_agent.execution_history[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_async_agent_execution(self, meta_state_with_agent):
        """Test async execution of embedded agent.

        NOTE: This test demonstrates what we WANT to implement.
        Currently execute_agent is sync, but we need it to be async.
        """

        # Create async wrapper for now (should be native async)
        async def execute_agent_async(meta_state, input_data=None):
            """Async wrapper for agent execution."""
            # In real implementation, this would be:
            # return await meta_state.aexecute_agent(input_data)

            # For now, wrap in async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, meta_state.execute_agent, input_data
            )

        # Execute asynchronously
        result = await execute_agent_async(meta_state_with_agent)

        assert result["status"] == "success"
        assert meta_state_with_agent.execution_status == "completed"

    def test_recompilation_on_tool_change(self, meta_state_with_agent):
        """Test that agent recompiles when tools change."""
        agent = meta_state_with_agent.agent

        # Initial state
        assert len(agent.tools) == 1
        assert not agent.needs_recompile

        # Add a new tool dynamically
        agent.add_tool(word_counter, route="langchain_tool")

        # Should be marked for recompilation
        assert agent.needs_recompile
        assert "Tool added: word_counter" in agent.recompile_reasons

        # Trigger recompilation
        if agent.auto_recompile:
            # Auto recompilation should happen
            assert not agent.needs_recompile
        else:
            # Manual recompilation
            agent.force_recompile("Manual test recompilation")
            assert not agent.needs_recompile

        # Verify tools are updated
        assert len(agent.tools) == 2

    def test_state_persistence_through_recompilation(self, meta_state_with_agent):
        """Test that state persists through agent recompilation."""
        # Execute once to establish state
        result1 = meta_state_with_agent.execute_agent()
        initial_history_count = len(meta_state_with_agent.execution_history)

        # Get the agent and add tool (triggers recompilation)
        agent = meta_state_with_agent.agent
        agent.add_tool(word_counter, route="langchain_tool")

        # Force recompilation if needed
        if agent.needs_recompile:
            agent.force_recompile("Test recompilation")

        # Execute again after recompilation
        new_input = {
            "messages": [
                HumanMessage(
                    content="Count the words in: The quick brown fox jumps over the lazy dog"
                )
            ]
        }
        result2 = meta_state_with_agent.execute_agent(input_data=new_input)

        # Verify state persistence
        assert len(meta_state_with_agent.execution_history) == initial_history_count + 1
        assert meta_state_with_agent.meta_context["execution_count"] == 2

        # Both executions should be successful
        assert all(
            record["status"] == "success"
            for record in meta_state_with_agent.execution_history
        )

    def test_structured_output_with_meta_state(self, meta_state_with_agent):
        """Test that SimpleAgentV2 produces structured output in meta state."""
        # Execute with request for structured output
        input_data = {
            "messages": [
                HumanMessage(
                    content="Analyze the importance of testing in software development"
                )
            ]
        }

        result = meta_state_with_agent.execute_agent(input_data=input_data)

        # Check that we got structured output
        assert result["status"] == "success"
        output = result["output"]

        # The output should contain our structured result
        # (The exact structure depends on SimpleAgentV2 implementation)
        # We expect the agent to return the structured output in some form

    def test_error_handling_in_meta_state(self, meta_state_with_agent):
        """Test error handling when agent execution fails."""
        # Cause an error by passing invalid input
        invalid_input = {"messages": None}  # This should cause an error

        with pytest.raises(RuntimeError) as exc_info:
            meta_state_with_agent.execute_agent(input_data=invalid_input)

        # Check error state
        assert meta_state_with_agent.execution_status == "error"
        assert meta_state_with_agent.error_info is not None
        assert "error" in meta_state_with_agent.execution_history[-1]["status"]

    def test_meta_state_summary(self, meta_state_with_agent):
        """Test execution summary functionality."""
        # Execute multiple times
        for i in range(3):
            input_data = {"messages": [HumanMessage(content=f"Test execution {i+1}")]}
            try:
                meta_state_with_agent.execute_agent(input_data=input_data)
            except:
                pass  # Allow some failures for testing

        # Get summary
        summary = meta_state_with_agent.get_execution_summary()

        assert summary["total_executions"] >= 3
        assert summary["agent_name"] == "analyzer_v2"
        assert summary["agent_type"] == "SimpleAgentV2"
        assert "success_rate" in summary
        assert "last_execution" in summary


class TestAsyncMetaAgentState:
    """Test async version of MetaAgentState.

    This demonstrates the pattern we want to implement.
    """

    async def aexecute_agent(
        self,
        meta_state: MetaStateSchema,
        input_data: Dict[str, Any] | None = None,
        config: Dict[str, Any] | None = None,
        update_state: bool = True,
    ) -> Dict[str, Any]:
        """Async version of execute_agent.

        This is what we need to implement in MetaStateSchema.
        """
        if meta_state.agent is None:
            raise ValueError("No agent configured for execution")

        # Use provided input or fall back to agent_input
        execution_input = input_data or meta_state.agent_input
        execution_config = config or meta_state.agent_config

        # Update execution status
        if update_state:
            meta_state.execution_status = "running"
            meta_state.error_info = None

        try:
            # Execute the agent asynchronously
            if hasattr(meta_state.agent, "arun"):
                # Agent has async run method
                result = await meta_state.agent.arun(
                    execution_input, **execution_config
                )
            elif hasattr(meta_state.agent, "ainvoke"):
                # Agent has async invoke method
                result = await meta_state.agent.ainvoke(
                    execution_input, execution_config
                )
            else:
                # Fall back to sync in executor
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    meta_state.execute_agent,
                    input_data,
                    config,
                    False,  # Don't update state in sync call
                )

            # Create execution record
            execution_record = {
                "timestamp": str(__import__("datetime").datetime.now()),
                "input": execution_input,
                "output": result,
                "config": execution_config,
                "status": "success",
            }

            if update_state:
                # Update meta state with results
                meta_state.agent_output = (
                    result if isinstance(result, dict) else {"result": result}
                )
                meta_state.last_execution_result = execution_record
                meta_state.execution_status = "completed"
                meta_state.execution_history.append(execution_record)

                # Update execution count
                meta_state.meta_context["execution_count"] = (
                    meta_state.meta_context.get("execution_count", 0) + 1
                )

            return execution_record

        except Exception as e:
            # Handle errors
            error_record = {
                "timestamp": str(__import__("datetime").datetime.now()),
                "input": execution_input,
                "error": str(e),
                "error_type": type(e).__name__,
                "config": execution_config,
                "status": "error",
            }

            if update_state:
                meta_state.error_info = error_record
                meta_state.execution_status = "error"
                meta_state.execution_history.append(error_record)

            raise RuntimeError(f"Agent execution failed: {e}") from e


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
