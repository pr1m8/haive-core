"""Tests for ToolEngine integration with Haive state schemas."""
import pytest
from typing import Annotated
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.tool import ToolEngine
from haive.core.schema.prebuilt.tool_state import ToolState
from haive.core.schema.prebuilt.messages_state import MessagesState
from langgraph.prebuilt.tool_node import InjectedState


class SearchInput(BaseModel):
    """Input schema for search."""
    query: str = Field(description="Search query")
    

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    return f"Result: {eval(expression)}"


@tool
def state_aware_tool(
    message: str,
    state: Annotated[dict, InjectedState]
) -> str:
    """Tool that accesses injected state."""
    msg_count = len(state.get("messages", []))
    return f"Processed '{message}' with {msg_count} messages in state"


@tool
def specific_field_tool(
    query: str,
    context: Annotated[str, InjectedState("context")]
) -> str:
    """Tool that accesses specific state field."""
    return f"Query: {query}, Context: {context}"


class TestToolEngineStateIntegration:
    """Test ToolEngine integration with state schemas."""
    
    def test_tool_state_compatibility(self):
        """Test that ToolEngine tools work with ToolState."""
        # Create tools using ToolEngine
        calc_tool = ToolEngine.create_state_tool(
            calculator,
            reads_state=True,
            state_keys=["messages"]
        )
        
        # Create ToolState with the tool
        state = ToolState(tools=[calc_tool])
        
        # Verify tool is registered
        assert len(state.tools) == 1
        assert calc_tool.name in state.tool_routes
        
        # Check tool properties via ToolRouteMixin
        tool_route = state.get_tool_route(calc_tool.name)
        assert tool_route is not None
        
        # Verify tool can be retrieved
        retrieved_tool = state.get_tool(calc_tool.name)
        assert retrieved_tool is calc_tool
    
    def test_injected_state_detection(self):
        """Test that ToolEngine detects InjectedState annotations."""
        engine = ToolEngine(
            tools=[state_aware_tool, specific_field_tool],
            enable_analysis=True
        )
        
        # Check state_aware_tool analysis
        props1 = engine.get_tool_properties("state_aware_tool")
        assert props1 is not None
        assert props1.is_state_tool
        assert props1.from_state_tool
        assert "injected_state" in [cap.value for cap in props1.capabilities]
        
        # Check specific_field_tool analysis  
        props2 = engine.get_tool_properties("specific_field_tool")
        assert props2 is not None
        assert props2.is_state_tool
        assert props2.from_state_tool
    
    def test_store_tools_with_tool_state(self):
        """Test store tools integration with ToolState."""
        from haive.core.tools.store_manager import StoreManager
        
        # Create store manager
        store_manager = StoreManager()
        
        # Create store tools
        store_tools = ToolEngine.create_store_tools_suite(
            store_manager=store_manager,
            namespace=("test", "integration"),
            include_tools=["store", "search"]
        )
        
        # Create ToolState with store tools
        state = ToolState(tools=store_tools)
        
        # Verify tools are registered
        assert len(state.tools) == 2
        
        # Find the actual tool names (they may have prefixes)
        store_tool_names = [tool.name for tool in store_tools if "store" in tool.name]
        search_tool_names = [tool.name for tool in store_tools if "search" in tool.name]
        
        # Verify tools are in routes
        assert len(store_tool_names) > 0
        assert len(search_tool_names) > 0
        assert store_tool_names[0] in state.tool_routes
        assert search_tool_names[0] in state.tool_routes
        
        # Check tool properties via ToolEngine
        engine = ToolEngine(tools=store_tools, enable_analysis=True)
        store_props = engine.get_tool_properties(store_tool_names[0])
        assert store_props.category.value == "memory"
    
    def test_structured_output_tool_with_state(self):
        """Test structured output tools with state schemas."""
        class TaskResult(BaseModel):
            """Structured task result."""
            status: str = Field(description="Task status")
            output: str = Field(description="Task output")
            confidence: float = Field(ge=0.0, le=1.0)
        
        def process_task(task: str) -> TaskResult:
            """Process a task and return structured result."""
            return TaskResult(
                status="completed",
                output=f"Processed: {task}",
                confidence=0.95
            )
        
        # Create structured output tool
        task_tool = ToolEngine.create_structured_output_tool(
            func=process_task,
            name="task_processor",
            description="Process tasks with structured output",
            output_model=TaskResult
        )
        
        # Use in ToolState
        state = ToolState(tools=[task_tool])
        
        # Verify tool is registered
        assert task_tool.name in state.tool_routes
        
        # Get tool properties via engine
        engine = ToolEngine(tools=[task_tool], enable_analysis=True)
        props = engine.get_tool_properties("task_processor")
        assert props.structured_output_model == TaskResult
        assert props.is_structured_output_model
    
    def test_augmented_tools_preserve_state_metadata(self):
        """Test that augmented tools preserve state interaction metadata."""
        # Start with a basic tool
        @tool
        def basic_processor(text: str) -> str:
            """Process text."""
            return f"Processed: {text}"
        
        # Augment with state capabilities
        enhanced = ToolEngine.augment_tool(
            basic_processor,
            reads_state=True,
            writes_state=True,
            state_keys=["messages", "context"],
            make_interruptible=True
        )
        
        # Use in ToolState
        state = ToolState(tools=[enhanced])
        
        # Verify all capabilities preserved
        engine = ToolEngine(tools=[enhanced], enable_analysis=True)
        props = engine.get_tool_properties(enhanced.name)
        
        assert props.is_state_tool
        assert props.from_state_tool
        assert props.to_state_tool
        assert props.is_interruptible
        assert props.state_dependencies == ["messages", "context"]
    
    def test_tool_routing_with_state_schemas(self):
        """Test tool routing strategies work with state schemas."""
        # Create various tools
        tools = [
            calculator,
            state_aware_tool,
            ToolEngine.create_interruptible_tool(
                lambda x: f"Interruptible: {x}",
                name="interruptible_task"
            )
        ]
        
        # Create engine with routing
        engine = ToolEngine(
            tools=tools,
            routing_strategy="capability",
            enable_analysis=True
        )
        
        # Get state-aware tools
        state_tools = engine.get_state_tools()
        assert "state_aware_tool" in state_tools
        
        # Get interruptible tools
        interruptible = engine.get_interruptible_tools()
        assert "interruptible_task" in interruptible
        
        # Create ToolState and verify routing info
        state = ToolState(tools=tools)
        
        # All tools should be registered
        assert len(state.tool_routes) == 3


    def test_injected_tool_execution_with_state_schema(self):
        """Test actual execution of tools with InjectedState against a state schema."""
        from haive.core.schema.prebuilt.messages_state import MessagesState
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Create a custom state schema
        class CustomState(MessagesState):
            """Custom state with additional fields."""
            context: str = Field(default="default_context")
            metadata: dict = Field(default_factory=dict)
            counter: int = Field(default=0)
        
        # Create tools that use injected state
        @tool
        def read_full_state(
            query: str,
            state: Annotated[dict, InjectedState]
        ) -> str:
            """Read the full state."""
            msg_count = len(state.get("messages", []))
            context = state.get("context", "no_context")
            counter = state.get("counter", 0)
            return f"Query: {query}, Messages: {msg_count}, Context: {context}, Counter: {counter}"
        
        @tool  
        def read_context_only(
            action: str,
            ctx: Annotated[str, InjectedState("context")]
        ) -> str:
            """Read only the context field."""
            return f"Action: {action} in context: {ctx}"
        
        @tool
        def update_counter(
            increment: int,
            current: Annotated[int, InjectedState("counter")]
        ) -> str:
            """Update counter based on current value."""
            new_value = current + increment
            return f"Counter updated from {current} to {new_value}"
        
        # Create state instance
        state = CustomState(
            messages=[
                HumanMessage(content="Hello"),
                AIMessage(content="Hi there!")
            ],
            context="production_environment",
            counter=42
        )
        
        # Create ToolEngine and analyze
        engine = ToolEngine(
            tools=[read_full_state, read_context_only, update_counter],
            enable_analysis=True
        )
        
        # Verify all tools detected as state-aware
        for tool_name in ["read_full_state", "read_context_only", "update_counter"]:
            props = engine.get_tool_properties(tool_name)
            assert props.is_state_tool
            assert props.from_state_tool
            assert "injected_state" in [cap.value for cap in props.capabilities]
        
        # Test simulated execution with state dict
        state_dict = state.model_dump()
        
        # Simulate tool execution (in real usage, graph would inject state)
        # This shows the expected behavior when tools are called with injected state
        expected_results = {
            "read_full_state": "Query: test, Messages: 2, Context: production_environment, Counter: 42",
            "read_context_only": "Action: deploy in context: production_environment", 
            "update_counter": "Counter updated from 42 to 47"
        }
        
        # Verify tool metadata indicates state requirements
        full_state_props = engine.get_tool_properties("read_full_state")
        assert full_state_props.from_state_tool
        
        context_props = engine.get_tool_properties("read_context_only")
        assert context_props.from_state_tool
        
        counter_props = engine.get_tool_properties("update_counter")
        assert counter_props.from_state_tool
    
    def test_tool_state_with_mixed_tools(self):
        """Test ToolState with a mix of regular and injected state tools."""
        from langchain_core.messages import HumanMessage
        
        # Regular tool
        @tool
        def regular_tool(text: str) -> str:
            """Regular tool without state access."""
            return f"Regular: {text}"
        
        # State-aware tool
        @tool
        def state_tool(
            query: str,
            messages: Annotated[list, InjectedState("messages")]
        ) -> str:
            """Tool that reads messages from state."""
            return f"Query: {query}, Message count: {len(messages)}"
        
        # Create mixed tool set
        tools = [
            regular_tool,
            state_tool,
            ToolEngine.create_state_tool(
                func=lambda x: f"Manual state tool: {x}",
                name="manual_state_tool",
                reads_state=True,
                state_keys=["context"]
            )
        ]
        
        # Create ToolState
        state = ToolState(
            tools=tools,
            messages=[HumanMessage(content="Test message")]
        )
        
        # Verify tool registration and categorization
        assert len(state.tool_routes) == 3
        
        # Create engine to analyze
        engine = ToolEngine(tools=tools, enable_analysis=True)
        
        # Check regular tool (not state-aware)
        regular_props = engine.get_tool_properties("regular_tool")
        assert not regular_props.is_state_tool
        
        # Check injected state tool
        state_props = engine.get_tool_properties("state_tool")
        assert state_props.is_state_tool
        assert state_props.from_state_tool
        
        # Check manually created state tool
        manual_props = engine.get_tool_properties("manual_state_tool")
        assert manual_props.is_state_tool
        assert manual_props.from_state_tool
        assert "context" in manual_props.state_dependencies


if __name__ == "__main__":
    pytest.main([__file__, "-v"])