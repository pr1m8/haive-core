"""Tests for the enhanced ToolEngine with universal typing."""
import pytest
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from haive.core.engine.tool import (
    ToolEngine,
    ToolLike,
    ToolType,
    ToolCategory,
    ToolCapability,
    ToolProperties,
)


class SearchInput(BaseModel):
    """Input for search tool."""
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")


class SearchResult(BaseModel):
    """Structured search result."""
    title: str
    url: str
    snippet: str


@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def test_tool_engine_creation():
    """Test basic ToolEngine creation."""
    engine = ToolEngine(
        tools=[calculator],
        enable_analysis=True
    )
    
    assert engine is not None
    assert engine.engine_type.value == "tool"
    assert len(engine._tool_properties) > 0


def test_tool_analysis():
    """Test tool property analysis."""
    engine = ToolEngine(
        tools=[calculator],
        enable_analysis=True
    )
    
    props = engine.get_tool_properties("calculator")
    assert props is not None
    assert props.name == "calculator"
    assert props.tool_type == ToolType.STRUCTURED_TOOL
    assert props.category == ToolCategory.COMPUTATION
    assert ToolCapability.STRUCTURED_OUTPUT in props.capabilities


def test_create_retriever_tool():
    """Test creating a retriever tool."""
    # Mock retriever for testing
    class MockRetriever:
        def get_relevant_documents(self, query: str):
            return [{"content": f"Result for {query}"}]
    
    retriever_tool = ToolEngine.create_retriever_tool(
        retriever=MockRetriever(),
        name="test_retriever",
        description="Test retrieval tool"
    )
    
    assert retriever_tool.name == "test_retriever"
    assert hasattr(retriever_tool, "__tool_type__")
    assert retriever_tool.__tool_type__ == ToolType.RETRIEVER_TOOL
    assert ToolCapability.RETRIEVER in retriever_tool.__tool_capabilities__


def test_create_state_tool():
    """Test creating a state-aware tool."""
    @tool
    def state_reader(state_key: str) -> str:
        """Read value from state."""
        return f"Value for {state_key}"
    
    state_tool = ToolEngine.create_state_tool(
        func=state_reader,
        reads_state=True,
        state_keys=["messages", "context"]
    )
    
    assert hasattr(state_tool, "__tool_capabilities__")
    assert ToolCapability.READS_STATE in state_tool.__tool_capabilities__
    assert ToolCapability.STATE_AWARE in state_tool.__tool_capabilities__
    assert state_tool.reads_state is True
    assert state_tool.state_dependencies == ["messages", "context"]


def test_create_interruptible_tool():
    """Test creating an interruptible tool."""
    @tool
    def long_task(duration: int) -> str:
        """Perform a long-running task."""
        return f"Task completed in {duration} seconds"
    
    interruptible = ToolEngine.create_interruptible_tool(
        func=long_task,
        interrupt_message="Task interrupted by user"
    )
    
    assert hasattr(interruptible, "is_interruptible")
    assert interruptible.is_interruptible is True
    assert hasattr(interruptible, "interrupt")
    assert ToolCapability.INTERRUPTIBLE in interruptible.__tool_capabilities__


def test_create_structured_output_tool():
    """Test creating a structured output tool."""
    def search_function(query: str, limit: int = 10) -> list[SearchResult]:
        """Search and return structured results."""
        return [
            SearchResult(
                title=f"Result {i}",
                url=f"https://example.com/{i}",
                snippet=f"Snippet for {query}"
            )
            for i in range(limit)
        ]
    
    structured_tool = ToolEngine.create_structured_output_tool(
        func=search_function,
        name="structured_search",
        description="Search with structured output",
        output_model=SearchResult
    )
    
    assert structured_tool.name == "structured_search"
    assert hasattr(structured_tool, "structured_output_model")
    assert structured_tool.structured_output_model == SearchResult
    assert ToolCapability.STRUCTURED_OUTPUT in structured_tool.__tool_capabilities__


def test_augment_tool():
    """Test augmenting an existing tool with multiple capabilities."""
    @tool
    def basic_tool(x: str) -> str:
        """A basic tool."""
        return f"Result: {x}"
    
    enhanced = ToolEngine.augment_tool(
        basic_tool,
        make_interruptible=True,
        reads_state=True,
        writes_state=True,
        state_keys=["data"],
        structured_output_model=SearchResult
    )
    
    # Check all enhancements were applied
    assert enhanced.is_interruptible is True
    assert enhanced.reads_state is True
    assert enhanced.writes_state is True
    assert enhanced.state_dependencies == ["data"]
    assert enhanced.structured_output_model == SearchResult
    
    # Check capabilities
    capabilities = enhanced.__tool_capabilities__
    assert ToolCapability.INTERRUPTIBLE in capabilities
    assert ToolCapability.READS_STATE in capabilities
    assert ToolCapability.WRITES_STATE in capabilities
    assert ToolCapability.STATE_AWARE in capabilities
    assert ToolCapability.STRUCTURED_OUTPUT in capabilities


def test_tool_routing():
    """Test tool routing strategies."""
    # Create tools with different capabilities
    retriever = ToolEngine.create_retriever_tool(
        retriever=MockRetriever(),
        name="retriever",
        description="Retrieves documents"
    )
    
    state_tool = ToolEngine.create_state_tool(
        func=lambda x: x,
        name="state_tool",
        reads_state=True
    )
    
    engine = ToolEngine(
        tools=[calculator, retriever, state_tool],
        routing_strategy="capability",
        enable_analysis=True
    )
    
    # Test capability queries
    retrievers = engine.get_tools_by_capability(ToolCapability.RETRIEVER)
    assert "retriever" in retrievers
    
    state_tools = engine.get_state_tools()
    assert "state_tool" in state_tools
    
    # Test category queries
    computation_tools = engine.get_tools_by_category(ToolCategory.COMPUTATION)
    assert "calculator" in computation_tools


def test_store_tools_suite():
    """Test creating store/memory tools suite."""
    # Mock StoreManager
    class MockStoreManager:
        pass
    
    # Would need actual store_tools module to test fully
    # This is a placeholder to show the pattern
    
    # tools = ToolEngine.create_store_tools_suite(
    #     store_manager=MockStoreManager(),
    #     namespace=("test", "namespace"),
    #     include_tools=["store", "search"]
    # )
    
    # assert len(tools) == 2
    # for tool in tools:
    #     assert tool.__tool_type__ == ToolType.STORE_TOOL
    #     assert tool.__tool_category__ == ToolCategory.MEMORY


class MockRetriever:
    """Mock retriever for testing."""
    def get_relevant_documents(self, query: str):
        return [{"content": f"Document about {query}"}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])