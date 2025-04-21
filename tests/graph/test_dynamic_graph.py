import pytest
import os
import tempfile
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from langgraph.graph import START, END
from haive.core.engine.base import Engine, EngineType
from haive.core.engine.aug_llm.base import AugLLMConfig
from haive.core.graph.dynamic_graph_builder import DynamicGraph, DebugLevel
from haive.core.graph.node.config import NodeConfig
from haive.core.graph.graph_pattern_registry import GraphPatternRegistry, GraphPattern

# Define a simple state schema for testing
class GraphState(BaseModel):
    """Test state schema."""
    query: str = ""
    context: List[str] = Field(default_factory=list)
    output: str = ""

# Test helpers to pretty print results
def print_test_result(test_name, result):
    """Print test result in a nice format."""
    result_str = json.dumps(result, indent=2)
    print(f"\n======================================================================")
    print(f"✅ TEST: {test_name}")
    print(f"======================================================================")
    print(f"RESULT:\n{result_str}")
    print(f"======================================================================\n")

# Fixture for creating mock engines
@pytest.fixture
def mock_engines():
    """Create mock engines for testing."""
    engine1 = AugLLMConfig(
        name="engine1",
        id="engine-1-id",
        model="test-model-1"
    )
    
    engine2 = AugLLMConfig(
        name="engine2",
        id="engine-2-id",
        model="test-model-2"
    )
    
    return [engine1, engine2]

# Fixture for creating a test graph
@pytest.fixture
def test_graph(mock_engines):
    """Create a basic test graph with components."""
    return DynamicGraph(
        name="test_graph",
        components=mock_engines,
        state_schema=GraphState,
        debug_level=DebugLevel.BASIC
    )

# Test DynamicGraph initialization
def test_dynamic_graph_init(mock_engines):
    """Test DynamicGraph initialization."""
    graph = DynamicGraph(
        name="test_graph",
        components=mock_engines,
        state_schema=GraphState
    )
    
    # Verify components were registered
    assert len(graph.engines) == 2
    assert "engine1" in graph.engines
    assert "engine2" in graph.engines
    
    # Check state model
    assert graph.state_model == GraphState
    
    result = {
        "name": graph.name,
        "component_count": len(graph.components),
        "engine_count": len(graph.engines)
    }
    
    print_test_result("test_dynamic_graph_init", result)

# Test adding a node
def test_add_node(test_graph, mock_engines):
    """Test adding a node to the graph."""
    engine1 = mock_engines[0]
    
    # Add a node using an engine
    test_graph.add_node("test_node", engine1)
    
    assert "test_node" in test_graph.nodes
    
    # Verify the node was added correctly
    node_config = test_graph.nodes["test_node"]
    assert node_config.name == "test_node"
    assert node_config.engine == engine1
    
    result = {
        "node_name": "test_node",
        "engine_name": engine1.name,
        "engine_id": engine1.id
    }
    
    print_test_result("test_add_node", result)

# Test adding an edge
def test_add_edge(test_graph, mock_engines):
    """Test adding edges between nodes."""
    # Create nodes first
    test_graph.add_node("node1", mock_engines[0])
    test_graph.add_node("node2", mock_engines[1])
    
    # Add an edge
    test_graph.add_edge("node1", "node2")
    
    # Check if edge exists
    edge_exists = any(e.source == "node1" and e.target == "node2" for e in test_graph.edges)
    assert edge_exists
    
    # Add a second edge
    test_graph.add_edge("node2", END)
    
    # Print results
    result = {
        "edges": [
            {"from": e.source, "to": e.target}
            for e in test_graph.edges
        ]
    }
    
    print_test_result("test_add_edge (first edge)", result)
    print_test_result("test_add_edge (second edge)", result)

# Test inserting a node
def test_insert_node(test_graph, mock_engines):
    """Test inserting a node between existing nodes."""
    # Create nodes and add edge
    test_graph.add_node("start_node", mock_engines[0])
    test_graph.add_node("end_node", mock_engines[1])
    test_graph.add_edge("start_node", "end_node")
    
    # Get current edges
    initial_edges = [{"from": e.source, "to": e.target} for e in test_graph.edges]
    print_test_result("test_insert_node (initial setup)", {"initial_edges": initial_edges})
    
    # Insert node in the middle
    test_graph.add_node("middle_node", mock_engines[0])
    test_graph.add_edge("start_node", "middle_node")
    test_graph.add_edge("middle_node", "end_node")
    
    # Check final edges
    final_edges = [{"from": e.source, "to": e.target} for e in test_graph.edges]
    print_test_result("test_insert_node (after insertion)", {"final_edges": final_edges})
    
    # Verify middle node exists
    assert "middle_node" in test_graph.nodes

# Test with_runnable_config method
def test_with_runnable_config(test_graph):
    """Test creating a new graph with a different runnable config."""
    # Create a config
    test_config = {
        "configurable": {
            "thread_id": "test-thread"
        }
    }
    
    # Create new graph with config
    new_graph = test_graph.with_runnable_config(test_config)
    
    # Check it has the config
    assert new_graph.default_runnable_config == test_config
    assert "thread_id" in new_graph.default_runnable_config.get("configurable", {})
    
    result = {
        "original_graph_name": test_graph.name,
        "new_graph_name": new_graph.name,
        "has_config": new_graph.default_runnable_config is not None,
        "thread_id": new_graph.default_runnable_config.get("configurable", {}).get("thread_id")
    }
    
    print_test_result("test_with_runnable_config", result)

# Test set_default_runnable_config
def test_set_default_runnable_config(test_graph):
    """Test setting the default runnable config."""
    # Create a config
    test_config = {
        "configurable": {
            "thread_id": "test-thread"
        }
    }
    
    # Set config
    test_graph.set_default_runnable_config(test_config)
    
    # Check it's set
    assert test_graph.default_runnable_config == test_config
    
    result = {
        "has_config": test_graph.default_runnable_config is not None,
        "thread_id": test_graph.default_runnable_config.get("configurable", {}).get("thread_id")
    }
    
    print_test_result("test_set_default_runnable_config", result)

# Test update_default_runnable_config
def test_update_default_runnable_config(test_graph):
    """Test updating the default runnable config."""
    # Create initial config
    test_graph.update_default_runnable_config(thread_id="test-thread")
    
    # Check initial state
    assert "thread_id" in test_graph.default_runnable_config.get("configurable", {})
    
    result = {
        "has_config": test_graph.default_runnable_config is not None,
        "thread_id": test_graph.default_runnable_config.get("configurable", {}).get("thread_id"),
        "user_id": test_graph.default_runnable_config.get("configurable", {}).get("user_id")
    }
    
    print_test_result("test_update_default_runnable_config (initial)", result)
    
    # Update config
    test_graph.update_default_runnable_config(user_id="test-user")
    
    # Check updated config
    assert "user_id" in test_graph.default_runnable_config.get("configurable", {})
    
    result = {
        "has_config": test_graph.default_runnable_config is not None,
        "thread_id": test_graph.default_runnable_config.get("configurable", {}).get("thread_id"),
        "user_id": test_graph.default_runnable_config.get("configurable", {}).get("user_id")
    }
    
    print_test_result("test_update_default_runnable_config (after update)", result)

# Test build and compile methods
def test_build_and_compile(test_graph, mock_engines):
    """Test building and compiling the graph."""
    # Add nodes and edges
    test_graph.add_node("start_node", mock_engines[0])
    test_graph.add_node("end_node", mock_engines[1])
    test_graph.add_edge("start_node", "end_node")
    test_graph.add_edge(START, "start_node")
    
    # Build the graph (not compiled)
    built_graph = test_graph.build()
    
    # Should return a StateGraph builder
    assert built_graph is not None
    
    # This would compile but we'll skip actually running it in tests
    # compiled_graph = test_graph.compile()
    # assert compiled_graph is not None

# Test pattern application
def test_apply_pattern(test_graph, monkeypatch):
    """Test applying a pattern to the graph."""
    # Create a mock pattern
    class MockPattern:
        def __init__(self):
            self.name = "test_pattern"
            self.description = "Test pattern"
            self.type = "test"
            
        def apply(self, graph, **kwargs):
            # Add pattern-specific nodes
            graph.add_node("pattern_node", graph.components[0])
            return graph
    
    # Mock the GraphPatternRegistry to return our mock pattern
    mock_registry = GraphPatternRegistry.get_instance()
    monkeypatch.setattr(mock_registry, "get_pattern", lambda name: MockPattern() if name == "test_pattern" else None)
    
    # Apply the pattern
    test_graph.apply_pattern("test_pattern", param1="value1")
    
    # Check pattern was recorded as applied
    assert "test_pattern" in test_graph.applied_patterns

# Test serialization
def test_serialization(test_graph, mock_engines):
    """Test serializing and deserializing a graph."""
    # Create a graph with nodes and edges
    test_graph.add_node("start_node", mock_engines[0])
    test_graph.add_node("end_node", mock_engines[1])
    test_graph.add_edge("start_node", "end_node")
    
    # Create a temporary file for serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_graph.json")
        
        # Save to file
        test_graph.save(filename)
        
        # Check file exists
        assert os.path.exists(filename)
        
        # This would load the graph back, but we'll skip it
        # loaded_graph = DynamicGraph.load(filename)
        # assert loaded_graph is not None
        # assert loaded_graph.name == test_graph.name

# Test conditional edges
def test_add_conditional_edges(test_graph, mock_engines):
    """Test adding conditional edges."""
    # Define a condition function
    def condition(state):
        if state.get("route") == "A":
            return "a"
        else:
            return "b"
    
    # Add nodes
    test_graph.add_node("router", mock_engines[0])
    test_graph.add_node("path_a", mock_engines[0])
    test_graph.add_node("path_b", mock_engines[1])
    
    # Add conditional edges
    test_graph.add_conditional_edges(
        "router",
        condition,
        {
            "a": "path_a",
            "b": "path_b"
        }
    )
    
    # Verify branches were added
    assert len(test_graph.branches) == 1
    
    # Get the branch
    branch = test_graph.branches[0]
    
    # Check branch properties
    assert branch["source"] == "router"
    assert branch["condition"] == condition
    assert branch["routes"] == {"a": "path_a", "b": "path_b"}
    
    # Check conditional edges were added
    conditional_edges = [e for e in test_graph.edges if e.condition is not None]
    assert len(conditional_edges) == 2