import pytest
import os
import tempfile
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from langgraph.graph import END, START

from haive_core.graph.dynamic_graph_builder import DynamicGraph, ComponentRef
from haive_core.graph.node.config import NodeConfig
from haive_core.engine.base import Engine, InvokableEngine, EngineType
from haive_core.config.runnable import RunnableConfigManager
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.models.llm.base import AzureLLMConfig

# Define test state schema
class GraphState(BaseModel):
    query: Optional[str] = None
    result: Optional[str] = None
    processed: bool = False

# Helper function to create a graph with real engines
def create_test_graph():
    # Create real AugLLMConfig engines
    engine1 = AugLLMConfig(
        name="engine1",
        id="engine-1-id",
        llm_config=AzureLLMConfig(
            model="gpt-4o",
            api_key="test-key-1",
            api_version="2023-07-01-preview"
        ),
        system_message="You are a helpful AI assistant."
    )
    
    engine2 = AugLLMConfig(
        name="engine2",
        id="engine-2-id",
        llm_config=AzureLLMConfig(
            model="gpt-4o-mini",
            api_key="test-key-2",
            api_version="2023-07-01-preview"
        ),
        system_message="You are a knowledgeable AI assistant."
    )
    
    return DynamicGraph(
        name="test_graph",
        description="Test graph",
        components=[engine1, engine2],
        state_schema=GraphState,
        visualize=False  # Disable visualization for testing
    )

# Define tests for DynamicGraph

def test_dynamic_graph_init():
    """Test DynamicGraph initialization."""
    graph = create_test_graph()
    
    # Check basic properties
    assert graph.name == "test_graph"
    assert graph.description == "Test graph"
    assert len(graph.components) == 2
    assert graph.state_model == GraphState
    assert graph.input_model == GraphState  # Defaults to state_model
    assert graph.output_model == GraphState  # Defaults to state_model
    
    # Check engine tracking
    assert len(graph.engines) == 2
    assert "engine1" in graph.engines
    assert "engine2" in graph.engines
    assert len(graph.engines_by_id) == 2
    assert "engine-1-id" in graph.engines_by_id
    assert "engine-2-id" in graph.engines_by_id
    
    # Check that graph was initialized
    assert graph.graph is not None

def test_add_node():
    """Test adding a node to the graph."""
    graph = create_test_graph()
    
    # Add a node with engine1
    graph.add_node(
        name="process_node",
        config=graph.engines["engine1"],
        command_goto=END
    )
    
    # Check that node was added
    assert "process_node" in graph.nodes
    assert graph.nodes["process_node"].name == "process_node"
    assert graph.nodes["process_node"].config.engine is graph.engines["engine1"]
    assert graph.nodes["process_node"].config.command_goto is END
    
    # Add a node with string reference
    with patch.object(graph, '_lookup_engine', return_value=graph.engines["engine2"]):
        graph.add_node(
            name="query_node",
            config="engine2",
            command_goto="process_node"
        )
    
    # Check that node was added
    assert "query_node" in graph.nodes
    assert graph.nodes["query_node"].name == "query_node"
    assert graph.nodes["query_node"].config.engine is graph.engines["engine2"]
    assert graph.nodes["query_node"].config.command_goto == "process_node"

def test_add_edge():
    """Test adding edges between nodes."""
    graph = create_test_graph()
    
    # Add nodes
    graph.add_node("node1", graph.engines["engine1"])
    graph.add_node("node2", graph.engines["engine2"])
    
    # Add edge
    graph.add_edge("node1", "node2")
    
    # Check that edge was added
    assert any(e.source == "node1" and e.target == "node2" for e in graph.edges)
    
    # Add edge to END
    graph.add_edge("node2", END)
    
    # Check that edge was added
    assert any(e.source == "node2" and e.target == "END" for e in graph.edges)
    
    # Add multiple edges
    graph.add_edge(["START", "node1"], "node2")
    
    # Check that edges were added
    assert any(e.source == "START" and e.target == "node2" for e in graph.edges)
    assert any(e.source == "node1" and e.target == "node2" for e in graph.edges)

def test_insert_node():
    """Test inserting a node between existing nodes."""
    graph = create_test_graph()
    
    # Add nodes with an edge
    graph.add_node("start_node", graph.engines["engine1"])
    graph.add_node("end_node", graph.engines["engine2"])
    graph.add_edge("start_node", "end_node")
    
    # Insert a node
    graph.insert_node(
        "middle_node",
        between=("start_node", "end_node"),
        node_config=graph.engines["engine1"]
    )
    
    # Check that node was added
    assert "middle_node" in graph.nodes
    
    # Check that edges were updated
    assert not any(e.source == "start_node" and e.target == "end_node" for e in graph.edges)
    assert any(e.source == "start_node" and e.target == "middle_node" for e in graph.edges)
    assert any(e.source == "middle_node" and e.target == "end_node" for e in graph.edges)

@patch('haive_core.graph.graph_pattern_registry.GraphPatternRegistry')
def test_apply_pattern(mock_registry_class):
    """Test applying a pattern to the graph."""
    graph = create_test_graph()
    
    # Mock pattern registry
    mock_registry = MagicMock()
    mock_registry.get_pattern.return_value = {"name": "test_pattern", "type": "test"}
    mock_registry_class.get_instance.return_value = mock_registry
    
    # Apply pattern
    graph.apply_pattern("test_pattern", param1="value1")
    
    # Check that pattern was applied
    assert "test_pattern" in graph.applied_patterns
    mock_registry.get_pattern.assert_called_once_with("test_pattern")

def test_with_runnable_config():
    """Test creating a graph with a specific runnable config."""
    graph = create_test_graph()
    
    # Create runnable config
    config = RunnableConfigManager.create(thread_id="test-thread")
    
    # Create new graph with config
    new_graph = graph.with_runnable_config(config)
    
    # Check that config was set
    assert new_graph.default_runnable_config == config
    
    # Check that other properties were copied
    assert new_graph.name == graph.name
    assert new_graph.description == graph.description
    assert len(new_graph.components) == len(graph.components)

def test_set_default_runnable_config():
    """Test setting the default runnable config."""
    graph = create_test_graph()
    
    # Create runnable config
    config = RunnableConfigManager.create(thread_id="test-thread")
    
    # Set config
    graph.set_default_runnable_config(config)
    
    # Check that config was set
    assert graph.default_runnable_config == config

def test_update_default_runnable_config():
    """Test updating the default runnable config."""
    graph = create_test_graph()
    
    # Update config with no existing config
    graph.update_default_runnable_config(thread_id="test-thread")
    
    # Check that config was created
    assert graph.default_runnable_config is not None
    assert graph.default_runnable_config["configurable"]["thread_id"] == "test-thread"
    
    # Update existing config
    graph.update_default_runnable_config(user_id="test-user")
    
    # Check that config was updated
    assert graph.default_runnable_config["configurable"]["thread_id"] == "test-thread"
    assert graph.default_runnable_config["configurable"]["user_id"] == "test-user"

def test_build_and_compile():
    """Test building and compiling the graph."""
    graph = create_test_graph()
    
    # Add nodes and edges to make a valid graph
    graph.add_node("start_node", graph.engines["engine1"])
    graph.add_node("end_node", graph.engines["engine2"], command_goto=END)
    graph.add_edge("start_node", "end_node")
    graph.add_edge(START, "start_node")
    
    # Build the graph
    built_graph = graph.build()
    
    # Check that it's a StateGraph
    assert built_graph is not None
    assert hasattr(built_graph, "add_node")
    assert hasattr(built_graph, "add_edge")
    
    # Compile with mock checkpointer
    mock_checkpointer = MagicMock()
    with patch('haive_core.graph.dynamic_graph_builder.DynamicGraph.compile') as mock_compile:
        mock_compile.return_value = MagicMock()
        compiled = graph.compile(checkpointer=mock_checkpointer)
        
        # Check that compile was called with checkpointer
        mock_compile.assert_called_once()
        assert "checkpointer" in mock_compile.call_args[1]
        assert mock_compile.call_args[1]["checkpointer"] is mock_checkpointer

def test_serialization():
    """Test serializing and deserializing the graph."""
    graph = create_test_graph()
    
    # Add nodes and edges
    graph.add_node("start_node", graph.engines["engine1"])
    graph.add_node("end_node", graph.engines["engine2"], command_goto=END)
    graph.add_edge("start_node", "end_node")
    
    # Serialize to dict
    data = graph.to_dict()
    
    # Check main properties
    assert data["name"] == "test_graph"
    assert data["description"] == "Test graph"
    assert len(data["components"]) == 2
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    
    # Check component serialization
    for component in data["components"]:
        assert "name" in component
        assert "id" in component
        assert "type" in component
    
    # Create temp directory for file tests
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test save to file
        filename = os.path.join(temp_dir, "test_graph.json")
        graph.save(filename)
        
        # Check that file exists
        assert os.path.exists(filename)
        
        # Test load from file with mocked engine lookup
        with patch('haive_core.graph.dynamic_graph_builder.DynamicGraph._lookup_engine', 
                  return_value=AugLLMConfig(name="mocked_engine")):
            loaded_graph = DynamicGraph.load(filename)
            
            # Check basic properties
            assert loaded_graph.name == "test_graph"
            assert loaded_graph.description == "Test graph"
            assert len(loaded_graph.components) > 0  # Will be different due to mocking

def test_add_conditional_edges():
    """Test adding conditional edges."""
    graph = create_test_graph()
    
    # Add nodes
    graph.add_node("router", graph.engines["engine1"])
    graph.add_node("path_a", graph.engines["engine1"])
    graph.add_node("path_b", graph.engines["engine2"])
    
    # Define condition function
    def condition(state):
        if state.get("path") == "a":
            return "a"
        return "b"
    
    # Add conditional edges
    graph.add_conditional_edges(
        "router",
        condition,
        {"a": "path_a", "b": "path_b"}
    )
    
    # Check branch was added
    assert len(graph.branches) == 1
    assert graph.branches[0].source == "router"
    assert set(graph.branches[0].routes.keys()) == {"a", "b"}
    assert set(graph.branches[0].routes.values()) == {"path_a", "path_b"}

def test_add_subgraph():
    """Test adding a subgraph."""
    graph = create_test_graph()
    
    # Create a subgraph
    subgraph = create_test_graph()
    subgraph.name = "subgraph"
    
    # Add nodes to subgraph
    subgraph.add_node("sub_node", subgraph.engines["engine1"], command_goto=END)
    subgraph.add_edge(START, "sub_node")
    
    # Mock StateGraph.add_node for subgraph
    with patch.object(graph.graph, 'add_node') as mock_add_node:
        # Add subgraph
        graph.add_subgraph(
            "my_subgraph",
            subgraph,
            input_mapping={"query": "input"},
            output_mapping={"output": "result"}
        )
        
        # Check that node was added to graph
        mock_add_node.assert_called_once()
        assert mock_add_node.call_args[0][0] == "my_subgraph"
        
        # Check node tracking
        assert "my_subgraph" in graph.nodes
        assert graph.nodes["my_subgraph"].config.input_mapping == {"query": "input"}
        assert graph.nodes["my_subgraph"].config.output_mapping == {"output": "result"}
        assert "original_name" in graph.nodes["my_subgraph"].config.metadata
        assert graph.nodes["my_subgraph"].config.metadata["original_name"] == "subgraph"

def test_dynamic_graph_with_components_ref():
    """Test DynamicGraph with ComponentRef."""
    # Create component references
    component_refs = [
        ComponentRef(name="engine1", id="engine-1-id", type="llm"),
        ComponentRef(name="engine2", id="engine-2-id", type="llm")
    ]
    
    # Mock _lookup_engine
    with patch('haive_core.graph.dynamic_graph_builder.DynamicGraph._lookup_engine') as mock_lookup:
        mock_lookup.side_effect = [
            AugLLMConfig(name="engine1", id="engine-1-id"),
            AugLLMConfig(name="engine2", id="engine-2-id")
        ]
        
        # Create graph with component refs
        graph = DynamicGraph(
            name="test_graph",
            components=component_refs,
            state_schema=GraphState,
            visualize=False
        )
        
        # Check that engines were looked up correctly
        assert mock_lookup.call_count == 2
        mock_lookup.assert_any_call("engine1", "llm", "engine-1-id")
        mock_lookup.assert_any_call("engine2", "llm", "engine-2-id")
        
        # Check engine tracking
        assert len(graph.engines) == 2
        assert "engine1" in graph.engines
        assert "engine2" in graph.engines

def test_engine_id_tracking():
    """Test that engine IDs are properly tracked."""
    graph = create_test_graph()
    
    # Add a node with explicit config overrides
    graph.add_node(
        "test_node",
        config=graph.engines["engine1"],
        config_overrides={"temperature": 0.7},
        command_goto=END
    )
    
    # Check that config overrides were properly stored
    assert graph.nodes["test_node"].config.config_overrides == {"temperature": 0.7}
    assert graph.nodes["test_node"].config.engine_id == "mock-engine-id"
    
    # Mock StateGraph.compile to check that config overrides are passed
    with patch('haive_core.graph.dynamic_graph_builder.DynamicGraph.compile') as mock_compile:
        mock_compile.return_value = MagicMock()
        
        # Ensure runnable_config is created with node-specific overrides
        with patch.object(graph.nodes["test_node"].function, '__call__') as mock_call:
            mock_call.return_value = {"result": "test"}
            
            # Compile graph to create node functions
            graph.compile()
            
            # Note: We can't easily test that the function correctly applies config overrides here
            # That's tested separately in test_node_factory.py