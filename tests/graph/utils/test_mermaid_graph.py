"""
Test module for the graph_visualization_utils.py module.

This demonstrates how to use the enhanced visualization functionality
with DynamicGraph instances.
"""

import os
import logging
import sys
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Adjust these imports to match your project structure
from haive.core.graph.dynamic_graph_builder import DynamicGraph
from haive.core.engine.aug_llm.base import AugLLMConfig
from haive.core.engine.retriever import RetrieverConfig
from haive.core.engine.vectorstore import VectorStoreConfig
from langgraph.graph import START, END

# Import the visualization utils
from haive.core.graph.utils.mermaid_visualizer import visualize_graph, replace_visualization_method

def create_sample_graph():
    """Create a sample graph for testing visualization."""
    
    # Create sample engines
    llm_engine = AugLLMConfig(
        name="gpt4_engine",
        id="llm-" + uuid.uuid4().hex[:8],
        model="gpt-4"
    )
    
    retriever_engine = RetrieverConfig(
        name="retriever_engine",
        id="retriever-" + uuid.uuid4().hex[:8]
    )
    
    vectorstore_engine = VectorStoreConfig(
        name="vectorstore_engine",
        id="vectorstore-" + uuid.uuid4().hex[:8]
    )
    
    # Create a graph
    graph = DynamicGraph(
        name="TestRAGWorkflow",
        components=[llm_engine, retriever_engine, vectorstore_engine],
        description="A test RAG workflow for visualization",
        visualize=True,
        debug_level="verbose"
    )
    
    # Create a basic RAG workflow
    graph.add_node("query_understanding", llm_engine)
    graph.add_node("retrieve", retriever_engine)
    graph.add_node("generate", llm_engine, command_goto=END)
    
    # Add a tool node with custom function
    def process_results(state):
        """Process retrieval results."""
        # In a real implementation, this would process the documents
        return {"processed_results": True}
    
    graph.add_node("process_results", process_results)
    
    # Create a simple, linear flow
    graph.add_edge(START, "query_understanding")
    graph.add_edge("query_understanding", "retrieve")
    graph.add_edge("retrieve", "process_results")
    graph.add_edge("process_results", "generate")
    
    # Add a conditional branch
    def route_by_query_complexity(state):
        """Route based on query complexity."""
        # In a real implementation, this would analyze the query
        query_length = len(state.get("query", ""))
        if query_length > 100:
            return "complex"
        elif query_length > 50:
            return "medium"
        else:
            return "simple"
    
    # Add branching nodes
    graph.add_node("complex_retrieval", retriever_engine, 
                  config_overrides={"k": 10})
    graph.add_node("simple_retrieval", retriever_engine,
                  config_overrides={"k": 3})
    
    # Add conditional routing
    graph.add_conditional_edges(
        "query_understanding",
        route_by_query_complexity,
        {
            "complex": "complex_retrieval",
            "simple": "simple_retrieval",
            "medium": "retrieve"  # Default retrieval
        }
    )
    
    # Connect branches back to main flow
    graph.add_edge("complex_retrieval", "process_results")
    graph.add_edge("simple_retrieval", "process_results")
    
    # Add error handling node
    graph.add_node("error_handler", llm_engine, command_goto="generate")
    
    # Make one node unreachable for testing
    graph.add_node("unreachable_node", llm_engine)
    
    return graph

def test_direct_visualization():
    """Test direct visualization without replacing methods."""
    print("\n=== Testing Direct Visualization ===")
    
    # Create sample graph
    graph = create_sample_graph()
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test HTML format
    html_file = os.path.join(output_dir, f"test_direct_html_{timestamp}.html")
    print(f"Generating HTML visualization: {html_file}")
    result_html = visualize_graph(
        graph=graph,
        output_file=html_file,
        open_browser=False  # Set to True to open in browser
    )
    print(f"HTML visualization generated: {result_html}")
    
    # Test PNG format
    png_file = os.path.join(output_dir, f"test_direct_png_{timestamp}.png")
    print(f"Generating PNG visualization: {png_file}")
    try:
        result_png = visualize_graph(
            graph=graph,
            output_file=png_file,
            format="png",
            open_browser=False
        )
        print(f"PNG visualization generated: {result_png}")
    except ImportError:
        print("PNG visualization requires requests library - skipped")

def test_method_replacement():
    """Test visualization after method replacement."""
    print("\n=== Testing Method Replacement ===")
    
    # Replace visualization method
    result = replace_visualization_method()
    print(f"Method replacement {'successful' if result else 'failed'}")
    
    if not result:
        print("Skipping this test as method replacement failed")
        return
    
    # Create sample graph
    graph = create_sample_graph()
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Test using replaced method
    output_file = os.path.join(output_dir, f"test_replaced_method_{timestamp}.html")
    print(f"Generating visualization using replaced method: {output_file}")
    
    # Now we can use the replaced method directly on the graph instance
    result = graph.visualize_graph(
        output_file=output_file,
        open_browser=False
    )
    print(f"Visualization generated: {result}")
    
    # Test using the dedicated Mermaid method
    output_file = os.path.join(output_dir, f"test_mermaid_method_{timestamp}.html")
    print(f"Generating visualization using dedicated Mermaid method: {output_file}")
    
    # Use the dedicated method
    result = graph.visualize_mermaid(
        output_file=output_file,
        open_browser=False,
        include_legend=True
    )
    print(f"Mermaid visualization generated: {result}")

def main():
    """Main test function."""
    print("=== Graph Visualization Test ===")
    
    # Run tests
    test_direct_visualization()
    test_method_replacement()
    
    print("\n=== All Tests Completed ===")

if __name__ == "__main__":
    main()