from typing import Dict, List, Optional, Union, Literal, Callable, Type
from src.haive.core.engine.aug_llm import AugLLMConfig
from src.haive.core.graph.NodeFactory import create_node_function
from src.haive.core.graph.StateSchemaManager import StateSchemaManager
from langgraph.graph import StateGraph, END
from src.haive.core.graph import NodeFactory
from langchain_core.tools import BaseTool
from src.haive.core.graph.branches import Branch
from pydantic import BaseModel
import json
import logging
logger = logging.getLogger(__name__)

class DynamicGraph:
    """
    A dynamic graph builder that automatically derives state schema from components.
    """
    
    def __init__(self, components=None, custom_fields=None, state_schema=None, build_type=None):
        """
        Initialize the graph builder with component-derived schema.
        
        Args:
            components: Optional list of AugLLMConfig objects to derive schema from
            custom_fields: Optional custom fields to add to the schema
            state_schema: Optional existing state schema to use instead of deriving
            build_type: Optional build type for specialized graph construction
        """
        from langgraph.graph import START
        from src.haive.core.graph.SchemaComposer import SchemaComposer
        
        # Initialize with empty schema if no components
        components = components or []
        if state_schema:
            print(f"DEBUG: has state_schema: {state_schema}")
            self.state_model = state_schema
        else:
            # Create schema manager from components
            self.schema_manager = SchemaComposer.create_schema_for_components(components)
            print(f"DEBUG: self.schema_manager: {self.schema_manager}") 
            self.state_model = self.schema_manager.get_model() if isinstance(self.schema_manager, StateSchemaManager) else self.schema_manager
            self.schema_manager = StateSchemaManager(self.state_model)
            self.schema_manager.pretty_print()
        
        # Add custom fields if provided
        if custom_fields:
            for name, (type_hint, default) in custom_fields.items():
                self.schema_manager.add_field(name, type_hint, default)
        
        # Create graph with derived schema
        self.graph = StateGraph(self.state_model)
        
        # Track nodes and edges
        self.nodes = {}
        self.edges = []
        self.branches = {}
        self.entry_point = None
        
        # Track structured output configuration
        self.structured_output_model = None
    
    def add_node(
        self, 
        name: str, 
        config: Union[AugLLMConfig, Callable], 
        command_goto: Optional[Union[str, Literal["END"]]] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None
    ):
        """
        Add a node to the graph.

        Args:
            name: Name of the node
            config: AugLLMConfig or callable node function
            command_goto: Where to route after this node
            input_mapping: Maps state fields to node inputs
            output_mapping: Maps node outputs to state fields

        Returns:
            Self for chaining
        """
        # If it's an AugLLMConfig, create a node function
        if isinstance(config, AugLLMConfig):
            from src.haive.core.graph.StateSchemaManager import StateSchemaManager as SchemaManager

            # Update schema if needed
            self.schema_manager = SchemaManager(self.state_model)
            self.state_model = self.schema_manager.get_model()
            self.graph = StateGraph(self.state_model)

            # Dynamically derive input mapping
            if input_mapping is None:
                expected_vars = set(config.prompt_template.input_variables) if config.prompt_template else set()
                input_mapping = {var: var for var in expected_vars}
            
            # Ensure output mapping is correctly set
            if output_mapping is None:
                if config.structured_output_model:
                    output_mapping = {config.structured_output_model.__name__.lower(): config.structured_output_model.__name__.lower()}
                    # Store structured output model for later use
                    self.structured_output_model = config.structured_output_model
            
            logger.debug(f"Using input_mapping: {input_mapping}")
            logger.debug(f"Using output_mapping: {output_mapping}")
            
            # Create node function with corrected mapping
            node_fn = create_node_function(
                config=config,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                command_goto=command_goto
            )
        else:
            # It's a callable, use directly
            node_fn = config

        # Add to graph
        self.graph.add_node(name, node_fn)
        self.nodes[name] = node_fn

        # Set as entry point if first node
        if self.entry_point is None:
            self.entry_point = name
            self.graph.set_entry_point(name)

        # Add edge if command_goto is specified and not END
        if command_goto is not None and command_goto != END:
            self.add_edge(name, command_goto)
        elif command_goto == END or command_goto == '__end__':
            # Explicitly add edge to END for proper routing
            self.add_edge(name, END)

        return self
    
    def add_structured_output_node(
        self,
        name: str = "structured_output",
        model: Optional[Type[BaseModel]] = None,
        command_goto: Optional[Union[str, Literal["END"]]] = END
    ):
        """
        Add a structured output node that formats responses according to a Pydantic model.
        
        Args:
            name: Name for the node
            model: Pydantic model to use (defaults to previously stored structured_output_model)
            command_goto: Where to route after this node
            
        Returns:
            Self for chaining
        """
        # Use provided model or previously stored one
        output_model = model or self.structured_output_model
        
        if not output_model:
            raise ValueError("No structured output model provided or previously configured")
            
        # Create structured output processor
        def process_structured_output(state):
            """Process the state to create structured output."""
            try:
                # Get the last message content
                messages = state.messages if hasattr(state, 'messages') else []
                if not messages:
                    return {"parsing_errors": "No messages found in state"}
                
                last_message = messages[-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
                # Store raw output
                result = {"raw_output": content}
                
                # Try to parse JSON from the content
                try:
                    # First, try to extract a JSON block if it exists
                    import re
                    json_match = re.search(r'```json\s*\n(.*?)\n\s*```', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # Try to find any JSON-like structure
                        json_match = re.search(r'({.*})', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                        else:
                            json_str = content
                    
                    # Parse the JSON
                    parsed_data = json.loads(json_str)
                    
                    # Validate against the model
                    structured_output = output_model(**parsed_data)
                    result["structured_output"] = structured_output
                    
                except (json.JSONDecodeError, Exception) as e:
                    # If JSON parsing or validation fails, log error
                    logger.warning(f"Failed to parse structured output: {e}")
                    result["parsing_errors"] = f"Error parsing output: {str(e)}"
                
                return result
                
            except Exception as e:
                logger.error(f"Error in structured output processor: {str(e)}")
                return {"parsing_errors": f"Processing error: {str(e)}"}
        
        # Add the node
        self.add_node(name, process_structured_output, command_goto)
        
        return self
    
    # Method to add to DynamicGraph class or use directly
    def add_node_with_conditional_edge(self, node_name, router_function, destinations, default_destination=None):
        """
        Add conditional edges from a node with directly specified router function.
        
        Args:
            node_name: Name of the source node
            router_function: Function that determines routing
            destinations: Mapping from function results to target nodes
            default_destination: Default target if result not in destinations
        """
        # Ensure the node exists
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} does not exist")
        
        # Directly add conditional edges to the graph
        self.graph.add_conditional_edges(
            node_name,
            router_function,
            destinations,
        )
        
        # Also register in internal tracking
        self.branches[node_name] = {
            'router': router_function,
            'destinations': destinations,
            'default': default_destination
        }
        
        return self
    
    def add_tool_node(
        self,
        name: str,
        tools: List[BaseTool],
        post_processor: Optional[Callable] = None,
        command_goto: Optional[Union[str, Literal["END"]]] = None
    ):
        """
        Add a tool node to the graph.
        
        Args:
            name: Name of the node
            tools: List of tools to use
            post_processor: Optional function to process tool results
            command_goto: Where to route after this node
            
        Returns:
            Self for chaining
        """
        # Update schema to include tool_results if not present
        if 'tool_results' not in self.schema_manager.fields:
            from typing import Dict, Any, List
            self.schema_manager.add_field('tool_results', List[Dict[str, Any]], default_factory=list)
            self.state_model = self.schema_manager.get_model()
            self.graph = StateGraph(self.state_model)
        
        # Create tool node
        node_fn = NodeFactory.create_tool_node(
            tools=tools,
            post_processor=post_processor,
            command_goto=command_goto
        )
        
        # Add to graph
        self.graph.add_node(name, node_fn)
        self.nodes[name] = node_fn
        
        # Set as entry point if first node
        if self.entry_point is None:
            self.entry_point = name
            self.graph.set_entry_point(name)
        
        # Add edge if command_goto is specified and not END
        if command_goto is not None and command_goto != END:
            self.add_edge(name, command_goto)
        else:
            # Explicitly add edge to END for proper routing
            self.add_edge(name, END)
        
        return self
    
    def add_edge(self, from_node: str, to_node: Union[str, Literal["END"]]):
        """
        Add an edge between nodes.
        
        Args:
            from_node: Source node name
            to_node: Target node name or END
            
        Returns:
            Self for chaining
        """
        # Record the edge
        self.edges.append((from_node, to_node))
        
        # Only add to graph if not END
        if to_node != END:
            self.graph.add_edge(from_node, to_node)
        else:
            # Explicitly add edge to END
            self.graph.add_edge(from_node, END)
        
        return self
    
    def add_conditional_edges(
    self,
    from_node: str,
    condition_or_branch: Union[Callable, Branch],
    routes: Optional[Dict[str, Union[str, Literal["END"]]]] = None
    ):
        """
        Add conditional edges based on either a routing function or a Branch object.

        Args:
            from_node (str): Source node name.
            condition_or_branch (Callable | Branch): Either a condition function or a Branch object.
            routes (Optional[Dict[str, Union[str, Literal["END"]]]]): 
                - If `condition_or_branch` is a function, this must be provided as a mapping from condition results to target nodes.
                - If `condition_or_branch` is a `Branch`, this is ignored.

        Returns:
            Self for chaining.
        """
        print(f"DEBUG: adding conditional edges from {from_node} to {condition_or_branch}")
        print(type(condition_or_branch))
        if isinstance(condition_or_branch, Branch):
            # If a Branch object is provided, let it handle routing internally
            self.branches[from_node] = condition_or_branch
            self.graph.add_conditional_edges(from_node, lambda state: condition_or_branch.evaluate(state))
        else:
            # Assume it's a condition function and requires explicit routes
            if routes is None:
                raise ValueError("Routes dictionary must be provided when using a condition function.")
            self.branches[from_node] = routes
            
            self.graph.add_conditional_edges(from_node, condition_or_branch, routes)

        return self
    
    def set_entry_point(self, node_name: str):
        """
        Set the entry point for the graph.
        
        Args:
            node_name: Name of the entry point node
            
        Returns:
            Self for chaining
        """
        self.entry_point = node_name
        self.graph.set_entry_point(node_name)
        return self
    
    def build(self, checkpointer=None):
        """
        Build and compile the graph.
        
        Args:
            checkpointer: Optional checkpoint saver
            
        Returns:
            Compiled graph application
        """
        from langgraph.graph import START
        
        if self.entry_point is None:
            raise ValueError("No entry point defined - add at least one node")
        
        # CRITICAL: Ensure START edge is present
        self.graph.set_entry_point(self.entry_point)
        
        # Rebuild all edges to ensure they're properly connected
        for from_node, to_node in self.edges:
            if to_node != END:
                self.graph.add_edge(from_node, to_node)
            else:
                # Explicitly add END edge
                self.graph.add_edge(from_node, END)
        
        return self.graph
    
    def remove_edge(self, from_node: str, to_node: Union[str, Literal["END"]]):
        """
        Remove a specific edge between two nodes.
        """
        if (from_node, to_node) in self.edges:
            self.edges.remove((from_node, to_node))

        if from_node in self.graph.graph:
            self.graph.graph[from_node] = [
                edge for edge in self.graph.graph[from_node] if edge.target != to_node
            ]

        logger.debug(f"Removed edge from {from_node} to {to_node}")
        return self

    def overwrite_edge(self, from_node: str, to_node: Union[str, Literal["END"]]):
        """
        Overwrite existing edge from `from_node` to a new `to_node`.
        """
        # Remove old edges from the record and graph
        self.edges = [edge for edge in self.edges if edge[0] != from_node]
        
        if from_node in self.graph.graph:
            self.graph.graph[from_node] = [
                edge for edge in self.graph.graph[from_node] if edge.target != END and edge.target != to_node
            ]

        # Add the new edge
        self.add_edge(from_node, to_node)
        logger.debug(f"Edge from {from_node} overwritten to {to_node}")
        return self

    def get_schema(self):
        """
        Get the current schema model.
        
        Returns:
            The Pydantic model for the state schema
        """
        return self.state_model
    
    def cut_after(self, node_name: str):
        """
        Remove all outgoing edges from a given node (e.g., to extend beyond it).
        """
        # Remove from edge list
        self.edges = [edge for edge in self.edges if edge[0] != node_name]

        # Remove from internal LangGraph structure
        if node_name in self.graph.graph:
            self.graph.graph[node_name] = []

        # Remove conditional branches
        if node_name in self.branches:
            del self.branches[node_name]

        return self