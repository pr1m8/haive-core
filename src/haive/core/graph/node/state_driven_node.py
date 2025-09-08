"""State-driven nodes that get behavior from state.

This module implements nodes whose behavior is determined at runtime
from the state, enabling true dynamic execution without recompilation.
"""

import logging
from typing import Any, Callable, Optional, Dict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StateDrivenNode:
    """Node that executes behavior from state.
    
    Instead of having fixed behavior at compile time, this node
    retrieves its behavior from the state at runtime. This enables:
    
    - Runtime behavior modification
    - Dynamic routing updates
    - Hot-swappable logic
    - No recompilation needed
    
    The state must have a 'nodes' dictionary containing callables
    and a 'routing_table' for dynamic routing.
    
    Examples:
        >>> # In state schema
        >>> class DynamicState(StateSchema):
        ...     nodes: dict[str, Callable] = Field(default_factory=dict)
        ...     routing_table: dict[str, list[str]] = Field(default_factory=dict)
        >>> 
        >>> # Create state-driven node
        >>> node = StateDrivenNode("processor")
        >>> 
        >>> # Add behavior to state
        >>> state.nodes["processor"] = lambda x: x.upper()
        >>> state.routing_table["processor"] = ["validator"]
        >>> 
        >>> # Execute - behavior comes from state!
        >>> result = node(state)
    """
    
    def __init__(self, name: str, fallback: Optional[Callable] = None):
        """Initialize state-driven node.
        
        Args:
            name: Node identifier used to lookup behavior in state
            fallback: Optional fallback behavior if not found in state
        """
        self.name = name
        self.fallback = fallback or self._default_fallback
        self._execution_count = 0
        self._last_behavior = None
    
    def __call__(self, state: Any) -> Any:
        """Execute node behavior from state.
        
        This method:
        1. Retrieves behavior from state.nodes[name]
        2. Executes the behavior
        3. Updates routing based on state.routing_table
        4. Returns modified state
        
        Args:
            state: State object containing nodes and routing
            
        Returns:
            Modified state after execution
        """
        self._execution_count += 1
        
        # Get behavior from state
        behavior = self._get_behavior_from_state(state)
        
        # Execute behavior
        if behavior:
            try:
                logger.debug(f"StateDrivenNode '{self.name}' executing behavior from state")
                result = behavior(state)
                
                # Track behavior changes
                if behavior != self._last_behavior:
                    logger.info(f"Node '{self.name}' behavior updated dynamically")
                    self._last_behavior = behavior
                
                # Update routing if needed
                self._update_routing(state)
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing state-driven behavior for '{self.name}': {e}")
                return self.fallback(state)
        else:
            logger.debug(f"No behavior found for '{self.name}', using fallback")
            return self.fallback(state)
    
    def _get_behavior_from_state(self, state: Any) -> Optional[Callable]:
        """Retrieve behavior from state.
        
        Args:
            state: State object
            
        Returns:
            Callable behavior or None
        """
        # Check for nodes dictionary in state
        if hasattr(state, 'nodes'):
            nodes = state.nodes
            if isinstance(nodes, dict) and self.name in nodes:
                behavior = nodes[self.name]
                if callable(behavior):
                    return behavior
                else:
                    logger.warning(f"Node '{self.name}' found in state but not callable")
        
        # Check for fallback behavior in state
        if hasattr(state, 'fallback_behaviors'):
            fallbacks = state.fallback_behaviors
            if isinstance(fallbacks, dict) and self.name in fallbacks:
                return fallbacks[self.name]
        
        return None
    
    def _update_routing(self, state: Any) -> None:
        """Update routing based on state.
        
        Args:
            state: State object
        """
        if hasattr(state, 'routing_table'):
            routing = state.routing_table
            if isinstance(routing, dict) and self.name in routing:
                next_nodes = routing[self.name]
                
                # Set next node(s) in state
                if next_nodes:
                    if len(next_nodes) == 1:
                        # Single next node
                        state.next_node = next_nodes[0]
                        logger.debug(f"Routing from '{self.name}' to '{next_nodes[0]}'")
                    else:
                        # Multiple next nodes (branching)
                        state.next_nodes = next_nodes
                        logger.debug(f"Branching from '{self.name}' to {next_nodes}")
    
    def _default_fallback(self, state: Any) -> Any:
        """Default fallback behavior.
        
        Args:
            state: State object
            
        Returns:
            Unmodified state
        """
        logger.debug(f"StateDrivenNode '{self.name}' using default passthrough")
        return state
    
    @property
    def execution_count(self) -> int:
        """Get number of times this node has executed."""
        return self._execution_count
    
    def reset_execution_count(self) -> None:
        """Reset execution counter."""
        self._execution_count = 0


class StateDrivenBranch:
    """Branch node that determines routing from state.
    
    Unlike StateDrivenNode which executes behavior, this node
    specifically handles conditional routing based on state.
    
    Examples:
        >>> branch = StateDrivenBranch("router")
        >>> 
        >>> # In state, define routing logic
        >>> state.branches["router"] = lambda s: "path_a" if s.score > 0.5 else "path_b"
        >>> 
        >>> # Branch will route based on state
        >>> next_node = branch(state)
    """
    
    def __init__(self, name: str, default_route: Optional[str] = None):
        """Initialize state-driven branch.
        
        Args:
            name: Branch identifier
            default_route: Default route if no logic in state
        """
        self.name = name
        self.default_route = default_route
    
    def __call__(self, state: Any) -> str:
        """Determine next node from state.
        
        Args:
            state: State object
            
        Returns:
            Name of next node to execute
        """
        # Check for branch logic in state
        if hasattr(state, 'branches'):
            branches = state.branches
            if isinstance(branches, dict) and self.name in branches:
                branch_func = branches[self.name]
                if callable(branch_func):
                    try:
                        next_node = branch_func(state)
                        logger.debug(f"Branch '{self.name}' routing to '{next_node}'")
                        return next_node
                    except Exception as e:
                        logger.error(f"Error in branch logic for '{self.name}': {e}")
        
        # Check static routing table
        if hasattr(state, 'routing_table'):
            routing = state.routing_table
            if isinstance(routing, dict) and self.name in routing:
                routes = routing[self.name]
                if routes:
                    return routes[0]
        
        # Use default route
        if self.default_route:
            logger.debug(f"Branch '{self.name}' using default route '{self.default_route}'")
            return self.default_route
        
        # No route found
        logger.warning(f"Branch '{self.name}' has no routing defined")
        return "END"


class DynamicNodeFactory:
    """Factory for creating state-driven nodes.
    
    This factory simplifies the creation of state-driven nodes
    and provides patterns for common use cases.
    """
    
    @staticmethod
    def create_node(name: str, **kwargs) -> StateDrivenNode:
        """Create a state-driven node.
        
        Args:
            name: Node name
            **kwargs: Additional node configuration
            
        Returns:
            StateDrivenNode instance
        """
        fallback = kwargs.get('fallback')
        return StateDrivenNode(name, fallback)
    
    @staticmethod
    def create_branch(name: str, default: str = None) -> StateDrivenBranch:
        """Create a state-driven branch.
        
        Args:
            name: Branch name
            default: Default route
            
        Returns:
            StateDrivenBranch instance
        """
        return StateDrivenBranch(name, default)
    
    @staticmethod
    def create_parallel_node(name: str, nodes: list[str]) -> StateDrivenNode:
        """Create a node that triggers parallel execution.
        
        Args:
            name: Node name
            nodes: List of nodes to execute in parallel
            
        Returns:
            StateDrivenNode that sets up parallel execution
        """
        def parallel_behavior(state):
            """Set up parallel execution."""
            state.parallel_nodes = nodes
            logger.info(f"Node '{name}' triggering parallel execution of {nodes}")
            return state
        
        node = StateDrivenNode(name)
        # Pre-configure with parallel behavior
        node.fallback = parallel_behavior
        return node
    
    @staticmethod
    def create_aggregator_node(name: str, aggregation_key: str = "results") -> StateDrivenNode:
        """Create a node that aggregates results.
        
        Args:
            name: Node name
            aggregation_key: Key in state to store aggregated results
            
        Returns:
            StateDrivenNode that aggregates parallel results
        """
        def aggregator_behavior(state):
            """Aggregate parallel results."""
            if hasattr(state, 'parallel_results'):
                results = state.parallel_results
                setattr(state, aggregation_key, results)
                logger.info(f"Node '{name}' aggregated {len(results)} results")
                # Clear parallel results
                delattr(state, 'parallel_results')
            return state
        
        node = StateDrivenNode(name)
        node.fallback = aggregator_behavior
        return node


class StateNodeSchema(BaseModel):
    """Schema for defining state-driven nodes in configuration.
    
    This allows nodes to be defined declaratively and created
    from configuration rather than code.
    """
    
    name: str = Field(..., description="Node identifier")
    type: str = Field(default="node", description="Node type: node, branch, parallel, aggregator")
    fallback_behavior: Optional[str] = Field(None, description="Name of fallback behavior")
    default_route: Optional[str] = Field(None, description="Default route for branches")
    parallel_nodes: Optional[list[str]] = Field(None, description="Nodes for parallel execution")
    aggregation_key: Optional[str] = Field(None, description="Key for aggregation")
    
    def create_node(self) -> Any:
        """Create the actual node from this schema.
        
        Returns:
            StateDrivenNode or StateDrivenBranch instance
        """
        factory = DynamicNodeFactory()
        
        if self.type == "branch":
            return factory.create_branch(self.name, self.default_route)
        elif self.type == "parallel":
            return factory.create_parallel_node(self.name, self.parallel_nodes or [])
        elif self.type == "aggregator":
            return factory.create_aggregator_node(self.name, self.aggregation_key or "results")
        else:
            return factory.create_node(self.name)