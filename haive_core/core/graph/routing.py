# src/haive/core/router/Router.py

import logging
from typing import (
    Dict, List, Any, Optional, Union, Callable, Type, Set, Tuple, 
    cast, Sequence, Literal
)
from src.haive.core.graph.routers.base import Route, RoutingConfig
from pydantic import BaseModel, Field, ValidationError

from langchain_core.messages import (
    AIMessage, SystemMessage, HumanMessage, BaseMessage, ToolMessage, AnyMessage, ToolCall
)
from langchain_core.tools import BaseTool, create_schema_from_function
from langchain_core.utils.pydantic import is_basemodel_subclass
from langgraph.graph import END
from langgraph.types import Command, Send

from src.haive.core.registry.registy import register_node
from src.haive.core.graph.state_schema import StateSchema
from src.haive.core.graph.ToolManager import tool_manager
from src.haive.core.graph.graph_builder2 import NodeType, NodeConfig
from src.haive.core.graph.routers.conditions import RouteCondition, ToolCallCondition, ContentCondition, StateValueCondition, FunctionCondition, CompositeCondition
# Set up logging
logger = logging.getLogger(__name__)



class ValidationConfig(BaseModel):
    """Configuration for tool call validation."""
    schemas: Dict[str, Type[BaseModel]] = Field(default_factory=dict, description="Tool schemas by name")
    format_error: Optional[Callable[[BaseException, Dict[str, Any], Type[BaseModel]], str]] = Field(
        default=None, description="Function to format validation errors"
    )
    auto_retry: bool = Field(default=True, description="Whether to automatically retry on validation errors")
    retry_destination: Optional[str] = Field(default=None, description="Node to route to for retries")
    add_error_metadata: bool = Field(default=True, description="Whether to add error metadata to tool messages")

def default_format_error(
    error: BaseException,
    call: Dict[str, Any],
    schema: Type[BaseModel],
) -> str:
    """Default error formatting function."""
    return f"{repr(error)}\n\nRespond after fixing all validation errors."

class Router:
    """
    Advanced router for conditional state transitions.
    
    This router supports:
    - Multiple routing conditions based on state, content, tool calls, etc.
    - Prioritization of routes
    - Tool call validation
    - Auto-retry for validation errors
    - Default routes for unmatched conditions
    """
    
    def __init__(self, 
                name: str, 
                default_destination: str = END,
                validation_config: Optional[ValidationConfig] = None):
        """
        Initialize a new router.
        
        Args:
            name: Router name
            default_destination: Default destination if no routes match
            validation_config: Configuration for tool call validation
        """
        self.name = name
        self.default_destination = default_destination
        self.routes: List[Route] = []
        self.validation_config = validation_config
        
        # Register as a node for discovery
        register_node(
            name, 
            tags=["router"], 
            metadata={"type": "router", "default_destination": default_destination}
        )
    
    def add_route(self, route: Route) -> 'Router':
        """
        Add a route to the router.
        
        Args:
            route: Route to add
            
        Returns:
            Self for chaining
        """
        self.routes.append(route)
        # Sort routes by priority (higher first)
        self.routes.sort(key=lambda r: r.condition.priority, reverse=True)
        return self
    
    def add_tool_route(self, 
                      name: str,
                      tool_names: List[str],
                      destination: str,
                      require_all: bool = False,
                      priority: int = 0,
                      description: Optional[str] = None) -> 'Router':
        """
        Add a route based on tool calls.
        
        Args:
            name: Route name
            tool_names: Names of tools to check for
            destination: Destination node
            require_all: Whether all tools must be present
            priority: Route priority
            description: Route description
            
        Returns:
            Self for chaining
        """
        condition = ToolCallCondition(
            tool_names=tool_names,
            require_all=require_all,
            priority=priority
        )
        
        route = Route(
            name=name,
            condition=condition,
            destination=destination,
            description=description or f"Route when tools called: {', '.join(tool_names)}"
        )
        
        return self.add_route(route)
    
    def add_content_route(self,
                         name: str,
                         keywords: List[str],
                         destination: str,
                         require_all: bool = False,
                         case_sensitive: bool = False,
                         message_type: Optional[str] = None,
                         priority: int = 0,
                         description: Optional[str] = None) -> 'Router':
        """
        Add a route based on message content.
        
        Args:
            name: Route name
            keywords: Keywords to check for
            destination: Destination node
            require_all: Whether all keywords must be present
            case_sensitive: Whether keyword matching is case sensitive
            message_type: Type of message to check
            priority: Route priority
            description: Route description
            
        Returns:
            Self for chaining
        """
        condition = ContentCondition(
            keywords=keywords,
            require_all=require_all,
            case_sensitive=case_sensitive,
            message_type=message_type,
            priority=priority
        )
        
        route = Route(
            name=name,
            condition=condition,
            destination=destination,
            description=description or f"Route when content contains: {', '.join(keywords)}"
        )
        
        return self.add_route(route)
    
    def add_state_route(self,
                       name: str,
                       key: str,
                       value: Any,
                       destination: str,
                       comparison: str = "==",
                       priority: int = 0,
                       description: Optional[str] = None) -> 'Router':
        """
        Add a route based on state values.
        
        Args:
            name: Route name
            key: State key to check
            value: Value to compare against
            destination: Destination node
            comparison: Comparison type
            priority: Route priority
            description: Route description
            
        Returns:
            Self for chaining
        """
        condition = StateValueCondition(
            key=key,
            value=value,
            comparison=comparison,
            priority=priority
        )
        
        route = Route(
            name=name,
            condition=condition,
            destination=destination,
            description=description or f"Route when {key} {comparison} {value}"
        )
        
        return self.add_route(route)
    
    def add_function_route(self,
                          name: str,
                          function: Callable[[Dict[str, Any]], bool],
                          destination: str,
                          priority: int = 0,
                          description: Optional[str] = None) -> 'Router':
        """
        Add a route based on a custom function.
        
        Args:
            name: Route name
            function: Function to evaluate
            destination: Destination node
            priority: Route priority
            description: Route description
            
        Returns:
            Self for chaining
        """
        condition = FunctionCondition(
            function=function,
            priority=priority
        )
        
        route = Route(
            name=name,
            condition=condition,
            destination=destination,
            description=description or f"Route based on function: {function.__name__}"
        )
        
        return self.add_route(route)
    
    def add_composite_route(self,
                           name: str,
                           conditions: List[RouteCondition],
                           destination: str,
                           operator: str = "and",
                           priority: int = 0,
                           description: Optional[str] = None) -> 'Router':
        """
        Add a route based on multiple conditions.
        
        Args:
            name: Route name
            conditions: List of conditions
            destination: Destination node
            operator: Logical operator (and, or, not)
            priority: Route priority
            description: Route description
            
        Returns:
            Self for chaining
        """
        condition = CompositeCondition(
            conditions=conditions,
            operator=operator,
            priority=priority
        )
        
        route = Route(
            name=name,
            condition=condition,
            destination=destination,
            description=description or f"Route based on {operator} of {len(conditions)} conditions"
        )
        
        return self.add_route(route)
    
    def set_validation_config(self, config: ValidationConfig) -> 'Router':
        """
        Set the validation configuration for tool calls.
        
        Args:
            config: Validation configuration
            
        Returns:
            Self for chaining
        """
        self.validation_config = config
        return self
    
    def add_tool_schema(self, tool_name: str, schema: Type[BaseModel]) -> 'Router':
        """
        Add a schema for tool validation.
        
        Args:
            tool_name: Name of the tool
            schema: Pydantic model for validation
            
        Returns:
            Self for chaining
        """
        # Initialize validation config if needed
        if not self.validation_config:
            self.validation_config = ValidationConfig(
                format_error=default_format_error
            )
        
        # Add schema
        self.validation_config.schemas[tool_name] = schema
        return self
    
    def add_tool_schemas(self, 
                        schemas: List[Union[BaseTool, Type[BaseModel], Callable]]) -> 'Router':
        """
        Add multiple schemas for tool validation.
        
        Args:
            schemas: List of tools, models, or functions to use as schemas
            
        Returns:
            Self for chaining
        """
        # Initialize validation config if needed
        if not self.validation_config:
            self.validation_config = ValidationConfig(
                format_error=default_format_error
            )
        
        # Process each schema
        for schema in schemas:
            if isinstance(schema, BaseTool):
                if schema.args_schema is None:
                    logger.warning(f"Tool {schema.name} does not have an args_schema defined")
                    continue
                elif not isinstance(schema.args_schema, type) or not is_basemodel_subclass(schema.args_schema):
                    logger.warning(f"Tool {schema.name} does not have a valid args_schema")
                    continue
                self.validation_config.schemas[schema.name] = schema.args_schema
            elif isinstance(schema, type) and issubclass(schema, BaseModel):
                self.validation_config.schemas[schema.__name__] = schema
            elif callable(schema):
                model = create_schema_from_function(f"{schema.__name__}Schema", schema)
                self.validation_config.schemas[schema.__name__] = model
            else:
                logger.warning(f"Unsupported schema type: {type(schema)}")
        
        return self
    
    def create_router_function(self) -> Callable[[Dict[str, Any]], Union[str, Command]]:
        """
        Create a router function for use in a graph.
        
        Returns:
            Router function
        """
        def router_function(state: Dict[str, Any]) -> Union[str, Command]:
            """Route based on state conditions."""
            logger.debug(f"Router {self.name} processing state")
            
            # First, check if we need to validate tool calls
            if self.validation_config and self.validation_config.schemas:
                validated_state = self._validate_tool_calls(state)
                if validated_state is not state:
                    # Validation produced updates
                    # Check if we need to retry
                    if self.validation_config.auto_retry and self.validation_config.retry_destination:
                        for msg in validated_state.get("messages", []):
                            if (isinstance(msg, ToolMessage) and 
                                msg.additional_kwargs and 
                                msg.additional_kwargs.get("is_error")):
                                logger.info(f"Validation error detected, routing to {self.validation_config.retry_destination}")
                                return Command(update=validated_state, goto=self.validation_config.retry_destination)
                    
                    # No retry needed, use the validated state
                    state = validated_state
            
            # Evaluate routes in priority order
            for route in self.routes:
                try:
                    if route.condition.evaluate(state):
                        logger.info(f"Route matched: {route.name} -> {route.destination}")
                        return route.destination
                except Exception as e:
                    logger.error(f"Error evaluating route {route.name}: {e}")
            
            # No routes matched, use default destination
            logger.info(f"No routes matched, using default: {self.default_destination}")
            return self.default_destination
        
        return router_function
    
    def _validate_tool_calls(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool calls in the state.
        
        Args:
            state: Current state
            
        Returns:
            Updated state with validation results
        """
        if not self.validation_config or not self.validation_config.schemas:
            return state
        
        # Extract messages
        messages = state.get("messages", [])
        if not messages:
            return state
        
        # Find the last AI message with tool calls
        ai_message = None
        for msg in reversed(messages):
            if (isinstance(msg, AIMessage) or 
                (hasattr(msg, "type") and msg.type == "ai")) and hasattr(msg, "tool_calls"):
                ai_message = msg
                break
        
        if not ai_message or not ai_message.tool_calls:
            return state
        
        # Create a copy of the state and messages for updates
        updated_state = dict(state)
        updated_messages = list(messages)
        
        # Process each tool call
        tool_messages = []
        validation_error = False
        
        for call in ai_message.tool_calls:
            # Skip if tool is not in our schemas
            if call["name"] not in self.validation_config.schemas:
                continue
            
            schema = self.validation_config.schemas[call["name"]]
            
            try:
                # Validate against schema
                output = schema.model_validate(call["args"])
                content = output.model_dump_json()
                
                # Create tool message
                tool_message = ToolMessage(
                    content=content,
                    name=call["name"],
                    tool_call_id=call["id"]
                )
                
            except ValidationError as e:
                # Format error
                format_error = self.validation_config.format_error or default_format_error
                error_content = format_error(e, call, schema)
                
                # Create error tool message
                tool_message = ToolMessage(
                    content=error_content,
                    name=call["name"],
                    tool_call_id=call["id"]
                )
                
                # Add error metadata if configured
                if self.validation_config.add_error_metadata:
                    tool_message.additional_kwargs = {"is_error": True}
                
                validation_error = True
            
            # Add to messages
            tool_messages.append(tool_message)
        
        # Add tool messages to state
        if tool_messages:
            updated_messages.extend(tool_messages)
            updated_state["messages"] = updated_messages
        
        return updated_state
    
    def to_node_config(self) -> NodeConfig:
        """
        Convert to a node configuration for use with NodeFactory.
        
        Returns:
            NodeConfig for this router
        """
        from src.haive.core.graph.graph_builder2 import NodeConfig, NodeType, RoutingConfig
        
        # Create routing config
        routing_config = RoutingConfig(
            default_destination=self.default_destination,
            is_dynamic=True
        )
        
        # Create the node config
        node_config = NodeConfig(
            name=self.name,
            function=self.create_router_function(),
            description=f"Router with {len(self.routes)} routes",
            node_type=NodeType.ROUTER,
            routing=routing_config,
            tags=["router"]
        )
        
        return node_config


# Create factory functions for easy creation
def create_router(name: str, default_destination: str = END) -> Router:
    """
    Create a new router.
    
    Args:
        name: Router name
        default_destination: Default destination
        
    Returns:
        Router instance
    """
    return Router(name=name, default_destination=default_destination)

def create_tool_router(name: str, 
                      tool_schemas: List[Union[BaseTool, Type[BaseModel], Callable]],
                      default_destination: str = END,
                      retry_destination: Optional[str] = None) -> Router:
    """
    Create a router with tool validation.
    
    Args:
        name: Router name
        tool_schemas: Schemas for tools
        default_destination: Default destination
        retry_destination: Destination for validation retries
        
    Returns:
        Router instance
    """
    router = Router(name=name, default_destination=default_destination)
    
    # Create validation config
    validation_config = ValidationConfig(
        auto_retry=retry_destination is not None,
        retry_destination=retry_destination,
        format_error=default_format_error,
        add_error_metadata=True
    )
    
    router.set_validation_config(validation_config)
    router.add_tool_schemas(tool_schemas)
    
    return router

def create_content_router(name: str,
                         routes: List[Tuple[str, List[str], str]],
                         default_destination: str = END) -> Router:
    """
    Create a router based on message content.
    
    Args:
        name: Router name
        routes: List of (route_name, keywords, destination) tuples
        default_destination: Default destination
        
    Returns:
        Router instance
    """
    router = Router(name=name, default_destination=default_destination)
    
    # Add routes
    for i, (route_name, keywords, destination) in enumerate(routes):
        router.add_content_route(
            name=route_name,
            keywords=keywords,
            destination=destination,
            priority=len(routes) - i  # Higher priority for earlier routes
        )
    
    return router