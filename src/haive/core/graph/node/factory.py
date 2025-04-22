# src/haive_core/graph/node/factory.py

import inspect
import logging
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

from haive.core.config.runnable import RunnableConfigManager
from haive.core.engine.base import Engine, InvokableEngine, NonInvokableEngine
from haive.core.graph.node.config import NodeConfig
from pydantic import BaseModel
# Configure logging

# Import rich logging components at the top of the file
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install as install_rich_traceback
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
import time
import inspect
import uuid

# Install rich traceback handler for better exception visualization
install_rich_traceback(show_locals=True)

# Configure rich console and logging
console = Console()
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(console=console, rich_tracebacks=True, tracebacks_show_locals=True)]
)

# Create a logger with the rich handler
logger = logging.getLogger("NodeFactory")

# Create a unique session ID for grouping related logs
session_id = str(uuid.uuid4())[:8]
class NodeFactory:
    """
    Factory for creating node functions with comprehensive engine support.
    
    Handles creation of node functions from different engine types and configurations,
    ensuring proper Command usage for control flow and engine ID targeting.
    """
    
    @classmethod
    def create_node_function(
        cls,
        config: Union[NodeConfig, Engine, Callable],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[RunnableConfig] = None,
        debug: bool = False
    ) -> Callable:
        """
        Create a node function with proper engine handling.
        
        Args:
            config: NodeConfig, Engine, or callable function
            command_goto: Optional next node to go to (ignored if NodeConfig provided)
            input_mapping: Optional mapping from state keys to engine input keys (ignored if NodeConfig provided)
            output_mapping: Optional mapping from engine output keys to state keys (ignored if NodeConfig provided)
            runnable_config: Optional default runtime configuration (ignored if NodeConfig provided)
            debug: Enable debug logging
            
        Returns:
            A node function compatible with LangGraph
        """
        # Create a function_id for tracking this specific creation process
        function_id = str(uuid.uuid4())[:6]
        console.rule(f"[bold blue]Creating Node Function (ID: {function_id})")
        
        # Log input parameters
        console.print(Panel(
            f"[bold]Input Parameters:[/bold]\n"
            f"config: {type(config).__name__}\n"
            f"command_goto: {command_goto}\n"
            f"input_mapping: {input_mapping}\n"
            f"output_mapping: {output_mapping}\n"
            f"debug: {debug}",
            title="Parameters",
            border_style="blue"
        ))
        
        # Convert to NodeConfig if not already
        start_time = time.time()
        if not isinstance(config, NodeConfig):
            node_name = getattr(config, "name", "unnamed_node")
            console.print(f"[yellow]Converting {type(config).__name__} to NodeConfig")
            node_config = NodeConfig(
                name=node_name,
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config,
                debug=debug
            )
            console.print(f"[green]Converted to NodeConfig in {(time.time() - start_time)*1000:.2f}ms")
        else:
            node_config = config
            console.print(f"[green]Using provided NodeConfig: {node_config.name}")
        
        # Resolve engine reference if needed
        console.print("[bold yellow]Resolving engine reference...")
        engine, engine_id = node_config.resolve_engine()
        
        # Create detailed engine info table
        engine_table = Table(title="Engine Information", show_header=True)
        engine_table.add_column("Property", style="cyan")
        engine_table.add_column("Value", style="green")
        
        engine_table.add_row("Type", type(engine).__name__)
        engine_table.add_row("Name", getattr(engine, "name", "N/A"))
        engine_table.add_row("ID", engine_id or "N/A")
        engine_table.add_row("Engine Type", str(getattr(engine, "engine_type", "N/A")))
        
        if hasattr(engine, "model"):
            engine_table.add_row("Model", getattr(engine, "model", "N/A"))
        
        console.print(engine_table)
        
        # Create appropriate node function based on engine type
        console.print(f"[bold blue]Creating node function for engine type: {type(engine).__name__}")
        
        if isinstance(engine, InvokableEngine):
            console.print("[green]Using InvokableEngine implementation")
            node_func = cls._create_invokable_engine_node(engine, node_config, function_id)
        elif isinstance(engine, NonInvokableEngine):
            console.print("[green]Using NonInvokableEngine implementation")
            node_func = cls._create_non_invokable_engine_node(engine, node_config, function_id)
        elif callable(engine):
            console.print("[green]Using callable implementation")
            node_func = cls._create_callable_node(engine, node_config, function_id)
        else:
            console.print(f"[red]Unknown engine type: {type(engine).__name__}. Creating generic node.")
            node_func = cls._create_generic_node(engine, node_config, function_id)
        
        # Add metadata to the function for serialization and node tracking
        node_func.__node_config__ = node_config
        node_func.__engine_id__ = engine_id
        node_func.__function_id__ = function_id
        
        # Log completion
        console.print(f"[bold green]Node function created successfully for {node_config.name}")
        console.rule()
        
        return node_func
        
    @classmethod
    def _create_invokable_engine_node(cls, engine: InvokableEngine, config: NodeConfig, function_id: str = None) -> Callable:
        """Create a node function for an invokable engine."""
        
        def node_function(state, runtime_config=None):
            """Node function that uses engine's invoke method."""
            try:
                # Process state into a dictionary
                processed_state = cls._preprocess_state(state)
                
                # Merge runtime configurations
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting and config overrides
                if engine_id := getattr(engine, "id", None):
                    merged_config = cls._ensure_engine_id_targeting(merged_config, engine_id)
                
                if config.config_overrides and merged_config:
                    merged_config = cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
                
                # --- PREPARE ENGINE INPUT ---
                
                # Extract input based on mapping configuration
                input_data = cls._extract_input(processed_state, config.input_mapping)
                
                # Apply special processing for direct message usage if configured
                if config.use_direct_messages and "messages" in processed_state:
                    # If messages exist in state and direct messaging is enabled,
                    # prepare them as BaseMessage objects for the engine
                    messages_data = processed_state["messages"]
                    if not isinstance(messages_data, list) or not all(isinstance(m, BaseMessage) for m in messages_data):
                        # Convert to BaseMessage objects if needed
                        messages = cls._prepare_messages(messages_data)
                    else:
                        messages = messages_data
                    
                    # Invoke engine with properly formatted messages
                    result = engine.invoke(messages, merged_config)
                else:
                    # Normal invocation with extracted input
                    result = engine.invoke(input_data, merged_config)
                
                # --- PROCESS ENGINE OUTPUT ---
                
                # Handle the result based on its type and the output mapping
                if "messages" in processed_state and isinstance(result, (BaseMessage, list)):
                    # If we have messages in state and result is message-related,
                    # process it with special message handling
                    return cls._handle_message_result(result, processed_state, 
                        processed_state.get("messages", []), config)
                else:
                    # Standard result handling with output mapping
                    return cls._handle_result(result, config.command_goto, config.output_mapping, processed_state)
                    
            except Exception as e:
                # Log error details
                logger.error(f"Error in node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error as Command
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
        
    @classmethod
    def _create_non_invokable_engine_node(cls, engine: NonInvokableEngine, config: NodeConfig) -> Callable:
        """
        Create a node function for a non-invokable engine.
        
        Args:
            engine: The non-invokable engine
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Node function that instantiates the engine."""
            try:
                # Process the state object into a dict if needed
                processed_state = cls._preprocess_state(state)
                
                # Merge runtime configs
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                
                # Apply engine ID targeting
                if engine_id := getattr(engine, "id", None):
                    merged_config = cls._ensure_engine_id_targeting(merged_config, engine_id)
                    
                # Apply node-specific config overrides
                if config.config_overrides and merged_config:
                    merged_config = cls._apply_config_overrides(merged_config, engine_id, config.config_overrides)
                
                # Instantiate the engine
                instance = engine.instantiate(merged_config)
                
                # Return the instance with Command for control flow
                return Command(update={"instance": instance}, goto=config.command_goto)
                
            except Exception as e:
                logger.error(f"Error instantiating engine in node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _create_callable_node(cls, func: Callable, config: NodeConfig, function_id: str) -> Callable:
        """
        Create a node function from a callable.
        
        Args:
            func: The callable function
            config: Node configuration
            function_id: The unique function ID for tracking
            
        Returns:
            Node function
        """
        # Check if function accepts config
        sig = inspect.signature(func)
        accepts_config = "config" in sig.parameters
        
        console.print(f"[bold blue]Creating callable node from function: {func.__name__ if hasattr(func, '__name__') else 'anonymous'}")
        console.print(f"[blue]Function accepts config parameter: {accepts_config}")
        
        def node_function(state, runtime_config=None):
            """Wrapped node function."""
            execution_id = str(uuid.uuid4())[:6]
            
            console.rule(f"[bold magenta]Executing Node: {config.name} (Function: {function_id}, Execution: {execution_id})")
            
            try:
                # Create execution info table
                exec_table = Table(title="Execution Information", show_header=True)
                exec_table.add_column("Property", style="cyan")
                exec_table.add_column("Value", style="green")
                
                exec_table.add_row("Node Name", config.name)
                exec_table.add_row("Function Name", func.__name__ if hasattr(func, "__name__") else "anonymous")
                exec_table.add_row("State Type", type(state).__name__)
                exec_table.add_row("Has Runtime Config", "Yes" if runtime_config else "No")
                
                console.print(exec_table)
                
                # Process the state object into a dict if needed
                console.print("[bold yellow]Preprocessing state...")
                start_time = time.time()
                processed_state = cls._preprocess_state(state)
                console.print(f"[green]State preprocessed in {(time.time() - start_time)*1000:.2f}ms")
                
                # Print state summary
                state_summary = {k: str(type(v).__name__) for k, v in processed_state.items()}
                console.print(Panel(
                    Pretty(state_summary),
                    title="State Keys and Types",
                    border_style="green"
                ))
                
                # Show full state for debugging if not too large
                state_size = len(str(processed_state))
                if config.debug and state_size < 10000:  # Only show full state if it's not too large
                    console.print(Panel(
                        Pretty(processed_state),
                        title="Full State Content",
                        border_style="blue"
                    ))
                else:
                    console.print(f"State content size: {state_size} characters (set debug=True to show)")
                
                # Merge runtime configs
                console.print("[bold yellow]Merging runtime configs...")
                start_time = time.time()
                merged_config = cls._merge_configs(config.runnable_config, runtime_config)
                console.print(f"[green]Configs merged in {(time.time() - start_time)*1000:.2f}ms")
                
                if merged_config:
                    console.print(Panel(
                        Pretty(merged_config),
                        title="Merged Config",
                        border_style="blue"
                    ))
                
                # Call original function with appropriate arguments
                console.print("[bold yellow]Calling wrapped function...")
                start_time = time.time()
                
                try:
                    if accepts_config:
                        console.print("[blue]Calling function with state and config")
                        result = func(processed_state, merged_config)
                    else:
                        console.print("[blue]Calling function with only state")
                        result = func(processed_state)
                    
                    console.print(f"[green]Function call completed in {(time.time() - start_time)*1000:.2f}ms")
                    console.print(f"[cyan]Result type: {type(result).__name__}")
                    
                    # Show result summary
                    if isinstance(result, dict):
                        console.print(Panel(
                            Pretty({k: str(type(v).__name__) for k, v in result.items()}),
                            title="Result Keys and Types",
                            border_style="green"
                        ))
                    elif hasattr(result, "model_dump"):
                        console.print(Panel(
                            Pretty(result.model_dump()),
                            title="Result Content (Pydantic Model)",
                            border_style="green"
                        ))
                    else:
                        console.print(Panel(
                            str(result)[:500] + ("..." if len(str(result)) > 500 else ""),
                            title="Result Content Preview",
                            border_style="green"
                        ))
                    
                    # Handle the result for output mapping and command
                    console.print("[bold yellow]Processing final result...")
                    start_time = time.time()
                    final_result = cls._handle_result(result, config.command_goto, config.output_mapping, processed_state)
                    console.print(f"[green]Result handled in {(time.time() - start_time)*1000:.2f}ms")
                    
                    # Show final result
                    console.print(f"[cyan]Final result type: {type(final_result).__name__}")
                    console.print(Panel(
                        Pretty(final_result),
                        title="Final Result",
                        border_style="blue"
                    ))
                    
                    return final_result
                    
                except Exception as e:
                    console.print(f"[bold red]Error calling wrapped function: {str(e)}")
                    console.print_exception()
                    raise
                    
            except Exception as e:
                console.print(f"[bold red]ERROR in node {config.name}: {str(e)}")
                console.print_exception()
                
                # Capture stack trace for debugging
                tb = traceback.format_exc()
                console.print(Panel(tb, title="Detailed Traceback", border_style="red"))
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                    "traceback": tb
                }
                
                console.print("[yellow]Returning error Command")
                error_result = Command(update={"error": error_data}, goto=config.command_goto)
                console.print(Panel(
                    Pretty(error_result),
                    title="Error Result",
                    border_style="red"
                ))
                
                return error_result
            finally:
                # End execution block
                console.rule(f"[bold magenta]End Execution: {config.name} (Execution: {execution_id})")
        
        return node_function
    
    @classmethod
    def _create_generic_node(cls, obj: Any, config: NodeConfig) -> Callable:
        """
        Create a node function for a non-engine, non-callable object.
        
        Args:
            obj: Any object
            config: Node configuration
            
        Returns:
            Node function
        """
        def node_function(state, runtime_config=None):
            """Generic node function that returns the object as-is."""
            try:
                # Return the object with Command for control flow
                return Command(update={"result": obj}, goto=config.command_goto)
                
            except Exception as e:
                logger.error(f"Error in generic node {config.name}: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Return error with command goto if specified
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
                
                return Command(update={"error": error_data}, goto=config.command_goto)
        
        return node_function
    
    @classmethod
    def _preprocess_state(cls, state: Any) -> Dict[str, Any]:
        """
        Preprocess state object into a dictionary.
        
        Args:
            state: State object (dict, BaseModel, etc.)
            
        Returns:
            Dictionary representation of state
        """
        # Handle different state types
        if isinstance(state, dict):
            # Already a dictionary
            return state
        elif hasattr(state, "model_dump"):
            # Pydantic v2 model
            return state.model_dump()
        elif hasattr(state, "dict"):
            # Pydantic v1 model
            return state.dict()
        elif hasattr(state, "__dict__"):
            # Object with __dict__ attribute
            return state.__dict__
        else:
            # Fallback - just return empty dict
            return {}
    
    @classmethod
    def _prepare_messages(cls, messages_data: Any) -> List[BaseMessage]:
        """
        Prepare messages data into a list of BaseMessage objects.
        
        Args:
            messages_data: Messages in various formats
            
        Returns:
            List of BaseMessage objects
        """
        start = time.time()
        console.print(f"[bold cyan]Preparing messages from data type: {type(messages_data).__name__}")
        
        if not messages_data:
            console.print("[yellow]No messages data provided - returning empty list")
            return []
        
        # Handle different message formats
        if isinstance(messages_data, list):
            console.print(f"[blue]Processing list of {len(messages_data)} items")
            normalized_messages = []
            
            for i, msg in enumerate(messages_data):
                if isinstance(msg, BaseMessage):
                    # Already a proper message
                    normalized_messages.append(msg)
                    console.print(f"[green]Item {i+1}: Already a BaseMessage (type: {msg.type})")
                elif isinstance(msg, tuple) and len(msg) >= 2:
                    # Process tuple messages: (role, content)
                    role, content = msg[0], msg[1]
                    console.print(f"[yellow]Item {i+1}: Converting tuple to Message (role: {role})")
                    
                    if role == "human" or role == "user":
                        normalized_messages.append(HumanMessage(content=content))
                    elif role in ["ai", "assistant"]:
                        normalized_messages.append(AIMessage(content=content))
                    elif role == "system":
                        normalized_messages.append(SystemMessage(content=content))
                    else:
                        # Use ChatMessage for custom roles
                        from langchain_core.messages import ChatMessage
                        normalized_messages.append(ChatMessage(role=role, content=content))
                elif isinstance(msg, dict) and "content" in msg:
                    # Dict-format message with at least content
                    role = msg.get("role", "human")
                    content = msg["content"]
                    console.print(f"[yellow]Item {i+1}: Converting dict to Message (role: {role})")
                    
                    if role == "human" or role == "user":
                        normalized_messages.append(HumanMessage(content=content))
                    elif role in ["ai", "assistant"]:
                        normalized_messages.append(AIMessage(content=content))
                    elif role == "system":
                        normalized_messages.append(SystemMessage(content=content))
                    else:
                        # Use ChatMessage for custom roles
                        from langchain_core.messages import ChatMessage
                        normalized_messages.append(ChatMessage(role=role, content=content))
                elif isinstance(msg, str):
                    # String message - treated as human message
                    console.print(f"[yellow]Item {i+1}: Converting string to HumanMessage")
                    normalized_messages.append(HumanMessage(content=msg))
                else:
                    # Unknown format - convert to string and use as human message
                    console.print(f"[red]Item {i+1}: Unknown format ({type(msg).__name__}) - converting to string HumanMessage")
                    normalized_messages.append(HumanMessage(content=str(msg)))
            
            console.print(f"[green]Normalized {len(messages_data)} messages to {len(normalized_messages)} BaseMessages")
            console.print(f"[dim]Preparation completed in {(time.time() - start)*1000:.2f}ms")
            return normalized_messages
        elif isinstance(messages_data, str):
            # Single string - treat as human message
            console.print("[yellow]Converting single string to HumanMessage")
            return [HumanMessage(content=messages_data)]
        elif isinstance(messages_data, BaseMessage):
            # Single message object
            console.print("[green]Already a BaseMessage - wrapping in list")
            return [messages_data]
        else:
            # Unknown format - convert to string and use as human message
            console.print(f"[red]Unknown format ({type(messages_data).__name__}) - converting to string HumanMessage")
            return [HumanMessage(content=str(messages_data))]
        
    @classmethod
    def _extract_input(cls, state: Dict[str, Any], input_mapping: Optional[Dict[str, str]]) -> Any:
        """
        Extract input from state based on mapping.
        
        Args:
            state: State dictionary
            input_mapping: Mapping from state keys to engine input keys
            
        Returns:
            Input data for engine
        """
        # Show detailed mapping info
        console.print(f"[bold cyan]Extracting input with mapping: {input_mapping}")
        
        start = time.time()
        # If no mapping, return state as-is
        if not input_mapping:
            console.print("[yellow]No input mapping provided - returning full state")
            return state
        
        # Apply mapping
        mapped_input = {}
        for state_key, input_key in input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]
                console.print(f"[green]Mapped: state[{state_key}] → input[{input_key}]")
            elif state_key == "messages" and "messages" not in state:
                # Special handling for messages field - many engines expect this
                # If asked for messages but none in state, provide empty list
                mapped_input[input_key] = []
                console.print(f"[yellow]No 'messages' in state - providing empty list for {input_key}")
            else:
                console.print(f"[red]Warning: State key '{state_key}' not found in state")
        
        # If only one key was mapped and we have that value, return it directly
        if len(input_mapping) == 1 and len(mapped_input) == 1:
            result = list(mapped_input.values())[0]
            console.print(f"[blue]Single mapped value - returning directly: {type(result).__name__}")
            return result
        
        # Return the mapped dictionary
        console.print(f"[blue]Returning mapped input with {len(mapped_input)} keys")
        console.print(f"[dim]Extraction completed in {(time.time() - start)*1000:.2f}ms")
        return mapped_input
    
    @classmethod
    def _handle_message_result(
        cls, 
        result: Any, 
        state: Dict[str, Any],
        input_messages: List[BaseMessage],
        config: NodeConfig
    ) -> Any:
        """
        Handle result specific to message-based interaction.
        
        Args:
            result: Engine result
            state: Original state
            input_messages: Input messages sent to engine
            config: Node configuration
            
        Returns:
            Processed result as Command or dict
        """
        start = time.time()
        console.print(f"[bold cyan]Handling message result of type: {type(result).__name__}")
        
        # Start with state updates
        updates = {}
        console.print("[blue]Processing result and building state updates")
        
        # Handle different result types
        if isinstance(result, BaseMessage):
            # Single message result - update messages with input + new message
            console.print("[green]Result is a single BaseMessage")
            all_messages = input_messages + [result]
            updates["messages"] = all_messages
            console.print(f"[blue]Updated messages list with {len(all_messages)} total messages")
            
            # Extract content if needed
            if config.extract_content:
                updates["content"] = result.content
                console.print(f"[blue]Extracted content ({len(result.content)} chars)")
                
        elif isinstance(result, list) and all(isinstance(msg, BaseMessage) for msg in result):
            # List of messages - use as the new messages
            console.print(f"[green]Result is a list of {len(result)} BaseMessages")
            all_messages = input_messages + result
            updates["messages"] = all_messages
            console.print(f"[blue]Updated messages list with {len(all_messages)} total messages")
            
            # Extract content from last message if needed
            if config.extract_content and result:
                updates["content"] = result[-1].content
                console.print(f"[blue]Extracted content from last message ({len(result[-1].content)} chars)")
                
        elif isinstance(result, dict):
            # Dictionary result
            console.print(f"[green]Result is a dictionary with {len(result)} keys")
            console.print(f"[dim]Keys: {', '.join(result.keys())}")
            
            # If it has a 'generations' key, extract messages
            if "generations" in result:
                try:
                    console.print("[yellow]Found 'generations' key - extracting message")
                    message = result["generations"][0][0].message
                    all_messages = input_messages + [message]
                    updates["messages"] = all_messages
                    console.print(f"[blue]Updated messages with generation (total: {len(all_messages)})")
                    
                    # Extract content if needed
                    if config.extract_content:
                        updates["content"] = message.content
                        console.print(f"[blue]Extracted content from generation ({len(message.content)} chars)")
                        
                except (IndexError, KeyError, AttributeError) as e:
                    # Fallback to full result
                    console.print(f"[red]Error extracting from generations: {str(e)}")
                    updates["result"] = result
                    console.print("[yellow]Falling back to full result")
            
            # If it has a structured output model field
            elif hasattr(config.engine, "structured_output_model") and config.engine.structured_output_model:
                model_name = config.engine.structured_output_model.__name__.lower()
                console.print(f"[blue]Engine has structured output model: {model_name}")
                
                if model_name in result:
                    # Result has a field matching the model name
                    updates[model_name] = result[model_name]
                    console.print(f"[green]Found structured output field: {model_name}")
                    
                # Always include the full result
                updates["output"] = result
                console.print("[blue]Added full result as 'output'")
                
                # Try to find an AI message in the result
                if "message" in result and isinstance(result["message"], BaseMessage):
                    # Update messages with the message
                    console.print("[green]Found BaseMessage in result 'message' field")
                    all_messages = input_messages + [result["message"]]
                    updates["messages"] = all_messages
                    console.print(f"[blue]Updated messages (total: {len(all_messages)})")
                
            else:
                # Apply output mapping if needed
                if config.output_mapping:
                    console.print(f"[blue]Applying output mapping: {config.output_mapping}")
                    for result_key, state_key in config.output_mapping.items():
                        if result_key in result:
                            updates[state_key] = result[result_key]
                            console.print(f"[green]Mapped: result[{result_key}] → state[{state_key}]")
                        else:
                            console.print(f"[yellow]Warning: Result key '{result_key}' not found in result")
                else:
                    # No output mapping - use entire result
                    console.print("[blue]No output mapping - using entire result")
                    updates.update(result)
                    
                # Keep messages in state
                if "messages" in state:
                    updates["messages"] = state["messages"]
                    console.print("[blue]Preserved existing messages from state")
                
                # Check if result has content that should be extracted
                if config.extract_content and "content" in result:
                    updates["content"] = result["content"]
                    console.print(f"[blue]Extracted content from result ({len(result['content'])} chars)")
        
        else:
            # Other result types - store as result
            console.print(f"[yellow]Unhandled result type: {type(result).__name__} - storing as 'result'")
            updates["result"] = result
            
            # Maintain existing messages
            if "messages" in state:
                updates["messages"] = state["messages"]
                console.print("[blue]Preserved existing messages from state")
        
        # Print summary of updates
        console.print(f"[green]Final updates contain {len(updates)} keys: {', '.join(updates.keys())}")
        
        # Return as Command if command_goto is specified
        if config.command_goto is not None:
            console.print(f"[blue]Creating Command with goto: {config.command_goto}")
            result = Command(update=updates, goto=config.command_goto)
        else:
            console.print("[blue]Returning updates dictionary (no command_goto specified)")
            result = updates
        
        console.print(f"[green]Message result handling completed in {(time.time() - start)*1000:.2f}ms")
        return result
        
    @classmethod
    def _handle_result(
        cls, 
        result: Any, 
        command_goto: Optional[str], 
        output_mapping: Optional[Dict[str, str]],
        original_state: Dict[str, Any] = None
    ) -> Any:
        """
        Handle different result types to ensure proper Command/Send pattern support.
        """
        logger.info(f"Handling result of type: {type(result).__name__}")
        
        # Handle already Command/Send results
        if isinstance(result, Command):
            logger.info("Result is already a Command")
            # If already Command, ensure goto is set if not already and one is provided
            if result.goto is None and command_goto is not None:
                logger.info(f"Adding command_goto: {command_goto} to existing Command")
                return Command(
                    update=result.update, 
                    goto=command_goto,
                    resume=result.resume,
                    graph=result.graph
                )
            return result
        elif isinstance(result, Send):
            logger.info("Result is a Send - returning as-is")
            return result
        elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
            logger.info(f"Result is a list of {len(result)} Send objects - returning as-is")
            return result
        
        # Special handling for Pydantic models
        if isinstance(result, BaseModel):
            logger.info(f"Result is a Pydantic model: {result.__class__.__name__}")
            processed_output = {}
            model_name = result.__class__.__name__.lower()
            processed_output[model_name] = result
            logger.info(f"Added model under key: {model_name}")
            
            # Apply output mapping if needed
            if output_mapping:
                logger.info(f"Applying output mapping to model: {output_mapping}")
                for output_key, state_key in output_mapping.items():
                    if output_key.startswith(f"{model_name}."):
                        field = output_key.split(".", 1)[1]
                        if hasattr(result, field):
                            processed_output[state_key] = getattr(result, field)
                            logger.info(f"Mapped {output_key} -> {state_key}")
        else:
            # Process output if it's not already a Command, Send, or Pydantic model
            processed_output = cls._process_output(result, output_mapping, original_state)
            logger.info(f"Processed output has {len(processed_output)} keys")
        
        # Wrap in Command if goto is specified
        if command_goto is not None:
            logger.info(f"Creating Command with goto: {command_goto}")
            return Command(update=processed_output, goto=command_goto)
        else:
            logger.info("Returning processed output directly (no Command)")
            return processed_output
    @classmethod
    def _process_output(
        cls, 
        output: Any, 
        output_mapping: Optional[Dict[str, str]],
        original_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process output according to mapping.
        
        Args:
            output: Output from engine or function
            output_mapping: Mapping from output keys to state keys
            original_state: Original state for preserving data
            
        Returns:
            Processed output
        """
        # Handle non-dict output
        if not isinstance(output, dict):
            # Special case: If output is a BaseMessage, extract content
            if isinstance(output, BaseMessage):
                result = {"content": output.content}
                
                # If we had messages in the original state, append this message
                if original_state and "messages" in original_state:
                    messages = original_state["messages"]
                    if isinstance(messages, list):
                        result["messages"] = messages + [output]
                
                return result
            
            # If output is a string and we have an output mapping with 'text' key,
            # assume this is from a StrOutputParser
            if isinstance(output, str) and output_mapping and 'text' in output_mapping:
                result = {output_mapping['text']: output}
                return result
                
            # Default handling - store as result
            return {"result": output}
        
        # Handle dictionary output
        result = {}
        
        # First check for structured output models in the result
        # Look for keys that match model class names in lowercase
        model_keys = [k for k in output.keys() if k.endswith("model") or k[0].islower() and k[0:1].isupper()]
        
        # If output has a structured model field, we need special handling
        if any(isinstance(output.get(k), BaseModel) for k in output):
            for k, v in output.items():
                if isinstance(v, BaseModel):
                    # Add the model directly under its name
                    result[k] = v
                    
                    # Also add individual fields if in output_mapping
                    if output_mapping:
                        model_name = k.lower()
                        for model_field in v.__dict__:
                            mapping_key = f"{model_name}.{model_field}"
                            if mapping_key in output_mapping:
                                result[output_mapping[mapping_key]] = getattr(v, model_field)
        
        # Apply standard output mapping if provided
        if output_mapping:
            for output_key, state_key in output_mapping.items():
                # Check for direct fields
                if output_key in output:
                    result[state_key] = output[output_key]
                # Check for nested fields (using dot notation)
                elif "." in output_key:
                    parts = output_key.split(".")
                    obj = output
                    valid_path = True
                    
                    # Navigate through the nested structure
                    for part in parts[:-1]:
                        if part in obj and isinstance(obj, dict):
                            obj = obj[part]
                        elif hasattr(obj, part):
                            obj = getattr(obj, part)
                        else:
                            valid_path = False
                            break
                    
                    # Set the value if we found a valid path
                    if valid_path:
                        last_part = parts[-1]
                        if isinstance(obj, dict) and last_part in obj:
                            result[state_key] = obj[last_part]
                        elif hasattr(obj, last_part):
                            result[state_key] = getattr(obj, last_part)
        
        # If we applied mappings but found nothing, or no mapping provided
        # include the original output
        if not result or not output_mapping:
            # For Pydantic models, convert to dict
            if isinstance(output, BaseModel):
                if hasattr(output, "model_dump"):
                    # Pydantic v2
                    result.update(output.model_dump())
                else:
                    # Pydantic v1
                    result.update(output.dict())
            else:
                # Regular dict
                result.update(output)
        
        return result
    
    
    @classmethod
    def _merge_configs(
        cls, 
        base_config: Optional[RunnableConfig], 
        override_config: Optional[RunnableConfig]
    ) -> Optional[RunnableConfig]:
        """
        Merge two RunnableConfigs.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override with
            
        Returns:
            Merged configuration or None if both inputs are None
        """
        # Handle None cases
        if base_config is None and override_config is None:
            return None
        elif base_config is None:
            return override_config
        elif override_config is None:
            return base_config
        
        # Use RunnableConfigManager for merging
        from haive.core.config.runnable import RunnableConfigManager
        return RunnableConfigManager.merge(base_config, override_config)
    
    @classmethod
    def _ensure_engine_id_targeting(
        cls, 
        config: Optional[Dict[str, Any]], 
        engine_id: str
    ) -> Dict[str, Any]:
        """
        Ensure a config includes targeting for a specific engine ID.
        
        Args:
            config: Config dictionary (may be None)
            engine_id: Engine ID to target
            
        Returns:
            Updated config
        """
        if not config:
            config = {}
            
        if "configurable" not in config:
            config["configurable"] = {}
            
        if "engine_configs" not in config["configurable"]:
            config["configurable"]["engine_configs"] = {}
            
        if engine_id not in config["configurable"]["engine_configs"]:
            config["configurable"]["engine_configs"][engine_id] = {}
            
        return config
    
    @classmethod
    def _apply_config_overrides(
        cls, 
        config: Dict[str, Any], 
        engine_id: Optional[str], 
        overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply node-specific configuration overrides to the config.
        
        Args:
            config: Config dictionary to modify
            engine_id: Optional engine ID to target
            overrides: Configuration overrides to apply
            
        Returns:
            Updated config
        """
        # Make a copy to avoid modifying the original
        config = config.copy()
        
        if "configurable" not in config:
            config["configurable"] = {}
        
        if engine_id:
            # Apply to engine-specific configuration
            if "engine_configs" not in config["configurable"]:
                config["configurable"]["engine_configs"] = {}
                
            if engine_id not in config["configurable"]["engine_configs"]:
                config["configurable"]["engine_configs"][engine_id] = {}
                
            # Update with overrides
            for key, value in overrides.items():
                config["configurable"]["engine_configs"][engine_id][key] = value
        else:
            # Apply to top-level configurable
            for key, value in overrides.items():
                config["configurable"][key] = value
                
        return config
    

    @classmethod
    def _debug_print_object(cls, obj, title="Object Details", max_depth=3):
        """Print detailed information about an object."""
        from rich.tree import Tree
        
        def _build_tree(obj, tree, depth=0):
            if depth > max_depth:
                tree.add("[dim]... (max depth reached)")
                return
                
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list, tuple, set)) and depth < max_depth:
                        subtree = tree.add(f"[bold blue]{k} [cyan]({type(v).__name__})")
                        _build_tree(v, subtree, depth + 1)
                    else:
                        tree.add(f"[bold blue]{k}: [green]{type(v).__name__}[/green] = {str(v)[:100]}")
            elif isinstance(obj, (list, tuple, set)):
                if len(obj) > 10:
                    tree.add(f"[dim](showing 10 of {len(obj)} items)")
                    items = list(obj)[:10]
                else:
                    items = obj
                    
                for i, item in enumerate(items):
                    if isinstance(item, (dict, list, tuple, set)) and depth < max_depth:
                        subtree = tree.add(f"[bold blue]{i} [cyan]({type(item).__name__})")
                        _build_tree(item, subtree, depth + 1)
                    else:
                        tree.add(f"[bold blue]{i}: [green]{type(item).__name__}[/green] = {str(item)[:100]}")
            else:
                for attr_name in dir(obj):
                    if attr_name.startswith("_"):
                        continue
                        
                    try:
                        attr = getattr(obj, attr_name)
                        if callable(attr):
                            continue
                            
                        if isinstance(attr, (dict, list, tuple, set)) and depth < max_depth:
                            subtree = tree.add(f"[bold blue]{attr_name} [cyan]({type(attr).__name__})")
                            _build_tree(attr, subtree, depth + 1)
                        else:
                            tree.add(f"[bold blue]{attr_name}: [green]{type(attr).__name__}[/green] = {str(attr)[:100]}")
                    except:
                        tree.add(f"[bold blue]{attr_name}: [red]<error accessing>")
        
        # Create tree visualization
        tree = Tree(f"[bold]{title} [cyan]({type(obj).__name__})")
        _build_tree(obj, tree)
        console.print(tree)