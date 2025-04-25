# src/haive/core/graph/node/factory.py

from typing import Any, Dict, List, Optional, Callable, Union, Awaitable
import logging
import asyncio
import traceback
from datetime import datetime
import os
import uuid
import inspect
from langgraph.types import Command, Send
from langgraph.graph import END

from haive.core.graph.node.config import NodeConfig
from haive.core.graph.node.registry import NodeTypeRegistry
from haive.core.registry.manager import RegistryManager
from haive.core.config.runnable import RunnableConfigManager
from haive.core.engine.base import Engine, EngineType

# Setup enhanced logging with rich if available
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.traceback import install as install_rich_traceback
    from rich.pretty import Pretty
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.tree import Tree
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.logging import RichHandler
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    
    # Install rich traceback handling
    install_rich_traceback(show_locals=True, width=120, word_wrap=True)
    
    # Create consoles for normal and debug output
    console = Console()
    debug_console = Console(stderr=True, width=120)
    
    # Set up the rich handler for logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=debug_console)]
    )
    
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    
# Setup logging
logger = logging.getLogger(__name__)

# Log setup status
if RICH_AVAILABLE:
    logger.info("Rich UI debugging enabled for NodeFactory")
    try:
        debug_console.print(Panel.fit(
            "[bold green]Rich UI Debugging Enabled[/bold green]\n"
            "[blue]Enhanced visualization for node creation and execution[/blue]",
            title="NodeFactory Debug Mode",
            border_style="green"
        ))
    except Exception as e:
        logger.error(f"Error initializing Rich UI: {e}")
else:
    logger.info("Rich UI not available, using standard logging")

class NodeFactory:
    """
    Factory for creating node functions with comprehensive engine support.
    
    Handles creation of node functions from different engine types and configurations,
    ensuring proper Command usage for control flow and engine ID targeting.
    """
    
    # Class-level registry reference
    _registry = None
    
    # Debug mode
    debug_mode = False
    rich_debug = RICH_AVAILABLE
    
    # Debug log path
    debug_log_path = None
    
    @classmethod
    def set_debug(cls, enabled: bool = True, rich_ui: bool = True, log_path: Optional[str] = None) -> None:
        """
        Set debug mode for the NodeFactory.
        
        Args:
            enabled: Whether to enable debugging
            rich_ui: Whether to use rich UI for debugging
            log_path: Optional path to write debug logs to
        """
        cls.debug_mode = enabled
        cls.rich_debug = rich_ui and RICH_AVAILABLE
        cls.debug_log_path = log_path
        
        if enabled:
            logger.setLevel(logging.DEBUG)
            logger.info(f"Debug mode enabled for NodeFactory (rich_ui={cls.rich_debug})")
            
            if cls.rich_debug:
                try:
                    debug_console.print(Panel.fit(
                        "[bold green]Debug Mode Enabled[/bold green]\n"
                        f"[blue]Log Path: {log_path or 'Not set'}[/blue]",
                        title="NodeFactory Configuration",
                        border_style="green"
                    ))
                except Exception as e:
                    logger.error(f"Error showing debug panel: {e}")
            
            if log_path:
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
                    
                    # Add file handler to logger
                    file_handler = logging.FileHandler(log_path)
                    file_handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    
                    logger.info(f"Debug logs will be written to {log_path}")
                except Exception as e:
                    logger.error(f"Error setting up debug log file: {e}")
        else:
            logger.setLevel(logging.INFO)
            logger.info("Debug mode disabled for NodeFactory")
    
    @classmethod
    def debug_log(cls, message: str, level: str = "debug") -> None:
        """
        Log a debug message if debug mode is enabled.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error, critical)
        """
        if not cls.debug_mode:
            return
            
        # Log with standard logger
        log_func = getattr(logger, level, logger.debug)
        log_func(message)
        
        # Log with rich if enabled
        if cls.rich_debug and RICH_AVAILABLE:
            level_style = {
                "debug": "[blue]DEBUG[/blue]",
                "info": "[green]INFO[/green]",
                "warning": "[yellow]WARNING[/yellow]",
                "error": "[red]ERROR[/red]",
                "critical": "[bold red]CRITICAL[/bold red]"
            }
            style = level_style.get(level, "[blue]DEBUG[/blue]")
            debug_console.print(f"{style} {message}")
            
        # Save to file if configured
        if cls.debug_log_path:
            try:
                with open(cls.debug_log_path, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] [{level.upper()}] {message}\n")
            except Exception as e:
                logger.warning(f"Could not write to debug log file: {e}")
                
    @classmethod
    def debug_dict(cls, title: str, data: Dict[str, Any]) -> None:
        """
        Log a dictionary with rich formatting if debug is enabled.
        
        Args:
            title: Title for the data
            data: Dictionary to log
        """
        if not cls.debug_mode:
            return
            
        # Log with standard logger
        logger.debug(f"{title}: {data}")
        
        # Log with rich if enabled
        if cls.rich_debug and RICH_AVAILABLE:
            debug_console.print(f"[bold]{title}:[/bold]")
            debug_console.print(Pretty(data))
            
        # Save to file if configured
        if cls.debug_log_path:
            try:
                import json
                with open(cls.debug_log_path, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"[{timestamp}] [DEBUG] {title}: {json.dumps(data, default=str)}\n")
            except Exception as e:
                logger.warning(f"Could not write to debug log file: {e}")
    
    @classmethod
    def get_registry(cls) -> NodeTypeRegistry:
        """Get the node type registry."""
        if cls._registry is None:
            cls.debug_log("Initializing NodeTypeRegistry")
            
            # Try to get from RegistryManager first
            try:
                manager = RegistryManager.get_instance()
                registry = manager.get_registry("node")
                if isinstance(registry, NodeTypeRegistry):
                    cls._registry = registry
                    cls.debug_log("Using NodeTypeRegistry from RegistryManager")
                else:
                    cls._registry = NodeTypeRegistry.get_instance()
                    cls.debug_log("Creating new NodeTypeRegistry instance")
            except (ImportError, AttributeError) as e:
                cls.debug_log(f"Error getting registry from manager: {e}", level="warning")
                cls._registry = NodeTypeRegistry.get_instance()
                cls.debug_log("Falling back to direct NodeTypeRegistry instance")
                
            # Register processors if needed
            if not cls._registry.node_processors:
                cls.debug_log("Registering default processors")
                cls._registry.register_default_processors()
                
        return cls._registry
    
    @classmethod
    def set_registry(cls, registry: Any) -> None:
        """Set the node type registry."""
        cls._registry = registry
        cls.debug_log(f"Set custom registry: {type(registry).__name__}")
    
    @classmethod
    def create_node_function(
        cls,
        config: Union[NodeConfig, Any],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        async_mode: Optional[bool] = None
    ) -> Callable:
        """
        Create a node function with proper engine handling.
        
        Args:
            config: NodeConfig, Engine, or callable function
            command_goto: Optional next node to go to (ignored if NodeConfig provided)
            input_mapping: Optional mapping from state keys to engine input keys
            output_mapping: Optional mapping from engine output keys to state keys
            runnable_config: Optional default runtime configuration
            debug: Enable debug logging
            async_mode: Whether to create an asynchronous node function
            
        Returns:
            A node function compatible with LangGraph
        """
        # Start progress tracking if rich is available
        if RICH_AVAILABLE and (debug or cls.debug_mode) and cls.rich_debug:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=debug_console
            ) as progress:
                # Create the task
                task = progress.add_task("[cyan]Creating node function...", total=1)
                
                # Call internal implementation
                result = cls._create_node_function_impl(
                    config, command_goto, input_mapping, output_mapping,
                    runnable_config, debug, async_mode, progress, task
                )
                
                # Mark task as complete
                progress.update(task, completed=1)
                
                return result
        else:
            # Call without progress tracking
            return cls._create_node_function_impl(
                config, command_goto, input_mapping, output_mapping,
                runnable_config, debug, async_mode
            )
    
    @classmethod
    def _create_node_function_impl(
        cls,
        config: Union[NodeConfig, Any],
        command_goto: Optional[str] = None,
        input_mapping: Optional[Dict[str, str]] = None,
        output_mapping: Optional[Dict[str, str]] = None,
        runnable_config: Optional[Dict[str, Any]] = None,
        debug: bool = False,
        async_mode: Optional[bool] = None,
        progress = None,
        task = None
    ) -> Callable:
        """Internal implementation for create_node_function with progress tracking."""
        # Enable verbose logging if debug mode
        debug_enabled = debug or cls.debug_mode
        
        if debug_enabled:
            cls.debug_log(f"Creating node function from config: {type(config).__name__}")
            
            # Log the config details
            if isinstance(config, NodeConfig):
                cls.debug_log(f"Using provided NodeConfig: {config.name}")
            elif isinstance(config, Engine):
                cls.debug_log(f"Using Engine: {getattr(config, 'name', 'unknown')} ({type(config).__name__})")
            elif callable(config):
                cls.debug_log(f"Using callable: {getattr(config, '__name__', 'unnamed')}")
            else:
                cls.debug_log(f"Using unknown config type: {type(config).__name__}")
        
        # Update progress if available
        if progress and task:
            progress.update(task, description="[cyan]Converting config to NodeConfig")
            
        # Convert to NodeConfig if not already
        if not isinstance(config, NodeConfig):
            # Create a NodeConfig
            node_config = NodeConfig(
                name=getattr(config, "name", "unnamed_node"),
                engine=config,
                command_goto=command_goto,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                runnable_config=runnable_config,
                debug=debug_enabled,
                rich_debug=cls.rich_debug,
                async_mode=async_mode  # May be None
            )
            
            if debug_enabled:
                cls.debug_log(f"Created NodeConfig with name: {node_config.name}")
        else:
            node_config = config
            # Update debug settings
            if debug_enabled:
                node_config.debug = True
                node_config.rich_debug = cls.rich_debug
            
            # Only override async_mode if explicitly set
            if async_mode is not None:
                node_config.async_mode = async_mode
                if debug_enabled:
                    cls.debug_log(f"Set async_mode to {async_mode}")
        
        # Show detailed config if debugging
        if debug_enabled and cls.rich_debug and RICH_AVAILABLE:
            config_table = Table(title=f"Node Configuration: {node_config.name}")
            config_table.add_column("Property", style="cyan")
            config_table.add_column("Value", style="green")
            
            # Add basic properties
            config_table.add_row("name", node_config.name)
            
            # Add engine info
            if isinstance(node_config.engine, Engine):
                config_table.add_row("engine", f"{getattr(node_config.engine, 'name', 'unnamed')} ({node_config.engine.__class__.__name__})")
                if hasattr(node_config.engine, "id"):
                    config_table.add_row("engine_id", getattr(node_config.engine, "id", "None"))
            elif isinstance(node_config.engine, str):
                config_table.add_row("engine", f"{node_config.engine} (reference)")
            elif callable(node_config.engine):
                config_table.add_row("engine", f"{getattr(node_config.engine, '__name__', 'callable')} (function)")
            else:
                config_table.add_row("engine", "None")
            
            # Add command flow
            if node_config.command_goto == END:
                config_table.add_row("command_goto", "END")
            elif isinstance(node_config.command_goto, str):
                config_table.add_row("command_goto", node_config.command_goto)
            elif isinstance(node_config.command_goto, Send):
                config_table.add_row("command_goto", f"Send to {node_config.command_goto.node}")
            elif isinstance(node_config.command_goto, list):
                goto_str = ", ".join(str(g) for g in node_config.command_goto)
                config_table.add_row("command_goto", f"[{goto_str}]")
            else:
                config_table.add_row("command_goto", "None")
                
            # Add other properties
            config_table.add_row("async_mode", str(node_config.async_mode))
            config_table.add_row("node_type", str(node_config.node_type))
            config_table.add_row("debug", str(node_config.debug))
            
            # Show mappings
            if node_config.input_mapping:
                mapping_str = ", ".join(f"{k}->{v}" for k, v in node_config.input_mapping.items())
                config_table.add_row("input_mapping", mapping_str)
            else:
                config_table.add_row("input_mapping", "None")
                
            if node_config.output_mapping:
                mapping_str = ", ".join(f"{k}->{v}" for k, v in node_config.output_mapping.items())
                config_table.add_row("output_mapping", mapping_str)
            else:
                config_table.add_row("output_mapping", "None")
            
            # Show the table
            debug_console.print(config_table)
        
        # Update progress if available
        if progress and task:
            progress.update(task, description="[cyan]Getting registry")
            
        # Get registry
        registry = cls.get_registry()
        node_config.registry = registry
        
        # Update progress if available
        if progress and task:
            progress.update(task, description="[cyan]Resolving engine reference")
            
        # Resolve engine reference if needed
        try:
            engine, engine_id = node_config.resolve_engine(registry)
            
            if debug_enabled:
                cls.debug_log(f"Resolved engine: {type(engine).__name__}, id={engine_id}")
            
            # Auto-detect async_mode if not specified
            if node_config.async_mode is None:
                # Check if engine supports async operations
                has_ainvoke = hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke"))
                is_async_func = asyncio.iscoroutinefunction(engine) if callable(engine) else False
                node_config.async_mode = has_ainvoke or is_async_func
                
                if debug_enabled:
                    cls.debug_log(f"Auto-detected async_mode={node_config.async_mode} (has_ainvoke={has_ainvoke}, is_async_func={is_async_func})")
            
            # Ensure engine has been converted to a runnable when appropriate
            if isinstance(engine, Engine):
                # Create runnable in advance to detect any initialization issues
                if debug_enabled:
                    cls.debug_log(f"Pre-creating runnable for engine {engine.__class__.__name__}")
                
                try:
                    runnable = engine.create_runnable(node_config.runnable_config)
                    if debug_enabled:
                        cls.debug_log(f"Successfully created runnable: {type(runnable).__name__}")
                except Exception as e:
                    if debug_enabled:
                        cls.debug_log(f"Error creating runnable: {e}", level="error")
                        cls.debug_log(traceback.format_exc(), level="error")
            
        except Exception as e:
            if debug_enabled:
                cls.debug_log(f"Error resolving engine: {e}", level="error")
                cls.debug_log(traceback.format_exc(), level="error")
            engine, engine_id = node_config.engine, None
        
        # Update progress if available
        if progress and task:
            progress.update(task, description="[cyan]Determining node type")
            
        # Determine node type
        if node_config.async_mode:
            # For async mode, prefer async node types
            if hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke")):
                node_type = "async_invokable"
                if debug_enabled:
                    cls.debug_log(f"Using async_invokable processor type")
            elif asyncio.iscoroutinefunction(engine):
                node_type = "async"
                if debug_enabled:
                    cls.debug_log(f"Using async processor type")
            else:
                # Fallback to regular node type detection
                node_type = node_config.determine_node_type()
                if debug_enabled:
                    cls.debug_log(f"Fell back to node type: {node_type}")
        else:
            node_type = node_config.determine_node_type()
            if debug_enabled:
                cls.debug_log(f"Determined node type: {node_type}")
        
        # Update progress if available
        if progress and task:
            progress.update(task, description=f"[cyan]Getting processor for {node_type}")
            
        # Get processor for this node type
        processor = registry.get_node_processor(node_type)
        
        # If no processor found, try to find by capability
        if processor is None:
            if debug_enabled:
                cls.debug_log(f"No processor for {node_type}, searching by capability")
                
            processor = registry.find_processor_for_engine(engine)
            
        # If still no processor, use generic
        if processor is None:
            if debug_enabled:
                cls.debug_log("No processor found, using generic")
                
            processor = registry.get_node_processor("generic")
            if processor is None:
                error_msg = f"No processor found for {node_type}"
                if debug_enabled:
                    cls.debug_log(error_msg, level="error")
                raise ValueError(error_msg)
        
        # Update progress if available
        if progress and task:
            progress.update(task, description=f"[cyan]Creating node function with {processor.__class__.__name__}")
            
        # Create node function
        try:
            node_func = processor.create_node_function(engine, node_config)
            if debug_enabled:
                cls.debug_log(f"Created node function with processor {processor.__class__.__name__}")
        except Exception as e:
            if debug_enabled:
                cls.debug_log(f"Error creating node function: {e}", level="error")
                cls.debug_log(traceback.format_exc(), level="error")
                
            # Create fallback error function
            def error_func(state, runtime_config=None):
                error_msg = f"Using fallback error function for {node_config.name}"
                if debug_enabled:
                    cls.debug_log(error_msg, level="error")
                    
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
                
                if node_config.command_goto:
                    return Command(update={"error": error_data}, goto=node_config.command_goto)
                return {"error": error_data}
                
            node_func = error_func
        
        # Add debug wrapper if needed
        if node_config.debug:
            # Use appropriate debug wrapper for sync or async
            if debug_enabled:
                cls.debug_log(f"Adding {'async' if node_config.async_mode else 'sync'} debug wrapper")
                
            if node_config.async_mode:
                node_func = cls._create_async_debug_wrapper(node_func, node_config)
            else:
                node_func = cls._create_debug_wrapper(node_func, node_config)
        
        # Add metadata
        node_func.__node_config__ = node_config
        node_func.__engine_id__ = engine_id
        node_func.__node_type__ = node_type
        node_func.__async_mode__ = node_config.async_mode
        node_func.__debug_id__ = f"node-{uuid.uuid4().hex[:8]}"
        
        # Update progress if available
        if progress and task:
            progress.update(task, description=f"[green]Node function created successfully")
            
        if debug_enabled:
            cls.debug_log(f"Node function created successfully for {node_config.name}")
            
            # Show function metadata if rich is available
            if cls.rich_debug and RICH_AVAILABLE:
                metadata_table = Table(title="Node Function Metadata")
                metadata_table.add_column("Property", style="cyan")
                metadata_table.add_column("Value", style="green")
                
                metadata_table.add_row("name", node_config.name)
                metadata_table.add_row("node_type", node_type)
                metadata_table.add_row("async_mode", str(node_config.async_mode))
                metadata_table.add_row("engine_id", str(engine_id))
                metadata_table.add_row("debug_id", node_func.__debug_id__)
                
                # Show signature if possible
                try:
                    sig = inspect.signature(node_func)
                    metadata_table.add_row("signature", str(sig))
                except Exception:
                    metadata_table.add_row("signature", "Not available")
                
                debug_console.print(metadata_table)
        
        return node_func
    
    @classmethod
    def _create_debug_wrapper(cls, node_func, config):
        """Create a debug wrapper for synchronous node functions."""
        # Create a unique ID for this function wrapper
        wrapper_id = f"debug-{uuid.uuid4().hex[:8]}"
        
        def debug_wrapper(state, runtime_config=None):
            """Debug wrapper for node functions."""
            function_logger = logging.getLogger(f"node.{config.name}")
            
            # Generate unique execution ID
            exec_id = f"exec-{uuid.uuid4().hex[:8]}"
            
            # Log execution start with rich UI if available
            if cls.rich_debug and RICH_AVAILABLE:
                debug_console.rule(f"[bold blue]Node Execution: [green]{config.name}[/green][/bold blue]")
                debug_console.print(f"[bold]Execution ID:[/bold] {exec_id}")
                debug_console.print(f"[bold]Node Type:[/bold] {getattr(node_func, '__node_type__', 'unknown')}")
                debug_console.print(f"[bold]Async Mode:[/bold] {getattr(node_func, '__async_mode__', 'unknown')}")
                
                # Show state info
                function_logger.debug(f"Node {config.name} called with state type: {type(state).__name__}")
                debug_console.print(f"[bold]State Type:[/bold] {type(state).__name__}")
                
                if isinstance(state, dict):
                    function_logger.debug(f"State keys: {list(state.keys())}")
                    debug_console.print(f"[bold]State Keys:[/bold] {list(state.keys())}")
                    
                    # Show sample of large keys
                    for key, value in state.items():
                        if isinstance(value, list) and len(value) > 5:
                            debug_console.print(f"[bold]{key}:[/bold] List with {len(value)} items (showing first 3)")
                            debug_console.print(Pretty(value[:3]))
                        elif isinstance(value, dict) and len(value) > 5:
                            debug_console.print(f"[bold]{key}:[/bold] Dict with {len(value)} items (showing first 3)")
                            preview = {k: value[k] for k in list(value.keys())[:3]}
                            debug_console.print(Pretty(preview))
                        elif isinstance(value, str) and len(value) > 500:
                            debug_console.print(f"[bold]{key}:[/bold] String with {len(value)} chars (showing first 500)")
                            debug_console.print(Text(value[:500] + "..."))
                
                # Show runtime config info if provided
                if runtime_config:
                    debug_console.print(f"[bold]Runtime Config:[/bold]")
                    debug_console.print(Pretty(runtime_config))
            else:
                # Standard logging
                function_logger.debug(f"Node {config.name} called with state type: {type(state).__name__}")
                if isinstance(state, dict):
                    function_logger.debug(f"State keys: {list(state.keys())}")
            
            # Track execution timing
            start_time = datetime.now()
            
            # Try to execute the node function
            try:
                # Execute the actual node function
                result = node_func(state, runtime_config)
                
                # Calculate execution time
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Log execution result with rich UI if available
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[bold green]Execution successful[/bold green] in {execution_time:.3f}s")
                    debug_console.print(f"[bold]Result type:[/bold] {type(result).__name__}")
                    
                    # Show different details based on result type
                    if isinstance(result, Command):
                        debug_console.print(f"[bold]Command:[/bold] goto={result.goto}")
                        if hasattr(result, "update") and result.update:
                            debug_console.print(f"[bold]Command update keys:[/bold] {list(result.update.keys()) if isinstance(result.update, dict) else 'Not a dict'}")
                            debug_console.print(f"[bold]Command update:[/bold]")
                            debug_console.print(Pretty(result.update))
                    elif isinstance(result, Send):
                        debug_console.print(f"[bold]Send:[/bold] node={result.node}")
                        debug_console.print(f"[bold]Send arg:[/bold]")
                        debug_console.print(Pretty(result.arg))
                    elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
                        debug_console.print(f"[bold]List of Send objects:[/bold] {len(result)} items")
                        for i, send in enumerate(result[:3]):  # Show first 3
                            debug_console.print(f"[bold]Send {i}:[/bold] node={send.node}")
                            debug_console.print(Pretty(send.arg))
                        if len(result) > 3:
                            debug_console.print(f"... and {len(result) - 3} more")
                    elif isinstance(result, dict):
                        debug_console.print(f"[bold]Result keys:[/bold] {list(result.keys())}")
                        debug_console.print(f"[bold]Result:[/bold]")
                        debug_console.print(Pretty(result))
                    else:
                        debug_console.print(f"[bold]Result:[/bold]")
                        debug_console.print(Pretty(result))
                else:
                    # Standard logging
                    function_logger.debug(f"Node {config.name} executed successfully in {execution_time:.3f}s")
                    function_logger.debug(f"Result type: {type(result).__name__}")
                
                # Return the result
                return result
                    
            except Exception as e:
                # Calculate execution time
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Log error with rich UI if available
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[bold red]Execution failed[/bold red] in {execution_time:.3f}s")
                    debug_console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    debug_console.print_exception()
                else:
                    # Standard logging
                    function_logger.error(f"Error in node {config.name}: {e}")
                    function_logger.error(traceback.format_exc())
                
                # Create error result
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                    "execution_id": exec_id
                }
                
                if config.command_goto:
                    return Command(update={"error": error_data}, goto=config.command_goto)
                return {"error": error_data}
            
            finally:
                # Close execution section in rich UI
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.rule()
        
        # Add metadata
        debug_wrapper.__name__ = f"debug_wrapper_{config.name}"
        debug_wrapper.__wrapper_id__ = wrapper_id
        debug_wrapper.__wrapped__ = node_func
        
        return debug_wrapper

    @classmethod
    def _create_async_debug_wrapper(cls, node_func, config):
        """Create a debug wrapper for asynchronous node functions."""
        # Create a unique ID for this function wrapper
        wrapper_id = f"async-debug-{uuid.uuid4().hex[:8]}"
        
        # We're creating a SYNCHRONOUS function that handles async internally
        # This is key for langgraph compatibility
        def async_wrapper(state, runtime_config=None):
            """Async-aware debug wrapper for node functions."""
            function_logger = logging.getLogger(f"node.{config.name}")
            
            # Generate unique execution ID
            exec_id = f"exec-{uuid.uuid4().hex[:8]}"
            
            # Log execution start with rich UI if available
            if cls.rich_debug and RICH_AVAILABLE:
                debug_console.rule(f"[bold blue]Async Node Execution: [green]{config.name}[/green][/bold blue]")
                debug_console.print(f"[bold]Execution ID:[/bold] {exec_id}")
                debug_console.print(f"[bold]Node Type:[/bold] {getattr(node_func, '__node_type__', 'unknown')}")
                debug_console.print(f"[bold]Async Mode:[/bold] True")
                
                # Show state info
                function_logger.debug(f"Async node {config.name} called with state type: {type(state).__name__}")
                debug_console.print(f"[bold]State Type:[/bold] {type(state).__name__}")
                
                if isinstance(state, dict):
                    function_logger.debug(f"State keys: {list(state.keys())}")
                    debug_console.print(f"[bold]State Keys:[/bold] {list(state.keys())}")
                    
                    # Show sample of large keys
                    for key, value in state.items():
                        if isinstance(value, list) and len(value) > 5:
                            debug_console.print(f"[bold]{key}:[/bold] List with {len(value)} items (showing first 3)")
                            debug_console.print(Pretty(value[:3]))
                        elif isinstance(value, dict) and len(value) > 5:
                            debug_console.print(f"[bold]{key}:[/bold] Dict with {len(value)} items (showing first 3)")
                            preview = {k: value[k] for k in list(value.keys())[:3]}
                            debug_console.print(Pretty(preview))
                        elif isinstance(value, str) and len(value) > 500:
                            debug_console.print(f"[bold]{key}:[/bold] String with {len(value)} chars (showing first 500)")
                            debug_console.print(Text(value[:500] + "..."))
                
                # Show runtime config info if provided
                if runtime_config:
                    debug_console.print(f"[bold]Runtime Config:[/bold]")
                    debug_console.print(Pretty(runtime_config))
            else:
                # Standard logging
                function_logger.debug(f"Async node {config.name} called with state type: {type(state).__name__}")
                if isinstance(state, dict):
                    function_logger.debug(f"State keys: {list(state.keys())}")
            
            # Track execution timing
            start_time = datetime.now()
            
            # Try to execute the node function
            try:
                # Run the function - with async handling inside
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.print("[bold yellow]Setting up async execution...[/bold yellow]")
                    
                # Check if we need to run a coroutine
                is_coro_func = asyncio.iscoroutinefunction(node_func)
                
                if is_coro_func:
                    # We need to run the coroutine
                    if cls.rich_debug and RICH_AVAILABLE:
                        debug_console.print("[bold yellow]Running coroutine function in event loop...[/bold yellow]")
                        
                    # Setup asyncio event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        if cls.rich_debug and RICH_AVAILABLE:
                            debug_console.print("[bold yellow]Creating new event loop...[/bold yellow]")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    # Run the coroutine
                    result = loop.run_until_complete(node_func(state, runtime_config))
                else:
                    # Direct invocation - the function should handle async internally
                    if cls.rich_debug and RICH_AVAILABLE:
                        debug_console.print("[bold yellow]Calling node function directly (should handle async internally)...[/bold yellow]")
                        
                    # Call the node function directly - should internally handle async
                    result = node_func(state, runtime_config)
                
                # Calculate execution time
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Log execution result with rich UI if available
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[bold green]Async execution successful[/bold green] in {execution_time:.3f}s")
                    debug_console.print(f"[bold]Result type:[/bold] {type(result).__name__}")
                    
                    # Show different details based on result type
                    if isinstance(result, Command):
                        debug_console.print(f"[bold]Command:[/bold] goto={result.goto}")
                        if hasattr(result, "update") and result.update:
                            debug_console.print(f"[bold]Command update keys:[/bold] {list(result.update.keys()) if isinstance(result.update, dict) else 'Not a dict'}")
                            debug_console.print(f"[bold]Command update:[/bold]")
                            debug_console.print(Pretty(result.update))
                    elif isinstance(result, Send):
                        debug_console.print(f"[bold]Send:[/bold] node={result.node}")
                        debug_console.print(f"[bold]Send arg:[/bold]")
                        debug_console.print(Pretty(result.arg))
                    elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
                        debug_console.print(f"[bold]List of Send objects:[/bold] {len(result)} items")
                        for i, send in enumerate(result[:3]):  # Show first 3
                            debug_console.print(f"[bold]Send {i}:[/bold] node={send.node}")
                            debug_console.print(Pretty(send.arg))
                        if len(result) > 3:
                            debug_console.print(f"... and {len(result) - 3} more")
                    elif isinstance(result, dict):
                        debug_console.print(f"[bold]Result keys:[/bold] {list(result.keys())}")
                        debug_console.print(f"[bold]Result:[/bold]")
                        debug_console.print(Pretty(result))
                    else:
                        debug_console.print(f"[bold]Result:[/bold]")
                        debug_console.print(Pretty(result))
                else:
                    # Standard logging
                    function_logger.debug(f"Async node {config.name} executed successfully in {execution_time:.3f}s")
                    function_logger.debug(f"Result type: {type(result).__name__}")
                
                # Return the result
                return result
                    
            except Exception as e:
                # Calculate execution time
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Log error with rich UI if available
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[bold red]Async execution failed[/bold red] in {execution_time:.3f}s")
                    debug_console.print(f"[bold red]Error:[/bold red] {str(e)}")
                    debug_console.print_exception()
                else:
                    # Standard logging
                    function_logger.error(f"Error in async node {config.name}: {e}")
                    function_logger.error(traceback.format_exc())
                
                # Create error result
                error_data = {
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat(),
                    "execution_id": exec_id
                }
                
                if config.command_goto:
                    return Command(update={"error": error_data}, goto=config.command_goto)
                return {"error": error_data}
            
            finally:
                # Close execution section in rich UI
                if cls.rich_debug and RICH_AVAILABLE:
                    debug_console.rule()
                    
        # Add metadata
        async_wrapper.__name__ = f"async_debug_wrapper_{config.name}"
        async_wrapper.__wrapper_id__ = wrapper_id
        async_wrapper.__wrapped__ = node_func
        
        return async_wrapper
    
    @classmethod
    def create_mapping_node(cls, 
                          item_provider: str,
                          target_node: str,
                          item_key: str = "item",
                          name: str = "mapping_node") -> Callable:
        """
        Create a node function that maps items to parallel processing.
        
        Args:
            item_provider: State key containing the items list
            target_node: Target node to send items to
            item_key: Key to use for each item in the target
            name: Name for the node
            
        Returns:
            Node function that creates Send objects
        """
        cls.debug_log(f"Creating mapping node: {name}")
        cls.debug_log(f"Items from: {item_provider}, Target: {target_node}, Item key: {item_key}")
        
        # Show visual representation if rich is available
        if cls.rich_debug and RICH_AVAILABLE:
            table = Table(title=f"Mapping Node Configuration: {name}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("item_provider", item_provider)
            table.add_row("target_node", target_node)
            table.add_row("item_key", item_key)
            
            debug_console.print(table)
        
        # For mapping nodes, we use a synchronous function that returns Send objects
        def map_items(state):
            """Map items from the state to Send objects."""
            cls.debug_log(f"Mapping node called with state type: {type(state).__name__}")
            
            # Extract items from state
            if isinstance(state, dict):
                items = state.get(item_provider, [])
            elif hasattr(state, "get"):
                items = state.get(item_provider, [])
            elif hasattr(state, item_provider):
                items = getattr(state, item_provider)
            else:
                items = []
                
            cls.debug_log(f"Found {len(items)} items to map")
            
            # Show items if rich is available
            if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                debug_console.print(f"[bold]Mapping {len(items)} items:[/bold]")
                if len(items) > 0:
                    # Show a sample of items (first 3)
                    debug_console.print(f"[bold]Item samples:[/bold]")
                    for i, item in enumerate(items[:3]):
                        debug_console.print(f"[bold]Item {i}:[/bold]")
                        debug_console.print(Pretty(item))
                    
                    if len(items) > 3:
                        debug_console.print(f"... and {len(items) - 3} more items")
            
            if not items:
                cls.debug_log("No items to map, returning empty list")
                return []
                
            # Create Send objects for each item
            result = [Send(target_node, {item_key: item}) for item in items]
            cls.debug_log(f"Created {len(result)} Send objects")
            
            # Show result if rich is available
            if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                debug_console.print(f"[bold green]Created {len(result)} Send objects[/bold green]")
                debug_console.print(f"[bold]First Send sample:[/bold] node={result[0].node}, key={item_key}")
            
            return result
        
        # Mark as mapping node for detection
        map_items.__mapping_node__ = True
        map_items.__name__ = name
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=map_items,
            node_type="mapping",
            async_mode=False,  # Use synchronous mapping
            debug=cls.debug_mode,
            rich_debug=cls.rich_debug
        )
        
        # Create the node function
        return cls.create_node_function(node_config)

    @classmethod
    def create_conditional_node(cls,
                            condition_func: Callable,
                            routes: Dict[Any, str],
                            default_route: Optional[str] = None,
                            name: str = "conditional_node") -> Callable:
        """
        Create a node function that routes based on a condition.
        
        Args:
            condition_func: Function that evaluates state and returns a key
            routes: Mapping from condition values to node names
            default_route: Default route if no match (or condition_func returns None)
            name: Name for the node
            
        Returns:
            Node function that routes based on condition
        """
        cls.debug_log(f"Creating conditional node: {name}")
        cls.debug_log(f"Routes: {routes}, Default: {default_route}")
        
        # Show visual representation if rich is available
        if cls.rich_debug and RICH_AVAILABLE:
            table = Table(title=f"Conditional Node Configuration: {name}")
            table.add_column("Condition", style="cyan")
            table.add_column("Route", style="green")
            
            for condition, route in routes.items():
                table.add_row(str(condition), route)
                
            if default_route:
                table.add_row("DEFAULT", default_route)
            
            debug_console.print(table)
            
            # Show condition function info
            if hasattr(condition_func, "__name__"):
                debug_console.print(f"[bold]Condition function:[/bold] {condition_func.__name__}")
            if hasattr(condition_func, "__module__"):
                debug_console.print(f"[bold]Condition module:[/bold] {condition_func.__module__}")
                
            # Show signature if possible
            try:
                sig = inspect.signature(condition_func)
                debug_console.print(f"[bold]Condition signature:[/bold] {sig}")
            except Exception:
                pass
        
        def route_by_condition(state):
            """Route based on condition function result."""
            cls.debug_log(f"Condition node called with state type: {type(state).__name__}")
            
            try:
                # Evaluate the condition
                result = condition_func(state)
                cls.debug_log(f"Condition function returned: {result}")
                
                # Show evaluation if rich is available
                if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                    debug_console.print(f"[bold]Condition result:[/bold] {result}")
                
                # Find matching route
                if result in routes:
                    target_route = routes[result]
                    cls.debug_log(f"Found matching route: {target_route}")
                    
                    if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                        debug_console.print(f"[bold green]Routing to:[/bold green] {target_route}")
                        
                    return Command(goto=target_route)
                
                # Fall back to default
                if default_route:
                    cls.debug_log(f"Using default route: {default_route}")
                    
                    if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                        debug_console.print(f"[bold yellow]No matching route, using default:[/bold yellow] {default_route}")
                        
                    return Command(goto=default_route)
                    
                # No route found
                error_msg = f"No route found for condition result: {result}"
                cls.debug_log(error_msg, level="warning")
                
                if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                    debug_console.print(f"[bold red]No route found for condition:[/bold red] {result}")
                    debug_console.print(f"[bold red]Available routes:[/bold red] {list(routes.keys())}")
                    
                return {"error": error_msg}
            except Exception as e:
                error_msg = f"Error in condition function: {str(e)}"
                cls.debug_log(error_msg, level="error")
                cls.debug_log(traceback.format_exc(), level="error")
                
                if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                    debug_console.print(f"[bold red]Error in condition function:[/bold red] {str(e)}")
                    debug_console.print_exception()
                    
                return {"error": error_msg}
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=route_by_condition,
            debug=cls.debug_mode,
            rich_debug=cls.rich_debug
        )
        
        # Create the node function
        return cls.create_node_function(node_config)

    @classmethod
    def create_error_handler_node(cls,
                                fallback_node: str = "END",
                                name: str = "error_handler") -> Callable:
        """
        Create a node function that handles errors.
        
        Args:
            fallback_node: Node to route to after handling error
            name: Name for the node
            
        Returns:
            Node function that handles errors
        """
        cls.debug_log(f"Creating error handler node: {name}")
        cls.debug_log(f"Fallback node: {fallback_node}")
        
        # Show visual representation if rich is available
        if cls.rich_debug and RICH_AVAILABLE:
            debug_console.print(Panel.fit(
                f"[bold]Error Handler Node: {name}[/bold]\n"
                f"Fallback route: [green]{fallback_node}[/green]",
                border_style="red"
            ))
        
        def handle_error(state):
            """Handle error state and route to fallback."""
            cls.debug_log(f"Error handler called with state type: {type(state).__name__}")
            
            # Show error handling if rich is available
            if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                debug_console.rule("[bold red]Error Handler Execution[/bold red]")
                debug_console.print(f"[bold]State type:[/bold] {type(state).__name__}")
                
                # Show error information if available in state
                if isinstance(state, dict) and "error" in state:
                    error_info = state["error"]
                    if isinstance(error_info, dict):
                        debug_console.print(f"[bold red]Error:[/bold red] {error_info.get('error', 'Unknown error')}")
                        if "error_type" in error_info:
                            debug_console.print(f"[bold red]Error type:[/bold red] {error_info['error_type']}")
                        if "traceback" in error_info:
                            debug_console.print(f"[bold red]Traceback:[/bold red]")
                            debug_console.print(Text(error_info["traceback"], style="dim"))
                    else:
                        debug_console.print(f"[bold red]Error:[/bold red] {error_info}")
            
            # Create a new state with error handled flag
            result = state.copy() if isinstance(state, dict) else {"state": state}
            result["error_handled"] = True
            result["handled_at"] = datetime.now().isoformat()
            cls.debug_log(f"Marked error as handled")
            
            # Show routing if rich is available
            if cls.rich_debug and RICH_AVAILABLE and cls.debug_mode:
                debug_console.print(f"[bold green]Routing to fallback node:[/bold green] {fallback_node}")
                debug_console.rule()
                
            # Route to fallback
            return Command(update=result, goto=fallback_node)
        
        # Create node config
        node_config = NodeConfig(
            name=name,
            engine=handle_error,
            debug=cls.debug_mode,
            rich_debug=cls.rich_debug
        )
        
        # Create the node function
        return cls.create_node_function(node_config)