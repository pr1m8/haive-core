"""
NodeConfig with rich UI debugging and tracebacks.

This module enhances the NodeConfig class with comprehensive debugging
capabilities using the rich library for visualization.
"""

import asyncio
import json
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Send
from pydantic import BaseModel, Field, model_validator

from haive.core.engine.retriever.retriever import (
    VectorStoreRetrieverConfig,
)

# Import rich if available
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.pretty import Pretty
    from rich.prompt import Prompt
    from rich.syntax import Syntax
    from rich.table import Table
    from rich.text import Text
    from rich.traceback import install as install_rich_traceback
    from rich.tree import Tree

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Set up logging
import logging

logger = logging.getLogger(__name__)

# Set up rich traceback if available
if RICH_AVAILABLE:
    install_rich_traceback(show_locals=True, word_wrap=True)
    console = Console()
    debug_console = Console(stderr=True, width=120)
else:
    # No-op console for fallback
    class DummyConsole:
        def print(self, *args, **kwargs):
            pass

        def log(self, *args, **kwargs):
            pass

        def rule(self, *args, **kwargs):
            pass

    console = DummyConsole()
    debug_console = DummyConsole()

# Import base classes - with fallbacks for testing
try:
    from haive.core.engine.base import Engine
    from haive.core.registry.base import AbstractRegistry
except ImportError:
    # Fallback for standalone testing
    class Engine:
        pass

    class AbstractRegistry:
        pass

    logger.warning("Using fallback Engine and AbstractRegistry classes")


class NodeConfig(BaseModel):
    """
    Configuration for a node in a graph with rich debugging capabilities.

    NodeConfig provides a standardized way to configure nodes in a graph,
    handling both engine-based nodes and callable functions.
    """

    # Debug flag - determines verbosity
    debug: bool = Field(default=False, description="Enable debug logging for this node")

    # Rich UI flag - determines visualization
    rich_debug: bool = Field(default=True, description="Enable rich UI debugging")
    # Added preserve_model field with default=True
    preserve_model: bool = Field(
        default=True,
        description="Preserve BaseModel instances instead of converting to dict",
    )
    # Debug log path for persistent logs
    debug_log_path: Optional[str] = Field(
        default=None, description="Path to save debug logs"
    )

    # Debug ID for tracking
    debug_id: str = Field(
        default_factory=lambda: f"node-{uuid.uuid4().hex[:8]}",
        description="Debug ID for tracking",
    )

    # Core configuration
    name: str = Field(description="Name of the node")
    engine: Optional[Union[Engine, str, Callable]] = Field(
        default=None,
        description="Engine, engine name, or callable function to use for this node",
    )

    # Unique engine identifier - used for targeting in runnable_config
    engine_id: Optional[str] = Field(
        default=None,
        description="Unique ID of the engine instance (auto-populated when resolving)",
    )

    # Control flow options with full Command/Send support
    command_goto: Optional[Union[str, Literal["END"], Send, List[Union[Send, str]]]] = (
        Field(
            default=None,
            description="Next node to go to after this node (or END, Send object, or list of Send objects)",
        )
    )

    # Mapping options
    input_mapping: Optional[Dict[str, str]] = Field(
        default=None, description="Mapping from state keys to engine input keys"
    )
    output_mapping: Optional[Dict[str, str]] = Field(
        default=None, description="Mapping from engine output keys to state keys"
    )

    # Runtime configuration
    runnable_config: Optional[RunnableConfig] = Field(
        default=None, description="Runtime configuration for this node"
    )

    # Configuration overrides at the node level
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engine configuration overrides specific to this node",
    )

    # Node type information
    node_type: Optional[str] = Field(
        default=None, description="Type of node function (auto-detected if None)"
    )
    async_mode: Optional[bool] = Field(
        default=None,
        description="Whether to operate in async mode (auto-detected if None)",
    )

    # Special handling flags
    use_direct_messages: Optional[bool] = Field(
        default=None,
        description="Whether to use messages field directly (auto-detected if None)",
    )

    extract_content: bool = Field(
        default=False,
        description="Extract content from messages and add as 'content' field",
    )

    preserve_state: bool = Field(
        default=True, description="Preserve state fields not affected by output mapping"
    )

    # Registry reference (not serialized)
    registry: Optional[AbstractRegistry] = Field(default=None, exclude=True)

    # Tracing related fields
    creation_stack: Optional[List[str]] = Field(
        default=None, description="Stack trace at creation time", exclude=True
    )
    creation_time: Optional[datetime] = Field(
        default_factory=datetime.now,
        description="Time when this config was created",
        exclude=True,
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for this node"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        """Initialize with debug information."""
        # Capture stack trace at creation time if debug is enabled
        if data.get("debug", False):
            data["creation_stack"] = traceback.format_stack()

        # Call parent constructor
        super().__init__(**data)

        # Debug initialization if requested
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            self._debug_initialization()

    def _debug_initialization(self):
        """Debug the initialization of this config with rich UI."""
        # Create debug header
        debug_console.rule(
            f"[bold blue]NodeConfig Initialization: [green]{self.name}[/green][/bold blue]"
        )

        # Show creation stack if available
        if self.creation_stack:
            debug_console.print("[bold]Creation Stack:[/bold]")
            for line in self.creation_stack[-10:]:  # Show last 10 lines
                debug_console.print(f"  [dim]{line.strip()}[/dim]")

        # Show basic config details
        debug_console.print(f"[bold]Debug ID:[/bold] {self.debug_id}")
        debug_console.print(f"[bold]Created at:[/bold] {self.creation_time}")

        # Show engine information
        if isinstance(self.engine, Engine):
            debug_console.print(
                f"[bold]Engine:[/bold] {getattr(self.engine, 'name', 'unnamed')} ([cyan]{self.engine.__class__.__name__}[/cyan])"
            )
            debug_console.print(
                f"[bold]Engine ID:[/bold] {getattr(self.engine, 'id', 'not set')}"
            )
        elif isinstance(self.engine, str):
            debug_console.print(
                f"[bold]Engine Reference:[/bold] [yellow]{self.engine}[/yellow] (string reference)"
            )
        elif callable(self.engine):
            debug_console.print(
                f"[bold]Engine:[/bold] [magenta]{getattr(self.engine, '__name__', 'unnamed')}[/magenta] (callable)"
            )
        else:
            debug_console.print("[bold]Engine:[/bold] [red]None[/red]")

        # Show mappings
        if self.input_mapping:
            table = Table(title="Input Mapping")
            table.add_column("State Key", style="cyan")
            table.add_column("Engine Input", style="green")
            for state_key, engine_key in self.input_mapping.items():
                table.add_row(state_key, engine_key)
            debug_console.print(table)

        if self.output_mapping:
            table = Table(title="Output Mapping")
            table.add_column("Engine Output", style="cyan")
            table.add_column("State Key", style="green")
            for engine_key, state_key in self.output_mapping.items():
                table.add_row(engine_key, state_key)
            debug_console.print(table)

        # Show control flow information
        if self.command_goto is not None:
            if self.command_goto == END:
                debug_console.print(
                    "[bold]Control Flow:[/bold] [bright_green]→ END[/bright_green]"
                )
            elif isinstance(self.command_goto, str):
                debug_console.print(
                    f"[bold]Control Flow:[/bold] [bright_green]→ {self.command_goto}[/bright_green]"
                )
            elif isinstance(self.command_goto, Send):
                debug_console.print(
                    f"[bold]Control Flow:[/bold] [bright_green]→ Send to {self.command_goto.node}[/bright_green]"
                )
            elif isinstance(self.command_goto, list):
                debug_console.print("[bold]Control Flow:[/bold] Multiple destinations:")
                for item in self.command_goto:
                    if isinstance(item, Send):
                        debug_console.print(
                            f"  [bright_green]→ Send to {item.node}[/bright_green]"
                        )
                    else:
                        debug_console.print(f"  [bright_green]→ {item}[/bright_green]")

        # Show remaining configuration
        debug_console.print("[bold]Additional Configuration:[/bold]")
        debug_console.print(
            f"  [dim]Node Type:[/dim] {self.node_type or 'auto-detect'}"
        )
        debug_console.print(
            f"  [dim]Async Mode:[/dim] {self.async_mode or 'auto-detect'}"
        )
        debug_console.print(
            f"  [dim]Use Direct Messages:[/dim] {self.use_direct_messages}"
        )
        debug_console.print(f"  [dim]Extract Content:[/dim] {self.extract_content}")
        debug_console.print(f"  [dim]Preserve State:[/dim] {self.preserve_state}")

        # Output overrides if present
        if self.config_overrides:
            debug_console.print("[bold]Config Overrides:[/bold]")
            for key, value in self.config_overrides.items():
                debug_console.print(f"  [dim]{key}:[/dim] {value}")

        debug_console.rule()

    @model_validator(mode="after")
    def validate_config(self):
        """Validate and normalize the configuration."""
        # Save original state for debugging
        original_state = {}
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            if hasattr(self, "model_dump"):
                original_state = self.model_dump(exclude={"creation_stack"})
            else:
                original_state = self.dict(exclude={"creation_stack"})

        # Perform validations
        try:
            # Convert "END" string to END constant
            if self.command_goto == "END":
                self.command_goto = END

            # Auto-populate engine_id if an engine with ID is provided
            if (
                self.engine
                and isinstance(self.engine, Engine)
                and self.engine_id is None
            ):
                if hasattr(self.engine, "id"):
                    self.engine_id = self.engine.id

            # Auto-detect use_direct_messages if None
            if self.use_direct_messages is None:
                self.use_direct_messages = self._detect_uses_messages()

            # Determine node type if not provided
            if self.node_type is None:
                self.node_type = self.determine_node_type()

            # Determine async mode if not provided and not in node type
            if self.async_mode is None and not self.node_type.startswith("async"):
                self.async_mode = asyncio.iscoroutinefunction(self.engine)

        except Exception as e:
            if self.debug:
                logger.exception(
                    f"Error validating NodeConfig for {self.name}: {str(e)}"
                )

                if self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[bold red]Error validating NodeConfig:[/bold red] {str(e)}"
                    )
                    debug_console.print_exception()
            raise

        # Debug validation changes if enabled
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            self._debug_validation_changes(original_state)

        return self

    def _debug_validation_changes(self, original_state: Dict[str, Any]):
        """Debug changes made during validation."""
        current_state = {}
        if hasattr(self, "model_dump"):
            current_state = self.model_dump(exclude={"creation_stack"})
        else:
            current_state = self.dict(exclude={"creation_stack"})

        # Find differences
        changes = []
        for key, new_value in current_state.items():
            if key in original_state:
                old_value = original_state[key]
                if old_value != new_value:
                    changes.append((key, old_value, new_value))

        if changes:
            debug_console.rule("[bold]NodeConfig Validation Changes[/bold]")
            table = Table(title=f"Changes for {self.name}")
            table.add_column("Field", style="cyan")
            table.add_column("Original", style="yellow")
            table.add_column("Updated", style="green")

            for key, old_value, new_value in changes:
                table.add_row(key, str(old_value), str(new_value))

            debug_console.print(table)
            debug_console.rule()

    def _detect_uses_messages(self) -> bool:
        """
        Detect if this node uses messages directly based on prompt variables and output models.

        This method intelligently determines whether direct messages should be used
        based on engine configuration, especially for AugLLMConfig with structured outputs.

        Returns:
            Boolean indicating if this node uses messages directly
        """
        try:
            # If it's an LLM engine, check if it might use messages
            if isinstance(self.engine, Engine) and hasattr(self.engine, "engine_type"):
                engine_type = self.engine.engine_type
                if engine_type and getattr(engine_type, "value", "") == "llm":
                    # For AugLLMConfig, we need special handling
                    if self.engine.__class__.__name__ == "AugLLMConfig":
                        # Check for structured output model first - this is a key indicator
                        structured_output_model = getattr(
                            self.engine, "structured_output_model", None
                        )
                        if structured_output_model:
                            # If we have a structured output model, we usually don't need messages
                            # We should also auto-derive the output mapping here
                            if not self.output_mapping:
                                # Get model class name for the output field name
                                # If it's a class, get its name, otherwise use "result"
                                if isinstance(structured_output_model, type):
                                    model_name = (
                                        structured_output_model.__name__.lower()
                                    )
                                    # Set output mapping to map the model result to a field of the same name
                                    self.output_mapping = {"result": model_name}
                                    self.debug_log(
                                        f"Auto-derived output mapping for structured model: {self.output_mapping}"
                                    )
                                else:
                                    # Default to mapping structured output to "result"
                                    self.output_mapping = {"result": "result"}
                            # With structured output, we typically don't need messages directly
                            return False

                        # Check for output parser
                        output_parser = getattr(self.engine, "output_parser", None)
                        if output_parser and not self.output_mapping:
                            # If we have an output parser, typically map content to output
                            self.output_mapping = {"content": "output"}
                            self.debug_log(
                                f"Auto-derived output mapping for output parser: {self.output_mapping}"
                            )
                            # With parsers, we may not need messages directly if input is mapped
                            if self.input_mapping:
                                return False

                        # Check prompt template variables
                        prompt_template = getattr(self.engine, "prompt_template", None)
                        if prompt_template and hasattr(
                            prompt_template, "input_variables"
                        ):
                            input_vars = prompt_template.input_variables

                            # If there are input variables, check if they all have mappings
                            if input_vars and "messages" not in input_vars:
                                # If no input mapping but we need specific vars, auto-create mapping
                                if not self.input_mapping and input_vars:
                                    self.input_mapping = {
                                        var: var for var in input_vars
                                    }
                                    self.debug_log(
                                        f"Auto-derived input mapping: {self.input_mapping}"
                                    )
                                    return False

                                # If all vars are already mapped, we don't need messages
                                if self.input_mapping and all(
                                    var in [v for _, v in self.input_mapping.items()]
                                    for var in input_vars
                                ):
                                    return False

                            # If explicitly requires messages, use them
                            if "messages" in input_vars:
                                return True

                        # Default for AugLLM - prefer specific mappings if available
                        return not bool(self.input_mapping)
                elif engine_type and getattr(engine_type, "value", "") == "retriever":
                    # For all retriever types, we don't use messages directly
                    self.debug_log(
                        f"Detected retriever engine: {type(self.engine).__name__}"
                    )

                    # Auto-derive input mapping if not provided
                    if not self.input_mapping:
                        # Map common state fields to retriever fields
                        if (
                            hasattr(self.engine, "input_schema")
                            and self.engine.input_schema
                        ):
                            input_fields = []
                            # Get field names from input schema
                            if hasattr(self.engine.input_schema, "model_fields"):
                                input_fields = list(
                                    self.engine.input_schema.model_fields.keys()
                                )
                            elif hasattr(self.engine.input_schema, "__fields__"):
                                input_fields = list(
                                    self.engine.input_schema.__fields__.keys()
                                )

                            # Create mappings based on schema fields
                            if "query" in input_fields:
                                if "query" in self.state_keys:
                                    self.input_mapping = {"query": "query"}
                                elif "question" in self.state_keys:
                                    self.input_mapping = {"question": "query"}
                                elif "content" in self.state_keys:
                                    self.input_mapping = {"content": "query"}
                                else:
                                    # Default to mapping from 'input' field
                                    self.input_mapping = {"input": "query"}
                        else:
                            # Default input mapping for retrievers
                            self.input_mapping = {"input": "query"}

                        self.debug_log(
                            f"Auto-derived input mapping for retriever: {self.input_mapping}"
                        )

                    # Auto-derive output mapping if not provided
                    if not self.output_mapping:
                        # Default to mapping 'documents' to state
                        if isinstance(self.engine, VectorStoreRetrieverConfig):
                            # Vector store retrievers return documents
                            self.output_mapping = {"documents": "documents"}
                        else:
                            # Generic mapping for any retriever
                            self.output_mapping = {"documents": "context"}

                        self.debug_log(
                            f"Auto-derived output mapping for retriever: {self.output_mapping}"
                        )

                    # Retrievers don't use messages directly
                    return False
        except Exception as e:
            self.debug_log(f"Error in _detect_uses_messages_field: {e}", level="error")

        # Default behavior - use messages for LLMs if no input mapping
        return not bool(self.input_mapping)

    def resolve_engine(self, registry=None) -> Tuple[Any, Optional[str]]:
        """
        Resolve engine reference to actual engine and its ID.

        Args:
            registry: Optional registry to use for lookup

        Returns:
            Tuple of (resolved engine, engine_id)
        """
        # Debug header if enabled
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            debug_console.rule(f"[bold]Resolving Engine for {self.name}[/bold]")
            debug_console.print(
                f"[bold]Original Engine Reference:[/bold] {self.engine}"
            )

        try:
            # Already resolved to a non-string (Engine, Callable, etc.)
            if not isinstance(self.engine, str):
                engine_id = None
                # Extract engine ID if possible
                if isinstance(self.engine, Engine) and hasattr(self.engine, "id"):
                    engine_id = self.engine.id
                    self.engine_id = engine_id

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Engine already resolved:[/green] {type(self.engine).__name__}"
                    )
                    if engine_id:
                        debug_console.print(f"[green]Engine ID:[/green] {engine_id}")

                return self.engine, engine_id

            # Try to lookup in registry
            engine_name = self.engine

            if registry is None:
                registry = self.registry

            if registry is None:
                # Try to import from haive.core if possible
                try:
                    from haive.core.engine.base import EngineRegistry

                    registry = EngineRegistry.get_instance()

                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[yellow]Using global EngineRegistry[/yellow]"
                        )
                except ImportError:
                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print("[red]No registry available[/red]")
                    return self.engine, None

            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.print(f"[bold]Looking up engine:[/bold] {engine_name}")
                debug_console.print(
                    f"[bold]Registry type:[/bold] {type(registry).__name__}"
                )

            # Try to find engine by name or ID
            engine = (
                registry.find_by_id(engine_name)
                if hasattr(registry, "find_by_id")
                else None
            )
            if engine:
                # Update engine reference
                self.engine = engine

                # Extract engine ID if available
                engine_id = None
                if hasattr(engine, "id"):
                    engine_id = engine.id
                    self.engine_id = engine_id

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Found engine by ID:[/green] {type(engine).__name__}"
                    )
                    if engine_id:
                        debug_console.print(f"[green]Engine ID:[/green] {engine_id}")

                return engine, engine_id

            # Try other lookup methods if find_by_id didn't work
            engine = registry.find(engine_name) if hasattr(registry, "find") else None
            if engine:
                self.engine = engine
                self.engine_id = getattr(engine, "id", None)

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Found engine by name:[/green] {type(engine).__name__}"
                    )
                    if self.engine_id:
                        debug_console.print(
                            f"[green]Engine ID:[/green] {self.engine_id}"
                        )

                return engine, self.engine_id

            # Not found - warn and return as is
            if self.debug:
                logger.warning(f"Engine '{engine_name}' not found in registry")

                if self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[red]Engine not found:[/red] {engine_name}")

            return self.engine, None

        except Exception as e:
            if self.debug:
                logger.exception(f"Error resolving engine for {self.name}: {str(e)}")

                if self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[bold red]Error resolving engine:[/bold red] {str(e)}"
                    )
                    debug_console.print_exception()

            # Return original engine on error
            return self.engine, None

        finally:
            # Close debug section
            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.rule()

    def determine_node_type(self) -> str:
        """
        Determine the most appropriate node type based on engine.

        Returns:
            Node type string
        """
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            debug_console.rule(f"[bold]Determining Node Type for {self.name}[/bold]")
            debug_console.print(
                f"[bold]Engine:[/bold] {type(self.engine).__name__ if self.engine is not None else 'None'}"
            )

        try:
            # Use explicit setting if provided
            if self.node_type:
                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Using explicit node type:[/green] {self.node_type}"
                    )
                return self.node_type

            engine = self.engine

            # Handle async mode explicitly
            if self.async_mode:
                # Check for AsyncInvokable
                if hasattr(engine, "ainvoke") and callable(engine.ainvoke):
                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[green]Detected async_invokable (explicit async mode)[/green]"
                        )
                    return "async_invokable"

                # Check for async functions
                if asyncio.iscoroutinefunction(engine):
                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[green]Detected async (explicit async mode)[/green]"
                        )
                    return "async"

            # Standard detection logic
            if asyncio.iscoroutinefunction(engine):
                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        "[green]Detected async (coroutine function)[/green]"
                    )
                return "async"

            # Check for AsyncInvokable
            if hasattr(engine, "ainvoke") and callable(engine.ainvoke):
                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print("[green]Detected async_invokable[/green]")
                return "async_invokable"

            # Check for Invokable
            if hasattr(engine, "invoke") and callable(engine.invoke):
                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print("[green]Detected invokable[/green]")
                return "invokable"

            # Check for mapping functions (based on signature return annotation)
            if callable(engine) and hasattr(engine, "__annotations__"):
                if "return" in engine.__annotations__:
                    return_type = engine.__annotations__["return"]
                    if "List[Send]" in str(return_type) or "list[Send]" in str(
                        return_type
                    ):
                        if self.debug and self.rich_debug and RICH_AVAILABLE:
                            debug_console.print(
                                "[green]Detected mapping function[/green]"
                            )
                        return "mapping"

            # Default to "callable" for callable functions
            if callable(engine):
                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print("[green]Detected callable[/green]")
                return "callable"

            # Generic for everything else
            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.print("[yellow]Defaulting to generic[/yellow]")
            return "generic"

        except Exception as e:
            if self.debug:
                logger.exception(
                    f"Error determining node type for {self.name}: {str(e)}"
                )

                if self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[bold red]Error determining node type:[/bold red] {str(e)}"
                    )
                    debug_console.print_exception()

            # Default to generic on error
            return "generic"

        finally:
            # Close debug section
            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.rule()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.

        Returns:
            Dictionary representation of the node config
        """
        if self.debug and self.rich_debug and RICH_AVAILABLE:
            debug_console.rule(f"[bold]Serializing NodeConfig {self.name}[/bold]")

        try:
            # Get dict representation based on Pydantic version
            if hasattr(self, "model_dump"):
                # Pydantic v2
                data = self.model_dump(exclude={"engine", "registry", "creation_stack"})
            else:
                # Pydantic v1
                data = self.dict(exclude={"engine", "registry", "creation_stack"})

            # Handle engine serialization
            if isinstance(self.engine, Engine):
                # Just store the engine ID and name for reference
                data["engine_ref"] = {
                    "id": getattr(self.engine, "id", None),
                    "name": getattr(self.engine, "name", None),
                    "engine_type": getattr(self.engine, "engine_type", None),
                }

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Serialized Engine as reference:[/green] {data['engine_ref']}"
                    )

            elif isinstance(self.engine, str):
                data["engine_ref"] = self.engine

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[green]Serialized Engine as string:[/green] {data['engine_ref']}"
                    )

            elif callable(self.engine):
                # For callables, store a reference if possible
                if hasattr(self.engine, "__name__"):
                    module = getattr(self.engine, "__module__", "")
                    data["engine_ref"] = f"function:{module}.{self.engine.__name__}"

                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            f"[green]Serialized Engine as function:[/green] {data['engine_ref']}"
                        )
                else:
                    data["engine_ref"] = "callable"

                    if self.debug and self.rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[yellow]Serialized Engine as generic callable[/yellow]"
                        )

            # Special handling for END in command_goto
            if self.command_goto is END:
                data["command_goto"] = "END"

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print("[green]Serialized command_goto as END[/green]")

            elif isinstance(self.command_goto, Send):
                # Handle Send objects
                data["command_goto"] = {
                    "type": "send",
                    "node": self.command_goto.node,
                    "arg": self._serialize_send_arg(self.command_goto.arg),
                }

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        "[green]Serialized command_goto as Send object[/green]"
                    )

            elif isinstance(self.command_goto, list) and any(
                isinstance(item, Send) for item in self.command_goto
            ):
                # Handle list containing Send objects
                data["command_goto"] = [
                    (
                        {
                            "type": "send",
                            "node": item.node,
                            "arg": self._serialize_send_arg(item.arg),
                        }
                        if isinstance(item, Send)
                        else item
                    )
                    for item in self.command_goto
                ]

                if self.debug and self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        "[green]Serialized command_goto as list with Send objects[/green]"
                    )

            # Show final result for debugging
            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.print("[bold]Serialization Result:[/bold]")
                debug_console.print(Pretty(data))

            return data

        except Exception as e:
            if self.debug:
                logger.exception(
                    f"Error serializing NodeConfig for {self.name}: {str(e)}"
                )

                if self.rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[bold red]Error serializing NodeConfig:[/bold red] {str(e)}"
                    )
                    debug_console.print_exception()

            # Return minimal representation on error
            return {"name": self.name, "error": str(e)}

        finally:
            # Close debug section
            if self.debug and self.rich_debug and RICH_AVAILABLE:
                debug_console.rule()

    def _serialize_send_arg(self, arg: Any) -> Any:
        """
        Serialize Send argument to ensure it's JSON serializable.

        Args:
            arg: The argument to serialize

        Returns:
            JSON serializable representation
        """
        try:
            if isinstance(arg, dict):
                return {k: self._serialize_send_arg(v) for k, v in arg.items()}
            elif isinstance(arg, list):
                return [self._serialize_send_arg(item) for item in arg]
            elif isinstance(arg, (str, int, float, bool, type(None))):
                return arg
            elif hasattr(arg, "model_dump"):
                return arg.model_dump()
            elif hasattr(arg, "dict"):
                return arg.dict()
            else:
                return str(arg)
        except Exception as e:
            if self.debug:
                logger.warning(f"Error serializing Send arg: {str(e)}")
            return str(arg)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry=None) -> "NodeConfig":
        """
        Create a NodeConfig from a dictionary representation.

        Args:
            data: Dictionary representation
            registry: Optional registry for engine lookup

        Returns:
            Instantiated NodeConfig
        """
        # Create a copy to avoid modifying the input
        config_data = data.copy()

        # Add debug info for tracing
        debug_mode = config_data.get("debug", False)
        rich_debug = config_data.get("rich_debug", True)

        if debug_mode and rich_debug and RICH_AVAILABLE:
            debug_console.rule(
                f"[bold]Deserializing NodeConfig: [green]{config_data.get('name', 'unnamed')}[/green][/bold]"
            )
            debug_console.print("[bold]Input data:[/bold]")
            debug_console.print(Pretty(config_data))

        try:
            # Handle engine references
            if "engine_ref" in config_data:
                ref = config_data.pop("engine_ref")

                if debug_mode and rich_debug and RICH_AVAILABLE:
                    debug_console.print(f"[bold]Engine Reference:[/bold] {ref}")

                if isinstance(ref, dict) and "id" in ref and "name" in ref:
                    # Engine reference with ID and name
                    engine = None

                    # Try to find the engine if registry provided
                    if registry:
                        # Try ID first
                        if ref["id"]:
                            engine = registry.find_by_id(ref["id"])

                            if debug_mode and rich_debug and RICH_AVAILABLE and engine:
                                debug_console.print(
                                    f"[green]Found engine by ID:[/green] {ref['id']}"
                                )

                        # Try name next
                        if engine is None and ref["name"] and hasattr(registry, "find"):
                            engine = registry.find(ref["name"])

                            if debug_mode and rich_debug and RICH_AVAILABLE and engine:
                                debug_console.print(
                                    f"[green]Found engine by name:[/green] {ref['name']}"
                                )

                    if engine:
                        config_data["engine"] = engine
                        config_data["engine_id"] = ref["id"]
                    else:
                        # Store name reference
                        config_data["engine"] = ref["name"]

                        if debug_mode and rich_debug and RICH_AVAILABLE:
                            debug_console.print(
                                f"[yellow]Engine not found, storing name reference:[/yellow] {ref['name']}"
                            )

                elif isinstance(ref, str):
                    # String reference (name, ID, or function path)
                    if ref.startswith("function:"):
                        # Function reference - try to import if possible
                        try:
                            import importlib

                            module_path, func_name = ref[9:].rsplit(".", 1)
                            module = importlib.import_module(module_path)
                            func = getattr(module, func_name)
                            config_data["engine"] = func

                            if debug_mode and rich_debug and RICH_AVAILABLE:
                                debug_console.print(
                                    f"[green]Imported function:[/green] {module_path}.{func_name}"
                                )
                        except (ImportError, AttributeError, ValueError) as e:
                            # Can't resolve - store as string
                            config_data["engine"] = ref

                            if debug_mode and rich_debug and RICH_AVAILABLE:
                                debug_console.print(
                                    f"[yellow]Could not import function, storing reference:[/yellow] {ref}"
                                )
                                debug_console.print(f"[dim]Error: {e}[/dim]")
                    else:
                        # Engine name or ID
                        config_data["engine"] = ref

                        if debug_mode and rich_debug and RICH_AVAILABLE:
                            debug_console.print(
                                f"[green]Storing engine reference:[/green] {ref}"
                            )

            # Handle command_goto serialization
            if "command_goto" in config_data:
                goto = config_data["command_goto"]

                if goto == "END":
                    config_data["command_goto"] = END

                    if debug_mode and rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[green]Converted command_goto to END constant[/green]"
                        )
                elif isinstance(goto, dict) and goto.get("type") == "send":
                    # Reconstruct Send object
                    from langgraph.types import Send

                    config_data["command_goto"] = Send(goto["node"], goto["arg"])

                    if debug_mode and rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[green]Reconstructed Send object for command_goto[/green]"
                        )
                elif isinstance(goto, list):
                    # Handle list containing Send dictionaries
                    from langgraph.types import Send

                    config_data["command_goto"] = [
                        (
                            Send(item["node"], item["arg"])
                            if isinstance(item, dict) and item.get("type") == "send"
                            else item
                        )
                        for item in goto
                    ]

                    if debug_mode and rich_debug and RICH_AVAILABLE:
                        debug_console.print(
                            "[green]Reconstructed list with Send objects for command_goto[/green]"
                        )

            # Create the NodeConfig
            if debug_mode and rich_debug and RICH_AVAILABLE:
                debug_console.print("[bold]Creating NodeConfig from data[/bold]")

            result = cls(**config_data)

            # Set registry
            result.registry = registry

            if debug_mode and rich_debug and RICH_AVAILABLE:
                debug_console.print(
                    f"[bold green]Successfully created NodeConfig: {result.name}[/bold green]"
                )

            return result

        except Exception as e:
            if debug_mode:
                logger.exception(f"Error creating NodeConfig from dict: {str(e)}")

                if rich_debug and RICH_AVAILABLE:
                    debug_console.print(
                        f"[bold red]Error creating NodeConfig:[/bold red] {str(e)}"
                    )
                    debug_console.print_exception()

            # Rethrow exception
            raise

        finally:
            # Close debug section
            if debug_mode and rich_debug and RICH_AVAILABLE:
                debug_console.rule()

    def __str__(self) -> str:
        """String representation of the node config."""
        engine_str = ""
        if isinstance(self.engine, Engine):
            engine_str = f"{getattr(self.engine, 'name', 'unknown')} ({type(self.engine).__name__})"
        elif isinstance(self.engine, str):
            engine_str = f"'{self.engine}' (reference)"
        elif callable(self.engine):
            engine_str = f"{getattr(self.engine, '__name__', 'callable')}"
        else:
            engine_str = "None"

        goto_str = ""
        if self.command_goto is END:
            goto_str = "END"
        elif isinstance(self.command_goto, str):
            goto_str = self.command_goto
        elif isinstance(self.command_goto, Send):
            goto_str = f"Send({self.command_goto.node})"
        elif isinstance(self.command_goto, list):
            goto_str = f"[{', '.join(str(i) for i in self.command_goto)}]"
        else:
            goto_str = "None"

        return (
            f"NodeConfig(name='{self.name}', engine={engine_str}, "
            f"command_goto={goto_str}, type={self.node_type or 'auto'})"
        )

    def debug_log(self, message: str, level: str = "debug") -> None:
        """
        Log a debug message if debug is enabled.

        Args:
            message: The message to log
            level: Log level (debug, info, warning, error, critical)
        """
        if not self.debug:
            return

        # Log with standard logger
        log_func = getattr(logger, level, logger.debug)
        log_func(f"[{self.name}] {message}")

        # Log with rich if enabled
        if self.rich_debug and RICH_AVAILABLE:
            level_style = {
                "debug": "[blue]DEBUG[/blue]",
                "info": "[green]INFO[/green]",
                "warning": "[yellow]WARNING[/yellow]",
                "error": "[red]ERROR[/red]",
                "critical": "[bold red]CRITICAL[/bold red]",
            }
            style = level_style.get(level, "[blue]DEBUG[/blue]")
            debug_console.print(f"{style} [{self.name}] {message}")

        # Save to file if configured
        if self.debug_log_path:
            try:
                with open(self.debug_log_path, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{timestamp}] [{level.upper()}] [{self.name}] {message}\n"
                    )
            except Exception as e:
                logger.warning(f"Could not write to debug log file: {e}")

    def debug_dict(self, title: str, data: Dict[str, Any]) -> None:
        """
        Log a dictionary with rich formatting if debug is enabled.

        Args:
            title: Title for the data
            data: Dictionary to log
        """
        if not self.debug:
            return

        # Log with standard logger
        logger.debug(f"[{self.name}] {title}: {data}")

        # Log with rich if enabled
        if self.rich_debug and RICH_AVAILABLE:
            debug_console.print(
                f"[bold blue][{self.name}][/bold blue] [bold]{title}:[/bold]"
            )
            debug_console.print(Pretty(data))

        # Save to file if configured
        if self.debug_log_path:
            try:
                with open(self.debug_log_path, "a") as f:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{timestamp}] [DEBUG] [{self.name}] {title}: {json.dumps(data, default=str)}\n"
                    )
            except Exception as e:
                logger.warning(f"Could not write to debug log file: {e}")
