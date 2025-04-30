"""
Agent - Base class for all agent implementations in the Haive framework.

This module provides the core agent architecture with consistent schema handling,
execution flows, persistence management, and rich debugging capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from typing import Any, Generic, TypeVar, Optional, Union, Dict, List, Type, cast
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# Haive core imports
from haive.core.config.runnable import RunnableConfigManager
from haive.core.engine.base import Engine, EngineType
from haive.core.persistence.types import CheckpointerType
from haive.core.persistence.base import CheckpointerConfig
from haive.core.persistence.postgres_config import PostgresCheckpointerConfig
from haive.core.persistence.handlers import (
    setup_checkpointer, 
    ensure_pool_open, 
    register_thread_if_needed, 
    prepare_merged_input
)
from haive.core.graph.dynamic_graph_builder import DynamicGraph
from haive.core.schema.schema_composer import SchemaComposer
from haive.core.utils.pydantic_utils import ensure_json_serializable
from haive.core.engine.agent.config import AgentConfig
# Rich UI imports
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.traceback import install as install_rich_traceback
    from rich.logging import RichHandler
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
from langgraph.graph import StateGraph
# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generics
TConfig = TypeVar("TConfig", bound="AgentConfig")
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

# Agent registry maps config classes to agent classes
AGENT_REGISTRY: Dict[Type["AgentConfig"], Type["Agent"]] = {}

def register_agent(config_class: Type["AgentConfig"]):
    """Register an agent class with its configuration class."""
    def decorator(agent_class: Type[Agent]):
        AGENT_REGISTRY[config_class] = agent_class
        return agent_class
    return decorator


class Agent(Generic[TConfig], ABC):
    """
    Base agent architecture class for all agent implementations.
    
    The Agent class provides a consistent framework for defining agent behavior,
    managing state, and executing workflows with rich debugging capabilities.
    """
    
    def __init__(self, config: TConfig, verbose: bool = False, rich_logging: bool = True):
        """
        Initialize the agent with its configuration.
        
        Args:
            config: Agent configuration
            verbose: Whether to enable verbose logging
            rich_logging: Whether to use rich UI for logging and debugging
        """
        self.config = config
        self.verbose = verbose
        self.rich_logging = rich_logging and RICH_AVAILABLE
        
        # Set up rich UI if available and requested
        if self.rich_logging:
            self._setup_rich_ui()
        
        # Configure logging
        self._setup_logging()
        
        # Print agent initialization banner
        self._log_agent_init()
        
        logger.info(f"Initializing agent: {config.name}")
        
        # Clear initialization order to prevent confusion
        # 1. Setup output directories
        self._setup_directories()
        
        # 2. Setup schemas first (needed for everything else)
        self._setup_schemas()
        
        # 3. Setup engines (needed for graph building)
        self._initialize_engines()
        
        # 4. Setup persistence (needed for compilation)
        self._setup_persistence()
        
        # 5. Setup runtime configuration
        self._setup_runtime_config()
        
        # Now we have all prerequisites to build the graph
        self._create_graph()
        
        # Allow subclass to set up workflow
        logger.info(f"Setting up workflow for {config.name}")
        self.graph = StateGraph(state_schema=self.state_schema,input=self.input_schema,\
            output=self.output_schema,config_schema=self.runnable_config)
        self.setup_workflow()
        
        # Compile the graph
        self.compile()
        
        # Generate visualization if requested
        if getattr(self.config, "visualize", True) and self.graph:
            self.visualize_graph()
        
        self._log_agent_ready()
    
    def _setup_rich_ui(self):
        """Configure rich UI for debugging."""
        if not RICH_AVAILABLE:
            logger.warning("Rich UI requested but rich library not installed. Install with: pip install rich")
            return
        
        # Install rich traceback handler
        install_rich_traceback(show_locals=True)
        
        # Set up console
        self.console = Console(record=True, width=120)
        
        # Create a logger console (can be different from main console)
        self.log_console = Console(stderr=True, width=100)
        
        # Create debug console for detailed output
        self.debug_console = Console(width=120)
        
        # Show initialization banner
        self.console.print(
            Panel.fit(
                f"[bold blue]Haive Agent: [green]{getattr(self.config, 'name', 'Unnamed')}[/green][/bold blue]",
                border_style="blue",
                title="Initializing",
                subtitle=f"v{getattr(self.config, 'version', '1.0.0')}"
            )
        )
    
    def _setup_logging(self):
        """Configure logging with rich or standard handlers."""
        # Get the agent's logger
        agent_logger = logging.getLogger(__name__)
        
        # Set level based on verbosity
        log_level = logging.DEBUG if self.verbose else logging.INFO
        agent_logger.setLevel(log_level)
        
        # Clear existing handlers
        if agent_logger.handlers:
            agent_logger.handlers.clear()
        
        if self.rich_logging and RICH_AVAILABLE:
            # Add rich handler
            rich_handler = RichHandler(
                console=getattr(self, "log_console", Console(stderr=True)),
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False
            )
            rich_handler.setLevel(log_level)
            agent_logger.addHandler(rich_handler)
            
            # Set format for rich handler
            rich_format = logging.Formatter("%(message)s")
            rich_handler.setFormatter(rich_format)
            
            # Set parent logger to avoid duplicate logs
            agent_logger.propagate = False
        else:
            # Add standard handler
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            handler.setLevel(log_level)
            agent_logger.addHandler(handler)
            
        # Set related loggers to appropriate level
        for module in ['haive.core.graph', 'haive.core.engine', 
                       'haive.core.schema',
                       'haive.core.graph.node',
                       'haive.core.engine.aug_llm']:
            module_logger = logging.getLogger(module)
            module_logger.setLevel(log_level)
            
            # Apply rich handler to module loggers if available
            if self.rich_logging and RICH_AVAILABLE:
                # Remove existing handlers
                if module_logger.handlers:
                    module_logger.handlers.clear()
                
                module_logger.addHandler(rich_handler)
                module_logger.propagate = False
    
    def _log_agent_init(self):
        """Display rich initialization information."""
        if not self.rich_logging or not RICH_AVAILABLE:
            return
        
        # Create a table for agent configuration
        table = Table(title=f"[bold]Agent Configuration: [green]{self.config.name}[/green][/bold]")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        # Add configuration details
        for key, value in self.config.__dict__.items():
            if key.startswith("_"):
                continue
                
            # Format complex objects
            if isinstance(value, dict):
                value_str = f"{len(value)} items"
            elif isinstance(value, list):
                value_str = f"{len(value)} entries"
            elif isinstance(value, BaseModel):
                value_str = f"{type(value).__name__}"
            elif callable(value):
                value_str = f"<function {value.__name__}>"
            else:
                value_str = str(value)
                
            # Truncate long values
            if len(value_str) > 50:
                value_str = value_str[:47] + "..."
                
            table.add_row(key, value_str)
        
        self.console.print(table)
    
    def _log_agent_ready(self):
        """Display rich ready notification."""
        if not self.rich_logging or not RICH_AVAILABLE:
            return
            
        self.console.print(
            Panel.fit(
                f"[bold green]Agent {self.config.name} Ready[/bold green]\n"
                f"[cyan]Workflow nodes:[/cyan] {len(self.graph.nodes) if hasattr(self.graph, 'nodes') else 0}\n"
                f"[cyan]Schema fields:[/cyan] {len(self.state_schema.model_fields) if hasattr(self.state_schema, 'model_fields') else 0}\n"
                f"[cyan]Persistence:[/cyan] {type(self.checkpointer).__name__}",
                border_style="green",
                title="Agent Ready",
                subtitle=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
    
    def _setup_directories(self):
        """Set up directories for outputs and artifacts."""
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set up state history directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state_history_dir = Path(self.config.output_dir) / "State_History"
        self.state_history_dir.mkdir(exist_ok=True)
        self.state_filename = self.state_history_dir / f"{self.config.name}_{timestamp}.json"
        
        # Set up graphs directory
        self.graphs_dir = Path(self.config.output_dir) / "Graphs"
        self.graphs_dir.mkdir(exist_ok=True)
        self.graph_image_path = self.graphs_dir / f"{self.config.name}_{timestamp}.png"
        
        # Set up debug logs directory
        self.debug_dir = Path(self.config.output_dir) / "Debug_Logs"
        self.debug_dir.mkdir(exist_ok=True)
        self.debug_log_path = self.debug_dir / f"{self.config.name}_{timestamp}.log"
        
        if self.rich_logging and RICH_AVAILABLE:
            if hasattr(self, "console"):
                self.console.print(f"[blue]Created output directories in:[/blue] {self.config.output_dir}")
        
        logger.debug(f"Output directories set up for {self.config.name}")
    
    def _setup_schemas(self):
        """Set up state, input, and output schemas with rich debugging."""
        if self.rich_logging and RICH_AVAILABLE:
            # Create schema setup progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Setting up schemas...[/bold blue]"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console
            ) as progress:
                state_task = progress.add_task("Setting up state schema...", total=100)
                input_task = progress.add_task("Setting up input schema...", total=100, start=False)
                output_task = progress.add_task("Setting up output schema...", total=100, start=False)
                
                # 1. Process state schema
                if hasattr(self.config, "state_schema") and self.config.state_schema is not None:
                    progress.update(state_task, advance=20, description="[bold blue]Building state schema from config...[/bold blue]")
                    
                    if isinstance(self.config.state_schema, dict):
                        # Build from dictionary definition
                        logger.debug(f"Building state schema from dictionary for {self.config.name}")
                        schema_composer = SchemaComposer(name=f"{self.config.name}State")
                        
                        # Progress through fields
                        field_count = len(self.config.state_schema)
                        for i, (field_name, field_info) in enumerate(self.config.state_schema.items()):
                            progress.update(state_task, advance=60/field_count)
                            
                            if isinstance(field_info, tuple):
                                # (type, default) format
                                field_type, default_value = field_info
                                schema_composer.add_field(field_name, field_type, default=default_value)
                            elif isinstance(field_info, dict):
                                # Dict with parameters
                                field_type = field_info.pop("type", Any)
                                schema_composer.add_field(field_name, field_type, **field_info)
                            else:
                                # Just a type
                                schema_composer.add_field(field_name, field_info)
                        
                        progress.update(state_task, advance=20, description="[bold blue]Building state schema class...[/bold blue]")
                        self.state_schema = schema_composer.build()
                    else:
                        # Use provided class
                        progress.update(state_task, advance=80, description="[bold blue]Using provided state schema...[/bold blue]")
                        logger.debug(f"Using provided state schema for {self.config.name}")
                        self.state_schema = self.config.state_schema
                else:
                    # Derive from components
                    progress.update(state_task, advance=50, description="[bold blue]Deriving state schema from components...[/bold blue]")
                    logger.debug(f"Deriving state schema for {self.config.name}")
                    self.state_schema = self.config.derive_schema()
                    progress.update(state_task, advance=30)
                    
                progress.update(state_task, completed=100, description="[bold green]State schema setup complete[/bold green]")
                progress.start_task(input_task)
                
                # 2. Process input schema
                if hasattr(self.config, "input_schema") and self.config.input_schema is not None:
                    progress.update(input_task, advance=20, description="[bold blue]Building input schema from config...[/bold blue]")
                    
                    if isinstance(self.config.input_schema, dict):
                        # Build from dictionary
                        logger.debug(f"Building input schema for {self.config.name}")
                        schema_composer = SchemaComposer(name=f"{self.config.name}Input")
                        
                        # Progress through fields
                        field_count = len(self.config.input_schema)
                        for i, (field_name, field_info) in enumerate(self.config.input_schema.items()):
                            progress.update(input_task, advance=60/field_count)
                            
                            if isinstance(field_info, tuple):
                                field_type, default_value = field_info
                                schema_composer.add_field(field_name, field_type, default=default_value)
                            elif isinstance(field_info, dict):
                                field_type = field_info.pop("type", Any)
                                schema_composer.add_field(field_name, field_type, **field_info)
                            else:
                                schema_composer.add_field(field_name, field_info)
                                
                        progress.update(input_task, advance=20, description="[bold blue]Building input schema class...[/bold blue]")
                        self.input_schema = schema_composer.build()
                    else:
                        # Use provided class
                        progress.update(input_task, advance=80, description="[bold blue]Using provided input schema...[/bold blue]")
                        logger.debug(f"Using provided input schema for {self.config.name}")
                        self.input_schema = self.config.input_schema
                else:
                    # Default to state schema
                    progress.update(input_task, advance=80, description="[bold blue]Using state schema as input schema...[/bold blue]")
                    logger.debug(f"Using state schema as input schema for {self.config.name}")
                    self.input_schema = self.state_schema
                
                progress.update(input_task, completed=100, description="[bold green]Input schema setup complete[/bold green]")
                progress.start_task(output_task)
                
                # 3. Process output schema
                if hasattr(self.config, "output_schema") and self.config.output_schema is not None:
                    progress.update(output_task, advance=20, description="[bold blue]Building output schema from config...[/bold blue]")
                    
                    if isinstance(self.config.output_schema, dict):
                        # Build from dictionary
                        logger.debug(f"Building output schema for {self.config.name}")
                        schema_composer = SchemaComposer(name=f"{self.config.name}Output")
                        
                        # Progress through fields
                        field_count = len(self.config.output_schema)
                        for i, (field_name, field_info) in enumerate(self.config.output_schema.items()):
                            progress.update(output_task, advance=60/field_count)
                            
                            if isinstance(field_info, tuple):
                                field_type, default_value = field_info
                                schema_composer.add_field(field_name, field_type, default=default_value)
                            elif isinstance(field_info, dict):
                                field_type = field_info.pop("type", Any)
                                schema_composer.add_field(field_name, field_type, **field_info)
                            else:
                                schema_composer.add_field(field_name, field_info)
                                
                        progress.update(output_task, advance=20, description="[bold blue]Building output schema class...[/bold blue]")
                        self.output_schema = schema_composer.build()
                    else:
                        # Use provided class
                        progress.update(output_task, advance=80, description="[bold blue]Using provided output schema...[/bold blue]")
                        logger.debug(f"Using provided output schema for {self.config.name}")
                        self.output_schema = self.config.output_schema
                else:
                    # Default to state schema
                    progress.update(output_task, advance=80, description="[bold blue]Using state schema as output schema...[/bold blue]")
                    logger.debug(f"Using state schema as output schema for {self.config.name}")
                    self.output_schema = self.state_schema
                
                progress.update(output_task, completed=100, description="[bold green]Output schema setup complete[/bold green]")
                
            # Print schema details
            self._debug_print_schemas()
        else:
            # Non-rich implementation (original logic)
            # 1. Process state schema
            if hasattr(self.config, "state_schema") and self.config.state_schema is not None:
                if isinstance(self.config.state_schema, dict):
                    # Build from dictionary definition
                    logger.debug(f"Building state schema from dictionary for {self.config.name}")
                    schema_composer = SchemaComposer(name=f"{self.config.name}State")
                    for field_name, field_info in self.config.state_schema.items():
                        if isinstance(field_info, tuple):
                            # (type, default) format
                            field_type, default_value = field_info
                            schema_composer.add_field(field_name, field_type, default=default_value)
                        elif isinstance(field_info, dict):
                            # Dict with parameters
                            field_type = field_info.pop("type", Any)
                            schema_composer.add_field(field_name, field_type, **field_info)
                        else:
                            # Just a type
                            schema_composer.add_field(field_name, field_info)
                    self.state_schema = schema_composer.build()
                else:
                    # Use provided class
                    logger.debug(f"Using provided state schema for {self.config.name}")
                    self.state_schema = self.config.state_schema
            else:
                # Derive from components
                logger.debug(f"Deriving state schema for {self.config.name}")
                self.state_schema = self.config.derive_schema()
            
            # 2. Process input schema (default to state schema if not provided)
            if hasattr(self.config, "input_schema") and self.config.input_schema is not None:
                if isinstance(self.config.input_schema, dict):
                    # Build from dictionary
                    logger.debug(f"Building input schema for {self.config.name}")
                    schema_composer = SchemaComposer(name=f"{self.config.name}Input")
                    for field_name, field_info in self.config.input_schema.items():
                        if isinstance(field_info, tuple):
                            field_type, default_value = field_info
                            schema_composer.add_field(field_name, field_type, default=default_value)
                        elif isinstance(field_info, dict):
                            field_type = field_info.pop("type", Any)
                            schema_composer.add_field(field_name, field_type, **field_info)
                        else:
                            schema_composer.add_field(field_name, field_info)
                    self.input_schema = schema_composer.build()
                else:
                    # Use provided class
                    logger.debug(f"Using provided input schema for {self.config.name}")
                    self.input_schema = self.config.input_schema
            else:
                # Default to state schema
                logger.debug(f"Using state schema as input schema for {self.config.name}")
                self.input_schema = self.state_schema
            
            # 3. Process output schema (default to state schema if not provided)
            if hasattr(self.config, "output_schema") and self.config.output_schema is not None:
                if isinstance(self.config.output_schema, dict):
                    # Build from dictionary
                    logger.debug(f"Building output schema for {self.config.name}")
                    schema_composer = SchemaComposer(name=f"{self.config.name}Output")
                    for field_name, field_info in self.config.output_schema.items():
                        if isinstance(field_info, tuple):
                            field_type, default_value = field_info
                            schema_composer.add_field(field_name, field_type, default=default_value)
                        elif isinstance(field_info, dict):
                            field_type = field_info.pop("type", Any)
                            schema_composer.add_field(field_name, field_type, **field_info)
                        else:
                            schema_composer.add_field(field_name, field_info)
                    self.output_schema = schema_composer.build()
                else:
                    # Use provided class
                    logger.debug(f"Using provided output schema for {self.config.name}")
                    self.output_schema = self.config.output_schema
            else:
                # Default to state schema
                logger.debug(f"Using state schema as output schema for {self.config.name}")
                self.output_schema = self.state_schema
        
        logger.debug(f"Schemas set up successfully for {self.config.name}")
    
    def _debug_print_schemas(self):
        """Print schema details using rich UI."""
        if not self.rich_logging or not RICH_AVAILABLE:
            return
            
        # Create a table for all schemas
        schema_table = Table(title="[bold]Schema Definitions[/bold]")
        schema_table.add_column("Schema", style="cyan")
        schema_table.add_column("Field", style="green")
        schema_table.add_column("Type", style="magenta")
        schema_table.add_column("Default", style="yellow")
        
        # Helper to extract field info
        def extract_field_info(schema):
            fields = {}
            if hasattr(schema, "model_fields"):
                # Pydantic v2
                for name, field in schema.model_fields.items():
                    fields[name] = {
                        "type": field.annotation,
                        "default": field.default
                    }
            elif hasattr(schema, "__fields__"):
                # Pydantic v1
                for name, field in schema.__fields__.items():
                    fields[name] = {
                        "type": field.type_,
                        "default": field.default
                    }
            return fields
        
        # Add state schema
        state_fields = extract_field_info(self.state_schema)
        for field_name, info in state_fields.items():
            type_str = str(info["type"]).replace("typing.", "")
            default_str = str(info["default"])
            if len(default_str) > 30:
                default_str = default_str[:27] + "..."
            schema_table.add_row("State", field_name, type_str, default_str)
        
        # Add input schema if different from state
        if self.input_schema != self.state_schema:
            input_fields = extract_field_info(self.input_schema)
            for field_name, info in input_fields.items():
                type_str = str(info["type"]).replace("typing.", "")
                default_str = str(info["default"])
                if len(default_str) > 30:
                    default_str = default_str[:27] + "..."
                schema_table.add_row("Input", field_name, type_str, default_str)
        
        # Add output schema if different from state
        if self.output_schema != self.state_schema and self.output_schema != self.input_schema:
            output_fields = extract_field_info(self.output_schema)
            for field_name, info in output_fields.items():
                type_str = str(info["type"]).replace("typing.", "")
                default_str = str(info["default"])
                if len(default_str) > 30:
                    default_str = default_str[:27] + "..."
                schema_table.add_row("Output", field_name, type_str, default_str)
        
        # Print the table
        self.console.print(schema_table)
        
        # Show shared schema notice if needed
        if self.input_schema == self.state_schema:
            self.console.print("[yellow]Note:[/yellow] Input schema is using the state schema")
        if self.output_schema == self.state_schema:
            self.console.print("[yellow]Note:[/yellow] Output schema is using the state schema")
    
    def _initialize_engines(self):
        """Initialize all engines from configuration with rich UI."""
        self.engines = {}
        
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            engine_table = Table(title="[bold]Engine Initialization[/bold]")
            engine_table.add_column("Name", style="cyan")
            engine_table.add_column("Type", style="green")
            engine_table.add_column("ID", style="yellow")
            engine_table.add_column("Model", style="magenta")
            engine_table.add_column("Status", style="blue")
            
            # Initialize main engine if present
            if hasattr(self.config, "engine") and self.config.engine:
                engine_name = "main"
                engine_type = getattr(self.config.engine, "engine_type", "unknown")
                engine_id = getattr(self.config.engine, "id", "not-set")
                engine_model = getattr(self.config.engine, "model", 
                                     getattr(self.config.engine, "model_name", "unknown"))
                                     
                with self.console.status(f"[bold blue]Initializing main engine...[/bold blue]"):
                    self.engine = self.config.engine
                    self.engines[engine_name] = self.engine
                
                engine_table.add_row(
                    engine_name, 
                    str(engine_type), 
                    str(engine_id), 
                    str(engine_model),
                    "[green]✓[/green]"
                )
            
            # Initialize additional engines
            for name, engine_config in getattr(self.config, "engines", {}).items():
                engine_type = getattr(engine_config, "engine_type", "unknown")
                engine_id = getattr(engine_config, "id", "not-set")
                engine_model = getattr(engine_config, "model", 
                                     getattr(engine_config, "model_name", "unknown"))
                                     
                with self.console.status(f"[bold blue]Initializing engine '{name}'...[/bold blue]"):
                    self.engines[name] = engine_config
                
                engine_table.add_row(
                    name, 
                    str(engine_type), 
                    str(engine_id), 
                    str(engine_model),
                    "[green]✓[/green]"
                )
            
            self.console.print(engine_table)
            self.console.print(f"[green]Successfully initialized {len(self.engines)} engines[/green]")
        else:
            # Initialize main engine if present
            if hasattr(self.config, "engine") and self.config.engine:
                engine_name = "main"
                logger.debug(f"Initializing main engine for {self.config.name}")
                self.engine = self.config.engine
                self.engines[engine_name] = self.engine
                logger.debug(f"Main engine initialized: {getattr(self.engine, 'name', 'unknown')}")
            
            # Initialize additional engines
            for name, engine_config in getattr(self.config, "engines", {}).items():
                logger.debug(f"Initializing engine '{name}' for {self.config.name}")
                self.engines[name] = engine_config
                logger.debug(f"Engine '{name}' initialized: {getattr(self.engines[name], 'name', 'unknown')}")
        
        logger.debug(f"Initialized {len(self.engines)} engines for {self.config.name}")
    
    def _setup_persistence(self):
        """Set up persistence with proper checkpointer."""
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            with self.console.status("[bold blue]Setting up persistence...[/bold blue]"):
                # Set up checkpointer from persistence configuration
                self.checkpointer = setup_checkpointer(self.config)
                
                # Add store if configured
                self.store = None
                if getattr(self.config, "add_store", False):
                    from langgraph.store.base import BaseStore
                    self.store = BaseStore()
            
            # Show persistence info
            persistence_type = type(self.checkpointer).__name__
            if "Postgres" in persistence_type:
                status_color = "blue"
                status_icon = "🐘"  # PostgreSQL elephant
            elif "Memory" in persistence_type:
                status_color = "yellow"
                status_icon = "💾"  # Memory icon
            else:
                status_color = "green"
                status_icon = "📦"  # Generic storage
                
            self.console.print(
                f"[bold]Persistence:[/bold] {status_icon} [{status_color}]{persistence_type}[/{status_color}]"
            )
            
            if hasattr(self, "store") and self.store:
                self.console.print("[bold]Store:[/bold] ✅ Enabled")
        else:
            # Set up checkpointer from persistence configuration
            self.checkpointer = setup_checkpointer(self.config)
            logger.debug(f"Checkpointer set up for {self.config.name}: {type(self.checkpointer).__name__}")
            
            # Add store if configured
            self.store = None
            if getattr(self.config, "add_store", False):
                from langgraph.store.base import BaseStore
                self.store = BaseStore()
                logger.debug("BaseStore added to agent")
    
    def _setup_runtime_config(self):
        """Set up default runtime configuration."""
        # Initialize runnable_config from config or create default
        if hasattr(self.config, "runnable_config") and self.config.runnable_config:
            self.runnable_config = self.config.runnable_config
        else:
            self.runnable_config = RunnableConfigManager.create(thread_id=str(uuid.uuid4()))
        
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            # Display the runtime configuration
            runtime_md = "## Runtime Configuration\n\n"
            runtime_md += "```json\n"
            runtime_md += json.dumps(self.runnable_config, indent=2, default=str)
            runtime_md += "\n```"
            
            self.console.print(Markdown(runtime_md))
            
        logger.debug(f"Runtime configuration set up for {self.config.name}")
    
    def _create_graph(self):
        """Create the DynamicGraph builder with proper schemas."""
        # Get all components for DynamicGraph
        components = list(self.engines.values())
        
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            with self.console.status("[bold blue]Creating graph builder...[/bold blue]") as status:
                # Create DynamicGraph with fully resolved schemas
                status.update("[bold blue]Setting up DynamicGraph...[/bold blue]")
                self.graph_builder = DynamicGraph(
                    name=self.config.name,
                    description=getattr(self.config, "description", f"Agent {self.config.name}"),
                    components=components,
                    # Pass fully resolved schemas
                    state_schema=self.state_schema,
                    input_schema=self.input_schema,
                    output_schema=self.output_schema,
                    default_runnable_config=self.runnable_config,
                    debug=self.verbose
                )
                
                # Apply default patterns if specified
                status.update("[bold blue]Applying default patterns...[/bold blue]")
                self._apply_default_patterns()
            
            # Show graph info
            self.console.print(f"[bold]Graph builder:[/bold] [green]Created successfully[/green]")
            
            # Display patterns if any were applied
            default_patterns = getattr(self.config, "default_patterns", [])
            if default_patterns:
                patterns_list = ""
                for pattern in default_patterns:
                    if isinstance(pattern, str):
                        patterns_list += f"- {pattern}\n"
                    elif isinstance(pattern, dict) and "name" in pattern:
                        patterns_list += f"- {pattern['name']}\n"
                
                if patterns_list:
                    self.console.print(f"[bold]Applied patterns:[/bold]\n{patterns_list}")
        else:
            # Create DynamicGraph with fully resolved schemas
            logger.debug(f"Creating DynamicGraph for {self.config.name}")
            self.graph_builder = DynamicGraph(
                name=self.config.name,
                description=getattr(self.config, "description", f"Agent {self.config.name}"),
                components=components,
                # Pass fully resolved schemas
                state_schema=self.state_schema,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                default_runnable_config=self.runnable_config,
                debug=self.verbose
            )
            
            # Apply default patterns if specified
            self._apply_default_patterns()
            logger.debug(f"DynamicGraph created for {self.config.name}")
        
        # No graph yet - will be built after setup_workflow
        self.graph = None
        self.app = None
    
    def _apply_default_patterns(self):
        """Apply default graph patterns specified in config."""
        default_patterns = getattr(self.config, "default_patterns", [])
        
        if not default_patterns:
            return
            
        for pattern_def in default_patterns:
            if isinstance(pattern_def, str):
                # Simple pattern name
                pattern_name = pattern_def
                pattern_params = {}
            elif isinstance(pattern_def, dict):
                # Pattern with parameters
                pattern_name = pattern_def.pop("name")
                pattern_params = pattern_def
            else:
                logger.warning(f"Invalid pattern definition: {pattern_def}")
                continue
                
            # Apply the pattern
            try:
                logger.debug(f"Applying default pattern '{pattern_name}' to {self.config.name}")
                self.graph_builder.apply_pattern(pattern_name, **pattern_params)
            except Exception as e:
                logger.error(f"Error applying pattern '{pattern_name}': {e}")
    
    @abstractmethod
    def setup_workflow(self) -> None:
        """
        Set up the workflow graph for this agent.
        
        This method must be implemented by concrete agent classes to define
        the agent's workflow structure.
        """
        pass
    
    def compile(self) -> None:
        """Compile the workflow graph into an executable app."""
        logger.info(f"Compiling graph for {self.config.name}")
        
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            with self.console.status("[bold blue]Compiling workflow graph...[/bold blue]") as status:
                # Build graph if not already built
                if not self.graph:
                    status.update("[bold blue]Building graph from builder...[/bold blue]")
                    self.graph = self.graph_builder.build()
                
                # Make sure checkpointer tables are set up if needed
                if hasattr(self.checkpointer, "setup"):
                    try:
                        # Ensure connection is open 
                        status.update("[bold blue]Setting up checkpointer tables...[/bold blue]")
                        ensure_pool_open(self.checkpointer)
                        
                        # Set up tables
                        self.checkpointer.setup()
                    except Exception as e:
                        self.console.print(f"[bold red]Error setting up checkpointer tables: {e}[/bold red]")
                
                # Compile the graph
                status.update("[bold blue]Compiling graph with checkpointer...[/bold blue]")
                start_time = time.time()
                self.app = self.graph.compile(checkpointer=self.checkpointer)
                compile_time = time.time() - start_time
            
            # Show compilation result
            self.console.print(f"[bold green]Graph compiled successfully[/bold green] in {compile_time:.2f} seconds")
            
            # Get node count
            node_count = len(self.graph.nodes) if hasattr(self.graph, "nodes") else 0
            self.console.print(f"[bold]Graph nodes:[/bold] {node_count}")
        else:
            # Build graph if not already built
            if not self.graph:
                logger.debug("Building graph from graph_builder")
                self.graph = self.graph_builder.build()
            
            # Make sure checkpointer tables are set up if needed
            if hasattr(self.checkpointer, "setup"):
                try:
                    # Ensure connection is open 
                    ensure_pool_open(self.checkpointer)
                    
                    # Set up tables
                    logger.debug("Setting up checkpointer tables")
                    self.checkpointer.setup()
                    logger.debug("Checkpointer tables set up successfully")
                except Exception as e:
                    logger.error(f"Error setting up checkpointer tables: {e}")
            
            # Compile the graph
            logger.debug("Compiling graph with checkpointer")
            self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        logger.info(f"Graph compiled successfully for {self.config.name}")
    
    def visualize_graph(self, output_path: Optional[str] = None) -> None:
        """
        Generate and save a visualization of the agent's graph.
        
        Args:
            output_path: Optional custom path for visualization output
        """
        if not output_path:
            output_path = self.graph_image_path
            
        logger.debug(f"Visualizing graph for {self.config.name}")
        
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            with self.console.status(f"[bold blue]Generating graph visualization...[/bold blue]"):
                # Use DynamicGraph's visualization capability
                from haive.core.utils.visualize_graph_utils import render_and_display_graph
                 # Fall back to compiled graph visualization if available
                if self.app and hasattr(self.app, "get_graph"):
                    try:
                        # Ensure directory exists
                        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                        
                        # Generate and save PNG
                        png_data = self.app.get_graph(xray=True).draw_mermaid_png()
                        with open(output_path, "wb") as f:
                            f.write(png_data)
                            
                        self.console.print(f"[bold green]Graph visualization saved to:[/bold green] {output_path}")
                    except Exception as e:
                        self.console.print(f"[bold red]Error visualizing graph: {e}[/bold red]")
                if hasattr(self.graph_builder, "visualize"):
                    try:
                        self.graph_builder.visualize_graph(output_path)
                        self.console.print(f"[bold green]Graph visualization saved to:[/bold green] {output_path}")
                        return
                    except Exception as e:
                        self.console.print(f"[bold yellow]Error using DynamicGraph visualization: {e}[/bold yellow]")
                
               
        else:
            # Use DynamicGraph's visualization capability
            if hasattr(self.graph_builder, "visualize"):
                try:
                    self.graph_builder.visualize_graph(output_path)
                    logger.info(f"Graph visualization saved to: {output_path}")
                    return
                except Exception as e:
                    logger.warning(f"Error using DynamicGraph visualization: {e}")
            
            # Fall back to compiled graph visualization if available
            if self.app and hasattr(self.app, "get_graph"):
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Generate and save PNG
                    png_data = self.app.get_graph(xray=True).draw_mermaid_png()
                    with open(output_path, "wb") as f:
                        f.write(png_data)
                        
                    logger.info(f"Graph visualization saved to: {output_path}")
                except Exception as e:
                    logger.error(f"Error visualizing graph: {e}")
    
    def apply_pattern(self, pattern_name: str, **kwargs) -> None:
        """
        Apply a graph pattern to the agent's workflow.
        
        Args:
            pattern_name: Name of the pattern to apply
            **kwargs: Pattern-specific parameters
        """
        if not hasattr(self, "graph_builder") or not self.graph_builder:
            logger.error("Cannot apply pattern: Graph builder is not set up")
            raise RuntimeError("Graph builder is not set up.")
            
        try:
            logger.info(f"Applying pattern '{pattern_name}' to {self.config.name}")
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                with self.console.status(f"[bold blue]Applying pattern '{pattern_name}'...[/bold blue]"):
                    self.graph_builder.apply_pattern(pattern_name, **kwargs)
                    
                    # Mark graph as needing rebuild
                    self.graph = None
                    self.app = None
                
                self.console.print(f"[bold green]Pattern '{pattern_name}' applied successfully[/bold green]")
            else:
                self.graph_builder.apply_pattern(pattern_name, **kwargs)
                
                # Mark graph as needing rebuild
                self.graph = None
                self.app = None
                
                logger.info(f"Pattern '{pattern_name}' applied successfully")
            
        except Exception as e:
            logger.error(f"Error applying pattern '{pattern_name}': {e}")
            raise RuntimeError(f"Failed to apply pattern '{pattern_name}': {e}") from e
    
    def _prepare_runnable_config(
        self, 
        thread_id: Optional[str] = None, 
        config: Optional[RunnableConfig] = None, 
        **kwargs
    ) -> RunnableConfig:
        """
        Prepare a runnable config with thread ID and other parameters.
        
        Args:
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional configuration parameters
            
        Returns:
            Prepared runnable configuration
        """
        # Start with the agent's base config
        base_config = getattr(self, "runnable_config", None)
        
        # Create new config with thread_id if provided, otherwise merge with existing
        if thread_id:
            # If thread_id is explicitly provided, use it as the primary ID
            runtime_config = RunnableConfigManager.create(
                thread_id=thread_id,
                user_id=kwargs.pop("user_id", None)
            )
            
            # Merge with base config if available
            if base_config:
                runtime_config = RunnableConfigManager.merge(base_config, runtime_config)
                
            # Merge with provided config if available
            if config:
                runtime_config = RunnableConfigManager.merge(runtime_config, config)
        elif config:
            # Start with provided config and merge with base
            if base_config:
                runtime_config = RunnableConfigManager.merge(base_config, config)
            else:
                runtime_config = config
        else:
            # Use base config or create new one
            if base_config:
                runtime_config = base_config
            else:
                runtime_config = RunnableConfigManager.create()
        
        # Ensure configurable section exists
        if "configurable" not in runtime_config:
            runtime_config["configurable"] = {}
            
        # Ensure thread_id exists
        if "thread_id" not in runtime_config["configurable"]:
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
            
        # Add other kwargs
        for key, value in kwargs.items():
            if key.startswith("configurable_"):
                # Handle configurable_ prefix for convenience
                param_name = key.replace("configurable_", "")
                runtime_config["configurable"][param_name] = value
            elif key == "configurable" and isinstance(value, dict):
                # Merge configurable dict
                for k, v in value.items():
                    runtime_config["configurable"][k] = v
            elif key == "engine_configs" and isinstance(value, dict):
                # Setup engine_configs section if not present
                if "engine_configs" not in runtime_config["configurable"]:
                    runtime_config["configurable"]["engine_configs"] = {}
                    
                # Merge engine configs
                for engine_id, engine_params in value.items():
                    if engine_id not in runtime_config["configurable"]["engine_configs"]:
                        runtime_config["configurable"]["engine_configs"][engine_id] = {}
                    runtime_config["configurable"]["engine_configs"][engine_id].update(engine_params)
            else:
                # Add to top level
                runtime_config[key] = value
        
        if self.rich_logging and RICH_AVAILABLE and self.verbose and hasattr(self, "debug_console"):
            # Log the runtime config in debug mode
            self.debug_console.print("[bold]Runtime Config:[/bold]")
            config_json = json.dumps(runtime_config, indent=2, default=str)
            self.debug_console.print(Syntax(config_json, "json", theme="monokai", line_numbers=True))
                
        return runtime_config
    
    def save_state_history(self, runnable_config=None) -> bool:
        """
        Save the current agent state to a JSON file.
        
        Args:
            runnable_config: Optional runnable configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not self.app:
            logger.error("Cannot save state history: Workflow graph not compiled")
            return False

        # Use provided runnable config or default
        runnable_config = runnable_config or self.runnable_config

        try:
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                with self.console.status("[bold blue]Saving state history...[/bold blue]") as status:
                    # Get state from app
                    state_json = self.app.get_state(runnable_config)

                    if not state_json:
                        self.console.print("[bold yellow]No state history available[/bold yellow]")
                        return False

                    # Ensure state is JSON serializable
                    state_json = ensure_json_serializable(state_json)

                    # Save to file
                    with open(self.state_filename, "w", encoding="utf-8") as f:
                        json.dump(state_json, f, indent=4)

                self.console.print(f"[bold green]State history saved to:[/bold green] {self.state_filename}")
            else:
                # Get state from app
                state_json = self.app.get_state(runnable_config)

                if not state_json:
                    logger.warning(f"No state history available for {self.config.name}")
                    return False

                # Ensure state is JSON serializable
                state_json = ensure_json_serializable(state_json)

                # Save to file
                with open(self.state_filename, "w", encoding="utf-8") as f:
                    json.dump(state_json, f, indent=4)

                logger.info(f"State history saved to: {self.state_filename}")
                
            return True
        except Exception as e:
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error saving state history: {e}[/bold red]")
            else:
                logger.error(f"Error saving state history: {e}")
            return False
    
    def _debug_input_processor(self, input_data, processed_input):
        """Debug input processing with rich UI."""
        if not self.rich_logging or not RICH_AVAILABLE or not hasattr(self, "debug_console"):
            return
            
        # Only show in verbose mode
        if not self.verbose:
            return
            
        self.debug_console.rule("[bold]Input Processing Debug[/bold]")
        
        # Show raw input
        self.debug_console.print("[bold cyan]Raw Input:[/bold cyan]")
        if isinstance(input_data, str):
            self.debug_console.print(f"[green]String Input:[/green] {input_data[:100]}{'...' if len(input_data) > 100 else ''}")
        elif isinstance(input_data, list):
            self.debug_console.print(f"[green]List Input:[/green] {len(input_data)} items")
            for i, item in enumerate(input_data[:3]):
                self.debug_console.print(f"  [dim]Item {i}:[/dim] {str(item)[:50]}{'...' if len(str(item)) > 50 else ''}")
            if len(input_data) > 3:
                self.debug_console.print(f"  [dim]... and {len(input_data) - 3} more items[/dim]")
        elif isinstance(input_data, dict):
            self.debug_console.print(f"[green]Dict Input:[/green] {len(input_data)} keys")
            syntax = Syntax(json.dumps(input_data, indent=2, default=str)[:500], "json", theme="monokai")
            self.debug_console.print(syntax)
        elif isinstance(input_data, BaseModel):
            self.debug_console.print(f"[green]Pydantic Model Input:[/green] {type(input_data).__name__}")
            if hasattr(input_data, "model_dump"):
                model_dict = input_data.model_dump()
            else:
                model_dict = input_data.dict()
            syntax = Syntax(json.dumps(model_dict, indent=2, default=str)[:500], "json", theme="monokai")
            self.debug_console.print(syntax)
        else:
            self.debug_console.print(f"[green]Other Input Type:[/green] {type(input_data).__name__}")
            
        # Show processed input
        self.debug_console.print("\n[bold cyan]Processed Input:[/bold cyan]")
        if isinstance(processed_input, dict):
            syntax = Syntax(json.dumps(processed_input, indent=2, default=str)[:1000], "json", theme="monokai")
            self.debug_console.print(syntax)
        elif isinstance(processed_input, BaseModel):
            if hasattr(processed_input, "model_dump"):
                model_dict = processed_input.model_dump()
            else:
                model_dict = processed_input.dict()
            syntax = Syntax(json.dumps(model_dict, indent=2, default=str)[:1000], "json", theme="monokai")
            self.debug_console.print(syntax)
        else:
            self.debug_console.print(f"[yellow]Unexpected processed input type:[/yellow] {type(processed_input).__name__}")
            
        self.debug_console.rule()
    
    def _prepare_input(self, input_data: Union[str, List[str], Dict[str, Any], BaseModel]) -> Any:
        """
        Prepare input for the agent based on the input type.
        
        Args:
            input_data: Input in various formats
            
        Returns:
            Processed input compatible with the graph
        """
        # If we have an input schema and the input is a simple scalar (string, number, etc.),
        # we need to convert it to match the schema's expected structure
        if self.config.input_schema:
            # Get the schema fields to determine the structure
            schema_fields = {}
            if hasattr(self.config.input_schema, "model_fields"):
                # Pydantic v2
                schema_fields = self.config.input_schema.model_fields
            elif hasattr(self.config.input_schema, "__fields__"):
                # Pydantic v1
                schema_fields = self.config.input_schema.__fields__
                
            # First we need to convert the input to the appropriate format
            if isinstance(input_data, str):
                # Convert string input to dict format
                if len(schema_fields) == 1:
                    # If schema has only one field, use the string as its value
                    field_name = list(schema_fields.keys())[0]
                    prepared_input = {field_name: input_data}
                else:
                    # Otherwise, create a basic structure with the string in multiple fields
                    prepared_input = {}
                    # Fill in commonly expected fields with the string value
                    for field_name in ["input", "query", "content", "question", "messages"]:
                        if field_name in schema_fields:
                            if field_name == "messages":
                                prepared_input[field_name] = [HumanMessage(content=input_data)]
                            else:
                                prepared_input[field_name] = input_data
                                
                    # If no matches found, use the first field in the schema
                    if not prepared_input and schema_fields:
                        field_name = list(schema_fields.keys())[0]
                        prepared_input = {field_name: input_data}
                        
            elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
                # Convert list of strings to dict format
                if len(schema_fields) == 1:
                    # If schema has only one field, use the list as its value
                    field_name = list(schema_fields.keys())[0]
                    prepared_input = {field_name: input_data}
                else:
                    # Otherwise create a structure with different formats
                    prepared_input = {}
                    joined_text = "\n".join(input_data)
                    
                    for field_name in schema_fields:
                        if field_name == "messages":
                            prepared_input[field_name] = [HumanMessage(content=text) for text in input_data]
                        elif "list" in str(schema_fields[field_name]).lower():
                            # If the field is a list type, use the original list
                            prepared_input[field_name] = input_data
                        else:
                            # Otherwise join the strings for text fields
                            prepared_input[field_name] = joined_text
                            
            elif isinstance(input_data, dict):
                # Use dict as is
                prepared_input = input_data
                
            elif isinstance(input_data, BaseModel):
                # Convert BaseModel to dict
                if hasattr(input_data, "model_dump"):
                    # Pydantic v2
                    prepared_input = input_data.model_dump()
                elif hasattr(input_data, "dict"):
                    # Pydantic v1
                    prepared_input = input_data.dict()
                else:
                    # Manual extraction
                    prepared_input = {}
                    for field in input_data.__annotations__:
                        if hasattr(input_data, field):
                            prepared_input[field] = getattr(input_data, field)
            else:
                # For other types, create a dictionary with the input as the value for each field
                prepared_input = {}
                for field_name in schema_fields:
                    if field_name == "messages":
                        prepared_input[field_name] = [HumanMessage(content=str(input_data))]
                    else:
                        prepared_input[field_name] = str(input_data)
            
            # Ensure all required fields have values
            for field_name, field_info in schema_fields.items():
                # Check if field is required
                is_required = False
                if hasattr(field_info, "is_required"):  # Pydantic v2
                    is_required = field_info.is_required
                elif hasattr(field_info, "required"):   # Pydantic v1
                    is_required = field_info.required
                    
                # Add default values for missing required fields
                if is_required and field_name not in prepared_input:
                    if field_name == "messages":
                        prepared_input[field_name] = []
                    else:
                        prepared_input[field_name] = ""
                        
            # Validate against the schema
            result = self.config.input_schema.model_validate(prepared_input)
            
            # Debug
            self._debug_input_processor(input_data, result)
            
            return result
        
        # When no input schema is available, use similar logic to original function
        
        # When input is a string
        if isinstance(input_data, str):
            # Initialize input dict
            prepared_input = {"messages": [HumanMessage(content=input_data)]}
            
            # Detect additional required inputs
            required_inputs = set()
            if hasattr(self, 'engine') and hasattr(self.engine, 'prompt_template') and hasattr(self.engine.prompt_template, 'input_variables'):
                required_inputs.update(self.engine.prompt_template.input_variables)
            
            # Add string to all required inputs
            for input_name in required_inputs:
                if input_name != 'messages':
                    prepared_input[input_name] = input_data
            
            result = self.state_schema(**prepared_input) if hasattr(self, 'state_schema') else prepared_input
            
            # Debug
            self._debug_input_processor(input_data, result)
            
            return result
        
        # Handle other types of input
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # List of strings - convert to messages
            messages = [HumanMessage(content=item) for item in input_data]
            
            prepared_input = {'messages': messages}
            
            # For other inputs, join the strings
            required_inputs = set()
            if hasattr(self, 'engine') and hasattr(self.engine, 'prompt_template') and hasattr(self.engine.prompt_template, 'input_variables'):
                required_inputs.update(self.engine.prompt_template.input_variables)
            
            for input_name in required_inputs:
                if input_name != 'messages':
                    prepared_input[input_name] = "\n".join(input_data)
            
            result = self.state_schema(**prepared_input) if hasattr(self, 'state_schema') else prepared_input
            
            # Debug
            self._debug_input_processor(input_data, result)
            
            return result
        
        elif isinstance(input_data, dict):
            # Dict input - use as is
            result = self.state_schema(**input_data) if hasattr(self, 'state_schema') else input_data
            
            # Debug
            self._debug_input_processor(input_data, result)
            
            return result
        
        elif isinstance(input_data, BaseModel):
            # Convert BaseModel to dict
            if hasattr(input_data, "model_dump"):
                # Pydantic v2
                model_dict = input_data.model_dump()
                result = self.state_schema(**model_dict) if hasattr(self, 'state_schema') else model_dict
            elif hasattr(input_data, "dict"):
                # Pydantic v1
                model_dict = input_data.dict()
                result = self.state_schema(**model_dict) if hasattr(self, 'state_schema') else model_dict
            else:
                # Manual extraction
                prepared_input = {}
                for field in input_data.__annotations__:
                    if hasattr(input_data, field):
                        prepared_input[field] = getattr(input_data, field)
                result = self.state_schema(**prepared_input) if hasattr(self, 'state_schema') else prepared_input
                
            # Debug
            self._debug_input_processor(input_data, result)
            
            return result
        
        # Fallback
        default_result = self.state_schema(messages=[HumanMessage(content=str(input_data))]) if hasattr(self, 'state_schema') else {
            "messages": [HumanMessage(content=str(input_data))]
        }
        
        # Debug
        self._debug_input_processor(input_data, default_result)
        
        return default_result
    
    def _debug_execution(self, method: str, thread_id: str, result: Any = None, error: Exception = None):
        """Log execution details using rich UI."""
        if not self.rich_logging or not RICH_AVAILABLE or not hasattr(self, "debug_console"):
            return
            
        # Only print in verbose mode
        if not self.verbose:
            return
            
        self.debug_console.rule(f"[bold]Execution Debug: {method}[/bold]")
        
        # Show thread ID
        self.debug_console.print(f"[cyan]Thread ID:[/cyan] {thread_id}")
        
        # Show execution result or error
        if error:
            self.debug_console.print(f"[bold red]Error:[/bold red] {str(error)}")
            if hasattr(error, "__traceback__"):
                import traceback
                tb_str = ''.join(traceback.format_tb(error.__traceback__))
                self.debug_console.print(f"[dim red]Traceback:[/dim red]\n{tb_str}")
        elif result is not None:
            self.debug_console.print("[bold green]Result:[/bold green]")
            if isinstance(result, dict):
                # Print top-level keys
                keys_str = ", ".join(result.keys())
                self.debug_console.print(f"[dim]Keys:[/dim] {keys_str}")
                
                # For large results, show summary
                if len(str(result)) > 1000:
                    self.debug_console.print(f"[dim]Large result ({len(str(result))} chars). Summary:[/dim]")
                    
                    # Show select fields that are typically interesting
                    for key in ["output", "answer", "response", "messages", "documents"]:
                        if key in result:
                            value = result[key]
                            if isinstance(value, list):
                                self.debug_console.print(f"[yellow]{key}:[/yellow] List with {len(value)} items")
                                if value and len(value) > 0:
                                    if isinstance(value[0], BaseMessage):
                                        self.debug_console.print(f"  [dim]Last message ({type(value[-1]).__name__}):[/dim]")
                                        self.debug_console.print(f"  {value[-1].content[:200]}{'...' if len(value[-1].content) > 200 else ''}")
                            else:
                                if isinstance(value, str):
                                    self.debug_console.print(f"[yellow]{key}:[/yellow] {value[:200]}{'...' if len(value) > 200 else ''}")
                                else:
                                    self.debug_console.print(f"[yellow]{key}:[/yellow] {str(value)[:200]}{'...' if len(str(value)) > 200 else ''}")
                else:
                    # Show full result
                    syntax = Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai")
                    self.debug_console.print(syntax)
            else:
                # Non-dict result
                self.debug_console.print(str(result)[:500] + ('...' if len(str(result)) > 500 else ''))
        
        self.debug_console.rule()
    
    def run(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel],
        thread_id: Optional[str] = None,
        debug: bool = True,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent with the given input and return the final state.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            debug: Whether to enable debug mode
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            logger.debug("Agent not compiled yet, compiling now")
            self.compile()

        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Show run info in rich UI
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            self.console.print(Panel.fit(
                f"[bold]Running Agent[/bold]\n"
                f"[cyan]Thread ID:[/cyan] {current_thread_id}",
                border_style="blue",
                title=self.config.name,
                subtitle=datetime.now().strftime("%H:%M:%S")
            ))
        
        logger.info(f"Running agent {self.config.name} with thread_id: {current_thread_id}")
        
        # Register thread in database if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)

        # Get previous state
        previous_state = None
        try:
            previous_state = self.app.get_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {current_thread_id}")
                
                # Show previous state in rich UI
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console") and self.verbose:
                    self.console.print("[bold]Found previous state:[/bold]")
                    if hasattr(previous_state, 'values'):
                        previous_values = previous_state.values
                        # Show key stats
                        if isinstance(previous_values, dict):
                            keys_str = ", ".join(previous_values.keys())
                            self.console.print(f"[dim]Keys:[/dim] {keys_str}")
                
        except Exception as e:
            logger.debug(f"No previous state found: {e}")
        
        # Process input data
        if isinstance(input_data, str):
            processed_input = self._prepare_input(input_data)
        else:
            processed_input = input_data

        # Run the agent
        try:
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                # Show animated status during execution
                with self.console.status("[bold green]Agent processing...[/bold green]", spinner="dots") as status:
                    start_time = time.time()
                    result = self.app.invoke(
                        processed_input,
                        config=runtime_config,
                        debug=debug
                    )
                    execution_time = time.time() - start_time
                
                # Show completion status
                self.console.print(f"[bold green]Execution complete in {execution_time:.2f} seconds[/bold green]")
                
                # Show result summary
                if isinstance(result, dict):
                    # Check for common output fields
                    output_fields = 0
                    table = Table(title="Result Summary", box=None)
                    table.add_column("Field", style="cyan")
                    table.add_column("Value", style="green")
                    
                    for field in ["output", "answer", "response"]:
                        if field in result and isinstance(result[field], str):
                            output = result[field]
                            display_output = output[:100] + ("..." if len(output) > 100 else "")
                            table.add_row(field, display_output)
                            output_fields += 1
                    
                    # Show message count if present
                    if "messages" in result and isinstance(result["messages"], list):
                        table.add_row("messages", f"{len(result['messages'])} messages")
                        output_fields += 1
                    
                    # Show document count if present
                    if "documents" in result and isinstance(result["documents"], list):
                        table.add_row("documents", f"{len(result['documents'])} documents")
                        output_fields += 1
                    
                    # Print table if we have output fields
                    if output_fields > 0:
                        self.console.print(table)
                
                # Save state history if requested
                if getattr(self.config, "save_history", True):
                    self.save_state_history(runtime_config)
            else:
                # Standard execution
                result = self.app.invoke(
                    processed_input,
                    config=runtime_config,
                    debug=debug
                )

                # Save state history if requested
                if getattr(self.config, "save_history", True):
                    self.save_state_history(runtime_config)
            
            # Debug execution results
            self._debug_execution("run", current_thread_id, result)
            
            return result

        except Exception as e:
            logger.error(f"Error running agent: {e}", exc_info=True)
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error running agent:[/bold red] {str(e)}")
            
            # Debug execution error
            self._debug_execution("run", current_thread_id, error=e)
            
            raise
    
    def _stream_formatter(self, chunk: Any) -> str:
        """Format stream chunks for rich output."""
        if isinstance(chunk, dict):
            # Extract content from common patterns
            if "delta" in chunk and "content" in chunk["delta"]:
                return chunk["delta"]["content"]
            elif "choices" in chunk and chunk["choices"] and "text" in chunk["choices"][0]:
                return chunk["choices"][0]["text"]
            elif any(key in chunk for key in ["content", "text", "output", "chunk"]):
                for key in ["content", "text", "output", "chunk"]:
                    if key in chunk and isinstance(chunk[key], str):
                        return chunk[key]
        return ""
    
    def stream(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel],
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        config: Optional[RunnableConfig] = None,
        debug: bool = True,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the agent execution.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            config: Optional runtime configuration
            debug: Whether to enable debug mode
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            logger.debug("Agent not compiled yet, compiling now")
            self.compile()

        # Prepare runtime config with thread_id
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Show stream info in rich UI
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            self.console.print(Panel.fit(
                f"[bold]Streaming Agent[/bold]\n"
                f"[cyan]Thread ID:[/cyan] {current_thread_id}\n"
                f"[cyan]Stream mode:[/cyan] {stream_mode}",
                border_style="blue",
                title=self.config.name,
                subtitle=datetime.now().strftime("%H:%M:%S")
            ))
        
        logger.info(f"Streaming agent {self.config.name} with thread_id: {current_thread_id}")
        
        # Register thread in database if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)

        # Get previous state
        previous_state = None
        try:
            previous_state = self.app.get_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {current_thread_id}")
        except Exception as e:
            logger.debug(f"No previous state found: {e}")

        # Process input
        if isinstance(input_data, str):
            processed_input = self._prepare_input(input_data)
        else:
            processed_input = input_data
        
        # Stream the execution
        try:
            # Rich streaming
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                # Show headers
                self.console.print("[bold]Agent Streaming:[/bold]")
                
                # Start streaming with stats tracking
                token_count = 0
                chunk_count = 0
                start_time = time.time()
                live_output = ""
                
                # Use live display for streamed content
                with Live("", console=self.console, refresh_per_second=10) as live:
                    # Process each chunk
                    for chunk in self.app.stream(
                        processed_input,
                        stream_mode=stream_mode,
                        config=runtime_config,
                        debug=debug
                    ):
                        # Count chunks
                        chunk_count += 1
                        
                        # Extract content for displaying
                        content = self._stream_formatter(chunk)
                        if content:
                            live_output += content
                            token_count += len(content.split())
                        
                        # Update live display
                        if stream_mode == "values":
                            # For values mode, just show the latest state values
                            live.update(f"[italic]{live_output}[/italic]")
                        else:
                            # For other modes, show raw chunks
                            live.update(Markdown(live_output))
                        
                        # Yield the chunk
                        yield chunk
                
                # Show streaming stats
                execution_time = time.time() - start_time
                self.console.print(f"[bold green]Streaming complete[/bold green]")
                self.console.print(f"[dim]Chunks: {chunk_count} | Tokens: {token_count} | Time: {execution_time:.2f}s | Rate: {token_count/execution_time:.1f} tokens/sec[/dim]")
            else:
                # Standard streaming
                for chunk in self.app.stream(
                    processed_input,
                    stream_mode=stream_mode,
                    config=runtime_config,
                    debug=debug
                ):
                    yield chunk
            
            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)
                
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error during streaming:[/bold red] {str(e)}")
            
            # Debug execution error
            self._debug_execution("stream", current_thread_id, error=e)
            
            raise
    
    async def arun(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel],
        thread_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent asynchronously with the given input.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            logger.debug("Agent not compiled yet, compiling now")
            self.compile()

        # Prepare runtime config
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Show run info in rich UI (when in async context)
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            self.console.print(Panel.fit(
                f"[bold]Async Running Agent[/bold]\n"
                f"[cyan]Thread ID:[/cyan] {current_thread_id}",
                border_style="blue",
                title=self.config.name,
                subtitle=datetime.now().strftime("%H:%M:%S")
            ))
        
        logger.info(f"Running agent {self.config.name} asynchronously with thread_id: {current_thread_id}")
        
        # Register thread in database if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)

        # Get previous state
        previous_state = None
        try:
            previous_state = await self.app.aget_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {current_thread_id}")
        except Exception as e:
            logger.debug(f"No previous state found: {e}")

        # Process input with schema validation
        if isinstance(input_data, str):
            processed_input = self._prepare_input(input_data)
        else:
            processed_input = input_data

        # Run the agent asynchronously
        try:
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                # Show status during execution
                with self.console.status("[bold green]Agent processing asynchronously...[/bold green]", spinner="dots") as status:
                    start_time = time.time()
                    result = await self.app.ainvoke(
                        processed_input,
                        config=runtime_config
                    )
                    execution_time = time.time() - start_time
                
                self.console.print(f"[bold green]Async execution complete in {execution_time:.2f} seconds[/bold green]")
            else:
                # Standard async execution
                result = await self.app.ainvoke(
                    processed_input,
                    config=runtime_config
                )

            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)
            
            # Debug execution results
            self._debug_execution("arun", current_thread_id, result)

            return result
        except Exception as e:
            logger.error(f"Error running agent asynchronously: {e}", exc_info=True)
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error running agent asynchronously:[/bold red] {str(e)}")
            
            # Debug execution error
            self._debug_execution("arun", current_thread_id, error=e)
            
            raise
    
    async def astream(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel],
        thread_id: Optional[str] = None,
        stream_mode: str = "values",
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the agent execution asynchronously.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            logger.debug("Agent not compiled yet, compiling now")
            self.compile()

        # Prepare runtime config with thread_id
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Show stream info in rich UI
        if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
            self.console.print(Panel.fit(
                f"[bold]Async Streaming Agent[/bold]\n"
                f"[cyan]Thread ID:[/cyan] {current_thread_id}\n"
                f"[cyan]Stream mode:[/cyan] {stream_mode}",
                border_style="blue",
                title=self.config.name,
                subtitle=datetime.now().strftime("%H:%M:%S")
            ))
        
        logger.info(f"Streaming agent {self.config.name} asynchronously with thread_id: {current_thread_id}")
        
        # Register thread in database if needed
        register_thread_if_needed(self.checkpointer, current_thread_id)

        # Get previous state
        previous_state = None
        try:
            previous_state = await self.app.aget_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {current_thread_id}")
        except Exception as e:
            logger.debug(f"No previous state found: {e}")

        # Process input
        if isinstance(input_data, str):
            processed_input = self._prepare_input(input_data)
        else:
            processed_input = input_data

        # Stream the execution asynchronously
        try:
            # Rich async streaming
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                # Show headers
                self.console.print("[bold]Agent Streaming Asynchronously:[/bold]")
                
                # Start streaming with stats tracking
                token_count = 0
                chunk_count = 0
                start_time = time.time()
                live_output = ""
                
                # Use live display for streamed content
                with Live("", console=self.console, refresh_per_second=10) as live:
                    # Process each chunk
                    async for chunk in self.app.astream(
                        processed_input,
                        stream_mode=stream_mode,
                        config=runtime_config
                    ):
                        # Count chunks
                        chunk_count += 1
                        
                        # Extract content for displaying
                        content = self._stream_formatter(chunk)
                        if content:
                            live_output += content
                            token_count += len(content.split())
                        
                        # Update live display
                        if stream_mode == "values":
                            # For values mode, just show the latest state values
                            live.update(f"[italic]{live_output}[/italic]")
                        else:
                            # For other modes, show raw chunks
                            live.update(Markdown(live_output))
                        
                        # Yield the chunk
                        yield chunk
                
                # Show streaming stats
                execution_time = time.time() - start_time
                self.console.print(f"[bold green]Async streaming complete[/bold green]")
                self.console.print(f"[dim]Chunks: {chunk_count} | Tokens: {token_count} | Time: {execution_time:.2f}s | Rate: {token_count/execution_time:.1f} tokens/sec[/dim]")
            else:
                # Standard async streaming
                async for chunk in self.app.astream(
                    processed_input,
                    stream_mode=stream_mode,
                    config=runtime_config
                ):
                    yield chunk
            
            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)
                
        except Exception as e:
            logger.error(f"Error during async streaming: {e}", exc_info=True)
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error during async streaming:[/bold red] {str(e)}")
            
            # Debug execution error
            self._debug_execution("astream", current_thread_id, error=e)
            
            raise
    
    def debug_schema_validation(self, data: Any, schema_type: str = "input") -> None:
        """
        Debug schema validation issues.
        
        Args:
            data: Data to validate
            schema_type: Type of schema to use ('input', 'state', or 'output')
        """
        if not self.rich_logging or not RICH_AVAILABLE or not hasattr(self, "debug_console"):
            return
            
        # Select the appropriate schema
        if schema_type == "input":
            schema = self.input_schema
            schema_name = "Input Schema"
        elif schema_type == "output":
            schema = self.output_schema
            schema_name = "Output Schema"
        else:
            schema = self.state_schema
            schema_name = "State Schema"
        
        self.debug_console.rule(f"[bold]{schema_name} Validation Debug[/bold]")
        
        # Show input data
        self.debug_console.print("[bold cyan]Validating Data:[/bold cyan]")
        if isinstance(data, dict):
            data_json = json.dumps(data, indent=2, default=str)
            self.debug_console.print(Syntax(data_json[:1000], "json", theme="monokai"))
            if len(data_json) > 1000:
                self.debug_console.print(f"[dim]... {len(data_json) - 1000} more characters[/dim]")
        elif isinstance(data, BaseModel):
            if hasattr(data, "model_dump"):
                model_dict = data.model_dump()
            else:
                model_dict = data.dict()
            data_json = json.dumps(model_dict, indent=2, default=str)
            self.debug_console.print(Syntax(data_json[:1000], "json", theme="monokai"))
        else:
            self.debug_console.print(f"[yellow]Data type:[/yellow] {type(data).__name__}")
            self.debug_console.print(str(data)[:500])
        
        # Show schema
        self.debug_console.print(f"\n[bold cyan]{schema_name}:[/bold cyan]")
        field_table = Table("Field", "Type", "Required", "Default")
        
        if hasattr(schema, "model_fields"):
            # Pydantic v2
            for name, field in schema.model_fields.items():
                field_table.add_row(
                    name,
                    str(field.annotation).replace("typing.", ""),
                    str(field.is_required()),
                    str(field.default)[:50]
                )
        elif hasattr(schema, "__fields__"):
            # Pydantic v1
            for name, field in schema.__fields__.items():
                field_table.add_row(
                    name,
                    str(field.type_).replace("typing.", ""),
                    str(field.required),
                    str(field.default)[:50]
                )
        
        self.debug_console.print(field_table)
        
        # Try validation
        self.debug_console.print("\n[bold cyan]Validation Result:[/bold cyan]")
        try:
            if isinstance(data, dict):
                validated = schema(**data)
                self.debug_console.print("[bold green]✓ Validation successful[/bold green]")
            elif isinstance(data, BaseModel):
                # Convert to dict first
                if hasattr(data, "model_dump"):
                    dict_data = data.model_dump()
                else:
                    dict_data = data.dict()
                validated = schema(**dict_data)
                self.debug_console.print("[bold green]✓ Validation successful[/bold green]")
            else:
                self.debug_console.print("[bold yellow]⚠ Cannot directly validate non-dict/non-model data[/bold yellow]")
                return
            
            # Show validation result structure
            if hasattr(validated, "model_dump"):
                result_dict = validated.model_dump()
            elif hasattr(validated, "dict"):
                result_dict = validated.dict()
            else:
                result_dict = validated
                
            result_json = json.dumps(result_dict, indent=2, default=str)
            self.debug_console.print(Syntax(result_json[:500], "json", theme="monokai"))
            if len(result_json) > 500:
                self.debug_console.print(f"[dim]... {len(result_json) - 500} more characters[/dim]")
                
        except Exception as e:
            self.debug_console.print(f"[bold red]✗ Validation failed:[/bold red] {str(e)}")
            
            # Try to identify missing fields
            if "required" in str(e).lower() and "missing" in str(e).lower():
                import re
                missing_fields = re.findall(r"'([^']*)'", str(e))
                if missing_fields:
                    self.debug_console.print(f"[yellow]Missing required fields:[/yellow] {', '.join(missing_fields)}")
                    
                    # Check data keys
                    if isinstance(data, dict):
                        self.debug_console.print(f"[dim]Available keys:[/dim] {', '.join(data.keys())}")
            
            # Try to identify type errors
            if "type_error" in str(e).lower():
                self.debug_console.print("[yellow]Type error detected - check field types[/yellow]")
        
        self.debug_console.rule()
    
    def inspect_state(self, thread_id: Optional[str] = None, config: Optional[RunnableConfig] = None) -> None:
        """
        Inspect the current state of the agent.
        
        Args:
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
        """
        if not self.app:
            logger.error("Cannot inspect state: Workflow graph not compiled")
            return

        # Prepare runtime config
        runtime_config = self._prepare_runnable_config(thread_id, config)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        # Get current state
        try:
            state = self.app.get_state(runtime_config)
            
            if not state:
                logger.warning(f"No state found for thread {current_thread_id}")
                
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                    self.console.print(f"[bold yellow]No state found for thread:[/bold yellow] {current_thread_id}")
                
                return
                
            # Extract values from state
            state_values = None
            if hasattr(state, 'values'):
                state_values = state.values
            elif hasattr(state, 'channel_values'):
                state_values = state.channel_values
            elif isinstance(state, dict):
                state_values = state
                
            if not state_values:
                logger.warning(f"Could not extract values from state for thread {current_thread_id}")
                
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                    self.console.print(f"[bold yellow]Could not extract values from state for thread:[/bold yellow] {current_thread_id}")
                
                return
            
            # Display state in rich UI
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.rule("[bold]State Inspection[/bold]")
                
                # Show thread and state metadata
                self.console.print(f"[bold cyan]Thread ID:[/bold cyan] {current_thread_id}")
                
                if hasattr(state, 'metadata') and state.metadata:
                    self.console.print("[bold cyan]Metadata:[/bold cyan]")
                    metadata_json = json.dumps(state.metadata, indent=2, default=str)
                    self.console.print(Syntax(metadata_json, "json", theme="monokai"))
                
                # Create table for state values
                state_table = Table(title=f"State Values for {self.config.name}")
                state_table.add_column("Key", style="cyan")
                state_table.add_column("Type", style="green")
                state_table.add_column("Value", style="yellow")
                
                for key, value in state_values.items():
                    # Format value preview
                    if isinstance(value, list):
                        if value and isinstance(value[0], BaseMessage):
                            # Message list - show count and last message
                            value_preview = f"{len(value)} messages"
                            if value:
                                last_msg = value[-1]
                                value_preview += f"\nLast ({type(last_msg).__name__}): {last_msg.content[:100]}{'...' if len(last_msg.content) > 100 else ''}"
                        else:
                            # Regular list - show length and sample
                            value_preview = f"{len(value)} items"
                            if value and len(value) > 0:
                                value_preview += f"\nSample: {str(value[0])[:100]}{'...' if len(str(value[0])) > 100 else ''}"
                    elif isinstance(value, dict):
                        # Dictionary - show keys
                        value_preview = f"{len(value)} keys: {', '.join(list(value.keys())[:5])}"
                        if len(value.keys()) > 5:
                            value_preview += f"... and {len(value.keys()) - 5} more"
                    elif isinstance(value, str):
                        # String - truncate if long
                        value_preview = value[:200] + ("..." if len(value) > 200 else "")
                    else:
                        # Other types - convert to string
                        value_preview = str(value)[:200] + ("..." if len(str(value)) > 200 else "")
                    
                    state_table.add_row(
                        key, 
                        type(value).__name__, 
                        value_preview
                    )
                
                self.console.print(state_table)
                self.console.rule()
            else:
                # Standard logging
                logger.info(f"State inspection for thread {current_thread_id}:")
                for key, value in state_values.items():
                    if isinstance(value, list):
                        logger.info(f"  {key}: List with {len(value)} items")
                    elif isinstance(value, dict):
                        logger.info(f"  {key}: Dict with {len(value)} keys")
                    elif isinstance(value, str):
                        logger.info(f"  {key}: String of length {len(value)}")
                    else:
                        logger.info(f"  {key}: {type(value).__name__}")
        
        except Exception as e:
            logger.error(f"Error inspecting state: {e}")
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error inspecting state:[/bold red] {str(e)}")
    
    def reset_state(self, thread_id: Optional[str] = None, config: Optional[RunnableConfig] = None) -> bool:
        """
        Reset the agent's state for a thread.
        
        Args:
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            
        Returns:
            True if successful, False otherwise
        """
        if not self.app:
            logger.error("Cannot reset state: Workflow graph not compiled")
            return False
            
        # Prepare runtime config
        runtime_config = self._prepare_runnable_config(thread_id, config)
        current_thread_id = runtime_config["configurable"]["thread_id"]
        
        try:
            # First, check if there's any state to delete
            has_state = False
            try:
                state = self.app.get_state(runtime_config)
                has_state = state is not None
            except Exception:
                # No state exists
                pass
                
            if not has_state:
                logger.info(f"No state exists for thread {current_thread_id} - nothing to reset")
                
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                    self.console.print(f"[bold yellow]No state exists for thread:[/bold yellow] {current_thread_id}")
                
                return True
                
            # Delete the state if it exists
            if hasattr(self.checkpointer, "delete_state"):
                # Use delete_state if available
                self.checkpointer.delete_state(config=runtime_config["configurable"])
                
                logger.info(f"State reset for thread {current_thread_id}")
                
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                    self.console.print(f"[bold green]State reset for thread:[/bold green] {current_thread_id}")
                
                return True
            else:
                # Fall back to clearing the database directly for Postgres persistence
                if "Postgres" in type(self.checkpointer).__name__:
                    try:
                        # Connect to database and delete state
                        if hasattr(self.checkpointer, 'conn'):
                            pool = self.checkpointer.conn
                            if pool:
                                # Ensure connection pool is usable
                                pool_opened = ensure_pool_open(self.checkpointer)
                                
                                # Delete the state
                                with pool.connection() as conn:
                                    with conn.cursor() as cursor:
                                        # Delete from snapshots table
                                        cursor.execute(
                                            "DELETE FROM snapshots WHERE config->>'thread_id' = %s",
                                            (current_thread_id,)
                                        )
                                        
                                        # Also delete from threads table if it exists
                                        try:
                                            cursor.execute(
                                                "DELETE FROM threads WHERE thread_id = %s",
                                                (current_thread_id,)
                                            )
                                        except Exception:
                                            # Threads table might not exist
                                            pass
                                
                                logger.info(f"State reset for thread {current_thread_id} (direct database access)")
                                
                                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                                    self.console.print(f"[bold green]State reset for thread:[/bold green] {current_thread_id}")
                                
                                return True
                    except Exception as e:
                        logger.error(f"Error resetting state via direct database access: {e}")
                
                logger.warning(f"Cannot reset state for thread {current_thread_id}: No delete_state method available")
                
                if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                    self.console.print(f"[bold yellow]Cannot reset state for thread:[/bold yellow] {current_thread_id} (method not available)")
                
                return False
                
        except Exception as e:
            logger.error(f"Error resetting state: {e}")
            
            if self.rich_logging and RICH_AVAILABLE and hasattr(self, "console"):
                self.console.print(f"[bold red]Error resetting state:[/bold red] {str(e)}")
            
            return False