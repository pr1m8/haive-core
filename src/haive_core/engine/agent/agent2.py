# src/haive/core/engine/agent/agent.py

from __future__ import annotations

import importlib
import json
import logging
import os
import uuid
from abc import ABC
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from typing import (
    Any,
    Generic,
    TypeVar,
    get_args,
    get_origin,
)

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, model_validator

from haive_core.config.constants import RESOURCES_DIR
from haive_core.config.runnable import RunnableConfigManager
from haive_core.engine.agent.persistence import (
    MemoryCheckpointerConfig,
    PostgresCheckpointerConfig,
)
from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.types import CheckpointerType
from haive_core.engine.aug_llm.base import AugLLMConfig
from haive_core.engine.base import Engine, EngineType, InvokableEngine
from haive_core.graph.dynamic_graph_builder import DynamicGraph
from haive_core.graph.graph_pattern_registry import GraphPatternRegistry as GraphRegistry
from haive_core.schema.schema_composer import SchemaComposer
from haive_core.schema.schema_manager import StateSchemaManager
from haive_core.utils.pydantic_utils import ensure_json_serializable
from haive_core.utils.visualize_graph_utils import render_and_display_graph

# Set up logging
logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Type variables for generics
TConfig = TypeVar("TConfig", bound="AgentConfig")
TIn = TypeVar("TIn")
TOut = TypeVar("TOut")

# -----------------------------------------------------
# Agent Registry - Maps config classes to agent classes
# -----------------------------------------------------
AGENT_REGISTRY: dict[type[AgentConfig], type[Agent]] = {}

def register_agent(config_class: type[AgentConfig]):
    """Register an agent class with its configuration class."""
    def decorator(agent_class: type[Agent]):
        AGENT_REGISTRY[config_class] = agent_class
        return agent_class
    return decorator


class AgentConfig(InvokableEngine[TIn, TOut], ABC):
    """Base configuration for an agent architecture.
    Extends InvokableEngine to provide a consistent interface with the Engine framework.
    """
    engine_type: EngineType = Field(default=EngineType.AGENT)
    name: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")

    # Primary engine (used as default processor)
    engine: Engine | str | None = None

    # Additional named engines
    engines: dict[str, Engine | str] = Field(default_factory=dict)

    # Schema definitions
    state_schema: type[BaseModel] | dict[str, Any] | None = None
    input_schema: type[BaseModel] | dict[str, Any] | None = None
    output_schema: type[BaseModel] | dict[str, Any] | None = None

    # Visualization and debugging
    visualize: bool = Field(default=True)
    output_dir: str = Field(default=RESOURCES_DIR)
    debug: bool = Field(default=False)
    save_history: bool = Field(default=True)

    # Runtime settings using RunnableConfigManager
    runnable_config: RunnableConfig = Field(
        default_factory=lambda: RunnableConfigManager.create(
            thread_id=str(uuid.uuid4()),
            recursion_limit=200
        )
    )

    # Storage
    add_store: bool = Field(default=False)

    # Agent-specific settings (to be overridden by subclasses)
    agent_settings: dict[str, Any] = Field(default_factory=dict)

    # =============================================
    # Persistence Configuration
    # =============================================
    # The persistence configuration to use
    persistence: CheckpointerConfig | None = Field(
        default_factory=lambda: (
            PostgresCheckpointerConfig() if POSTGRES_AVAILABLE
            else MemoryCheckpointerConfig()
        ),
        description="Persistence configuration for state checkpointing"
    )

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def ensure_engine(self):
        """Ensure at least one engine is available."""
        if not self.engine and not self.engines:
            self.engine = AugLLMConfig()
        return self

    def derive_schema(self) -> type[BaseModel]:
        """Derive state schema from components and engines."""
        # Get all components including engines
        all_components = []
        if self.engine:
            all_components.append(self.engine)
        all_components.extend(self.engines.values())

        # Create schema name
        schema_name = f"{self.name.replace('-', '_').title()}State"

        # Use SchemaComposer to build schema
        return SchemaComposer.compose_schema(all_components, name=schema_name)

    def resolve_engine(self, engine_ref=None) -> Engine:
        """Resolve an engine reference to an actual engine.
        
        Args:
            engine_ref: Engine reference (name or object) or None to use default engine
            
        Returns:
            Resolved Engine object
        """
        # Use the provided reference, default engine, or first from engines dict
        ref = engine_ref or self.engine or next(iter(self.engines.values()), None)

        if ref is None:
            raise ValueError("No engine specified and no default engine available")

        # If it's already an Engine object, return it
        if isinstance(ref, Engine):
            return ref

        # If it's a string, look it up in the registry
        if isinstance(ref, str):
            # Try each engine type
            from haive_core.engine.base import EngineRegistry
            registry = EngineRegistry.get_instance()
            for engine_type in EngineType:
                engine = registry.get(engine_type, ref)
                if engine:
                    return engine

            raise ValueError(f"Engine '{ref}' not found in registry")

        raise TypeError(f"Unsupported engine reference type: {type(ref)}")

    def build_agent(self) -> Agent:
        """Build an agent instance from this configuration."""
        # Try to find agent class in registry
        agent_class = AGENT_REGISTRY.get(self.__class__)

        # Try class attribute if not in registry
        if agent_class is None and hasattr(self.__class__, "agent_class"):
            agent_class = self.__class__.agent_class

        # Try to resolve by naming convention
        if agent_class is None:
            agent_class = self._resolve_agent_class_by_name()

        if agent_class is None:
            raise TypeError(f"No agent class found for {self.__class__.__name__}")

        # Instantiate and return the agent
        return agent_class(config=self)

    def _resolve_agent_class_by_name(self) -> type[Agent] | None:
        """Try to resolve agent class by naming convention."""
        import importlib

        agent_class_name = self.__class__.__name__.replace("Config", "")

        # Try same module
        try:
            module = importlib.import_module(self.__class__.__module__)
            return getattr(module, agent_class_name, None)
        except (ImportError, AttributeError):
            pass

        # Try sibling module
        try:
            base_module = self.__class__.__module__.rsplit(".", 1)[0]
            for suffix in ["agent", "impl", ""]:
                try:
                    agent_module = importlib.import_module(f"{base_module}.{suffix}")
                    return getattr(agent_module, agent_class_name, None)
                except (ImportError, AttributeError):
                    continue
        except Exception:
            pass

        return None

    def create_runnable(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Create a runnable instance from this agent config.
        
        Implements the InvokableEngine interface.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            Built and compiled agent application
        """
        # Build the agent
        agent = self.build_agent()

        # Apply runtime config if provided
        if runnable_config:
            # Create a merged config
            merged_config = RunnableConfigManager.merge(
                self.runnable_config,
                runnable_config
            )

            # Update agent's runnable_config
            agent.runnable_config = merged_config

        # Return the built and compiled agent app
        return agent.app

    def derive_input_schema(self) -> type[BaseModel]:
        """Derive input schema for this agent.
        
        Implements the Engine interface.
        
        Returns:
            Pydantic model for input schema
        """
        # Use provided schema if available
        if self.input_schema:
            if isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
                return self.input_schema
            if isinstance(self.input_schema, dict):
                schema_manager = StateSchemaManager(self.input_schema, name=f"{self.name}Input")
                return schema_manager.get_model()

        # Try to derive from class type hints
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is InvokableEngine:
                args = get_args(base_cls)
                if len(args) >= 1:
                    # Extract TIn from InvokableEngine[TIn, TOut]
                    in_type = args[0]
                    if in_type is not TIn:  # Not the generic parameter itself
                        if isinstance(in_type, type) and issubclass(in_type, BaseModel):
                            return in_type

        # Default to deriving from state schema
        return self.derive_schema()

    def derive_output_schema(self) -> type[BaseModel]:
        """Derive output schema for this agent.
        
        Implements the Engine interface.
        
        Returns:
            Pydantic model for output schema
        """
        # Use provided schema if available
        if self.output_schema:
            if isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
                return self.output_schema
            if isinstance(self.output_schema, dict):
                schema_manager = StateSchemaManager(self.output_schema, name=f"{self.name}Output")
                return schema_manager.get_model()

        # Try to derive from class type hints
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is InvokableEngine:
                args = get_args(base_cls)
                if len(args) >= 2:
                    # Extract TOut from InvokableEngine[TIn, TOut]
                    out_type = args[1]
                    if out_type is not TOut:  # Not the generic parameter itself
                        if isinstance(out_type, type) and issubclass(out_type, BaseModel):
                            return out_type

        # Default to deriving from state schema
        return self.derive_schema()

    def invoke(self, input_data: TIn, runnable_config: RunnableConfig | None = None) -> TOut:
        """Invoke the agent with input data.
        
        Implements the InvokableEngine interface.
        
        Args:
            input_data: Input data for the agent
            runnable_config: Optional runtime configuration
            
        Returns:
            Output from the agent
        """
        # Build the agent
        agent = self.build_agent()

        # Extract thread ID from runnable_config if present
        thread_id = None
        if runnable_config and "configurable" in runnable_config and "thread_id" in runnable_config["configurable"]:
            thread_id = runnable_config["configurable"]["thread_id"]

        # Run with input data and config
        return agent.run(input_data, thread_id=thread_id, config=runnable_config)

    async def ainvoke(self, input_data: TIn, runnable_config: RunnableConfig | None = None) -> TOut:
        """Asynchronously invoke the agent with input data.
        
        Implements the InvokableEngine interface.
        
        Args:
            input_data: Input data for the agent
            runnable_config: Optional runtime configuration
            
        Returns:
            Output from the agent
        """
        # Build the agent
        agent = self.build_agent()

        # Extract thread ID from runnable_config if present
        thread_id = None
        if runnable_config and "configurable" in runnable_config and "thread_id" in runnable_config["configurable"]:
            thread_id = runnable_config["configurable"]["thread_id"]

        # Run with input data and config
        return await agent.arun(input_data, thread_id=thread_id, config=runnable_config)

    def apply_runnable_config(self, runnable_config: RunnableConfig | None = None) -> dict[str, Any]:
        """Extract parameters from runnable_config relevant to this agent.
        
        Args:
            runnable_config: Runtime configuration to extract from
            
        Returns:
            Dictionary of relevant parameters
        """
        # Start with common parameters from base class
        params = super().apply_runnable_config(runnable_config)

        if not runnable_config or "configurable" not in runnable_config:
            return params

        configurable = runnable_config["configurable"]

        # Extract agent-specific parameters
        agent_params = ["thread_id", "user_id", "save_history", "debug"]
        for param in agent_params:
            if param in configurable:
                params[param] = configurable[param]

        # Extract engine parameters if engine configs are present
        if "engine_configs" in configurable:
            for engine_name, engine_config in configurable["engine_configs"].items():
                # Handle primary engine
                if engine_name == self.name or (self.engine and hasattr(self.engine, "name") and engine_name == self.engine.name):
                    params.update(engine_config)
                # Handle named engines
                elif engine_name in self.engines:
                    if "engines" not in params:
                        params["engines"] = {}
                    params["engines"][engine_name] = engine_config

        return params

    def get_schema_fields(self) -> dict[str, tuple[type, Any]]:
        """Get schema fields for this agent.
        
        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Any

        # Basic fields every agent state should have
        fields = {
            "messages": (list[BaseMessage], []),
            "__runnable_config__": (dict[str, Any], {}),
        }

        # Add fields from input schema
        input_schema = self.derive_input_schema()
        if hasattr(input_schema, "model_fields"):
            # Pydantic v2
            for name, field_info in input_schema.model_fields.items():
                if name not in fields:
                    fields[name] = (field_info.annotation, field_info.default)
        else:
            # Pydantic v1
            for name, field_info in input_schema.__fields__.items():
                if name not in fields:
                    fields[name] = (field_info.type_, field_info.default)

        # Add fields from output schema
        output_schema = self.derive_output_schema()
        if hasattr(output_schema, "model_fields"):
            # Pydantic v2
            for name, field_info in output_schema.model_fields.items():
                if name not in fields:
                    fields[name] = (field_info.annotation, field_info.default)
        else:
            # Pydantic v1
            for name, field_info in output_schema.__fields__.items():
                if name not in fields:
                    fields[name] = (field_info.type_, field_info.default)

        return fields

    def to_dict(self) -> dict[str, Any]:
        """Convert agent config to a dictionary.
        
        Returns:
            Dictionary representation of the agent config
        """
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"input_schema", "output_schema"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"input_schema", "output_schema"})

        # Convert engines to serializable format
        if "engine" in data and isinstance(data["engine"], Engine):
            data["engine"] = data["engine"].extract_params()

        if "engines" in data:
            serialized_engines = {}
            for name, engine in data["engines"].items():
                if isinstance(engine, Engine):
                    serialized_engines[name] = engine.extract_params()
                else:
                    serialized_engines[name] = engine
            data["engines"] = serialized_engines

        # Convert persistence config
        if "persistence" in data and data["persistence"] is not None:
            if hasattr(data["persistence"], "to_dict"):
                data["persistence"] = data["persistence"].to_dict()

        return data


class Agent(Generic[TConfig], ABC):
    """Base agent architecture class.
    Defines how an agent works - its implementation and behavior.
    """
    def __init__(self, config: TConfig):
        """Initialize the agent with its configuration."""
        self.config = config

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

        # Set up checkpointer from persistence configuration
        self.checkpointer = self._setup_checkpointer()

        # Initialize runnable_config from config
        self.runnable_config = getattr(self.config, "runnable_config",
            RunnableConfigManager.create(thread_id=str(uuid.uuid4())))

        # Add store if configured
        if getattr(self.config, "add_store", False):
            self.store = BaseStore()
        else:
            self.store = None

        # Initialize engines and prepare state
        self._initialize_engines()
        self._setup_schemas()
        self._setup_output_paths()

        # Create graph and set up workflow
        self._create_initial_graph()
        self.setup_workflow()

        # Compile the graph
        self.app = None
        self.compile()

        # Generate visualization if requested
        if getattr(self.config, "visualize", True) and self.app:
            self.visualize_graph()

    def _initialize_engines(self):
        """Initialize all engines with improved runnable_config integration."""
        self.engines = {}

        # Initialize main engine if present
        if self.config.engine:
            self.engine = self._build_engine(self.config.engine, "main")
            self.engines["main"] = self.engine

        # Initialize additional engines
        for name, engine_config in getattr(self.config, "engines", {}).items():
            self.engines[name] = self._build_engine(engine_config, name)

        # Connect engines to agent's runnable_config if they don't have their own
        for name, engine in self.engines.items():
            if isinstance(engine, InvokableEngine) and not hasattr(engine, "_default_runnable_config"):
                # This adds a reference but doesn't modify the original engine
                engine._default_runnable_config = self.runnable_config



    def _build_engine(self, config, name=None):
        """Build an engine from configuration with improved registry integration."""
        # Try to resolve from string reference
        if isinstance(config, str):
            # Try to find in engine registry first
            from haive_core.engine.base import EngineRegistry, EngineType
            registry = EngineRegistry.get_instance()
            for engine_type in EngineType:
                engine = registry.get(engine_type, config)
                if engine:
                    config = engine
                    break

            # If still a string, try graph registry
            if isinstance(config, str):
                graph_registry = GraphRegistry.get_instance()
                components = graph_registry.list_components(tags=["engine"])
                for component in components:
                    if component.name == config:
                        if component.metadata.get("source_module"):
                            try:
                                module = importlib.import_module(component.metadata["source_module"])
                                class_obj = getattr(module, component.name)
                                if issubclass(class_obj, Engine):
                                    config = class_obj(name=name or component.name)
                                    break
                            except (ImportError, AttributeError) as e:
                                logger.warning(f"Could not load engine {config}: {e}")

            # If still a string, we couldn't resolve it
            if isinstance(config, str):
                raise ValueError(f"Engine '{config}' not found in registry")

        # Build based on engine type with runnable_config integration
        if isinstance(config, AugLLMConfig):
            # Pass agent's runnable_config if the engine doesn't have one
            runnable_config = getattr(config, "default_runnable_config", None) or self.runnable_config
            return config.create_runnable(runnable_config)
        if hasattr(config, "create_runnable"):
            runnable_config = getattr(config, "default_runnable_config", None) or self.runnable_config
            return config.create_runnable(runnable_config)

        # Return as-is if already built
        return config

    def _setup_schemas(self):
        """Set up state, input, and output schemas."""
        # Resolve or derive state schema
        if getattr(self.config, "state_schema", None) is None:
            self.state_schema = self.config.derive_schema()
        elif isinstance(self.config.state_schema, dict):
            schema_manager = StateSchemaManager(self.config.state_schema, name=f"{self.config.name}State")
            self.state_schema = schema_manager.get_model()
        else:
            self.state_schema = self.config.state_schema

        # Process input schema - default to state schema if not provided
        if getattr(self.config, "input_schema", None) is None:
            self.input_schema = self.state_schema
        elif isinstance(self.config.input_schema, dict):
            schema_manager = StateSchemaManager(self.config.input_schema, name=f"{self.config.name}Input")
            self.input_schema = schema_manager.get_model()
        else:
            self.input_schema = self.config.input_schema

        # Process output schema - default to state schema if not provided
        if getattr(self.config, "output_schema", None) is None:
            self.output_schema = self.state_schema
        elif isinstance(self.config.output_schema, dict):
            schema_manager = StateSchemaManager(self.config.output_schema, name=f"{self.config.name}Output")
            self.output_schema = schema_manager.get_model()
        else:
            self.output_schema = self.config.output_schema

    def _setup_output_paths(self):
        """Set up paths for output files (state history, graph images)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up state history directory and file
        self.state_history_dir = os.path.join(self.config.output_dir, "State_History")
        os.makedirs(self.state_history_dir, exist_ok=True)
        self.state_filename = os.path.join(
            self.state_history_dir,
            f"{self.config.name}_{timestamp}.json"
        )

        # Set up graphs directory and file
        self.graphs_dir = os.path.join(self.config.output_dir, "Graphs")
        os.makedirs(self.graphs_dir, exist_ok=True)
        self.graph_image_path = os.path.join(
            self.graphs_dir,
            f"{self.config.name}_{timestamp}.png"
        )

    def _create_initial_graph(self):
        """Create the initial graph based on the schemas."""
        if hasattr(self, "input_schema") and hasattr(self, "output_schema"):
            self.graph = StateGraph(
                self.state_schema,
                input=self.input_schema,
                output=self.output_schema
            )
        else:
            self.graph = StateGraph(self.state_schema)

    def _setup_checkpointer(self):
        """Set up the appropriate checkpointer based on persistence configuration.
        
        Returns:
            A configured checkpointer instance
        """
        # Import MemorySaver

        # Handle explicit request for memory persistence
        if not hasattr(self.config, "persistence") or self.config.persistence is None:
            logger.info(f"No persistence config for {self.config.name}. Using memory checkpointer.")
            return MemorySaver()

        # Handle dictionary config
        if isinstance(self.config.persistence, dict):
            # Check for explicit memory type
            if self.config.persistence.get("type", "memory") == "memory":
                logger.info(f"Using memory checkpointer (per config) for {self.config.name}")
                return MemorySaver()

            # Only continue with postgres if it's explicitly requested and available
            if self.config.persistence.get("type") == "postgres" and POSTGRES_AVAILABLE:
                try:
                    # Import only when needed
                    from langgraph.checkpoint.postgres import PostgresSaver
                    from psycopg_pool import ConnectionPool

                    # Get connection parameters
                    db_host = self.config.persistence.get("db_host", "localhost")
                    db_port = self.config.persistence.get("db_port", 5432)
                    db_name = self.config.persistence.get("db_name", "postgres")
                    db_user = self.config.persistence.get("db_user", "postgres")
                    db_pass = self.config.persistence.get("db_pass", "postgres")
                    ssl_mode = self.config.persistence.get("ssl_mode", "disable")

                    # Create connection URI
                    import urllib.parse
                    encoded_pass = urllib.parse.quote_plus(str(db_pass))
                    db_uri = f"postgresql://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"
                    if ssl_mode:
                        db_uri += f"?sslmode={ssl_mode}"

                    # Create connection pool
                    pool = ConnectionPool(
                        conninfo=db_uri,
                        min_size=self.config.persistence.get("min_pool_size", 1),
                        max_size=self.config.persistence.get("max_pool_size", 5),
                        kwargs={
                            "autocommit": self.config.persistence.get("auto_commit", True),
                            "prepare_threshold": self.config.persistence.get("prepare_threshold", 0),
                        },
                        open=True  # Explicitly open the pool to avoid "not open yet" errors
                    )

                    # Create checkpointer
                    checkpointer = PostgresSaver(pool)

                    # Set flag for setup if needed
                    if self.config.persistence.get("setup_needed", False):
                        self.postgres_setup_needed = True

                    logger.info(f"Using PostgreSQL checkpointer for {self.config.name}")
                    return checkpointer

                except Exception as e:
                    logger.error(f"Failed to setup PostgreSQL checkpointer: {e}")
                    logger.warning(f"Falling back to memory checkpointer for {self.config.name}")

        # Handle CheckpointerConfig objects
        elif hasattr(self.config.persistence, "type"):
            if self.config.persistence.type == CheckpointerType.memory:
                logger.info(f"Using memory checkpointer for {self.config.name}")
                return MemorySaver()

            if self.config.persistence.type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
                try:
                    # Try to use create_checkpointer method if available
                    if hasattr(self.config.persistence, "create_checkpointer"):
                        checkpointer = self.config.persistence.create_checkpointer()

                        # Set flag for table setup if needed
                        if hasattr(self.config.persistence, "setup_needed") and self.config.persistence.setup_needed:
                            self.postgres_setup_needed = True

                        logger.info(f"Using PostgreSQL checkpointer for {self.config.name}")
                        return checkpointer
                except Exception as e:
                    logger.error(f"Failed to create PostgreSQL checkpointer: {e}")

        # Default to memory checkpointer for any other case
        logger.info(f"Using memory checkpointer (default) for {self.config.name}")
        return MemorySaver()

    def setup_workflow(self) -> None:
        """Set up the workflow graph for this agent using DynamicGraph.
        
        This is an improved implementation that subclasses should override.
        """
        # Create a DynamicGraph with the agent's engines and schema
        self.graph_builder = DynamicGraph(
            name=self.config.name,
            components=list(self.config.engines.values()),
            state_schema=self.state_schema,
            default_runnable_config=self.runnable_config
        )

        # Add default routing if not implemented by subclass
        if isinstance(self.engine, Engine):
            self.graph_builder.add_node(
                name="process",
                config=self.engine,
                command_goto=END
            )
            self.graph_builder.set_entry_point("process")

        # Build the graph
        self.graph = self.graph_builder.build_graph()

        logger.warning(f"Using default workflow for {self.config.name}. Subclasses should override setup_workflow().")

    def compile(self) -> None:
        """Compile the workflow graph into an executable app."""
        if not self.graph:
            raise RuntimeError("Graph is not set up.")

        # Setup PostgreSQL tables if needed
        persistence_config = getattr(self.config, "persistence", None)
        postgres_setup_needed = getattr(self, "postgres_setup_needed", False)

        # Check if using PostgreSQL with proper attribute check
        is_postgres = False
        if persistence_config:
            if hasattr(persistence_config, "is_postgres"):
                is_postgres = persistence_config.is_postgres
            elif hasattr(persistence_config, "type"):
                from haive_core.engine.agent.persistence import CheckpointerType
                is_postgres = persistence_config.type == CheckpointerType.postgres

        # Ensure pool is open if using PostgreSQL
        opened_pool = None
        if persistence_config and is_postgres and postgres_setup_needed:
            opened_pool = self._ensure_pool_open()

            try:
                # Set up tables
                if hasattr(persistence_config, "use_async") and persistence_config.use_async:
                    # For async setup, we need to run this differently
                    import asyncio
                    try:
                        asyncio.run(self.checkpointer.setup())
                    except RuntimeError:
                        # If already in an event loop, use create_task
                        loop = asyncio.get_event_loop()
                        loop.create_task(self.checkpointer.setup())
                else:
                    self.checkpointer.setup()

                # Clear the flag so we don't try to set up again
                self.postgres_setup_needed = False
                logger.info(f"PostgreSQL tables set up for {self.config.name}")
            except Exception as e:
                logger.error(f"Error setting up PostgreSQL tables: {e}")

        # Compile the app
        if self.app is None:
            self.app = self.graph.compile(checkpointer=self.checkpointer)
            logger.info(f"Workflow compiled successfully for {self.config.name}")
        else:
            logger.info(f"Workflow already compiled for {self.config.name}")

    def visualize_graph(self) -> None:
        """Generate and save a visualization of the graph."""
        if self.graph and self.app:
            render_and_display_graph(self.app, output_name=self.graph_image_path)
            logger.info(f"Graph visualization saved to {self.graph_image_path}")
        else:
            logger.warning(f"Cannot visualize graph for {self.config.name}: Graph is not set up or compiled")

    def run(
        self,
        input_data: str | list[str] | dict[str, Any] | BaseModel={},
        thread_id: str | None = None,
        config: RunnableConfig | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Run the agent with the given input and return the final state.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.compile()

        # Prepare runtime configuration
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Process input data
        processed_input = self._prepare_input(input_data)

        # Register thread in PostgreSQL if needed
        thread_id = RunnableConfigManager.get_thread_id(runtime_config)
        if thread_id and hasattr(self.checkpointer, "conn"):
            self._register_thread_if_needed(thread_id)

        # Try to get existing state for this thread
        previous_state = None
        try:
            logger.debug(f"Checking for previous state with thread_id: {thread_id}")
            previous_state = self.app.get_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {thread_id}")
        except Exception as e:
            logger.debug(f"No previous state found or error retrieving: {e}")

        # Run the agent
        try:
            logger.info(f"Running agent {self.config.name} with thread_id: {thread_id}")
            result = self.app.invoke(
                processed_input,
                config=runtime_config,
                debug=True
            )

            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)

            return result

        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e

    def stream(
        self,
        input_data: str | list[str] | dict[str, Any] | BaseModel,
        thread_id: str | None = None,
        stream_mode: str = "values",
        config: RunnableConfig | None = None,
        **kwargs
    ) -> Generator[dict[str, Any], None, None]:
        """Stream the agent execution.
        
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
            self.compile()

        # Process input based on type
        processed_input = self._prepare_input(input_data)

        # Prepare runnable config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Stream the execution
        logger.info(f"Streaming agent {self.config.name}")
        for output in self.app.stream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config,
            debug=getattr(self.config, "debug", False)
        ):
            yield output

        # Save state history if requested
        if getattr(self.config, "save_history", True):
            self.save_state_history(runnable_config)

    async def arun(
        self,
        input_data: str | list[str] | dict[str, Any] | BaseModel,
        thread_id: str | None = None,
        config: RunnableConfig | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """Run the agent asynchronously with the given input.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            config: Optional runtime configuration
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.compile()

        # Process input based on type
        processed_input = self._prepare_input(input_data)

        # Prepare runnable config
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Ensure PostgreSQL pool is open if needed
        opened_pool = self._ensure_pool_open()

        # Run the agent asynchronously
        try:
            logger.info(f"Running agent {self.config.name} asynchronously with thread_id: {RunnableConfigManager.get_thread_id(runtime_config)}")
            result = await self.app.ainvoke(
                processed_input,
                config=runtime_config
            )

            # Get the final state (if available)
            try:
                final_state = await self.app.aget_state(runtime_config)
                if not final_state:
                    final_state = result
            except (AttributeError, NotImplementedError):
                # Fallback if aget_state is not available
                final_state = result

            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)

            return final_state
        except Exception as e:
            logger.error(f"Error running agent asynchronously: {e}")
            raise e

    async def astream(
        self,
        input_data: str | list[str] | dict[str, Any] | BaseModel,
        thread_id: str | None = None,
        stream_mode: str = "values",
        config: RunnableConfig | None = None,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream the agent execution asynchronously.
        
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
            self.compile()

        # Process input based on type
        processed_input = self._prepare_input(input_data)

        # Prepare runnable config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Stream the execution asynchronously
        logger.info(f"Streaming agent {self.config.name} asynchronously")
        async for output in self.app.astream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config,
            debug=getattr(self.config, "debug", False)
        ):
            yield output

        # Save state history if requested
        if getattr(self.config, "save_history", True):
            self.save_state_history(runnable_config)

    def save_state_history(self, runnable_config=None) -> None:
        """Save the current agent state to a JSON file.
        
        Args:
            runnable_config: Optional runnable configuration
        """
        if not self.app:
            logger.error("Cannot save state history: Workflow graph not compiled")
            return

        # Use provided runnable config or default
        runnable_config = runnable_config or self.runnable_config

        try:
            # Get state from app
            state_json = self.app.get_state(runnable_config)

            if not state_json:
                logger.warning(f"No state history available for {self.config.name}")
                return

            # Ensure state is JSON serializable
            state_json = ensure_json_serializable(state_json)

            # Save to file
            with open(self.state_filename, "w", encoding="utf-8") as f:
                json.dump(state_json, f, indent=4)

            logger.info(f"State history saved to: {self.state_filename}")
        except Exception as e:
            logger.error(f"Error saving state history: {e!s}")

    def _register_thread_if_needed(self, thread_id: str) -> None:
        """Register a thread in the PostgreSQL database if needed.
        
        Args:
            thread_id: Thread ID to register
        """
        if hasattr(self.checkpointer, "conn"):
            try:
                pool = self.checkpointer.conn
                if pool:
                    # Ensure connection pool is usable
                    pool_opened = self._ensure_pool_open()

                    # Register the thread
                    with pool.connection() as conn:
                        with conn.cursor() as cursor:
                            # Check columns
                            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name='threads'")
                            columns = [row[0] for row in cursor.fetchall()]
                            logger.debug(f"Threads table columns: {columns}")

                            # Insert the thread if not exists
                            cursor.execute(
                                "INSERT INTO threads (thread_id) VALUES (%s) ON CONFLICT DO NOTHING",
                                (thread_id,)
                            )
                            logger.info(f"Thread {thread_id} registered in PostgreSQL")
            except Exception as e:
                logger.warning(f"Error registering thread: {e}")

    def _prepare_runnable_config(self, thread_id=None, config=None, **kwargs) -> RunnableConfig:
        """Prepare runnable config with thread ID and other parameters.
        
        Args:
            thread_id: Optional thread ID for persistence
            config: Optional base configuration
            **kwargs: Additional parameters
            
        Returns:
            Prepared RunnableConfig
        """
        # Start with base config from the agent
        base_config = getattr(self, "runnable_config", {})

        # If explicit config provided, start with that instead
        if config:
            # Use RunnableConfigManager to merge configs
            runtime_config = RunnableConfigManager.merge(base_config, config)
        else:
            # Start with a copy of the base config
            runtime_config = json.loads(json.dumps(base_config))

        # Ensure configurable section exists
        if "configurable" not in runtime_config:
            runtime_config["configurable"] = {}

        # Use provided thread_id or keep existing one
        if thread_id:
            runtime_config["configurable"]["thread_id"] = thread_id
        elif "thread_id" not in runtime_config["configurable"]:
            # Generate a new thread ID if none exists
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())

        # Add other kwargs
        for key, value in kwargs.items():
            # If it's a configurable param, add to configurable section
            if key.startswith("configurable_"):
                param_name = key.replace("configurable_", "")
                runtime_config["configurable"][param_name] = value
            elif key == "configurable" and isinstance(value, dict):
                # Merge configurable section
                for k, v in value.items():
                    runtime_config["configurable"][k] = v
            else:
                # Otherwise add to top level
                runtime_config[key] = value

        # Register thread in PostgreSQL if needed
        thread_id = runtime_config["configurable"]["thread_id"]
        if hasattr(self, "checkpointer") and hasattr(self.checkpointer, "conn"):
            self._register_thread_if_needed(thread_id)

        return runtime_config

    def _prepare_input(self, input_data: str | list[str] | dict[str, Any] | BaseModel) -> Any:
        """Prepare input for the agent based on the input type.
        
        Args:
            input_data: Input in various formats
            
        Returns:
            Processed input compatible with the graph
        """
        # When input is a string
        if isinstance(input_data, str):
            # Initialize input dict
            prepared_input = {"messages": [HumanMessage(content=input_data)]}

            # Detect additional required inputs
            required_inputs = set()
            if hasattr(self, "engine") and hasattr(self.engine, "prompt_template") and hasattr(self.engine.prompt_template, "input_variables"):
                required_inputs.update(self.engine.prompt_template.input_variables)

            # Add string to all required inputs
            for input_name in required_inputs:
                if input_name != "messages":
                    prepared_input[input_name] = input_data

            return self.state_schema(**prepared_input) if hasattr(self, "state_schema") else prepared_input

        # Handle other types of input
        if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # List of strings - convert to messages
            messages = [HumanMessage(content=item) for item in input_data]

            prepared_input = {"messages": messages}

            # For other inputs, join the strings
            required_inputs = set()
            if hasattr(self, "engine") and hasattr(self.engine, "prompt_template") and hasattr(self.engine.prompt_template, "input_variables"):
                required_inputs.update(self.engine.prompt_template.input_variables)

            for input_name in required_inputs:
                if input_name != "messages":
                    prepared_input[input_name] = "\n".join(input_data)

            return self.state_schema(**prepared_input) if hasattr(self, "state_schema") else prepared_input

        if isinstance(input_data, dict):
            # Dict input - use as is
            return self.state_schema(**input_data) if hasattr(self, "state_schema") else input_data

        if isinstance(input_data, BaseModel):
            # Convert BaseModel to dict
            if hasattr(input_data, "model_dump"):
                # Pydantic v2
                model_dict = input_data.model_dump()
                return self.state_schema(**model_dict) if hasattr(self, "state_schema") else model_dict
            if hasattr(input_data, "dict"):
                # Pydantic v1
                model_dict = input_data.dict()
                return self.state_schema(**model_dict) if hasattr(self, "state_schema") else model_dict
            # Manual extraction
            prepared_input = {}
            for field in input_data.__annotations__:
                if hasattr(input_data, field):
                    prepared_input[field] = getattr(input_data, field)
            return self.state_schema(**prepared_input) if hasattr(self, "state_schema") else prepared_input

        # Fallback
        return self.state_schema(messages=[HumanMessage(content=str(input_data))]) if hasattr(self, "state_schema") else {
            "messages": [HumanMessage(content=str(input_data))]
        }

    def _ensure_pool_open(self):
        """Ensure that any PostgreSQL connection pool is properly opened.
        This should be called before any operation that uses the checkpointer.
        
        Returns:
            The opened pool if one was found and opened, None otherwise
        """
        opened_pool = None
        try:
            # Check for connection pools in the checkpointer
            if hasattr(self.checkpointer, "conn"):
                conn = self.checkpointer.conn

                # Import here to avoid dependency issues
                try:
                    from psycopg_pool.base import BaseConnectionPool

                    # Check if it's a pool
                    if isinstance(conn, BaseConnectionPool):
                        # Check if the pool is already open
                        try:
                            if hasattr(conn, "is_open"):
                                is_open = conn.is_open()
                            else:
                                # Older versions might not have is_open()
                                is_open = getattr(conn, "_opened", False)

                            # Open the pool if needed
                            if not is_open:
                                logger.info(f"Opening PostgreSQL connection pool for {self.config.name}")
                                try:
                                    conn.open()
                                    opened_pool = conn
                                    logger.info("Successfully opened pool")
                                except Exception as e:
                                    logger.error(f"Error opening pool: {e}")

                                    # Try a different approach with direct pool access
                                    if hasattr(conn, "_pool"):
                                        logger.info("Trying alternative pool opening method")
                                        conn._pool = [] if not hasattr(conn, "_pool") or conn._pool is None else conn._pool
                                        conn._opened = True
                                        opened_pool = conn
                        except Exception as e:
                            logger.error(f"Error checking if pool is open: {e}")
                            # Last ditch effort - try direct attribute manipulation
                            if hasattr(conn, "_pool"):
                                conn._pool = [] if not hasattr(conn, "_pool") or conn._pool is None else conn._pool
                                conn._opened = True
                                opened_pool = conn
                except ImportError:
                    logger.debug("psycopg_pool not available")

            # Additional check for other types of pools or connections
            if not opened_pool and hasattr(self.checkpointer, "setup"):
                # If the checkpointer has a setup method but no connection was found,
                # just make sure tables are set up
                logger.debug("No pool found but checkpointer has setup method")
                try:
                    self.checkpointer.setup()
                except Exception as e:
                    logger.error(f"Error setting up checkpointer: {e}")

        except Exception as e:
            logger.error(f"Error ensuring pool is open: {e}")

        return opened_pool

    def _close_pool_if_needed(self, pool=None):
        """Close a PostgreSQL connection pool if it was previously opened.
        This should be called in finally blocks after operations.
        
        Args:
            pool: The pool to close. If None, will try to find the pool 
                from the checkpointer.
        """
        if pool is None:
            # Try to find a pool from the checkpointer
            try:
                if hasattr(self.checkpointer, "conn"):
                    pool = self.checkpointer.conn
            except AttributeError:
                return

        # Close the pool if it's a ConnectionPool
        try:
            from psycopg_pool.pool import ConnectionPool
            if isinstance(pool, ConnectionPool) and pool.is_open():
                logger.debug(f"Closing PostgreSQL connection pool for {self.config.name}")
                # Uncomment to actually close the pool - generally not recommended
                # unless you're sure you won't need it again
                # pool.close()
        except (ImportError, AttributeError):
            pass

        # Close the pool if it's an AsyncConnectionPool
        try:
            from psycopg_pool.pool import AsyncConnectionPool
            if isinstance(pool, AsyncConnectionPool) and pool.is_open():
                logger.debug(f"Closing async PostgreSQL connection pool for {self.config.name}")
                # Uncomment to actually close the pool
                # import asyncio
                # try:
                #     asyncio.run(pool.close())
                # except RuntimeError:
                #     loop = asyncio.get_event_loop()
                #     task = loop.create_task(pool.close())
        except (ImportError, AttributeError):
            pass
