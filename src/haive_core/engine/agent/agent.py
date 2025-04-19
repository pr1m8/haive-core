from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from datetime import datetime
from typing import (
    Any,
    Generic,
    TypeVar,
)

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel

from haive_core.config.runnable import RunnableConfigManager

# Import persistence-related functionality
from haive_core.engine.agent.persistence.types import CheckpointerType
from haive_core.engine.base import InvokableEngine
from haive_core.graph.dynamic_graph_builder import DynamicGraph
from haive_core.schema.schema_composer import SchemaComposer
from haive_core.utils.pydantic_utils import ensure_json_serializable

# Set up logging
logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
from haive_core.engine.agent.config import AgentConfig

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

        # Create the graph builder - centralized here for all agents
        self._create_graph_builder()

        # Allow subclass to set up workflow
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

            # If still a string, we couldn't resolve it
            if isinstance(config, str):
                raise ValueError(f"Engine '{config}' not found in registry")

        # Build based on engine type with runnable_config integration
        if hasattr(config, "create_runnable"):
            runnable_config = getattr(config, "default_runnable_config", None) or self.runnable_config
            return config.create_runnable(runnable_config)

        # Return as-is if already built
        return config

    def _setup_schemas(self):
        """Set up state, input, and output schemas."""
        # Resolve or derive state schema
        if hasattr(self.config, "state_schema") and self.config.state_schema is not None:
            if isinstance(self.config.state_schema, dict):
                schema_composer = SchemaComposer(name=f"{self.config.name}State")
                schema_composer.add_fields_from_dict(self.config.state_schema)
                self.state_schema = schema_composer.build()
            else:
                self.state_schema = self.config.state_schema
        else:
            # Derive from components if no schema provided
            self.state_schema = self.config.derive_schema()

        # Process input schema - default to state schema if not provided
        if hasattr(self.config, "input_schema") and self.config.input_schema is not None:
            if isinstance(self.config.input_schema, dict):
                schema_composer = SchemaComposer(name=f"{self.config.name}Input")
                schema_composer.add_fields_from_dict(self.config.input_schema)
                self.input_schema = schema_composer.build()
            else:
                self.input_schema = self.config.input_schema
        else:
            self.input_schema = self.state_schema

        # Process output schema - default to state schema if not provided
        if hasattr(self.config, "output_schema") and self.config.output_schema is not None:
            if isinstance(self.config.output_schema, dict):
                schema_composer = SchemaComposer(name=f"{self.config.name}Output")
                schema_composer.add_fields_from_dict(self.config.output_schema)
                self.output_schema = schema_composer.build()
            else:
                self.output_schema = self.config.output_schema
        else:
            self.output_schema = self.state_schema

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

    def _create_graph_builder(self):
        """Create the DynamicGraph builder with proper schema and configuration.
        This is centralized in the base class to avoid duplication.
        """
        # Get all components for schema derivation
        components = list(self.engines.values())

        # Create the DynamicGraph
        self.graph_builder = DynamicGraph(
            name=self.config.name,
            description=f"Agent {self.config.name}",
            components=components,
            state_schema=self.state_schema,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            default_runnable_config=self.runnable_config,
            include_runnable_config=False  # IMPORTANT: Never include in schema
        )

    def _setup_checkpointer(self):
        """Set up the appropriate checkpointer based on persistence configuration.
        Prefers PostgreSQL if available, falls back to memory otherwise.
        
        Returns:
            A configured checkpointer instance
        """
        # If no persistence config specified
        if not hasattr(self.config, "persistence") or self.config.persistence is None:
            # Default to PostgreSQL if available
            if POSTGRES_AVAILABLE:
                try:
                    from haive_core.engine.agent.persistence import PostgresCheckpointerConfig
                    # Use default values
                    postgres_config = PostgresCheckpointerConfig()
                    checkpointer = postgres_config.create_checkpointer()
                    logger.info(f"Using default PostgreSQL checkpointer for {self.config.name}")

                    # Set flag to set up tables later
                    self.postgres_setup_needed = True
                    return checkpointer
                except Exception as e:
                    logger.warning(f"Failed to create default PostgreSQL checkpointer: {e}")
                    logger.warning(f"Falling back to memory checkpointer for {self.config.name}")

            # Fall back to memory
            logger.info(f"Using memory checkpointer for {self.config.name} (no persistence config)")
            return MemorySaver()

        # Use the provided persistence configuration
        persistence_config = self.config.persistence

        # Handle CheckpointerConfig objects
        if hasattr(persistence_config, "type"):
            if persistence_config.type == CheckpointerType.memory:
                logger.info(f"Using memory checkpointer for {self.config.name}")
                return MemorySaver()

            if persistence_config.type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
                try:
                    # Try to use create_checkpointer method
                    if hasattr(persistence_config, "create_checkpointer"):
                        checkpointer = persistence_config.create_checkpointer()

                        # Set flag for setup if needed
                        if getattr(persistence_config, "setup_needed", False):
                            self.postgres_setup_needed = True

                        logger.info(f"Using PostgreSQL checkpointer for {self.config.name}")
                        return checkpointer
                except Exception as e:
                    logger.error(f"Failed to create PostgreSQL checkpointer: {e}")

        # Default to memory checkpointer as fallback
        logger.info(f"Using memory checkpointer (default) for {self.config.name}")
        return MemorySaver()

    @abstractmethod
    def setup_workflow(self) -> None:
        """Set up the workflow graph for this agent.
        
        This method must be implemented by concrete agent classes to define
        the agent's workflow structure.
        """

    def compile(self) -> None:
        """Compile the workflow graph into an executable app."""
        if not hasattr(self, "graph") or not self.graph:
            raise RuntimeError("Graph is not set up.")

        # Setup PostgreSQL tables if needed
        if hasattr(self, "postgres_setup_needed") and self.postgres_setup_needed:
            # Ensure pool is open
            self._ensure_pool_open()

            try:
                # Set up tables
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
        try:
            from haive_core.utils.visualize_graph_utils import render_and_display_graph
            if self.graph and self.app:
                render_and_display_graph(self.app, output_name=self.graph_image_path)
                logger.info(f"Graph visualization saved to {self.graph_image_path}")
            else:
                logger.warning(f"Cannot visualize graph for {self.config.name}: Graph not set up or compiled")
        except ImportError as e:
            logger.warning(f"Cannot visualize graph: required modules not available: {e}")

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
        current_thread_id = runtime_config["configurable"]["thread_id"]

        # Register thread in PostgreSQL if needed
        if hasattr(self.checkpointer, "conn"):
            self._register_thread_if_needed(current_thread_id)

        # Get previous state for this thread
        previous_state = None
        try:
            previous_state = self.app.get_state(runtime_config)
            if previous_state:
                logger.info(f"Found previous state for thread {current_thread_id}")
        except Exception as e:
            logger.debug(f"No previous state found or error retrieving: {e}")

        # Process input data based on type
        processed_input = self._prepare_input(input_data, previous_state)

        # Run the agent
        try:
            logger.info(f"Running agent {self.config.name} with thread_id: {current_thread_id}")
            result = self.app.invoke(
                processed_input,
                config=runtime_config
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

        # Prepare runtime config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Get previous state
        previous_state = None
        try:
            previous_state = self.app.get_state(runnable_config)
        except Exception:
            pass

        # Process input based on type
        processed_input = self._prepare_input(input_data, previous_state)

        # Stream the execution
        logger.info(f"Streaming agent {self.config.name}")
        for output in self.app.stream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config
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

        # Prepare runtime config
        runtime_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Get previous state
        previous_state = None
        try:
            previous_state = await self.app.aget_state(runtime_config)
        except Exception:
            pass

        # Process input based on type
        processed_input = self._prepare_input(input_data, previous_state)

        # Run the agent asynchronously
        try:
            logger.info(f"Running agent {self.config.name} asynchronously")
            result = await self.app.ainvoke(
                processed_input,
                config=runtime_config
            )

            # Save state history if requested
            if getattr(self.config, "save_history", True):
                self.save_state_history(runtime_config)

            return result
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

        # Prepare runtime config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, config, **kwargs)

        # Get previous state
        previous_state = None
        try:
            previous_state = await self.app.aget_state(runnable_config)
        except Exception:
            pass

        # Process input based on type
        processed_input = self._prepare_input(input_data, previous_state)

        # Stream the execution asynchronously
        logger.info(f"Streaming agent {self.config.name} asynchronously")
        async for output in self.app.astream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config
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
            import copy
            runtime_config = copy.deepcopy(base_config)

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

        return runtime_config

    def _prepare_input(self, input_data: Any, previous_state: dict[str, Any] | None = None) -> dict[str, Any]:
        """Prepare input for the agent based on the input type.
        
        Args:
            input_data: Input in various formats
            previous_state: Optional previous state to merge with
            
        Returns:
            Processed input compatible with the graph
        """
        # When input is a string
        if isinstance(input_data, str):
            # Initialize input dict with a message
            prepared_input = {"messages": [HumanMessage(content=input_data)]}

            # If we have previous state, merge messages
            if previous_state and "messages" in previous_state:
                all_messages = list(previous_state["messages"])
                all_messages.append(HumanMessage(content=input_data))
                prepared_input["messages"] = all_messages

                # Copy other fields from previous state
                for key, value in previous_state.items():
                    if key != "messages" and not key.startswith("__"):
                        prepared_input[key] = value

            # Validate against schema if available
            if hasattr(self, "state_schema"):
                return self.state_schema(**prepared_input)
            return prepared_input

        # List of strings - convert to messages
        if isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            messages = [HumanMessage(content=item) for item in input_data]

            # If we have previous state, merge messages
            if previous_state and "messages" in previous_state:
                all_messages = list(previous_state["messages"])
                all_messages.extend(messages)
                prepared_input = dict(previous_state)
                prepared_input["messages"] = all_messages
            else:
                prepared_input = {"messages": messages}

            # Validate against schema if available
            if hasattr(self, "state_schema"):
                return self.state_schema(**prepared_input)
            return prepared_input

        # Dict input - merge with previous state if available
        if isinstance(input_data, dict):
            if previous_state:
                # Start with previous state
                merged_input = dict(previous_state)

                # Handle messages specially - append rather than replace
                if "messages" in input_data and "messages" in previous_state:
                    merged_input["messages"] = list(previous_state["messages"])

                    # Handle both list and single message cases
                    if isinstance(input_data["messages"], list):
                        merged_input["messages"].extend(input_data["messages"])
                    else:
                        merged_input["messages"].append(input_data["messages"])

                # Add other fields
                for key, value in input_data.items():
                    if key != "messages":
                        merged_input[key] = value

                # Validate against schema if available
                if hasattr(self, "state_schema"):
                    return self.state_schema(**merged_input)
                return merged_input
            # No previous state, use input as-is
            if hasattr(self, "state_schema"):
                return self.state_schema(**input_data)
            return input_data

        # BaseModel input - convert to dict
        if isinstance(input_data, BaseModel):
            # Convert to dict
            if hasattr(input_data, "model_dump"):
                # Pydantic v2
                model_dict = input_data.model_dump()
            else:
                # Pydantic v1
                model_dict = input_data.dict()

            # Handle previous state merging
            if previous_state:
                # Merge with previous state
                return self._prepare_input(model_dict, previous_state)

            # Validate against schema if available
            if hasattr(self, "state_schema"):
                return self.state_schema(**model_dict)
            return model_dict

        # Already contains BaseMessage objects
        if hasattr(input_data, "messages") and isinstance(input_data.messages, list) and len(input_data.messages) > 0 and isinstance(input_data.messages[0], BaseMessage):
            if previous_state:
                # Merge with previous state
                merged_input = dict(previous_state)

                # Handle messages specially - append rather than replace
                if hasattr(merged_input, "messages"):
                    merged_input["messages"] = list(merged_input["messages"])
                    merged_input["messages"].extend(input_data.messages)
                else:
                    merged_input["messages"] = input_data.messages

                # Validate against schema if available
                if hasattr(self, "state_schema"):
                    return self.state_schema(**merged_input)
                return merged_input
            # No previous state
            if hasattr(self, "state_schema"):
                return self.state_schema(messages=input_data.messages)
            return {"messages": input_data.messages}

        # Fallback - convert to string message
        fallback_input = {
            "messages": [HumanMessage(content=str(input_data))]
        }

        # Handle previous state merging
        if previous_state:
            return self._prepare_input(fallback_input, previous_state)

        # Validate against schema if available
        if hasattr(self, "state_schema"):
            return self.state_schema(**fallback_input)
        return fallback_input

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
                    self._ensure_pool_open()

                    # Register the thread
                    with pool.connection() as conn:
                        with conn.cursor() as cursor:
                            # Check if threads table exists
                            cursor.execute("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = 'threads'
                                );
                            """)
                            table_exists = cursor.fetchone()[0]

                            if not table_exists:
                                logger.debug("Creating threads table")
                                cursor.execute("""
                                    CREATE TABLE IF NOT EXISTS threads (
                                        thread_id VARCHAR(255) PRIMARY KEY,
                                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                        last_access TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                                        metadata JSONB DEFAULT '{}'::jsonb
                                    );
                                """)

                            # Insert the thread if not exists
                            cursor.execute("""
                                INSERT INTO threads (thread_id, last_access) 
                                VALUES (%s, CURRENT_TIMESTAMP) 
                                ON CONFLICT (thread_id) 
                                DO UPDATE SET last_access = CURRENT_TIMESTAMP
                            """, (thread_id,))

                            logger.info(f"Thread {thread_id} registered in PostgreSQL")
            except Exception as e:
                logger.warning(f"Error registering thread: {e}")

    def _ensure_pool_open(self):
        """Ensure that any PostgreSQL connection pool is properly opened.
        
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
                                logger.info("Opening PostgreSQL connection pool")
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
