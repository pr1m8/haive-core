# src/haive/core/engine/agent/agent.py

from __future__ import annotations
import os
import json
import uuid
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Any, Dict, List, Optional, Type, Union, ClassVar, TypeVar, Generic, Generator, AsyncGenerator

from pydantic import BaseModel, Field, field_validator, model_validator
from src.haive.core.utils.pydantic_utils import ensure_json_serializable
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore

from src.haive.core.engine.aug_llm import AugLLMConfig
from src.haive.core.graph.SchemaComposer import SchemaComposer
from src.haive.core.graph.StateSchemaManager import StateSchemaManager
from src.haive.core.utils.visualize_graph_utils import render_and_display_graph
from src.config.constants import RESOURCES_DIR
from src.haive.core.engine.agent.persistence.types import CheckpointerType
from src.haive.core.engine.agent.persistence.base import CheckpointerConfig
from src.haive.core.engine.agent.persistence import (
    load_checkpointer_config,
    PostgresCheckpointerConfig,
    MemoryCheckpointerConfig,
    MongoDBCheckpointerConfig
)
from src.haive.core.engine.base import Engine, EngineRegistry, EngineType

# Set up logging
logger = logging.getLogger(__name__)

# Check if PostgreSQL dependencies are available
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Type variables for generics
TConfig = TypeVar('TConfig', bound='AgentConfig')

# -----------------------------------------------------
# Agent Registry - Maps config classes to agent classes
# -----------------------------------------------------
AGENT_REGISTRY: Dict[Type['AgentConfig'], Type['Agent']] = {}

def register_agent(config_class: Type['AgentConfig']):
    """Register an agent class with its configuration class."""
    def decorator(agent_class: Type['Agent']):
        AGENT_REGISTRY[config_class] = agent_class
        return agent_class
    return decorator

from langgraph.prebuilt import create_react_agent
class AgentConfig(BaseModel, ABC):
    """
    Base configuration for an agent architecture.
    """
    name: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    
    # Primary engine (used as default processor)
    engine: Optional[Union[Engine, str]] = None
    
    # Additional named engines
    engines: Dict[str, Union[Engine, str]] = Field(default_factory=dict)
    
    # Schema definitions
    state_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    input_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    output_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    
    # Visualization and debugging
    visualize: bool = Field(default=True)
    output_dir: str = Field(default=RESOURCES_DIR)
    debug: bool = Field(default=False)
    save_history: bool = Field(default=True)
    
    # Runtime settings
    runnable_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": 200
        }
    )
    
    # Storage
    add_store: bool = Field(default=False)
    
    # Agent-specific settings (to be overridden by subclasses)
    agent_settings: Dict[str, Any] = Field(default_factory=dict)

    # =============================================
    # Persistence Configuration
    # =============================================
    # The persistence configuration to use
    persistence: Optional[CheckpointerConfig] = Field(
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
    
    def derive_schema(self) -> Type[BaseModel]:
        """Derive state schema from components and engines."""
        # Get all components including engines
        all_components = []
        if self.engine:
            all_components.append(self.engine)
        all_components.extend(self.engines.values())
        
        # Create schema name
        schema_name = f"{self.name.replace('-', '_').title()}State"
        
        # Use SchemaComposer to build schema
        return SchemaComposer.create_schema_for_components(all_components, name=schema_name)
    
    def resolve_engine(self, engine_ref=None):
        """
        Resolve an engine reference to an actual engine.
        
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
            registry = EngineRegistry.get_instance()
            for engine_type in EngineType:
                engine = registry.get(engine_type, ref)
                if engine:
                    return engine
                    
            raise ValueError(f"Engine '{ref}' not found in registry")
            
        raise TypeError(f"Unsupported engine reference type: {type(ref)}")
    
    def build_agent(self) -> 'Agent':
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
    
    def _resolve_agent_class_by_name(self) -> Optional[Type['Agent']]:
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
    
class Agent(Generic[TConfig], ABC):
    """
    Base agent architecture class.
    Defines how an agent works - its implementation and behavior.
    """
    def __init__(self, config: TConfig):
        """Initialize the agent with its configuration."""
        self.config = config
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Set up checkpointer from persistence configuration
        self.checkpointer = self._setup_checkpointer()
        
        # Add store if configured 
        if config.add_store:
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
        if config.visualize and self.app:
            self.visualize_graph()
    
    def _initialize_engines(self):
        """Initialize all engines (LLMs, runnables, etc.)."""
        self.engines = {}
        
        # Initialize main engine if present
        if self.config.engine:
            self.engine = self._build_engine(self.config.engine)
            self.engines["main"] = self.engine
        
        # Initialize additional engines
        for name, engine_config in getattr(self.config, 'engines', {}).items():
            self.engines[name] = self._build_engine(engine_config)
    
    def _build_engine(self, config):
        """Build an engine from configuration."""
        if isinstance(config, str):
            # Try to resolve from registry
            registry = EngineRegistry.get_instance()
            for engine_type in EngineType:
                engine = registry.get(engine_type, config)
                if engine:
                    config = engine
                    break
            if isinstance(config, str):
                raise ValueError(f"Engine '{config}' not found in registry")
        
        # Build based on engine type
        if isinstance(config, AugLLMConfig):
            return config.create_runnable()
        elif hasattr(config, "create_runnable"):
            return config.create_runnable()
        return config  # Return as-is if already built
    
    def _setup_schemas(self):
        """Set up state, input, and output schemas."""
        # Resolve or derive state schema
        if self.config.state_schema is None:
            self.state_schema = self.config.derive_schema()
        elif isinstance(self.config.state_schema, dict):
            schema_manager = StateSchemaManager(self.config.state_schema, name=f"{self.config.name}State")
            self.state_schema = schema_manager.get_model()
        else:
            self.state_schema = self.config.state_schema
        
        # Process input schema - default to state schema if not provided
        if self.config.input_schema is None:
            self.input_schema = self.state_schema
        elif isinstance(self.config.input_schema, dict):
            schema_manager = StateSchemaManager(self.config.input_schema, name=f"{self.config.name}Input")
            self.input_schema = schema_manager.get_model()
        else:
            self.input_schema = self.config.input_schema
        
        # Process output schema - default to state schema if not provided
        if self.config.output_schema is None:
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
        print(f"input_schema: {self.input_schema}")
        print(f"output_schema: {self.output_schema}")   
        print(f"state_schema: {self.state_schema}")
        if hasattr(self, 'input_schema') and hasattr(self, 'output_schema'):
            self.graph = StateGraph(
                self.state_schema,
                input=self.input_schema,
                output=self.output_schema
            )
        else:
            self.graph = StateGraph(self.state_schema)
    
    

    def _setup_checkpointer(self):
        """
        Set up the appropriate checkpointer based on persistence configuration.
        
        Returns:
            A configured checkpointer instance
        """
        # Import MemorySaver
        from langgraph.checkpoint.memory import MemorySaver
        
        # Handle explicit request for memory persistence
        if not hasattr(self.config, 'persistence') or self.config.persistence is None:
            logger.info(f"No persistence config for {self.config.name}. Using memory checkpointer.")
            return MemorySaver()
        
        # Handle dictionary config (which is what's passed from tests)
        if isinstance(self.config.persistence, dict):
            # Check for explicit memory type
            if self.config.persistence.get('type', 'memory') == 'memory':
                logger.info(f"Using memory checkpointer (per config) for {self.config.name}")
                return MemorySaver()
                
            # Only continue with postgres if it's explicitly requested and available
            if self.config.persistence.get('type') == 'postgres' and POSTGRES_AVAILABLE:
                try:
                    # Import only when needed
                    from psycopg_pool import ConnectionPool
                    from langgraph.checkpoint.postgres import PostgresSaver
                    
                    # Get connection parameters
                    db_host = self.config.persistence.get('db_host', 'localhost')
                    db_port = self.config.persistence.get('db_port', 5432)
                    db_name = self.config.persistence.get('db_name', 'postgres')
                    db_user = self.config.persistence.get('db_user', 'postgres')
                    db_pass = self.config.persistence.get('db_pass', 'postgres')
                    ssl_mode = self.config.persistence.get('ssl_mode', 'disable')
                    
                    # Create connection URI
                    import urllib.parse
                    encoded_pass = urllib.parse.quote_plus(str(db_pass))
                    db_uri = f"postgresql://{db_user}:{encoded_pass}@{db_host}:{db_port}/{db_name}"
                    if ssl_mode:
                        db_uri += f"?sslmode={ssl_mode}"
                    
                    # Create connection pool
                    pool = ConnectionPool(
                        conninfo=db_uri,
                        min_size=self.config.persistence.get('min_pool_size', 1),
                        max_size=self.config.persistence.get('max_pool_size', 5),
                        kwargs={
                            "autocommit": self.config.persistence.get('auto_commit', True),
                            "prepare_threshold": self.config.persistence.get('prepare_threshold', 0),
                        },
                        open=True  # Explicitly open the pool to avoid "not open yet" errors
                    )
                    
                    # Create checkpointer
                    checkpointer = PostgresSaver(pool)
                    
                    # Set flag for setup if needed
                    if self.config.persistence.get('setup_needed', False):
                        self.postgres_setup_needed = True
                        
                    logger.info(f"Using PostgreSQL checkpointer for {self.config.name}")
                    return checkpointer
                    
                except Exception as e:
                    logger.error(f"Failed to setup PostgreSQL checkpointer: {e}")
                    logger.warning(f"Falling back to memory checkpointer for {self.config.name}")
        
        # Handle CheckpointerConfig objects
        elif hasattr(self.config.persistence, 'type'):
            if self.config.persistence.type == CheckpointerType.memory:
                logger.info(f"Using memory checkpointer for {self.config.name}")
                return MemorySaver()
                
            if self.config.persistence.type == CheckpointerType.postgres and POSTGRES_AVAILABLE:
                try:
                    # Try to use create_checkpointer method if available
                    if hasattr(self.config.persistence, 'create_checkpointer'):
                        checkpointer = self.config.persistence.create_checkpointer()
                        
                        # Set flag for table setup if needed
                        if hasattr(self.config.persistence, 'setup_needed') and self.config.persistence.setup_needed:
                            self.postgres_setup_needed = True
                            
                        logger.info(f"Using PostgreSQL checkpointer for {self.config.name}")
                        return checkpointer
                except Exception as e:
                    logger.error(f"Failed to create PostgreSQL checkpointer: {e}")
        
        # Default to memory checkpointer for any other case
        logger.info(f"Using memory checkpointer (default) for {self.config.name}")
        return MemorySaver()
    
    @abstractmethod
    def setup_workflow(self) -> None:
        """Set up the workflow graph for this agent."""
        pass
    
        
    # Update the compile method in src/haive/core/engine/agent/agent.py to fix PostgreSQL detection

    def compile(self) -> None:
        """Compile the workflow graph into an executable app."""
        if not self.graph:
            raise RuntimeError("Graph is not set up.")
            
        # Setup PostgreSQL tables if needed
        persistence_config = getattr(self.config, 'persistence', None)
        postgres_setup_needed = getattr(self, 'postgres_setup_needed', False)
        
        # Check if using PostgreSQL with proper attribute check
        is_postgres = False
        if persistence_config:
            if hasattr(persistence_config, 'is_postgres'):
                is_postgres = persistence_config.is_postgres
            elif hasattr(persistence_config, 'type'):
                from src.haive.core.engine.agent.persistence.types import CheckpointerType
                is_postgres = persistence_config.type == CheckpointerType.postgres
        
        # Ensure pool is open if using PostgreSQL
        opened_pool = None
        if persistence_config and is_postgres and postgres_setup_needed:
            opened_pool = self._ensure_pool_open()
            
            try:
                # Set up tables
                if hasattr(persistence_config, 'use_async') and persistence_config.use_async:
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
    
    # Update the run method in src/haive/core/engine/agent/agent.py to ensure thread registration
    # This is a more streamlined approach that avoids the need to modify all methods

    # Update the run method in Agent class to maintain conversation history

    def run(
        self,
        input_data: Union[str, List[str], Dict[str, Any], BaseModel]={},
        thread_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run the agent with the given input and return the final state.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.compile()
            
        # Ensure thread_id for persistence
        runtime_config = self.config.runnable_config.copy()
        if "configurable" not in runtime_config:
            runtime_config["configurable"] = {}
        
        # Use provided thread_id or create new one
        if thread_id:
            runtime_config["configurable"]["thread_id"] = thread_id
        elif "thread_id" not in runtime_config["configurable"]:
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # Get the current thread_id
        current_thread_id = runtime_config["configurable"]["thread_id"]
        logger.info(f"Using thread_id: {current_thread_id}")
        
        # Merge with other kwargs
        for key, value in kwargs.items():
            runtime_config[key] = value
        
        # Check if using PostgreSQL checkpointer
        is_postgres = False
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            is_postgres = isinstance(self.checkpointer, PostgresSaver)
        except ImportError:
            pass
        
        # If using PostgreSQL, explicitly register the thread before continuing
        if is_postgres:
            self._register_thread_in_postgres(current_thread_id)
        
        # For string inputs with message-based schemas, use LangGraph format directly
        if isinstance(input_data, str) and hasattr(self.state_schema, '__annotations__'):
            # Check if state has messages field
            has_messages = 'messages' in self.state_schema.__annotations__
            if has_messages:
                # Use LangGraph tuple format for messages
                input_data = {"messages": [("human", input_data)]}
        
        # Process input data (directly use as-is rather than trying complex transformations)
        try:
            # For most inputs, just use the input as-is
            # LangGraph will correctly handle merging with previous state
            result = self.app.invoke(
                input_data,
                config=runtime_config,
                debug=self.config.debug
            )
            
            # Save state history if requested
            if self.config.save_history:
                self.save_state_history(runtime_config)
            
            return result
        
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            raise e

    def _prepare_input_for_schema(self, input_data: Any, previous_state: Optional[Any] = None) -> Any:
        """
        Prepare input data based on the state schema and previous state.
        This method handles different input types and merges with previous state if available.
        
        Args:
            input_data: The input in various formats
            previous_state: Optional previous state from the checkpointer
            
        Returns:
            Input data compatible with the agent's state schema
        """
        # If input matches our schema exactly, use it directly
        if isinstance(input_data, self.state_schema):
            return input_data
        
        # Handle StateSnapshot objects - extract the values field
        if previous_state and hasattr(previous_state, 'values'):
            previous_values = previous_state.values
        else:
            previous_values = previous_state if isinstance(previous_state, dict) else {}
        
        # Convert to dictionary if it's a BaseModel
        if isinstance(input_data, BaseModel):
            if hasattr(input_data, "model_dump"):
                input_data = input_data.model_dump()
            elif hasattr(input_data, "dict"):
                input_data = input_data.dict()
            else:
                # Manual conversion
                input_dict = {}
                for field in input_data.__annotations__:
                    if hasattr(input_data, field):
                        input_dict[field] = getattr(input_data, field)
                input_data = input_dict
        
        # Convert string to appropriate format based on schema
        if isinstance(input_data, str):
            # Check if state schema has a 'messages' field
            has_messages = False
            for field_name, field_info in self.state_schema.__annotations__.items():
                if field_name == 'messages':
                    has_messages = True
                    break
            
            if has_messages:
                # If schema has messages, create a human message
                if previous_values and 'messages' in previous_values:
                    # Preserve previous messages and add new one
                    from langchain_core.messages import HumanMessage
                    new_message = HumanMessage(content=input_data)
                    input_data = {'messages': previous_values['messages'] + [new_message]}
                else:
                    # No previous messages, create fresh list
                    from langchain_core.messages import HumanMessage
                    input_data = {'messages': [HumanMessage(content=input_data)]}
            else:
                # For schemas without messages, use plain text format
                # Use a generic field name that might exist in the schema
                for field_name in ['input', 'query', 'text', 'content']:
                    if field_name in self.state_schema.__annotations__:
                        return self.state_schema(**{field_name: input_data})
                
                # If no matching field found, use the first field as a fallback
                field_name = next(iter(self.state_schema.__annotations__), None)
                if field_name:
                    return self.state_schema(**{field_name: input_data})
                
                # Last resort - just make a dict
                input_data = {'input': input_data}
        
        # If it's a list of strings
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Similar approach to string but with multiple messages
            has_messages = 'messages' in self.state_schema.__annotations__
            
            if has_messages:
                from langchain_core.messages import HumanMessage
                new_messages = [HumanMessage(content=item) for item in input_data]
                
                if previous_values and 'messages' in previous_values:
                    input_data = {'messages': previous_values['messages'] + new_messages}
                else:
                    input_data = {'messages': new_messages}
            else:
                # Join strings for schemas without messages
                joined_text = "\n".join(input_data)
                for field_name in ['input', 'query', 'text', 'content']:
                    if field_name in self.state_schema.__annotations__:
                        return self.state_schema(**{field_name: joined_text})
                
                # Fallback to first field
                field_name = next(iter(self.state_schema.__annotations__), None)
                if field_name:
                    return self.state_schema(**{field_name: joined_text})
                
                # Last resort
                input_data = {'input': joined_text}
        
        # Handle dictionary input
        if isinstance(input_data, dict):
            # If previous values exist, merge with it to preserve history
            if previous_values:
                # Create a merged state preserving previous values
                merged_state = dict(previous_values)
                
                # Update only the fields provided in input_data
                for key, value in input_data.items():
                    if key == 'messages' and key in merged_state:
                        # Special handling for messages to append rather than replace
                        if isinstance(value, list) and isinstance(merged_state[key], list):
                            # Only append messages that aren't already there
                            existing_message_ids = set()
                            for msg in merged_state[key]:
                                if hasattr(msg, 'id'):
                                    existing_message_ids.add(msg.id)
                            
                            # Append only new messages
                            for msg in value:
                                if not hasattr(msg, 'id') or msg.id not in existing_message_ids:
                                    merged_state[key].append(msg)
                        else:
                            # If not lists or other structure, replace
                            merged_state[key] = value
                    else:
                        # For other fields, simply update
                        merged_state[key] = value
                
                # Return merged state as our input
                return self.state_schema(**merged_state)
        
        # If we get here, just try to create a state object with the input data
        try:
            return self.state_schema(**input_data)
        except Exception as e:
            logger.warning(f"Error creating state from input data: {e}")
            # Last resort: If nothing else works, try to use the input as-is
            return input_data

    def _register_thread_if_needed(self, thread_id: str) -> None:
        """
        Register a thread in the PostgreSQL database if needed.
        
        Args:
            thread_id: Thread ID to register
        """
        if hasattr(self.checkpointer, 'conn'):
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
            
    def _ensure_pool_open(self) -> bool:
        """
        Ensure that the PostgreSQL connection pool is open.
        
        Returns:
            bool: True if the pool is open, False otherwise
        """
        try:
            if hasattr(self.checkpointer, 'conn') and self.checkpointer.conn:
                pool = self.checkpointer.conn
                # Check if the pool is usable by getting a connection
                with pool.connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT 1")
                        result = cursor.fetchone()
                        if result and result[0] == 1:
                            logger.debug("PostgreSQL connection pool is open")
                            return True
            return False
        except Exception as e:
            logger.warning(f"Error checking PostgreSQL connection pool: {e}")
            return False
            
    def stream(
        self, 
        input_data: Union[str, List[str], Dict[str, Any], BaseModel], 
        thread_id: Optional[str] = None,
        stream_mode: str = "values", 
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the agent execution.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            self.compile()
            
        # Process input based on type
        processed_input = self._prepare_input(input_data)
        
        # Prepare runnable config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, **kwargs)
        
        # Stream the execution
        logger.info(f"Streaming agent {self.config.name}")
        for output in self.app.stream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config,
            debug=self.config.debug
        ):
            yield output
            
        # Save state history if requested
        if self.config.save_history:
            self.save_state_history(runnable_config)
    
    async def arun(
    self,
    input_data: Union[str, List[str], Dict[str, Any], BaseModel],
    thread_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
        """
        Run the agent asynchronously with the given input.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            **kwargs: Additional runtime configuration
            
        Returns:
            Final state or output
        """
        if not self.app:
            self.compile()
            
        # Process input based on type
        processed_input = self._prepare_input(input_data)
        
        # Ensure thread_id for persistence
        runtime_config = self.config.runnable_config.copy()
        if "configurable" not in runtime_config:
            runtime_config["configurable"] = {}
        
        # Use provided thread_id or create new one
        if thread_id:
            runtime_config["configurable"]["thread_id"] = thread_id
        elif "thread_id" not in runtime_config["configurable"]:
            runtime_config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # Merge with other kwargs
        for key, value in kwargs.items():
            runtime_config[key] = value
        
        # Ensure PostgreSQL pool is open if needed
        opened_pool = self._ensure_pool_open()
        
        # Run the agent asynchronously
        try:
            logger.info(f"Running agent {self.config.name} asynchronously with thread_id: {runtime_config['configurable']['thread_id']}")
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
            if self.config.save_history:
                self.save_state_history()
                
            return final_state
        except Exception as e:
            logger.error(f"Error running agent asynchronously: {e}")
            raise e
            
    async def astream(
        self, 
        input_data: Union[str, List[str], Dict[str, Any], BaseModel], 
        thread_id: Optional[str] = None,
        stream_mode: str = "values", 
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the agent execution asynchronously.
        
        Args:
            input_data: Input data in various formats
            thread_id: Optional thread ID for persistence
            stream_mode: Stream mode (values, updates, debug, etc.)
            **kwargs: Additional runtime configuration
            
        Yields:
            State updates during execution
        """
        if not self.app:
            self.compile()
            
        # Process input based on type
        processed_input = self._prepare_input(input_data)
        
        # Prepare runnable config with thread_id
        runnable_config = self._prepare_runnable_config(thread_id, **kwargs)
        
        # Stream the execution asynchronously
        logger.info(f"Streaming agent {self.config.name} asynchronously")
        async for output in self.app.astream(
            processed_input,
            stream_mode=stream_mode,
            config=runnable_config,
            debug=self.config.debug
        ):
            yield output
            
        # Save state history if requested
        if self.config.save_history:
            self.save_state_history(runnable_config)
    
    def save_state_history(self, runnable_config=None) -> None:
        """
        Save the current agent state to a JSON file.
        
        Args:
            runnable_config: Optional runnable configuration
        """
        if not self.app:
            logger.error(f"Cannot save state history: Workflow graph not compiled")
            return

        # Use provided runnable config or default from config
        runnable_config = runnable_config or self.config.runnable_config
            
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
            logger.error(f"Error saving state history: {str(e)}")
    
    
    def _register_postgres_thread(self, conn):
        """
        Register a thread in the PostgreSQL database before using it.
        This ensures the thread exists before any checkpoints reference it.
        
        Args:
            conn: Connection pool to use
        """
        if not hasattr(self.config, 'runnable_config') or not self.config.runnable_config:
            logger.warning("No runnable_config found, can't register thread")
            return
            
        if 'configurable' not in self.config.runnable_config:
            logger.warning("No configurable section in runnable_config, can't register thread")
            return
            
        thread_id = self.config.runnable_config['configurable'].get('thread_id')
        if not thread_id:
            logger.warning("No thread_id found in runnable_config, can't register thread")
            return
        
        logger.info(f"Registering thread {thread_id} in PostgreSQL database")
        
        try:
            with conn.connection() as connection:
                with connection.cursor() as cursor:
                    # First check if thread already exists
                    cursor.execute("SELECT 1 FROM threads WHERE thread_id = %s", (thread_id,))
                    thread_exists = cursor.fetchone() is not None
                    
                    if not thread_exists:
                        # Insert the thread with a name based on agent name
                        cursor.execute(
                            "INSERT INTO threads (thread_id, name, metadata) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                            (thread_id, f"Thread for {self.config.name}", '{}')
                        )
                        logger.info(f"Thread {thread_id} registered successfully")
                    else:
                        logger.debug(f"Thread {thread_id} already exists in database")
        except Exception as e:
            logger.error(f"Error registering thread in PostgreSQL: {e}")

    # Then, update the _prepare_runnable_config method to ensure thread is registered before use:

    def _prepare_runnable_config(self, thread_id=None, **kwargs) -> Dict[str, Any]:
        """
        Prepare runnable config with thread ID and other parameters.
        
        Args:
            thread_id: Optional thread ID for persistence
            **kwargs: Additional runnable configuration parameters
            
        Returns:
            Runnable configuration dictionary with proper thread ID
        """
        # Start with base runnable config from agent config
        runnable_config = self.config.runnable_config.copy()
        
        # Ensure configurable section exists
        if "configurable" not in runnable_config:
            runnable_config["configurable"] = {}
        
        # Use provided thread_id or keep existing one
        if thread_id:
            runnable_config["configurable"]["thread_id"] = thread_id
        elif "thread_id" not in runnable_config["configurable"]:
            # Generate a new thread ID if none exists
            runnable_config["configurable"]["thread_id"] = str(uuid.uuid4())
        
        # Add other kwargs
        for key, value in kwargs.items():
            # If it's a configurable param, add to configurable section
            if key.startswith("configurable_"):
                param_name = key.replace("configurable_", "")
                runnable_config["configurable"][param_name] = value
            else:
                # Otherwise add to top level
                runnable_config[key] = value
        
        # Register the thread in PostgreSQL if we're using PostgreSQL persistence
        if hasattr(self, 'checkpointer') and hasattr(self.checkpointer, 'conn'):
            try:
                # Check if this is a PostgreSQL saver
                from langgraph.checkpoint.postgres import PostgresSaver
                if isinstance(self.checkpointer, PostgresSaver):
                    # Ensure the thread is registered
                    self._register_postgres_thread(self.checkpointer.conn)
            except (ImportError, AttributeError):
                pass
        
        return runnable_config
    
    def _prepare_input(self, input_data: Union[str, List[str], Dict[str, Any], BaseModel]) -> Any:
        """
        Prepare input for the agent based on the input type.
        
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
            if hasattr(self, 'engine') and hasattr(self.engine, 'prompt_template') and hasattr(self.engine.prompt_template, 'input_variables'):
                required_inputs.update(self.engine.prompt_template.input_variables)
            
            # Add string to all required inputs
            for input_name in required_inputs:
                if input_name != 'messages':
                    prepared_input[input_name] = input_data
            
            return self.state_schema(**prepared_input)
        
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
            
            return self.state_schema(**prepared_input)
        
        elif isinstance(input_data, dict):
            # Dict input - use as is
            return self.state_schema(**input_data)
        
        elif isinstance(input_data, BaseModel):
            # Convert BaseModel to dict
            if hasattr(input_data, "model_dump"):
                return self.state_schema(**input_data.model_dump())
            elif hasattr(input_data, "dict"):
                return self.state_schema(**input_data.dict())
            else:
                # Manual extraction
                prepared_input = {}
                for field in input_data.__annotations__:
                    if hasattr(input_data, field):
                        prepared_input[field] = getattr(input_data, field)
                return self.state_schema(**prepared_input)
        
        # Fallback
        return self.state_schema(messages=[HumanMessage(content=str(input_data))])
    
        
    # Update the _ensure_pool_open method to be more robust

    
    def _register_thread_in_postgres(self, thread_id: str) -> None:
        """
        Explicitly register a thread in the PostgreSQL database.
        This ensures the thread exists in the threads table before any checkpoints reference it.
        
        Args:
            thread_id: The thread ID to register
        """
        # Only proceed if we're using PostgreSQL
        if not hasattr(self.checkpointer, 'conn'):
            return
            
        logger.info(f"Explicitly registering thread {thread_id} in PostgreSQL database")
        
        try:
            # Get the connection pool
            pool = self.checkpointer.conn
            if not pool:
                logger.warning("No connection pool available for thread registration")
                return
                
            # Ensure the pool is open
            self._ensure_pool_open()
            
            # Connect and register the thread
            with pool.connection() as conn:
                with conn.cursor() as cursor:
                    # Create the thread with minimal metadata
                    cursor.execute(
                        """
                        INSERT INTO threads (thread_id) 
                        VALUES (%s)
                        ON CONFLICT (thread_id) DO NOTHING
                        """,
                        (thread_id,)
                    )
                    # Commit the transaction explicitly to ensure it's saved
                    if not getattr(pool, 'kwargs', {}).get('autocommit', False):
                        conn.commit()
                        
                    logger.info(f"Thread {thread_id} successfully registered in PostgreSQL")
                    
        except Exception as e:
            logger.error(f"Error registering thread in PostgreSQL: {e}")
    def _close_pool_if_needed(self, pool=None):
        """
        Close a PostgreSQL connection pool if it was previously opened.
        This should be called in finally blocks after operations.
        
        Args:
            pool: The pool to close. If None, will try to find the pool 
                from the checkpointer.
        """
        if pool is None:
            # Try to find a pool from the checkpointer
            try:
                if hasattr(self.checkpointer, 'conn'):
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