from __future__ import annotations
# src/haive/core/engine/agent/agent.py


import os
import json
import uuid
import logging
from datetime import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union, ClassVar, TypeVar, \
    Generic, Generator, AsyncGenerator, Tuple, Literal, get_args, get_origin, cast, Self

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from haive_core.engine.base import Engine, InvokableEngine, EngineType, EngineRegistry
from haive_core.engine.aug_llm import AugLLMConfig
from haive_core.schema.schema_composer import SchemaComposer
from haive_core.schema.state_schema import StateSchema
from haive_core.graph.dynamic_graph_builder import DynamicGraph
from haive_core.config.runnable import RunnableConfigManager

# Import persistence-related functionality

from haive_core.engine.agent.persistence.base import CheckpointerConfig
from haive_core.engine.agent.persistence.handlers import (
    setup_checkpointer, 
    ensure_pool_open, 
    close_pool_if_needed, 
    register_thread_if_needed
)

# Import state handling functionality
from haive_core.engine.agent.utils.state_handling import (
    process_input, 
    prepare_merged_input
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from haive_core.engine.agent.agent import Agent

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generics
TConfig = TypeVar('TConfig', bound='AgentConfig')
TIn = TypeVar('TIn')
TOut = TypeVar('TOut')

# -----------------------------------------------------
# Agent Registry - Maps config classes to agent classes
# -----------------------------------------------------
from haive_core.engine.agent.registry import AGENT_REGISTRY, register_agent
class AgentConfig(InvokableEngine[TIn, TOut], ABC):
    """
    Base configuration for an agent architecture.
    
    AgentConfig extends InvokableEngine to provide a consistent interface
    for creating and using agents within the Engine framework.
    
    Each AgentConfig is paired with an Agent implementation class that defines
    the actual behavior and workflow.
    """
    engine_type: EngineType = Field(default=EngineType.AGENT)
    name: str = Field(default_factory=lambda: f"agent_{uuid.uuid4().hex[:8]}")
    
    # Primary engine (used as default processor)
    engine: Optional[Union[Engine, str]] = None
    
    # Additional named engines
    engines: Dict[str, Union[Engine, str]] = Field(default_factory=dict)
    
    # Schema definitions
    state_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    input_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    output_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    config_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None
    
    # Parent agent for subgraphs
    parent: Optional[Union['AgentConfig', str]] = None
    
    # Node definition (for simple agents)
    node_name: str = Field(default="process")
    
    # Graph patterns
    auto_end: bool = Field(default=True, description="Automatically add END edge from last node")
    
    # Visualization and debugging
    visualize: bool = Field(default=True)
    output_dir: str = Field(default="resources")
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
    agent_settings: Dict[str, Any] = Field(default_factory=dict)

    # Persistence Configuration
    persistence: Optional[CheckpointerConfig] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @model_validator(mode="after")
    def ensure_engine(self):
        """Ensure at least one engine is available."""
        if not self.engine and not self.engines:
            self.engine = AugLLMConfig()
        return self
    
    def derive_schema(self) -> Type[StateSchema]:
        """Derive state schema from components and engines."""
        # Get all components including engines
        all_components = []
        if self.engine:
            all_components.append(self.engine)
        all_components.extend(self.engines.values())
        
        # Create schema name
        schema_name = f"{self.name.replace('-', '_').title()}State"
        
        # Use SchemaComposer to build schema
        return SchemaComposer.compose(
            all_components, 
            name=schema_name,
            include_runnable_config=True
        )
    
    def resolve_engine(self, engine_ref=None) -> Engine:
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
            for engine_type in registry.engines:
                engine = registry.get(engine_type, ref)
                if engine:
                    return engine
                    
            raise ValueError(f"Engine '{ref}' not found in registry")
            
        raise TypeError(f"Unsupported engine reference type: {type(ref)}")
    
    def resolve_parent(self) -> Optional['AgentConfig']:
        """
        Resolve parent reference to an actual AgentConfig.
        
        Returns:
            Resolved parent AgentConfig or None
        """
        if not self.parent:
            return None
            
        # If parent is already an AgentConfig, return it
        if isinstance(self.parent, AgentConfig):
            return self.parent
            
        # If it's a string, look it up in the registry
        if isinstance(self.parent, str):
            # Try to find in registry
            registry = EngineRegistry.get_instance()
            parent = registry.get(EngineType.AGENT, self.parent)
            if parent and isinstance(parent, AgentConfig):
                return parent
                
            logger.warning(f"Parent agent '{self.parent}' not found in registry")
            
        return None
    
    def build_agent(self) -> 'Agent':
        """Build an agent instance from this configuration."""
        # Try to find agent class in registry
        from haive_core.engine.agent.registry import resolve_agent_class
        agent_class = resolve_agent_class(self.__class__)
        
        if agent_class is None:
            raise TypeError(f"No agent class found for {self.__class__.__name__}")
        
        # Instantiate and return the agent
        return agent_class(config=self)
    
    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Any:
        """
        Create a runnable instance from this agent config.
        
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
    
    def derive_input_schema(self) -> Type[BaseModel]:
        """
        Derive input schema for this agent.
        
        Returns:
            Pydantic model for input schema
        """
        # Use provided schema if available
        if self.input_schema:
            if isinstance(self.input_schema, type) and issubclass(self.input_schema, BaseModel):
                # Ensure it's a StateSchema
                if not issubclass(self.input_schema, StateSchema):
                    return SchemaComposer.compose_as_state_schema(
                        [self.input_schema], 
                        name=f"{self.name}Input"
                    )
                return self.input_schema
            elif isinstance(self.input_schema, dict):
                return SchemaComposer.compose_as_state_schema(
                    [self.input_schema], 
                    name=f"{self.name}Input"
                )
        
        # Try to derive from class type hints
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is InvokableEngine:
                args = get_args(base_cls)
                if len(args) >= 1:
                    # Extract TIn from InvokableEngine[TIn, TOut]
                    in_type = args[0]
                    if in_type is not TIn:  # Not the generic parameter itself
                        if isinstance(in_type, type) and issubclass(in_type, BaseModel):
                            # Ensure it's a StateSchema
                            if not issubclass(in_type, StateSchema):
                                return SchemaComposer.compose_as_state_schema(
                                    [in_type], 
                                    name=f"{self.name}Input"
                                )
                            return in_type
        
        # Default to deriving from state schema
        return self.derive_schema()
    
    def derive_output_schema(self) -> Type[BaseModel]:
        """
        Derive output schema for this agent.
        
        Returns:
            Pydantic model for output schema
        """
        # Use provided schema if available
        if self.output_schema:
            if isinstance(self.output_schema, type) and issubclass(self.output_schema, BaseModel):
                # Ensure it's a StateSchema
                if not issubclass(self.output_schema, StateSchema):
                    return SchemaComposer.compose_as_state_schema(
                        [self.output_schema], 
                        name=f"{self.name}Output"
                    )
                return self.output_schema
            elif isinstance(self.output_schema, dict):
                return SchemaComposer.compose_as_state_schema(
                    [self.output_schema], 
                    name=f"{self.name}Output"
                )
        
        # Try to derive from class type hints
        for base_cls in self.__class__.__orig_bases__:
            if get_origin(base_cls) is InvokableEngine:
                args = get_args(base_cls)
                if len(args) >= 2:
                    # Extract TOut from InvokableEngine[TIn, TOut]
                    out_type = args[1]
                    if out_type is not TOut:  # Not the generic parameter itself
                        if isinstance(out_type, type) and issubclass(out_type, BaseModel):
                            # Ensure it's a StateSchema
                            if not issubclass(out_type, StateSchema):
                                return SchemaComposer.compose_as_state_schema(
                                    [out_type], 
                                    name=f"{self.name}Output"
                                )
                            return out_type
        
        # Default to deriving from state schema
        return self.derive_schema()
    
    def invoke(self, input_data: TIn, runnable_config: Optional[RunnableConfig] = None) -> TOut:
        """
        Invoke the agent with input data.
        
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
    
    async def ainvoke(self, input_data: TIn, runnable_config: Optional[RunnableConfig] = None) -> TOut:
        """
        Asynchronously invoke the agent with input data.
        
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
    
    def apply_runnable_config(self, runnable_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """
        Extract parameters from runnable_config relevant to this agent.
        
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
        
        # Extract ALL agent parameters from fields
        for field in self.model_fields:
            if field in configurable and field not in params:
                params[field] = configurable[field]
        
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
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert agent config to a dictionary.
        
        Returns:
            Dictionary representation of the agent config
        """
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"input_schema", "output_schema", "config_schema"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"input_schema", "output_schema", "config_schema"})
        
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
        
        # Convert parent to name if it's an AgentConfig
        if "parent" in data and isinstance(data["parent"], AgentConfig):
            data["parent"] = data["parent"].name
        
        # Convert persistence config
        if "persistence" in data and data["persistence"] is not None:
            if hasattr(data["persistence"], "to_dict"):
                data["persistence"] = data["persistence"].to_dict()
        
        # Convert schemas if available
        if hasattr(self, "state_schema") and isinstance(self.state_schema, type):
            data["state_schema"] = self.state_schema.__name__
        if hasattr(self, "input_schema") and isinstance(self.input_schema, type):
            data["input_schema"] = self.input_schema.__name__
        if hasattr(self, "output_schema") and isinstance(self.output_schema, type):
            data["output_schema"] = self.output_schema.__name__
        if hasattr(self, "config_schema") and isinstance(self.config_schema, type):
            data["config_schema"] = self.config_schema.__name__
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        """
        Create an AgentConfig from a dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            AgentConfig instance
        """
        # Create a copy to avoid modifying the input
        config_data = data.copy()
        
        # Handle engine references
        if "engine" in config_data and isinstance(config_data["engine"], str):
            registry = EngineRegistry.get_instance()
            for engine_type in registry.engines:
                engine = registry.get(engine_type, config_data["engine"])
                if engine:
                    config_data["engine"] = engine
                    break
        
        # Handle engine dict
        if "engines" in config_data:
            engines = {}
            for name, engine_ref in config_data["engines"].items():
                if isinstance(engine_ref, str):
                    registry = EngineRegistry.get_instance()
                    for engine_type in registry.engines:
                        engine = registry.get(engine_type, engine_ref)
                        if engine:
                            engines[name] = engine
                            break
                    if name not in engines:
                        engines[name] = engine_ref
                else:
                    engines[name] = engine_ref
            config_data["engines"] = engines
        
        # Handle parent reference
        if "parent" in config_data and isinstance(config_data["parent"], str):
            registry = EngineRegistry.get_instance()
            parent = registry.get(EngineType.AGENT, config_data["parent"])
            if parent:
                config_data["parent"] = parent
        
        # Remove schema names if present
        for schema_key in ["state_schema", "input_schema", "output_schema", "config_schema"]:
            if schema_key in config_data and isinstance(config_data[schema_key], str):
                # We can't resolve class names here, so remove them
                del config_data[schema_key]
        
        # Process persistence configuration
        if "persistence" in config_data and config_data["persistence"] is not None:
            if isinstance(config_data["persistence"], dict):
                from haive_core.engine.agent.persistence import load_checkpointer_config
                config_data["persistence"] = load_checkpointer_config(config_data["persistence"])
        
        # Create instance
        return cls(**config_data)

