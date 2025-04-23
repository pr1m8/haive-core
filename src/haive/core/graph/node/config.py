# src/haive/core/graph/node/config.py
"""
Configuration for a node in a graph.
TODO: Fix up the conisistelycn in aug llm and llm. 
Returns:
    _type_: _description_
"""
from typing import Dict, Optional, Union, Any, List, Callable, Literal, Type, Tuple
from pydantic import BaseModel, Field, model_validator
from langgraph.graph import END
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig
import asyncio
import inspect
import json

from haive.core.engine.base import Engine
from haive.core.registry.base import AbstractRegistry

class NodeConfig(BaseModel):
    """
    Configuration for a node in a graph.
    
    NodeConfig provides a standardized way to configure nodes in a graph,
    handling both engine-based nodes and callable functions.
    """
    name: str = Field(description="Name of the node")
    engine: Optional[Union[Engine, str, Callable]] = Field(
        default=None, 
        description="Engine, engine name, or callable function to use for this node"
    )
    
    # Unique engine identifier - used for targeting in runnable_config
    engine_id: Optional[str] = Field(
        default=None,
        description="Unique ID of the engine instance (auto-populated when resolving)"
    )
    
    # Control flow options with full Command/Send support
    command_goto: Optional[Union[str, Literal["END"], Send, List[Union[Send, str]]]] = Field(
        default=None,
        description="Next node to go to after this node (or END, Send object, or list of Send objects)"
    )
    
    # Mapping options
    input_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from state keys to engine input keys"
    )
    output_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from engine output keys to state keys"
    )
    
    # Runtime configuration
    runnable_config: Optional[RunnableConfig] = Field(
        default=None,
        description="Runtime configuration for this node"
    )
    
    # Configuration overrides at the node level
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Engine configuration overrides specific to this node"
    )
    
    # Node type information
    node_type: Optional[str] = Field(
        default=None,
        description="Type of node function (auto-detected if None)"
    )
    async_mode: Optional[bool] = Field(
    default=None,
    description="Whether to operate in async mode (auto-detected if None)"
    )
    
    # Special handling flags
    use_direct_messages: Optional[bool] = Field(
        default=None, 
        description="Whether to use messages field directly (auto-detected if None)"
    )
    
    extract_content: bool = Field(
        default=False,
        description="Extract content from messages and add as 'content' field"
    )
    
    preserve_state: bool = Field(
        default=True,
        description="Preserve state fields not affected by output mapping"
    )
    
    # Registry reference (not serialized)
    registry: Optional[AbstractRegistry] = Field(
        default=None,
        exclude=True
    )
    
    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug logging for this node"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this node"
    )
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @model_validator(mode='after')
    def validate_config(self):
        """Validate and normalize the configuration."""
        # Convert "END" string to END constant
        if self.command_goto == "END":
            self.command_goto = END
            
        # Auto-populate engine_id if an engine with ID is provided
        if self.engine and isinstance(self.engine, Engine) and self.engine_id is None:
            if hasattr(self.engine, "id"):
                self.engine_id = getattr(self.engine, "id")
                
        # Auto-detect use_direct_messages if None
        if self.use_direct_messages is None:
            self.use_direct_messages = self._detect_uses_messages()
            
        return self
    
    def _detect_uses_messages(self) -> bool:
        """Detect if this node configuration likely uses messages directly."""
        # If it's an LLM engine, check if it might use messages
        if isinstance(self.engine, Engine) and hasattr(self.engine, "engine_type"):
            engine_type = getattr(self.engine, "engine_type")
            if engine_type and getattr(engine_type, "value", "") == "llm":
                # Check for uses_messages_field attribute
                if hasattr(self.engine, "uses_messages_field"):
                    return getattr(self.engine, "uses_messages_field")
                    
                # Check if it's an AugLLMConfig
                if self.engine.__class__.__name__ == "AugLLMConfig":
                    return True
                    
                # Default to True for LLMs
                return True
        
        # For non-LLM engines, defer to input mapping
        if self.input_mapping:
            # Check if any state key maps to 'messages'
            return any(input_key == "messages" for _, input_key in self.input_mapping.items())
            
        # Default behavior - not using direct messages
        return False
    
    def resolve_engine(self, registry=None) -> Tuple[Any, Optional[str]]:
        """
        Resolve engine reference to actual engine and its ID.
        
        Args:
            registry: Optional registry to use for lookup
            
        Returns:
            Tuple of (resolved engine, engine_id)
        """
        # Already resolved to a non-string (Engine, Callable, etc.)
        if not isinstance(self.engine, str):
            engine_id = None
            # Extract engine ID if possible
            if isinstance(self.engine, Engine) and hasattr(self.engine, "id"):
                engine_id = getattr(self.engine, "id")
                self.engine_id = engine_id
            
            return self.engine, engine_id
            
        # Try to lookup in registry
        engine_name = self.engine
        
        if registry is None:
            registry = self.registry
            
        if registry is None:
            from haive.core.engine.base import EngineRegistry
            registry = EngineRegistry.get_instance()
            
        # Try to find engine by name or ID
        engine = registry.find_by_id(engine_name)
        if engine:
            # Update engine reference
            self.engine = engine
            
            # Extract engine ID if available
            engine_id = None
            if hasattr(engine, "id"):
                engine_id = getattr(engine, "id")
                self.engine_id = engine_id
            
            return engine, engine_id
                
        # Try other lookup methods if find_by_id didn't work
        engine = registry.find(engine_name) if hasattr(registry, "find") else None
        if engine:
            self.engine = engine
            self.engine_id = getattr(engine, "id", None)
            return engine, self.engine_id
                
        # Not found - return as is
        return self.engine, None
    
    def determine_node_type(self) -> str:
        """
        Determine the most appropriate node type based on engine.
        
        Returns:
            Node type string
        """
        if self.node_type:
            return self.node_type
            
        engine = self.engine
        
        # Handle async mode explicitly
        if self.async_mode:
            # Check for AsyncInvokable
            if hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke")):
                return "async_invokable"
            
            # Check for async functions
            if asyncio.iscoroutinefunction(engine):
                return "async"
        
        # Standard detection logic
        if asyncio.iscoroutinefunction(engine):
            return "async"
            
        # Check for AsyncInvokable
        if hasattr(engine, "ainvoke") and callable(getattr(engine, "ainvoke")):
            return "async_invokable"
            
        # Check for Invokable
        if hasattr(engine, "invoke") and callable(getattr(engine, "invoke")):
            return "invokable"
            
        # Check for mapping functions (based on signature return annotation)
        if callable(engine) and hasattr(engine, "__annotations__"):
            if "return" in engine.__annotations__:
                return_type = engine.__annotations__["return"]
                if "List[Send]" in str(return_type) or "list[Send]" in str(return_type):
                    return "mapping"
        
        # Default to "callable" for callable functions
        if callable(engine):
            return "callable"
            
        # Generic for everything else
        return "generic"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.
        
        Returns:
            Dictionary representation of the node config
        """
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"engine", "registry"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"engine", "registry"})
        
        # Handle engine serialization
        if isinstance(self.engine, Engine):
            # Just store the engine ID and name for reference
            data["engine_ref"] = {
                "id": getattr(self.engine, "id", None),
                "name": getattr(self.engine, "name", None),
                "engine_type": getattr(self.engine, "engine_type", None)
            }
        elif isinstance(self.engine, str):
            data["engine_ref"] = self.engine
        elif callable(self.engine):
            # For callables, store a reference if possible
            if hasattr(self.engine, "__name__"):
                module = getattr(self.engine, "__module__", "")
                data["engine_ref"] = f"function:{module}.{self.engine.__name__}"
            else:
                data["engine_ref"] = "callable"
        
        # Special handling for END in command_goto
        if self.command_goto is END:
            data["command_goto"] = "END"
        elif isinstance(self.command_goto, Send):
            # Handle Send objects
            data["command_goto"] = {
                "type": "send",
                "node": self.command_goto.node,
                "arg": self._serialize_send_arg(self.command_goto.arg)
            }
        elif isinstance(self.command_goto, list) and any(isinstance(item, Send) for item in self.command_goto):
            # Handle list containing Send objects
            data["command_goto"] = [
                {"type": "send", "node": item.node, "arg": self._serialize_send_arg(item.arg)} 
                if isinstance(item, Send) else item
                for item in self.command_goto
            ]
        
        return data
    
    def _serialize_send_arg(self, arg: Any) -> Any:
        """Serialize Send argument to ensure it's JSON serializable."""
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry=None) -> 'NodeConfig':
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
        
        # Handle engine references
        if "engine_ref" in config_data:
            ref = config_data.pop("engine_ref")
            
            if isinstance(ref, dict) and "id" in ref and "name" in ref:
                # Engine reference with ID and name
                engine = None
                
                # Try to find the engine if registry provided
                if registry:
                    # Try ID first
                    if ref["id"]:
                        engine = registry.find_by_id(ref["id"])
                    
                    # Try name next
                    if engine is None and ref["name"] and hasattr(registry, "find"):
                        engine = registry.find(ref["name"])
                
                if engine:
                    config_data["engine"] = engine
                    config_data["engine_id"] = ref["id"]
                else:
                    # Store name reference
                    config_data["engine"] = ref["name"] 
                
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
                    except (ImportError, AttributeError, ValueError):
                        # Can't resolve - store as string
                        config_data["engine"] = ref
                else:
                    # Engine name or ID
                    config_data["engine"] = ref
        
        # Handle command_goto serialization
        if "command_goto" in config_data:
            goto = config_data["command_goto"]
            
            if goto == "END":
                config_data["command_goto"] = END
            elif isinstance(goto, dict) and goto.get("type") == "send":
                # Reconstruct Send object
                from langgraph.types import Send
                config_data["command_goto"] = Send(goto["node"], goto["arg"])
            elif isinstance(goto, list):
                # Handle list containing Send dictionaries
                from langgraph.types import Send
                config_data["command_goto"] = [
                    Send(item["node"], item["arg"]) 
                    if isinstance(item, dict) and item.get("type") == "send" 
                    else item
                    for item in goto
                ]
        
        # Create the NodeConfig
        result = cls(**config_data)
        
        # Set registry
        result.registry = registry
        
        return result