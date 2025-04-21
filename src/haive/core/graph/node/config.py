# src/haive_core/graph/node/config.py

from typing import Dict, Optional, Union, Any, List, Callable, Literal, Type, Tuple
from pydantic import BaseModel, Field, model_validator
from langgraph.graph import END
from langgraph.types import Command, Send
from langchain_core.runnables import RunnableConfig

from haive.core.engine.base import Engine

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
    
    # Control flow options
    command_goto: Optional[Union[str, Literal["END"], Send, List[Union[Send, str]]]] = Field(
        default=None,
        description="Next node to go to after this node (or END)"
    )
    
    # Mapping options
    input_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from state keys to engine input keys (e.g., {'query': 'input', 'context': 'documents'})"
    )
    output_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapping from engine output keys to state keys (e.g., {'output': 'result', 'confidence': 'score'})"
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
    
    # Special handling flags
    use_direct_messages: Optional[bool] = Field(
        default=None, 
        description="Force direct usage of messages field if present (auto-detected if None)"
    )
    
    extract_content: Optional[bool] = Field(
        default=None,
        description="Extract content from messages and make it available as 'content' in state"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this node"
    )
    
    # Debug mode
    debug: bool = Field(
        default=False,
        description="Enable debug logging for this node"
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
            from haive.core.engine.base import EngineRegistry, EngineType
            registry = EngineRegistry.get_instance()
            
        # Try each engine type
        for engine_type in registry.engines:
            engine = registry.get(engine_type, engine_name)
            if engine:
                # Update engine reference
                self.engine = engine
                
                # Extract engine ID if available
                engine_id = None
                if hasattr(engine, "id"):
                    engine_id = getattr(engine, "id")
                    self.engine_id = engine_id
                
                return engine, engine_id
                
        # Not found - return as is
        return self.engine, None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a serializable dictionary.
        
        Returns:
            Dictionary representation of the node config
        """
        # Use model_dump or dict based on Pydantic version
        if hasattr(self, "model_dump"):
            # Pydantic v2
            data = self.model_dump(exclude={"engine"})
        else:
            # Pydantic v1
            data = self.dict(exclude={"engine"})
        
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
        elif self.engine is not None:
            # For callables, store a reference if possible
            if hasattr(self.engine, "__name__"):
                data["engine_ref"] = f"function:{self.engine.__name__}"
            else:
                data["engine_ref"] = "callable"
        
        # Special handling for END in command_goto
        if self.command_goto is END:
            data["command_goto"] = "END"
        
        return data
    
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
        # Create a copy of the data to avoid modifying the input
        config_data = data.copy()
        
        # Handle engine references
        if "engine_ref" in config_data:
            ref = config_data.pop("engine_ref")
            
            if isinstance(ref, dict) and "id" in ref and "name" in ref:
                # Try to look up by ID first, then by name
                engine = None
                
                if registry is None:
                    from haive.core.engine.base import EngineRegistry
                    registry = EngineRegistry.get_instance()
                
                # Try to find by ID if available
                if ref["id"]:
                    engine = registry.find(ref["id"])
                
                # Fall back to name lookup
                if engine is None and ref["name"]:
                    engine = registry.find(ref["name"])
                
                if engine:
                    config_data["engine"] = engine
                    config_data["engine_id"] = ref["id"]
                else:
                    # Fall back to name as string
                    config_data["engine"] = ref["name"]
            elif isinstance(ref, str):
                # Handle function references or engine names
                if ref.startswith("function:"):
                    # We can't easily resolve function references here
                    # This would need application-specific handling
                    config_data["engine"] = None
                else:
                    config_data["engine"] = ref
            
        # Handle special values
        if "command_goto" in config_data and config_data["command_goto"] == "END":
            config_data["command_goto"] = END
            
        # Create the NodeConfig
        return cls(**config_data)