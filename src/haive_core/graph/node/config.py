# src/haive/core/graph/node/config.py

import inspect
import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Tuple, Union

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Send
from pydantic import BaseModel, Field, ValidationError, model_validator

from haive_core.engine.base import Engine

# Configure colored logging
try:
    import colorlog

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s[%(name)s]%(reset)s %(white)s%(message)s%(reset)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
    )
    
    logger = colorlog.getLogger(__name__)
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    
except ImportError:
    # Fallback to standard logging if colorlog is not installed
    logging.basicConfig(
        format='%(levelname)-8s [%(name)s] %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.warning("colorlog not installed. Using standard logging. Install with: pip install colorlog")


class NodeConfig(BaseModel):
    """Configuration for a node in a graph.
    
    NodeConfig provides a standardized way to configure nodes in a graph,
    handling both engine-based nodes and callable functions.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this node configuration")
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
    command_goto: Optional[Union[str, Literal["END"], Send, list[Union[Send, str]]]] = Field(
        default=None,
        description="Next node to go to after this node (or END)"
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

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for this node"
    )
    
    # Debug flag
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode for this node"
    )
    
    # Validation and logging details (renamed without leading underscores)
    created_at: datetime = Field(default_factory=datetime.now, description="When this configuration was created")
    validation_history: Dict[str, Any] = Field(default_factory=dict, description="History of validation events")
    resolution_history: list[Dict[str, Any]] = Field(default_factory=list, description="History of engine resolution attempts")

    model_config = {
        "arbitrary_types_allowed": True
    }

    @model_validator(mode="after")
    def validate_config(self):
        """Validate and normalize the configuration.
        
        Performs comprehensive validation of the node configuration and logs
        any issues that might cause problems during execution.
        """
        validation_event = {
            "timestamp": datetime.now().isoformat(),
            "issues": [],
            "success": True
        }
        
        try:
            # Convert "END" string to END constant
            if self.command_goto == "END":
                logger.debug(f"Converting 'END' string to END constant for node {self.name}")
                self.command_goto = END
            
            # Check for potential issues
            
            # 1. Validate engine field
            if self.engine is None:
                logger.warning(f"Node {self.name} has no engine specified")
                validation_event["issues"].append({"type": "warning", "message": "No engine specified"})
            elif isinstance(self.engine, str):
                logger.debug(f"Node {self.name} uses string reference to engine: {self.engine}")
            elif not isinstance(self.engine, (Engine, Callable)):
                warning_msg = f"Node {self.name} has engine of unexpected type: {type(self.engine)}"
                logger.warning(warning_msg)
                validation_event["issues"].append({"type": "warning", "message": warning_msg})
            
            # 2. Check input mapping consistency
            if self.input_mapping:
                for state_key, input_key in self.input_mapping.items():
                    if not isinstance(state_key, str) or not isinstance(input_key, str):
                        warning_msg = f"Non-string keys in input_mapping: {state_key} -> {input_key}"
                        logger.warning(warning_msg)
                        validation_event["issues"].append({"type": "warning", "message": warning_msg})
            
            # 3. Check output mapping consistency
            if self.output_mapping:
                for output_key, state_key in self.output_mapping.items():
                    if not isinstance(output_key, str) or not isinstance(state_key, str):
                        warning_msg = f"Non-string keys in output_mapping: {output_key} -> {state_key}"
                        logger.warning(warning_msg)
                        validation_event["issues"].append({"type": "warning", "message": warning_msg})
            
            # 4. Check for runnable_config structure issues
            if self.runnable_config:
                if not isinstance(self.runnable_config, dict):
                    warning_msg = f"runnable_config is not a dictionary: {type(self.runnable_config)}"
                    logger.warning(warning_msg)
                    validation_event["issues"].append({"type": "warning", "message": warning_msg})
                elif "configurable" not in self.runnable_config:
                    warning_msg = "runnable_config is missing 'configurable' key"
                    logger.warning(warning_msg)
                    validation_event["issues"].append({"type": "warning", "message": warning_msg})
            
            # 5. Check config_overrides for serializable values
            if self.config_overrides:
                try:
                    # Test serialization to catch potential issues
                    json.dumps(self.config_overrides)
                except (TypeError, OverflowError) as e:
                    warning_msg = f"config_overrides contains non-serializable values: {str(e)}"
                    logger.warning(warning_msg)
                    validation_event["issues"].append({"type": "warning", "message": warning_msg})
            
            # Auto-populate engine_id if an engine with ID is provided
            if self.engine and isinstance(self.engine, Engine) and self.engine_id is None:
                if hasattr(self.engine, "id") and getattr(self.engine, "id") is not None:
                    self.engine_id = getattr(self.engine, "id")
                    logger.debug(f"Auto-populated engine_id: {self.engine_id} for node {self.name}")
            
            # Add node_id to metadata for traceability if not present
            if "node_id" not in self.metadata:
                self.metadata["node_id"] = self.id
                
            # Record validation success or failure
            if validation_event["issues"]:
                validation_event["success"] = False
                logger.warning(f"Node {self.name} has {len(validation_event['issues'])} validation issues")
            else:
                logger.debug(f"Node {self.name} configuration validated successfully")
                
            self.validation_history = validation_event
                
        except Exception as e:
            error_msg = f"Error during node configuration validation: {str(e)}"
            logger.error(error_msg)
            validation_event["success"] = False
            validation_event["issues"].append({"type": "error", "message": error_msg})
            self.validation_history = validation_event

        return self

    def resolve_engine(self, registry=None, retry=True) -> Tuple[Any, Optional[str]]:
        """Resolve engine reference to actual engine and its ID.
        
        Args:
            registry: Optional registry to use for lookup
            retry: Whether to retry with different search strategies if initial lookup fails
            
        Returns:
            Tuple of (resolved engine, engine_id)
        """
        resolution_event = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "engine_type": None,
            "engine_id": None,
            "search_path": []
        }
        
        start_time = datetime.now()
        
        try:
            # Already resolved to a non-string (Engine, Callable, etc.)
            if not isinstance(self.engine, str):
                engine_id = None
                
                # Extract engine type for logging
                engine_type = type(self.engine).__name__
                resolution_event["engine_type"] = engine_type
                
                # Extract engine ID if possible
                if isinstance(self.engine, Engine) and hasattr(self.engine, "id"):
                    engine_id = getattr(self.engine, "id")
                    if engine_id:
                        self.engine_id = engine_id
                        resolution_event["engine_id"] = engine_id
                        logger.debug(f"Using existing engine of type {engine_type} with ID {engine_id} for node {self.name}")
                    else:
                        logger.debug(f"Using existing engine of type {engine_type} with no ID for node {self.name}")
                elif callable(self.engine) and not isinstance(self.engine, type):
                    func_name = getattr(self.engine, "__name__", str(self.engine))
                    logger.debug(f"Using callable function '{func_name}' for node {self.name}")
                else:
                    logger.debug(f"Using object of type {engine_type} for node {self.name}")
                
                resolution_event["success"] = True
                self.resolution_history.append(resolution_event)
                return self.engine, engine_id
            
            # Need to resolve string reference
            engine_name = self.engine
            resolution_event["search_path"].append(f"string:{engine_name}")
            logger.debug(f"Resolving engine reference '{engine_name}' for node {self.name}")
            
            if registry is None:
                from haive_core.engine.base import EngineRegistry
                registry = EngineRegistry.get_instance()
            
            # First attempt: direct lookup by name for each engine type
            engine = None
            for engine_type in registry.engines:
                resolution_event["search_path"].append(f"registry:{engine_type}:{engine_name}")
                engine = registry.get(engine_type, engine_name)
                if engine:
                    # Update engine reference
                    self.engine = engine
                    
                    # Log success
                    engine_type_name = engine_type.name if hasattr(engine_type, "name") else str(engine_type)
                    logger.info(f"Resolved engine '{engine_name}' as {engine_type_name} for node {self.name}")
                    resolution_event["engine_type"] = engine_type_name
                    
                    # Extract engine ID if available
                    engine_id = None
                    if hasattr(engine, "id"):
                        engine_id = getattr(engine, "id")
                        if engine_id:
                            self.engine_id = engine_id
                            resolution_event["engine_id"] = engine_id
                            logger.debug(f"Found engine ID: {engine_id}")
                    
                    resolution_event["success"] = True
                    
                    # Record resolution time
                    resolution_time = (datetime.now() - start_time).total_seconds() * 1000
                    resolution_event["resolution_time_ms"] = resolution_time
                    
                    # Store resolution event
                    self.resolution_history.append(resolution_event)
                    
                    return engine, engine_id
            
            # Second attempt: try registry.find method if available
            if retry and hasattr(registry, "find"):
                resolution_event["search_path"].append(f"registry.find:{engine_name}")
                engine = registry.find(engine_name)
                if engine:
                    # Update engine reference
                    self.engine = engine
                    
                    # Log success
                    engine_type = getattr(engine, "engine_type", "unknown")
                    logger.info(f"Resolved engine '{engine_name}' via registry.find as {engine_type} for node {self.name}")
                    resolution_event["engine_type"] = str(engine_type)
                    
                    # Extract engine ID if available
                    engine_id = None
                    if hasattr(engine, "id"):
                        engine_id = getattr(engine, "id")
                        if engine_id:
                            self.engine_id = engine_id
                            resolution_event["engine_id"] = engine_id
                            logger.debug(f"Found engine ID: {engine_id}")
                    
                    resolution_event["success"] = True
                    
                    # Record resolution time
                    resolution_time = (datetime.now() - start_time).total_seconds() * 1000
                    resolution_event["resolution_time_ms"] = resolution_time
                    
                    # Store resolution event
                    self.resolution_history.append(resolution_event)
                    
                    return engine, engine_id
            
            # Not found in registry
            logger.warning(f"Could not resolve engine '{engine_name}' for node {self.name}")
            resolution_event["search_path"].append("not_found")
            
            # Record resolution time even for failure
            resolution_time = (datetime.now() - start_time).total_seconds() * 1000
            resolution_event["resolution_time_ms"] = resolution_time
            
            # Store resolution event
            self.resolution_history.append(resolution_event)
            
            return self.engine, None
            
        except Exception as e:
            # Log error
            logger.error(f"Error resolving engine for node {self.name}: {str(e)}")
            
            # Record resolution failure
            resolution_event["error"] = str(e)
            resolution_time = (datetime.now() - start_time).total_seconds() * 1000
            resolution_event["resolution_time_ms"] = resolution_time
            
            # Store resolution event
            self.resolution_history.append(resolution_event)
            
            return self.engine, None

        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary.
        
        Returns:
            Dictionary representation of the node config
        """
        # Use model_dump in Pydantic v2
        data = self.model_dump(exclude={"engine", "created_at", "validation_history", "resolution_history"})
        
        # Add metadata about serialization
        data["serialized_at"] = datetime.now().isoformat()
        
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
            if callable(self.engine) and hasattr(self.engine, "__name__"):
                if hasattr(self.engine, "__module__") and self.engine.__module__ != "__main__":
                    data["engine_ref"] = f"function:{self.engine.__module__}.{self.engine.__name__}"
                else:
                    data["engine_ref"] = f"function:{self.engine.__name__}"
            else:
                data["engine_ref"] = f"callable:{type(self.engine).__name__}"
        
        # Special handling for END in command_goto
        if self.command_goto is END:
            data["command_goto"] = "END"
        
        # Add validation status
        if self.validation_history:
            validation_status = "valid" if self.validation_history.get("success", False) else "issues"
            issue_count = len(self.validation_history.get("issues", []))
            data["validation_status"] = f"{validation_status}:{issue_count}"
        
        # Add resolution status
        if self.resolution_history:
            last_resolution = self.resolution_history[-1]
            resolution_status = "resolved" if last_resolution.get("success", False) else "unresolved"
            data["resolution_status"] = resolution_status
        
        # Preserve only original metadata keys provided by the user
        # (remove node_id that was auto-added during initialization)
        if "metadata" in data and "node_id" in data["metadata"]:
            # Create a copy to avoid modifying the original
            metadata_copy = data["metadata"].copy()
            if "node_id" in metadata_copy:
                del metadata_copy["node_id"]
            data["metadata"] = metadata_copy
        
        return data

    def get_debug_info(self) -> Dict[str, Any]:
        """Get detailed debug information about this node configuration.
        
        Returns:
            Dictionary with comprehensive debug information
        """
        debug_info = {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "engine_info": self._get_engine_info(),
            "validation": self.validation_history,
            "resolution_history": self.resolution_history,
            "mappings": {
                "input": self.input_mapping,
                "output": self.output_mapping
            },
            "config": {
                "has_runnable_config": self.runnable_config is not None,
                "overrides": self.config_overrides
            },
            "control_flow": {
                "command_goto": "END" if self.command_goto is END else self.command_goto
            },
            "metadata": self.metadata,
            "debug_mode": self.debug_mode
        }
        
        return debug_info
    
    def _get_engine_info(self) -> Dict[str, Any]:
        """Get information about the configured engine.
        
        Returns:
            Dictionary with engine information
        """
        if self.engine is None:
            return {"type": "none"}
        
        if isinstance(self.engine, str):
            return {
                "type": "reference",
                "reference": self.engine
            }
        
        if isinstance(self.engine, Engine):
            engine_info = {
                "type": "engine",
                "engine_type": getattr(self.engine, "engine_type", "unknown"),
                "name": getattr(self.engine, "name", "unnamed"),
                "id": getattr(self.engine, "id", None)
            }
            
            # Add module info if available
            if hasattr(self.engine, "__module__"):
                engine_info["module"] = self.engine.__module__
            
            return engine_info
        
        if callable(self.engine) and not isinstance(self.engine, type):
            callable_info = {
                "type": "callable",
                "name": getattr(self.engine, "__name__", "unnamed")
            }
            
            # Add module info if available
            if hasattr(self.engine, "__module__"):
                callable_info["module"] = self.engine.__module__
            
            # Add signature if available
            try:
                sig = inspect.signature(self.engine)
                callable_info["signature"] = str(sig)
                callable_info["parameters"] = list(sig.parameters.keys())
            except Exception:
                pass
            
            return callable_info
        
        # Generic object
        return {
            "type": "object",
            "class": type(self.engine).__name__
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], registry=None) -> "NodeConfig":
        """Create a NodeConfig from a dictionary representation.
        
        Args:
            data: Dictionary representation
            registry: Optional registry for engine lookup
            
        Returns:
            Instantiated NodeConfig
        """
        # Create a copy of the data to avoid modifying the input
        config_data = data.copy()
        
        # Remove metadata fields
        for field in ["serialized_at", "validation_status", "resolution_status"]:
            if field in config_data:
                del config_data[field]
        
        try:
            # Handle engine references
            if "engine_ref" in config_data:
                ref = config_data.pop("engine_ref")
                
                if isinstance(ref, dict) and "id" in ref and "name" in ref:
                    # Try to look up by ID first, then by name
                    engine = None
                    
                    if registry is None:
                        from haive_core.engine.base import EngineRegistry
                        registry = EngineRegistry.get_instance()
                    
                    # Try to find by ID if available
                    if ref["id"]:
                        logger.debug(f"Looking up engine by ID: {ref['id']}")
                        engine = registry.find(ref["id"])
                    
                    # Fall back to name lookup
                    if engine is None and ref["name"]:
                        logger.debug(f"Looking up engine by name: {ref['name']}")
                        engine = registry.find(ref["name"])
                    
                    if engine:
                        logger.debug(f"Found engine: {engine}")
                        config_data["engine"] = engine
                        config_data["engine_id"] = ref["id"]
                    else:
                        # Fall back to name as string
                        logger.warning(f"Could not find engine with ID {ref['id']} or name {ref['name']}")
                        config_data["engine"] = ref["name"]
                elif isinstance(ref, str):
                    # Handle function references or engine names
                    if ref.startswith("function:"):
                        # Try to resolve function reference
                        function_path = ref.replace("function:", "")
                        logger.debug(f"Trying to resolve function reference: {function_path}")
                        
                        try:
                            module_path, func_name = function_path.rsplit('.', 1)
                            module = __import__(module_path, fromlist=[func_name])
                            function = getattr(module, func_name)
                            config_data["engine"] = function
                            logger.debug(f"Successfully resolved function reference: {function_path}")
                        except Exception as e:
                            logger.warning(f"Could not resolve function reference {function_path}: {e}")
                            config_data["engine"] = None
                    else:
                        # Engine name
                        config_data["engine"] = ref
                        logger.debug(f"Using engine name: {ref}")
            
            # Handle special values
            if "command_goto" in config_data and config_data["command_goto"] == "END":
                config_data["command_goto"] = END
                logger.debug("Converting 'END' string to END constant")
            
            # Create the NodeConfig
            instance = cls(**config_data)
            logger.debug(f"Created NodeConfig for {instance.name} from dictionary")
            return instance
            
        except ValidationError as e:
            logger.error(f"Validation error creating NodeConfig from dictionary: {e}")
            
            # Try to extract name for error context
            node_name = data.get("name", "unknown")
            
            # Create a basic config with minimal information
            logger.warning(f"Creating minimal NodeConfig for {node_name} due to validation error")
            return cls(
                name=node_name,
                engine=None,
                metadata={"error": str(e), "original_data": data}
            )
        except Exception as e:
            logger.error(f"Error creating NodeConfig from dictionary: {e}")
            
            # Try to extract name for error context
            node_name = data.get("name", "unknown")
            
            # Create a basic config with minimal information
            logger.warning(f"Creating minimal NodeConfig for {node_name} due to error")
            return cls(
                name=node_name,
                engine=None,
                metadata={"error": str(e), "original_data": data}
            )