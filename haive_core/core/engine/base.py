from enum import Enum
from typing import Dict, Any, Optional, Union, Type, ClassVar, List
from pydantic import BaseModel, Field
import uuid
import logging

logger = logging.getLogger(__name__)

class EngineType(str, Enum):
    """Types of engines the system can use."""
    LLM = "llm"
    VECTOR_STORE = "vector_store"
    RETRIEVER = "retriever"
    TOOL = "tool"
    EMBEDDINGS = "embeddings"
    DOCUMENT_LOADER = "document_loader"
    DOCUMENT_TRANSFORMER = "document_transformer"
    AGENT = "agent"

class Engine(BaseModel):
    """Base class for all engine configurations."""
    name: str = Field(default_factory=lambda: f"engine_{uuid.uuid4().hex[:8]}", description="Unique name for this engine")
    engine_type: EngineType = Field(description="Type of engine")
    
    def create_runnable(self):
        """Create a runnable instance from this engine config."""
        raise NotImplementedError("Subclasses must implement this method")
    
    def register(self):
        """Register this engine in the global registry."""
        return EngineRegistry.get_instance().register(self)
    
    @classmethod
    def get(cls, name: str):
        """Get an engine of this type from the registry."""
        registry = EngineRegistry.get_instance()
        return registry.get(cls.engine_type, name)
    
    @classmethod
    def list(cls):
        """List all registered engines of this type."""
        registry = EngineRegistry.get_instance()
        return registry.list(cls.engine_type)

class EngineRegistry:
    """Central registry for all engines in the system."""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.engines = {engine_type: {} for engine_type in EngineType}
    
    def register(self, engine):
        """Register an engine."""
        self.engines[engine.engine_type][engine.name] = engine
        return engine
    
    def get(self, engine_type, name):
        """Get an engine by type and name."""
        return self.engines[engine_type].get(name)
    
    def list(self, engine_type):
        """List all engines of a type."""
        return list(self.engines[engine_type].keys())
    
    def get_all(self, engine_type):
        """Get all engines of a type."""
        return self.engines[engine_type]