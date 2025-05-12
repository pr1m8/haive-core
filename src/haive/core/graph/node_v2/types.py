from enum import Enum
from types import FunctionType
from typing import Dict

from git import Optional
from pydantic import BaseModel, Field, computed_field, field_validator

from haive.core.engine.base import Engine, EngineType


class NodeType(Enum):
    """Enum representing the type of a node."""
    # Predefined
    START = "start"
    END = "end"
    # Core
    ROUTE = "route"
    FUNCTION = "function"
    ENGINE = "engine"
    # Tool & Validation
    TOOL='tool'
    VALIDATION='validation'
    # Other
    INTERRUPT = "interrupt"
    
class Node(BaseModel):
    name: str = Field(description="The name of the node.")
    node_type: NodeType = Field(description="The type of the node.")

    @field_validator('name',mode='after')
    def ensure_valid_name(cls):
        if cls.name =='':
            raise ValueError("Name cannot be empty")
        elif cls.name=='__start__':
            raise ValueError("Name cannot be '__start__'")
        elif cls.name=='__end__':
            raise ValueError("Name cannot be '__end__'")
        return cls.name
    
    #@classmethod
    
    #node_config: 
    #engine_type: Optional[EngineType] = Field(description="The type of engine to use for the node.")
    
    #@field_validator('engine_type',mode='after')
    #def ensure_engine_type(cls):
    #    if cls.node_type == NodeType.ENGINE:    
    #        return cls.engine_type
    #    return None
class EngineNode(Node):
    node_type: NodeType = NodeType.ENGINE
    engine_type: EngineType
    engine: Engine
    
    
    @computed_field
    @property
    def name(self)->str:
        return f"{self.engine_type.value}_{self.name}"
    
    @computed_field
    @property 
    def id(self)->str:
        return f"{self.engine_type.value}_{self.id}"


class ValidationNode(Node):
    node_type: NodeType = NodeType.VALIDATION
    validation_type: ValidationType
    validation_message: str
    
    
class FunctionNode(Node):
    node_type: NodeType = NodeType.FUNCTION
    function_type: FunctionType
    function_message: str
    
class ToolNode(Node):
    node_type: NodeType = NodeType.TOOL
    tool_type: ToolType
    tool_message: str
    
    
    
    

   
    
class InterruptNode(Node):
    node_type: NodeType = NodeType.INTERRUPT
    
    interrupt_message: str
    
    
    def 