from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.haive.core.graph.routers.conditions import RouteCondition
from typing import Callable, List
class RoutingConfig(BaseModel):
    """Configuration for node routing behavior."""
    default_destination: str = Field(..., description="Default destination if no other routing applies")
    condition_map: Dict[Any, str] = Field(
        default_factory=dict, description="Map of condition values to destinations"
    )
    condition_function: Optional[Callable[[Dict[str, Any]], Any]] = Field(
        default=None, description="Function that determines routing"
    )
    allowed_destinations: List[str] = Field(
        default_factory=list, description="List of allowed destinations (for validation)"
    )
    is_dynamic: bool = Field(default=False, description="Whether routing is determined at runtime")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional routing metadata")

class Route(BaseModel):
    """Defines a routing path with conditions."""
    name: str = Field(..., description="Name of the route")
    condition: RouteCondition = Field(..., description="Condition that triggers this route")
    destination: str = Field(..., description="Destination node or END")
    description: Optional[str] = Field(default=None, description="Description of the route")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
