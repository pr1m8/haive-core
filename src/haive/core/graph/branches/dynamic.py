"""
Dynamic output mapping based on state.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

# Import from common utilities
from haive.core.graph.common import CallableReference, extract_field

from .types import ComparisonType


class OutputMapping(BaseModel):
    """Output mapping configuration."""

    field_mappings: Dict[str, str] = Field(default_factory=dict)
    condition: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class DynamicMappingConfig(BaseModel):
    """Configuration for dynamic output mapping."""

    mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    function_ref: Optional[CallableReference] = None
    key: Optional[str] = None
    value: Optional[Any] = None
    comparison: ComparisonType = ComparisonType.EQUALS
    default_node: str = "END"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_mapping(self, state: Any) -> Tuple[str, Optional[Dict[str, str]]]:
        """
        Determine which mapping to use based on state.

        Args:
            state: State object

        Returns:
            Tuple of (node_name, output_mapping)
        """
        # Determine next node
        next_node = self.default_node

        # Use function if available
        if self.function_ref:
            func = self.function_ref.resolve()
            if func:
                try:
                    result = func(state)
                    if isinstance(result, str):
                        next_node = result
                except Exception as e:
                    pass
        # Otherwise use key/value comparison
        elif self.key:
            field_value = extract_field(state, self.key)
            if field_value is not None:
                # Perform comparison
                if (
                    self.comparison == ComparisonType.EQUALS
                    and field_value == self.value
                ):
                    next_node = "true_dest"
                elif (
                    self.comparison == ComparisonType.NOT_EQUALS
                    and field_value != self.value
                ):
                    next_node = "true_dest"
                # Add more comparisons as needed

        # Get mapping for this node
        mapping = self.mappings.get(next_node)

        return next_node, mapping
