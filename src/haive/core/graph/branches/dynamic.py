"""Dynamic output mapping based on state."""

import logging
from typing import Any, Self

from langgraph.graph import END
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.branches.types import ComparisonType
from haive.core.graph.common.field_utils import extract_field
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import StateLike

logger = logging.getLogger(__name__)


class OutputMapping(BaseModel):
    """Output mapping configuration."""

    field_mappings: dict[str, str] = Field(default_factory=dict)
    condition: str | None = None
    model_config = {"arbitrary_types_allowed": True}


class DynamicMapping(BaseModel):
    """Configuration for dynamic output mapping."""

    mappings: dict[str, dict[str, str]] = Field(default_factory=dict)
    function_ref: CallableReference | None = None
    key: str | None = None
    value: Any | None = None
    comparison: ComparisonType = ComparisonType.EQUALS
    default_node: str = END
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_mappings(self) -> Self:
        """Validate Mappings.

        Returns:
            [TODO: Add return description]
        """
        for _key, mapping_data in self.mappings.items():
            if "mapping" in mapping_data and isinstance(mapping_data["mapping"], dict):
                pass
        return self

    def get_mapping(self, state: StateLike) -> tuple[str, dict[str, str] | None]:
        """Determine which mapping to use based on state.

        Args:
            state: State object

        Returns:
            Tuple of (node_name, output_mapping)
        """
        next_node = self.default_node
        if self.function_ref:
            func = self.function_ref.resolve()
            if func:
                try:
                    result = func(state)
                    if isinstance(result, str):
                        next_node = result
                except Exception as e:
                    logger.exception(f"Error in dynamic mapping function: {e}")
        elif self.key:
            field_value = extract_field(state, self.key)
            if field_value is not None:
                comparison_result = False
                if self.comparison == ComparisonType.EQUALS:
                    comparison_result = field_value == self.value
                elif self.comparison == ComparisonType.NOT_EQUALS:
                    comparison_result = field_value != self.value
                elif self.comparison == ComparisonType.GREATER_THAN:
                    comparison_result = field_value > self.value
                elif self.comparison == ComparisonType.LESS_THAN:
                    comparison_result = field_value < self.value
                elif self.comparison == ComparisonType.GREATER_EQUALS:
                    comparison_result = field_value >= self.value
                elif self.comparison == ComparisonType.LESS_EQUALS:
                    comparison_result = field_value <= self.value
                if comparison_result:
                    if "high_score_route" in self.mappings:
                        next_node = "high_score_route"
                    else:
                        non_default_routes = [
                            k for k in self.mappings if k != self.default_node
                        ]
                        if non_default_routes:
                            next_node = non_default_routes[0]
        mapping = self.mappings.get(next_node)
        return (next_node, mapping)
