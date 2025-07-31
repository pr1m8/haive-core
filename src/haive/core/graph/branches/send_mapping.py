"""Send mapping functionality for routing and state transformation."""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from haive.core.graph.common.field_utils import extract_field

# Import from common utilities
from haive.core.graph.common.types import StateLike

logger = logging.getLogger(__name__)


class SendMapping(BaseModel):
    """Mapping configuration for generating Send objects."""

    node: str = Field(..., description="Target node name")
    fields: dict[str, str] = Field(
        default_factory=dict, description="Field mapping from state to Send arg"
    )
    condition: str | None = Field(
        None, description="Optional condition expression to evaluate"
    )
    transform: dict[str, Callable] | None = Field(
        None, description="Transformations to apply to fields"
    )

    model_config = {"arbitrary_types_allowed": True}

    def create_send(self, state: StateLike) -> Any | None:
        """Create a Send object from state using this mapping."""
        from langgraph.types import Send

        # Check condition if specified
        if self.condition:
            try:
                # Simple condition check - could be expanded to full expression
                # evaluation
                condition_value = extract_field(state, self.condition)
                if not condition_value:
                    return None
            except Exception as e:
                logger.exception(f"Error evaluating condition {self.condition}: {e}")
                return None

        # Extract and transform values for Send
        arg = {}
        for target_field, source_field in self.fields.items():
            # Extract value
            value = extract_field(state, source_field)

            # Apply transformation if specified
            if self.transform and target_field in self.transform:
                try:
                    value = self.transform[target_field](value)
                except Exception as e:
                    logger.exception(f"Error applying transform to {target_field}: {e}")

            arg[target_field] = value

        return Send(self.node, arg)


class SendGenerator(BaseModel):
    """Generator for Send objects based on lists or collections."""

    target_node: str = Field(..., description="Target node name")
    collection_field: str = Field(
        ..., description="State field containing the collection"
    )
    item_field: str = Field(
        "item", description="Field name for the item in the send object"
    )
    filter_function: Callable[[Any], bool] | None = Field(
        None, description="Function to filter items"
    )

    model_config = {"arbitrary_types_allowed": True}

    def create_sends(self, state: StateLike) -> list[Any]:
        """Create multiple Send objects from a collection in state."""
        from langgraph.types import Send

        # Extract collection
        collection = extract_field(state, self.collection_field)
        if not collection or not isinstance(collection, list | tuple | set):
            return []

        sends = []

        # Process each item
        for item in collection:
            # Apply filter if specified
            if self.filter_function and not self.filter_function(item):
                continue

            # Create Send object for this item
            sends.append(Send(self.target_node, {self.item_field: item}))

        return sends


class SendMappingList(BaseModel):
    """Collection of send mappings."""

    mappings: list[SendMapping] = Field(default_factory=list)
    generators: list[SendGenerator] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def create_sends(self, state: StateLike) -> list[Any]:
        """Apply all mappings and generators to state."""
        sends = []

        # Apply standard mappings
        for mapping in self.mappings:
            send_obj = mapping.create_send(state)
            if send_obj:
                sends.append(send_obj)

        # Apply generators
        for generator in self.generators:
            sends.extend(generator.create_sends(state))

        return sends
