"""
Branch component implementation for the Haive graph system.

This module defines the Branch class which represents a decision point in a graph
that routes execution based on state conditions.
"""

import logging
import uuid
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
)

from langgraph.graph import END
from langgraph.types import Command, Send
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.branches.types import BranchMode, BranchResult, ComparisonType
from haive.core.graph.common.field_utils import extract_field, get_last_message_content
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, StateLike

# Set up logging
logger = logging.getLogger(__name__)

# Type variables for generic parameters
T = TypeVar("T", bound=StateLike)
C = TypeVar("C", bound=ConfigLike)
O = TypeVar("O", bound=NodeOutput)


class Branch(BaseModel, Generic[T, C, O]):
    """
    Unified branch for dynamic routing based on state values.

    This class provides conditional routing functionality in a graph,
    handling different routing modes and comparison types.
    """

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")

    # Connection information
    source_node: Optional[str] = Field(default=None)
    destinations: Dict[Union[bool, str], str] = Field(
        default_factory=lambda: {True: "continue", False: END}
    )
    default: Optional[str] = None

    # Evaluation properties
    mode: BranchMode = BranchMode.DIRECT
    key: Optional[str] = None
    value: Any = None
    comparison: Union[ComparisonType, str] = ComparisonType.EQUALS
    function: Optional[Callable] = None
    function_ref: Optional[CallableReference] = None
    allow_none: bool = False
    message_key: str = "messages"

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def setup_function_and_mappings(self) -> "Branch":
        """Set up function and mappings after initialization."""
        # Resolve function from reference
        if self.function_ref and not self.function:
            resolved = self.function_ref.resolve()
            # Handle case where resolve returns a tuple
            if isinstance(resolved, tuple) and len(resolved) > 0:
                self.function = resolved[0]  # Extract the function from the tuple
            else:
                self.function = resolved

        return self

    @model_validator(mode="after")
    def validate_destinations_and_default(self) -> "Branch":
        """Validate destinations and set default if needed."""
        if not self.destinations:
            self.destinations = {True: "continue", False: END}
        elif len(self.destinations) == 1:
            # If only one destination is provided, map it to True and END to False
            true_dest = next(iter(self.destinations.values()))
            self.destinations = {True: true_dest, False: END}
        # Only set default if we have multiple destinations and no default is set
        # AND we don't have a list of destinations (True/False mapping)
        elif (
            len(self.destinations) > 1
            and self.default is None
            and not all(isinstance(k, bool) for k in self.destinations.keys())
        ):
            self.default = END
        return self

    def __call__(self, state: T, config: Optional[C] = None) -> O:
        """
        Make Branch directly callable for use in conditional edges.

        Args:
            state: Current state
            config: Optional configuration

        Returns:
            Routing result (node name, Send object, Command, etc.)
        """
        try:
            # Regular branch evaluation
            result = self.evaluate(state)

            # Handle None results
            if result is None:
                # If we have a default destination, use it
                if self.default is not None:
                    return self.default
                # Otherwise use False route
                if False in self.destinations:
                    return self.destinations[False]
                # Last resort - return first destination
                if self.destinations:
                    return next(iter(self.destinations.values()))
                return END  # Final fallback

            return result

        except Exception as e:
            logger.error(f"Error in branch evaluation: {e}")
            # Same fallback logic as None
            if self.default is not None:
                return self.default
            if False in self.destinations:
                return self.destinations[False]
            if self.destinations:
                return next(iter(self.destinations.values()))
            return END

    def evaluate(self, state: T) -> O:
        """
        Evaluate the branch against state.

        Args:
            state: Current state

        Returns:
            Routing result
        """
        try:
            # Handle different branch modes
            if self.mode == BranchMode.FUNCTION and self.function:
                return self._process_result(self.function(state), state)

            # Default direct mode
            # Handle existence checks
            if self.comparison in [ComparisonType.EXISTS, ComparisonType.NOT_EXISTS]:
                result = self._check_exists(state)
                if self.comparison == ComparisonType.NOT_EXISTS:
                    result = not result
                return self._get_destination(result)

            # Handle special message comparisons
            elif self.comparison == ComparisonType.MESSAGE_CONTAINS:
                result = self._check_message_contains(state)
                return self._get_destination(result)

            # Regular field comparison
            else:
                field_value = extract_field(state, self.key)

                # Handle None values
                if field_value is None and not self.allow_none:
                    logger.warning(
                        f"Field '{self.key}' is None and allow_none is False"
                    )
                    return self.default

                # Perform comparison
                result = self._compare(field_value)
                return self._get_destination(result)

        except Exception as e:
            logger.error(f"Error evaluating branch: {e}")
            return self.default

    def _process_result(self, result: Any, state: T) -> O:
        """Process the result of a branch evaluation."""
        # Handle Send objects
        if isinstance(result, Send):
            return BranchResult(send_objects=[result])

        # Handle lists of Send objects
        if isinstance(result, list) and all(isinstance(item, Send) for item in result):
            return BranchResult(send_objects=result)

        # Handle Command objects
        if isinstance(result, Command):
            return BranchResult(command_object=result)

        # Handle regular destination lookup
        if isinstance(result, (bool, str)):
            return self._get_destination(result)

        # Fallback
        return self.default

    def _get_destination(self, result: Union[bool, str]) -> str:
        """Get destination from result."""
        return self.destinations.get(result, self.default)

    def _compare(self, value: Any) -> bool:
        """Compare a value against the branch's value."""
        comparison = self.comparison
        target = self.value

        # Convert string comparison to enum if needed
        if isinstance(comparison, str):
            try:
                comparison = ComparisonType(comparison)
            except ValueError:
                # Use as string if not an enum value
                pass

        # Perform comparison
        if comparison == ComparisonType.EQUALS:
            return value == target
        elif comparison == ComparisonType.NOT_EQUALS:
            return value != target
        elif comparison == ComparisonType.GREATER_THAN:
            return value > target
        elif comparison == ComparisonType.LESS_THAN:
            return value < target
        elif comparison == ComparisonType.GREATER_EQUALS:
            return value >= target
        elif comparison == ComparisonType.LESS_EQUALS:
            return value <= target
        elif comparison == ComparisonType.IN:
            return value in target if isinstance(target, (list, tuple, set)) else False
        elif comparison == ComparisonType.CONTAINS:
            if isinstance(value, str):
                return target in value
            if isinstance(value, (list, tuple, set, dict)):
                return target in value
            return False
        elif comparison == ComparisonType.IS:
            return value is target
        elif comparison == ComparisonType.IS_NOT:
            return value is not target
        elif comparison == ComparisonType.MATCHES:
            import re

            if isinstance(value, str) and isinstance(target, str):
                return bool(re.match(target, value))
            return False
        elif comparison == ComparisonType.STARTS_WITH:
            return value.startswith(target) if isinstance(value, str) else False
        elif comparison == ComparisonType.ENDS_WITH:
            return value.endswith(target) if isinstance(value, str) else False

        logger.warning(f"Unknown comparison type: {comparison}")
        return False

    def _check_exists(self, state: T) -> bool:
        """Check if a field exists in state."""
        if self.key is None:
            return False

        value = extract_field(state, self.key)
        return value is not None

    def _check_message_contains(self, state: T) -> bool:
        """Check if the last message contains specified text."""
        content = get_last_message_content(state, self.message_key)
        if not content or not isinstance(content, str):
            return False

        # Check if content contains value
        return self.value in content
