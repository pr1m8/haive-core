"""
Branch system for dynamic routing based on state values.

This module provides the Branch class and various factory methods
for creating branches for different routing scenarios.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Union

from langgraph.types import Send

from haive.core.graph.branches.branch import Branch, BranchConfig
from haive.core.graph.branches.dynamic import DynamicMappingConfig
from haive.core.graph.branches.send_mapping import (
    SendGenerator,
    SendMapping,
    SendMappingList,
)
from haive.core.graph.branches.types import (
    BranchMode,
    BranchResultModel,
    ComparisonType,
)

# Import common utilities we want to re-export
from haive.core.graph.common import extract_field, get_field_value

# Re-export key classes
__all__ = [
    "Branch",
    "BranchConfig",
    "ComparisonType",
    "BranchMode",
    "BranchResultModel",
    "SendMapping",
    "SendGenerator",
    "SendMappingList",
    "extract_field",
    "get_field_value",
]

# Factory functions for common branch types


def key_equals(
    key: str, value: Any, true_dest: str = "continue", false_dest: str = "END"
) -> Branch:
    """Create a Branch that checks if a key equals a value."""
    return Branch(
        key=key,
        value=value,
        comparison=ComparisonType.EQUALS,
        destinations={True: true_dest, False: false_dest},
    )


def key_exists(
    key: str, true_dest: str = "continue", false_dest: str = "END"
) -> Branch:
    """Create a Branch that checks if a key exists in the state."""
    return Branch(
        key=key,
        comparison=ComparisonType.EXISTS,
        destinations={True: true_dest, False: false_dest},
    )


def from_function(
    function: Callable[[Any], Union[bool, str]],
    destinations: Optional[Dict[Union[bool, str], str]] = None,
    default: str = "END",
) -> Branch:
    """Create a Branch from a function."""
    return Branch(
        function=function,
        destinations=destinations,
        default=default,
        mode=BranchMode.FUNCTION,
    )


def chain(*branches: Branch, default: str = "END") -> Branch:
    """Chain multiple branches together, evaluating them in sequence."""
    branch = Branch(mode=BranchMode.CHAIN, default=default)
    branch._chain_branches = list(branches)
    return branch


def conditional(
    condition: Callable[[Any], bool],
    if_true: Union[str, Branch],
    if_false: Union[str, Branch],
    default: str = "END",
) -> Branch:
    """Create a Branch with conditional evaluation."""
    branch = Branch(mode=BranchMode.CONDITION, default=default)
    branch._condition = condition
    branch._true_branch = if_true
    branch._false_branch = if_false
    return branch


def message_contains(
    text: str,
    true_dest: str = "continue",
    false_dest: str = "END",
    message_key: str = "messages",
) -> Branch:
    """Create a Branch that checks if the last message contains specific text."""
    return Branch(
        value=text,
        comparison=ComparisonType.MESSAGE_CONTAINS,
        destinations={True: true_dest, False: false_dest},
        message_key=message_key,
    )


def send_mapper(
    function: Optional[Callable[[Any], List[Send]]] = None,
    mappings: Optional[List[SendMapping]] = None,
    generators: Optional[List[SendGenerator]] = None,
) -> Branch:
    """
    Create a Branch that generates Send objects.

    Args:
        function: Function that takes state and returns Send objects
        mappings: List of SendMapping configurations
        generators: List of SendGenerator configurations

    Returns:
        Branch configured as a Send mapper
    """
    branch = Branch(function=function, mode=BranchMode.SEND_MAPPER)

    # Set up send mapping list
    branch.send_mapping_list = SendMappingList(
        mappings=mappings or [], generators=generators or []
    )

    return branch


def create_from_send_function(mapper_function: Callable[[Any], List[Send]]) -> Branch:
    """
    Create a Send mapper branch from a function that returns Send objects.

    This is a convenience method for the common map_summaries pattern.

    Args:
        mapper_function: Function that returns list of Send objects

    Returns:
        Branch configured as a Send mapper
    """
    return send_mapper(function=mapper_function)
