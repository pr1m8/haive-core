"""
Core Branch implementation for dynamic routing.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, cast

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict, Field

# Import from common utilities
from haive.core.graph.common import (
    CallableReference,
    extract_base_field,
    extract_field,
    extract_fields_from_function,
    get_last_message_content,
)

from .dynamic import DynamicMappingConfig
from .send_mapping import SendGenerator, SendMapping, SendMappingList
from .types import BranchMode, BranchResultModel, ComparisonType

logger = logging.getLogger(__name__)


class BranchConfig(BaseModel):
    """
    Configuration for a Branch.

    This serves as a serializable representation of a Branch.
    """

    key: Optional[str] = None
    value: Any = None
    comparison: Union[ComparisonType, str] = ComparisonType.EQUALS
    function_ref: Optional[CallableReference] = None
    destinations: Dict[Union[bool, str], str] = Field(
        default_factory=lambda: {True: "continue", False: "END"}
    )
    default: str = "END"
    allow_none: bool = False
    message_key: str = "messages"
    mode: BranchMode = BranchMode.DIRECT

    # For Send operations
    send_mappings: List[SendMapping] = Field(default_factory=list)
    send_generators: List[SendGenerator] = Field(default_factory=list)

    # For dynamic mapping
    dynamic_mapping: Optional[DynamicMappingConfig] = None

    # For chain branches
    sub_branches: List["BranchConfig"] = Field(default_factory=list)

    # For condition branches
    condition_ref: Optional[CallableReference] = None
    true_branch: Optional[Union[str, "BranchConfig"]] = None
    false_branch: Optional[Union[str, "BranchConfig"]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_branch(self) -> "Branch":
        """Convert config to a Branch instance."""
        # Create base branch
        branch = Branch(
            key=self.key,
            value=self.value,
            comparison=self.comparison,
            destinations=self.destinations,
            default=self.default,
            allow_none=self.allow_none,
            message_key=self.message_key,
            mode=self.mode,
        )

        # Resolve function
        if self.function_ref:
            branch.function = self.function_ref.resolve()

        # Set Send mappings
        branch.send_mapping_list = SendMappingList(
            mappings=self.send_mappings, generators=self.send_generators
        )

        # Set dynamic mapping
        branch.dynamic_mapping = self.dynamic_mapping

        # Set sub branches for chain
        if self.sub_branches:
            branch._chain_branches = [sb.to_branch() for sb in self.sub_branches]

        # Set condition
        if self.condition_ref:
            branch._condition = self.condition_ref.resolve()

            # Set true branch
            if isinstance(self.true_branch, BranchConfig):
                branch._true_branch = self.true_branch.to_branch()
            else:
                branch._true_branch = self.true_branch

            # Set false branch
            if isinstance(self.false_branch, BranchConfig):
                branch._false_branch = self.false_branch.to_branch()
            else:
                branch._false_branch = self.false_branch

        return branch

    @classmethod
    def from_branch(cls, branch: "Branch") -> "BranchConfig":
        """Create config from a Branch instance."""
        config = cls(
            key=branch.key,
            value=branch.value,
            comparison=branch.comparison,
            destinations=branch.destinations,
            default=branch.default,
            allow_none=branch.allow_none,
            message_key=branch.message_key,
            mode=branch.mode,
        )

        # Convert function to reference
        if branch.function:
            config.function_ref = CallableReference.from_callable(branch.function)

        # Add Send mappings
        if hasattr(branch, "send_mapping_list"):
            config.send_mappings = branch.send_mapping_list.mappings
            config.send_generators = branch.send_mapping_list.generators

        # Add dynamic mapping
        if hasattr(branch, "dynamic_mapping") and branch.dynamic_mapping:
            config.dynamic_mapping = branch.dynamic_mapping

        # Add sub branches for chain
        if hasattr(branch, "_chain_branches") and branch._chain_branches:
            config.sub_branches = [cls.from_branch(sb) for sb in branch._chain_branches]

        # Add condition
        if hasattr(branch, "_condition") and branch._condition:
            config.condition_ref = CallableReference.from_callable(branch._condition)

            # Add true branch
            if hasattr(branch, "_true_branch"):
                if isinstance(branch._true_branch, Branch):
                    config.true_branch = cls.from_branch(branch._true_branch)
                else:
                    config.true_branch = branch._true_branch

            # Add false branch
            if hasattr(branch, "_false_branch"):
                if isinstance(branch._false_branch, Branch):
                    config.false_branch = cls.from_branch(branch._false_branch)
                else:
                    config.false_branch = branch._false_branch

        return config


class Branch:
    """
    Enhanced branch for dynamic routing based on state values.
    """

    def __init__(
        self,
        key: Optional[str] = None,
        value: Any = None,
        comparison: Union[ComparisonType, str] = ComparisonType.EQUALS,
        function: Optional[Callable[[Any], Any]] = None,
        destinations: Optional[Dict[Union[bool, str], str]] = None,
        default: str = "END",
        allow_none: bool = False,
        message_key: str = "messages",
        evaluator: Optional[Callable] = None,
        mode: BranchMode = BranchMode.DIRECT,
    ):
        """
        Initialize a branch.

        Args:
            key: State key to check
            value: Value to compare against
            comparison: Comparison operation
            function: Callable that returns bool or route key
            destinations: Mapping of result to destination node
            default: Default destination
            allow_none: Whether to allow None values
            message_key: Key for accessing messages in state
            evaluator: Direct evaluator function
            mode: Branch evaluation mode
        """
        self.key = key
        self.value = value
        self.comparison = comparison
        self.function = function
        self.destinations = destinations or {True: "continue", False: "END"}
        self.default = default
        self.allow_none = allow_none
        self.message_key = message_key
        self._evaluator = evaluator
        self.mode = mode

        # For Send operations
        self.send_mapping_list = SendMappingList()

        # For dynamic mapping
        self.dynamic_mapping = None

        # For chain branches
        self._chain_branches = []

        # For condition branches
        self._condition = None
        self._true_branch = None
        self._false_branch = None

    def __call__(self, state: Any, config: Optional[Any] = None) -> Any:
        """
        Make Branch directly callable for use in conditional edges.

        Args:
            state: State object
            config: Optional runtime configuration

        Returns:
            Node output (Command, Send, string, etc.)
        """
        result = self.evaluate(state)

        if isinstance(result, BranchResultModel):
            # Handle Send objects
            if result.is_send:
                if len(result.send_objects) == 1:
                    return result.send_objects[0]
                return result.send_objects

            # Handle Command
            if result.is_command:
                return result.command_object

            # Handle dynamic mapping with next node
            if result.has_mapping and result.next_node:
                # Create command with goto and dynamic mapping in extra
                return Command(
                    goto=result.next_node,
                    update={"output_mapping": result.output_mapping},
                )

            # Handle simple next node
            if result.next_node:
                return result.next_node

        # Fallback for string result
        if isinstance(result, str):
            return result

        # Final fallback
        return self.default

    def evaluate(self, state: Any) -> Union[str, BranchResultModel]:
        """
        Evaluate the branch against state.

        Args:
            state: State object

        Returns:
            Either a string node name or BranchResultModel
        """
        try:
            # Use custom evaluator if provided
            if self._evaluator:
                return self._process_result(self._evaluator(state), state)

            # Handle different branch modes
            if self.mode == BranchMode.FUNCTION and self.function:
                return self._process_result(self.function(state), state)

            elif self.mode == BranchMode.CHAIN and self._chain_branches:
                return self._evaluate_chain(state)

            elif self.mode == BranchMode.CONDITION and self._condition:
                return self._evaluate_condition(state)

            elif self.mode == BranchMode.SEND_MAPPER:
                return self._evaluate_send_mapper(state)

            elif self.mode == BranchMode.DYNAMIC and self.dynamic_mapping:
                return self._evaluate_dynamic(state)

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

    def _process_result(self, result: Any, state: Any) -> Union[str, BranchResultModel]:
        """Process the result of a branch evaluation."""
        # Handle Send objects
        if isinstance(result, Send):
            return BranchResultModel(send_objects=[result])

        # Handle lists of Send objects
        if isinstance(result, list) and all(isinstance(item, Send) for item in result):
            return BranchResultModel(send_objects=result)

        # Handle Command objects
        if isinstance(result, Command):
            return BranchResultModel(command_object=result)

        # Handle dictionaries with special keys
        if isinstance(result, dict):
            if "_output_mapping" in result:
                # Handle dynamic output mapping
                mapping = result.pop("_output_mapping")
                next_node = result.pop("_next_node", self.default)
                return BranchResultModel(next_node=next_node, output_mapping=mapping)

            # Handle dictionaries with send objects
            if "_send_objects" in result:
                send_objects = result["_send_objects"]
                if isinstance(send_objects, list) and all(
                    isinstance(item, Send) for item in send_objects
                ):
                    return BranchResultModel(send_objects=send_objects)

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
        elif comparison == ComparisonType.HAS_LENGTH:
            try:
                return len(value) == target
            except (TypeError, AttributeError):
                return False

        logger.warning(f"Unknown comparison type: {comparison}")
        return False

    def _check_exists(self, state: Any) -> bool:
        """Check if a field exists in state."""
        if self.key is None:
            return False

        value = extract_field(state, self.key)
        return value is not None

    def _check_message_contains(self, state: Any) -> bool:
        """Check if the last message contains specified text."""
        content = get_last_message_content(state, self.message_key)
        if not content or not isinstance(content, str):
            return False

        # Check if content contains value
        return self.value in content

    def _evaluate_chain(self, state: Any) -> Union[str, BranchResultModel]:
        """Evaluate a chain of branches."""
        for branch in self._chain_branches:
            result = branch.evaluate(state)

            # If result is BranchResultModel with content, return it
            if isinstance(result, BranchResultModel) and (
                result.is_send or result.is_command or result.has_mapping
            ):
                return result

            # If string result is not the branch's default, return it
            if isinstance(result, str) and result != branch.default:
                return result

        # If all branches returned default, use this branch's default
        return self.default

    def _evaluate_condition(self, state: Any) -> Union[str, BranchResultModel]:
        """Evaluate a conditional branch."""
        if not self._condition:
            return self.default

        try:
            condition_result = self._condition(state)

            if condition_result:
                if isinstance(self._true_branch, Branch):
                    return self._true_branch.evaluate(state)
                return self._true_branch or self.default
            else:
                if isinstance(self._false_branch, Branch):
                    return self._false_branch.evaluate(state)
                return self._false_branch or self.default

        except Exception as e:
            logger.error(f"Error in condition evaluation: {e}")
            return self.default

    def _evaluate_send_mapper(self, state: Any) -> BranchResultModel:
        """Evaluate a mapper branch that creates Send objects."""
        sends = []

        # Use send_mapping_list
        if self.send_mapping_list:
            sends.extend(self.send_mapping_list.create_sends(state))

        # Use function if available and no sends yet
        if not sends and self.function:
            try:
                result = self.function(state)

                # Handle list of Send objects
                if isinstance(result, list):
                    sends.extend([obj for obj in result if isinstance(obj, Send)])

                # Handle single Send object
                elif isinstance(result, Send):
                    sends.append(result)

            except Exception as e:
                logger.error(f"Error in mapper function: {e}")

        return BranchResultModel(send_objects=sends)

    def _evaluate_dynamic(self, state: Any) -> BranchResultModel:
        """Evaluate a dynamic branch that selects output mapping based on state."""
        if not self.dynamic_mapping:
            return BranchResultModel(next_node=self.default)

        # Get mapping
        next_node, output_mapping = self.dynamic_mapping.get_mapping(state)

        return BranchResultModel(next_node=next_node, output_mapping=output_mapping)

    def extract_field_references(self) -> Set[str]:
        """Extract field references used by this branch."""
        fields = set()

        # Add direct key reference
        if self.key:
            fields.add(extract_base_field(self.key))

        # Add message key if used
        if self.comparison == ComparisonType.MESSAGE_CONTAINS:
            fields.add(self.message_key)

        # Add fields from send mappings
        if hasattr(self, "send_mapping_list"):
            for mapping in self.send_mapping_list.mappings:
                if mapping.condition:
                    fields.add(extract_base_field(mapping.condition))
                for source_field in mapping.fields.values():
                    fields.add(extract_base_field(source_field))

            for generator in self.send_mapping_list.generators:
                fields.add(extract_base_field(generator.collection_field))

        # Extract fields from function
        if self.function:
            fields.update(extract_fields_from_function(self.function))

        # Include fields from chain branches
        for branch in self._chain_branches:
            fields.update(branch.extract_field_references())

        return fields

    def to_config(self) -> BranchConfig:
        """Convert to a BranchConfig for serialization."""
        return BranchConfig.from_branch(self)

    @classmethod
    def from_config(cls, config: BranchConfig) -> "Branch":
        """Create Branch from BranchConfig."""
        return config.to_branch()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        config = self.to_config()
        return config.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Branch":
        """Create from a dictionary representation."""
        config = BranchConfig.model_validate(data)
        return cls.from_config(config)
