"""Core Branch implementation for dynamic routing.

This module provides a unified Branch class that replaces the need for
separate ConditionalEdge objects, consolidating all routing logic.
"""

import logging
import uuid
from collections.abc import Callable
from typing import Any, Self, Union

from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field, model_validator

from haive.core.graph.branches.dynamic import DynamicMapping
from haive.core.graph.branches.send_mapping import (
    SendGenerator,
    SendMapping,
    SendMappingList,
)
from haive.core.graph.branches.types import BranchMode, BranchResult, ComparisonType
from haive.core.graph.common.field_utils import (
    extract_base_field,
    extract_field,
    get_last_message_content,
)
from haive.core.graph.common.references import CallableReference
from haive.core.graph.common.types import ConfigLike, NodeOutput, StateLike

logger = logging.getLogger(__name__)


class Branch(BaseModel):
    """Unified branch for dynamic routing based on state values.

    This class combines the functionality of branch routing and conditional edges,
    providing a single unified interface for graph routing.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default_factory=lambda: f"branch_{uuid.uuid4().hex[:8]}")
    source_node: str | None = Field(default=None)
    destinations: dict[bool | str, str] = Field(
        default_factory=lambda: {True: "continue", False: END}
    )
    default: str | None = None
    mode: BranchMode = BranchMode.DIRECT
    key: str | None = None
    value: Any = None
    comparison: ComparisonType | str = ComparisonType.EQUALS
    function: Callable | None = None
    function_ref: CallableReference | None = None
    allow_none: bool = False
    message_key: str = "messages"
    send_mappings: list[SendMapping] = Field(default_factory=list)
    send_generators: list[SendGenerator] = Field(default_factory=list)
    dynamic_mapping: DynamicMapping | None = None
    chain_branches: list["Branch"] = Field(default_factory=list)
    condition_ref: CallableReference | None = None
    true_branch: Union[str, "Branch"] | None = None
    false_branch: Union[str, "Branch"] | None = None
    send_mapping_list: SendMappingList | None = None
    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def setup_function_and_mappings(self) -> Self:
        """Set up function and mappings after initialization."""
        if self.function_ref and (not self.function):
            resolved = self.function_ref.resolve()
            if isinstance(resolved, tuple) and len(resolved) > 0:
                self.function = resolved[0]
            else:
                self.function = resolved
        if not self.send_mapping_list and (self.send_mappings or self.send_generators):
            self.send_mapping_list = SendMappingList(
                mappings=self.send_mappings, generators=self.send_generators
            )
        return self

    @model_validator(mode="after")
    def validate_destinations_and_default(self) -> Self:
        """Validate destinations and set default if needed."""
        if not self.destinations:
            self.destinations = {True: "continue", False: END}
        elif len(self.destinations) == 1:
            single_key = next(iter(self.destinations.keys()))
            if isinstance(single_key, bool) or single_key in ["continue", "default"]:
                true_dest = next(iter(self.destinations.values()))
                self.destinations = {True: true_dest, False: END}
        elif (
            len(self.destinations) > 1
            and self.default is None
            and (not all(isinstance(k, bool) for k in self.destinations))
        ):
            self.default = END
        return self

    def __call__(
        self, state: StateLike, config: ConfigLike | None = None
    ) -> NodeOutput:
        """Make Branch directly callable for use in conditional edges."""
        try:
            if self.mode == BranchMode.DYNAMIC and self.dynamic_mapping:
                next_node, output_mapping = self.dynamic_mapping.get_mapping(state)
                return Command(
                    goto=next_node,
                    update={"output_mapping": output_mapping} if output_mapping else {},
                )
            result = self.evaluate(state)
            if result is None:
                if self.default is not None:
                    return self.default
                if False in self.destinations:
                    return self.destinations[False]
                if self.destinations:
                    return next(iter(self.destinations.values()))
                return END
            return result
        except NameError as ne:
            error_msg = str(ne).lower()
            if "model" not in error_msg:
                logger.exception(f"NameError in branch evaluation: {ne}")
            if self.default is not None:
                return self.default
            if False in self.destinations:
                return self.destinations[False]
            if self.destinations:
                return next(iter(self.destinations.values()))
            return END
        except Exception as e:
            logger.exception(f"Error in branch evaluation: {e}")
            if self.default is not None:
                return self.default
            if False in self.destinations:
                return self.destinations[False]
            if self.destinations:
                return next(iter(self.destinations.values()))
            return END

    def evaluator(self, state: dict[str, Any]) -> str:
        """Evaluate the branch condition and return the next node.

        Args:
            state: Current graph state

        Returns:
            Name of the next node
        """
        try:
            result = self.function(state)
            if result in self.destinations:
                return self.destinations[result]
            if self.default is not None:
                return self.default
            if self.destinations:
                return next(iter(self.destinations.values()))
            return ""
        except NameError as ne:
            error_msg = str(ne).lower()
            if "model" not in error_msg:
                pass
            return self.default if self.default else ""
        except Exception:
            return self.default if self.default else ""

    def evaluate(self, state: StateLike) -> NodeOutput:
        """Evaluate the branch against state."""
        try:
            if self.mode == BranchMode.FUNCTION and self.function:
                return self._process_result(self.function(state), state)
            if self.mode == BranchMode.CHAIN and self.chain_branches:
                return self._evaluate_chain(state)
            if self.mode == BranchMode.CONDITION and self.condition_ref:
                return self._evaluate_condition(state)
            if self.mode == BranchMode.SEND_MAPPER:
                return self._evaluate_send_mapper(state)
            if self.mode == BranchMode.DYNAMIC and self.dynamic_mapping:
                return self._evaluate_dynamic(state)
            if self.comparison in [ComparisonType.EXISTS, ComparisonType.NOT_EXISTS]:
                result = self._check_exists(state)
                if self.comparison == ComparisonType.NOT_EXISTS:
                    result = not result
                return self._get_destination(result)
            if self.comparison == ComparisonType.MESSAGE_CONTAINS:
                result = self._check_message_contains(state)
                return self._get_destination(result)
            field_value = extract_field(state, self.key)
            if field_value is None and (not self.allow_none):
                logger.warning(f"Field '{self.key}' is None and allow_none is False")
                return self.default
            result = self._compare(field_value)
            return self._get_destination(result)
        except NameError as ne:
            error_msg = str(ne).lower()
            if "model" not in error_msg:
                logger.exception(f"NameError in branch evaluation: {ne}")
            return self.default
        except Exception as e:
            logger.exception(f"Error evaluating branch: {e}")
            return self.default

    def _process_result(self, result: Any, state: StateLike) -> NodeOutput:
        """Process the result of a branch evaluation."""
        from langgraph.types import Command

        if isinstance(result, Send):
            return BranchResult(send_objects=[result])
        if isinstance(result, list) and all(isinstance(item, Send) for item in result):
            return BranchResult(send_objects=result)
        if isinstance(result, Command):
            return BranchResult(command_object=result)
        if isinstance(result, dict):
            if "_output_mapping" in result:
                mapping = result.pop("_output_mapping")
                next_node = result.pop("_next_node", self.default)
                return BranchResult(next_node=next_node, output_mapping=mapping)
            if "_send_objects" in result:
                send_objects = result["_send_objects"]
                if isinstance(send_objects, list) and all(
                    isinstance(item, Send) for item in send_objects
                ):
                    return BranchResult(send_objects=send_objects)
        if isinstance(result, bool | str):
            return self._get_destination(result)
        return self.default

    def _get_destination(self, result: bool | str) -> str:
        """Get destination from result."""
        return self.destinations.get(result, self.default)

    def _compare(self, value: Any) -> bool:
        """Compare a value against the branch's value."""
        comparison = self.comparison
        target = self.value
        if isinstance(comparison, str):
            try:
                comparison = ComparisonType(comparison)
            except ValueError:
                pass
        if comparison == ComparisonType.EQUALS:
            return value == target
        if comparison == ComparisonType.NOT_EQUALS:
            return value != target
        if comparison == ComparisonType.GREATER_THAN:
            return value > target
        if comparison == ComparisonType.LESS_THAN:
            return value < target
        if comparison == ComparisonType.GREATER_EQUALS:
            return value >= target
        if comparison == ComparisonType.LESS_EQUALS:
            return value <= target
        if comparison == ComparisonType.IN:
            return value in target if isinstance(target, list | tuple | set) else False
        if comparison == ComparisonType.CONTAINS:
            if isinstance(value, str):
                return target in value
            if isinstance(value, list | tuple | set | dict):
                return target in value
            return False
        if comparison == ComparisonType.IS:
            return value is target
        if comparison == ComparisonType.IS_NOT:
            return value is not target
        if comparison == ComparisonType.MATCHES:
            import re

            if isinstance(value, str) and isinstance(target, str):
                return bool(re.match(target, value))
            return False
        if comparison == ComparisonType.STARTS_WITH:
            return value.startswith(target) if isinstance(value, str) else False
        if comparison == ComparisonType.ENDS_WITH:
            return value.endswith(target) if isinstance(value, str) else False
        if comparison == ComparisonType.HAS_LENGTH:
            try:
                return len(value) == target
            except (TypeError, AttributeError):
                return False
        logger.warning(f"Unknown comparison type: {comparison}")
        return False

    def _check_exists(self, state: StateLike) -> bool:
        """Check if a field exists in state."""
        if self.key is None:
            return False
        value = extract_field(state, self.key)
        return value is not None

    def _check_message_contains(self, state: StateLike) -> bool:
        """Check if the last message contains specified text."""
        content = get_last_message_content(state, self.message_key)
        if not content or not isinstance(content, str):
            return False
        return self.value in content

    def _evaluate_chain(self, state: StateLike) -> NodeOutput:
        """Evaluate a chain of branches."""
        for branch in self.chain_branches:
            result = branch.evaluate(state)
            if isinstance(result, BranchResult) and (
                result.is_send or result.is_command or result.has_mapping
            ):
                return result
            if isinstance(result, str) and result != branch.default:
                return result
        return self.default

    def _evaluate_condition(self, state: StateLike) -> NodeOutput:
        """Evaluate a conditional branch."""
        if not self.condition_ref:
            return self.default
        try:
            condition_func = self.condition_ref.resolve()
            if not condition_func:
                return self.default
            condition_result = condition_func(state)
            if condition_result:
                if isinstance(self.true_branch, Branch):
                    return self.true_branch.evaluate(state)
                return self.true_branch or self.default
            if isinstance(self.false_branch, Branch):
                return self.false_branch.evaluate(state)
            return self.false_branch or self.default
        except Exception as e:
            logger.exception(f"Error in condition evaluation: {e}")
            return self.default

    def _evaluate_send_mapper(self, state: StateLike) -> BranchResult:
        """Evaluate a mapper branch that creates Send objects."""
        sends = []
        if self.send_mapping_list:
            sends.extend(self.send_mapping_list.create_sends(state))
        if not sends and self.function:
            try:
                result = self.function(state)
                if isinstance(result, list):
                    from langgraph.types import Send

                    sends.extend([obj for obj in result if isinstance(obj, Send)])
                elif hasattr(result, "node") and hasattr(result, "arg"):
                    sends.append(result)
            except Exception as e:
                logger.exception(f"Error in mapper function: {e}")
        return BranchResult(send_objects=sends)

    def _evaluate_dynamic(self, state: StateLike) -> Any:
        """Evaluate a dynamic branch that selects output mapping based on state."""
        from langgraph.types import Command

        if not self.dynamic_mapping:
            return Command(goto=self.default)
        next_node, output_mapping = self.dynamic_mapping.get_mapping(state)
        return Command(
            goto=next_node,
            update={"output_mapping": output_mapping} if output_mapping else {},
        )

    def extract_field_references(self) -> set[str]:
        """Extract field references used by this branch."""
        fields = set()
        if self.key:
            fields.add(extract_base_field(self.key))
        if self.comparison == ComparisonType.MESSAGE_CONTAINS:
            fields.add(self.message_key)
        if self.send_mapping_list:
            for mapping in self.send_mapping_list.mappings:
                if mapping.condition:
                    fields.add(extract_base_field(mapping.condition))
                for source_field in mapping.fields.values():
                    fields.add(extract_base_field(source_field))
            for generator in self.send_mapping_list.generators:
                fields.add(extract_base_field(generator.collection_field))
        return fields
