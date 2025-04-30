# src/haive/core/graph/branches/__init__.py

import inspect
import logging
from collections.abc import Callable
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union, get_type_hints

from pydantic import BaseModel, Field

#from haive.core.graph.graph_pattern_registry import register_graph_component

logger = logging.getLogger(__name__)

class Branch:
    """Generalized branching logic for dynamic routing based on state values.
    
    Supports:
    - Direct state key comparisons (e.g., state["iterations"] > 3)
    - Function-based evaluation (e.g., lambda state: state["score"] > 5)
    - Multiple comparison operators (==, !=, >, <, >=, <=, in, contains, is, is not, exists, not exists)
    - Custom multi-path routing with conditions
    """

    COMPARISON_TYPES = Literal[
        "==", "!=", ">", "<", ">=", "<=", "in", "contains", "is", "is not", "exists", "not exists"
    ]

    def __init__(
        self,
        key: str | None = None,
        value: Any = None,
        comparison: COMPARISON_TYPES = "==",
        function: Callable[[dict[str, Any]], bool] | None = None,
        destinations: dict[bool | str, str] | None = None,
        default: str = "END",
        allow_none: bool = False,
        evaluator: Callable | None = None
    ):
        """Args:
        key (Optional[str]): State key to check (e.g., "iterations", "messages").
        value (Any): Value to compare against.
        comparison (COMPARISON_TYPES): Type of comparison (==, !=, >, <, >=, <=, in, contains, is, is not, exists, not exists).
        function (Optional[Callable]): Function that takes state dict and returns a bool.
        destinations (Optional[Dict[Union[bool, str], str]]): Custom routing based on evaluation results.
        default (str): Default destination if no match.
        allow_none (bool): If True, allows None values in state.
        evaluator (Optional[Callable]): Direct evaluator function for advanced cases.
        """
        self.key = key
        self.value = value
        self.comparison = comparison
        self.function = function
        self.destinations = destinations or {True: "continue", False: "END"}
        self.default = default
        self.allow_none = allow_none

        # Store direct evaluator if provided
        self._evaluator = evaluator

        # Parse function signature to derive input schema
        self.input_schema = None
        if function:
            try:
                hints = get_type_hints(function)
                if "state" in hints:
                    self.input_schema = hints["state"]
            except:
                pass

    @property
    def evaluator(self) -> Callable:
        """Get the evaluator function for this branch."""
        if self._evaluator:
            return self._evaluator

        # Create evaluator function based on configuration
        def evaluate_branch(state: dict[str, Any]) -> str:
            return self.evaluate(state)

        return evaluate_branch

    def evaluate(self, state: dict[str, Any]) -> str:
        """Evaluate the condition and return the next step.
        
        Args:
            state (Dict[str, Any]): Current state dictionary.
            
        Returns:
            str: Next node name (e.g., "execute", "retry", "END").
        """
        try:
            # Use function if provided
            if self.function:
                result = self.function(state)
            # Perform existence check for "exists" or "not exists"
            elif self.comparison in ["exists", "not exists"]:
                result = self.exists(state)
                if self.comparison == "not exists":
                    result = not result
            else:
                # Get state value
                state_value = state.get(self.key)

                # Handle None values
                if state_value is None and not self.allow_none:
                    logger.warning(f"State key '{self.key}' is None and allow_none is False.")
                    return self.default

                # Perform comparison
                result = self._compare(state_value)

        except Exception as e:
            logger.error(f"Error evaluating branch: {e}")
            return self.default

        # Determine next step
        return self.destinations.get(result, self.default)

    def _compare(self, state_value: Any) -> bool:
        """Helper method to perform comparison."""
        if self.comparison == "==":
            return state_value == self.value
        if self.comparison == "!=":
            return state_value != self.value
        if self.comparison == ">":
            return state_value > self.value
        if self.comparison == "<":
            return state_value < self.value
        if self.comparison == ">=":
            return state_value >= self.value
        if self.comparison == "<=":
            return state_value <= self.value
        if self.comparison == "in":
            return state_value in self.value if isinstance(self.value, (list, set, tuple)) else False
        if self.comparison == "contains":
            return self.value in state_value if isinstance(state_value, str) else False
        if self.comparison == "is":
            return state_value is self.value
        if self.comparison == "is not":
            return state_value is not self.value
        logger.warning(f"Unknown comparison type: {self.comparison}")
        return False

    def exists(self, state: dict[str, Any]) -> bool:
        """Checks whether the given key exists in the state and is not None.
        
        Args:
            state (Dict[str, Any]): The current state dictionary.
            
        Returns:
            bool: True if key exists and is not None, False otherwise.
        """
        if self.key is None:
            return False
        return self.key in state and state[self.key] is not None

    @classmethod
    def from_function(cls, function: Callable[[dict[str, Any]], bool | str],
                    destinations: dict[bool | str, str] | None = None,
                    default: str = "END") -> "Branch":
        """Create a Branch from a function.
        
        Args:
            function: Function that takes state and returns bool or route key
            destinations: Mapping of function outputs to node names
            default: Default destination if no match
            
        Returns:
            Configured Branch
        """
        return cls(function=function, destinations=destinations, default=default)

    @classmethod
    def key_equals(cls, key: str, value: Any,
                  true_dest: str = "continue", false_dest: str = "END") -> "Branch":
        """Create a Branch that checks if a key equals a value.
        
        Args:
            key: State key to check
            value: Value to compare against
            true_dest: Destination if true
            false_dest: Destination if false
            
        Returns:
            Configured Branch
        """
        return cls(
            key=key,
            value=value,
            comparison="==",
            destinations={True: true_dest, False: false_dest}
        )

    @classmethod
    def key_exists(cls, key: str,
                  true_dest: str = "continue", false_dest: str = "END") -> "Branch":
        """Create a Branch that checks if a key exists in the state.
        
        Args:
            key: State key to check
            true_dest: Destination if key exists
            false_dest: Destination if key doesn't exist
            
        Returns:
            Configured Branch
        """
        return cls(
            key=key,
            comparison="exists",
            destinations={True: true_dest, False: false_dest}
        )

    @classmethod
    def key_not_exists(cls, key: str,
                       true_dest: str = "continue", false_dest: str = "END") -> "Branch":
        """Create a Branch that checks if a key does not exist in the state.
        
        Args:
            key: State key to check
            true_dest: Destination if key doesn't exist
            false_dest: Destination if key exists
            
        Returns:
            Configured Branch
        """
        return cls(
            key=key,
            comparison="not exists",
            destinations={True: true_dest, False: false_dest}
        )

    @classmethod
    def key_greater_than(cls, key: str, value: Any,
                        true_dest: str = "continue", false_dest: str = "END") -> "Branch":
        """Create a Branch that checks if a key is greater than a value.
        
        Args:
            key: State key to check
            value: Value to compare against
            true_dest: Destination if true
            false_dest: Destination if false
            
        Returns:
            Configured Branch
        """
        return cls(
            key=key,
            value=value,
            comparison=">",
            destinations={True: true_dest, False: false_dest}
        )

    @classmethod
    def key_less_than(cls, key: str, value: Any,
                     true_dest: str = "continue", false_dest: str = "END") -> "Branch":
        """Create a Branch that checks if a key is less than a value.
        
        Args:
            key: State key to check
            value: Value to compare against
            true_dest: Destination if true
            false_dest: Destination if false
            
        Returns:
            Configured Branch
        """
        return cls(
            key=key,
            value=value,
            comparison="<",
            destinations={True: true_dest, False: false_dest}
        )

    @classmethod
    def multi_condition(cls, function: Callable[[dict[str, Any]], str],
                       destinations: dict[str, str],
                       default: str = "END") -> "Branch":
        """Create a Branch with multiple possible outputs based on a function.
        
        Args:
            function: Function that takes state and returns a string key
            destinations: Mapping of function output strings to node names
            default: Default destination if output not in destinations
            
        Returns:
            Configured Branch
        """
        # Create an evaluator function that maps function results to destinations
        def evaluator(state: dict[str, Any]) -> str:
            try:
                result = function(state)
                return destinations.get(result, default)
            except Exception as e:
                logger.error(f"Error in multi_condition: {e}")
                return default

        return cls(evaluator=evaluator)

    @classmethod
    def chain(cls, *branches: "Branch", default: str = "END") -> "Branch":
        """Chain multiple branches together, evaluating them in sequence.
        
        Args:
            *branches: Sequence of Branch objects to evaluate
            default: Default destination if all branches return their defaults
            
        Returns:
            Configured Branch that evaluates each branch in sequence
        """
        def chain_evaluator(state: dict[str, Any]) -> str:
            for branch in branches:
                result = branch.evaluate(state)
                # If the result is not the branch's default, return it
                if result != branch.default:
                    return result
            # If all branches returned their defaults, return the overall default
            return default

        return cls(evaluator=chain_evaluator)

    @classmethod
    def conditional(cls, condition: Callable[[dict[str, Any]], bool],
                   if_true: Union[str, "Branch"],
                   if_false: Union[str, "Branch"],
                   default: str = "END") -> "Branch":
        """Create a Branch with conditional evaluation of other branches.
        
        Args:
            condition: Function that takes state and returns a boolean
            if_true: Branch or node name to use if condition is True
            if_false: Branch or node name to use if condition is False
            default: Default destination if evaluation fails
            
        Returns:
            Configured Branch that conditionally evaluates other branches
        """
        def conditional_evaluator(state: dict[str, Any]) -> str:
            try:
                result = condition(state)
                if result:
                    # Handle if_true
                    if isinstance(if_true, Branch):
                        return if_true.evaluate(state)
                    return if_true
                # Handle if_false
                if isinstance(if_false, Branch):
                    return if_false.evaluate(state)
                return if_false
            except Exception as e:
                logger.error(f"Error in conditional branch: {e}")
                return default

        return cls(evaluator=conditional_evaluator)

    def to_dict(self) -> dict[str, Any]:
        """Convert the branch to a serializable dictionary.
        
        Returns:
            Dictionary representation of the branch
        """
        result = {
            "key": self.key,
            "value": self.value,
            "comparison": self.comparison,
            "destinations": self.destinations,
            "default": self.default,
            "allow_none": self.allow_none
        }

        # Handle function
        if self.function:
            # Serialize function if possible (module.name format)
            if hasattr(self.function, "__module__") and hasattr(self.function, "__name__"):
                result["function"] = f"{self.function.__module__}.{self.function.__name__}"

        # Handle evaluator
        if self._evaluator:
            # Serialize evaluator if possible
            if hasattr(self._evaluator, "__module__") and hasattr(self._evaluator, "__name__"):
                result["evaluator"] = f"{self._evaluator.__module__}.{self._evaluator.__name__}"

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Branch":
        """Create a Branch from a dictionary representation.
        
        Args:
            data: Dictionary representation of a branch
            
        Returns:
            Instantiated Branch object
        """
        # Handle function string reference
        function = None
        if "function" in data and isinstance(data["function"], str):
            try:
                import importlib
                module_path, func_name = data["function"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                function = getattr(module, func_name)
            except (ValueError, ImportError, AttributeError) as e:
                logger.warning(f"Could not import function {data.get('function')}: {e}")

        # Handle evaluator string reference
        evaluator = None
        if "evaluator" in data and isinstance(data["evaluator"], str):
            try:
                import importlib
                module_path, func_name = data["evaluator"].rsplit(".", 1)
                module = importlib.import_module(module_path)
                evaluator = getattr(module, func_name)
            except (ValueError, ImportError, AttributeError) as e:
                logger.warning(f"Could not import evaluator {data.get('evaluator')}: {e}")

        # Create the branch
        return cls(
            key=data.get("key"),
            value=data.get("value"),
            comparison=data.get("comparison", "=="),
            function=function,
            destinations=data.get("destinations", {True: "continue", False: "END"}),
            default=data.get("default", "END"),
            allow_none=data.get("allow_none", False),
            evaluator=evaluator
        )
