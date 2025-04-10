# src/haive/core/graph/branches.py

from typing import Dict, Any, Optional, Union, Callable, List, Literal
import logging
from langgraph.graph import END

logger = logging.getLogger(__name__)

class Branch:
    """
    Flexible branching logic for dynamic routing in LangGraph workflows.
    """
    
    def __init__(
        self,
        function: Optional[Callable[[Dict[str, Any]], Any]] = None,
        key: Optional[str] = None,
        value: Any = None,
        comparison: str = "==",
        destinations: Optional[Dict[Any, str]] = None,
        default: str = END
    ):
        """
        Initialize a branch with a condition and destinations.
        
        Args:
            function: Custom function that takes state and returns a condition value
            key: State key to check (for simpler conditions)
            value: Value to compare against (for simpler conditions)
            comparison: Comparison operator (==, !=, >, <, >=, <=, in, contains, exists)
            destinations: Mapping from condition values to destinations
            default: Default destination if no condition matches
        """
        self.function = function
        self.key = key
        self.value = value
        self.comparison = comparison
        self.destinations = destinations or {True: "continue", False: END}
        self.default = default
    
    def evaluate(self, state: Dict[str, Any]) -> str:
        """
        Evaluate the condition and return the next node to go to.
        
        Args:
            state: Current state dictionary
            
        Returns:
            Name of the next node
        """
        try:
            # Convert state to dict if needed
            state_dict = state.model_dump() if hasattr(state, "model_dump") else dict(state)
            
            # Use custom function if provided
            if self.function:
                result = self.function(state_dict)
            else:
                # Use key/value/comparison
                if self.key is None:
                    logger.warning("Branch has no key and no function")
                    return self.default
                
                if self.key not in state_dict:
                    logger.warning(f"Branch key '{self.key}' not found in state")
                    return self.default
                
                state_value = state_dict[self.key]
                result = self._compare(state_value, self.value, self.comparison)
            
            # Determine next node based on result
            if result in self.destinations:
                return self.destinations[result]
            
            return self.default
        except Exception as e:
            logger.error(f"Error evaluating branch: {e}")
            return self.default
    
    @staticmethod
    def _compare(value1: Any, value2: Any, op: str) -> bool:
        """Perform comparison between values."""
        if op == "==":
            return value1 == value2
        elif op == "!=":
            return value1 != value2
        elif op == ">":
            return value1 > value2
        elif op == "<":
            return value1 < value2
        elif op == ">=":
            return value1 >= value2
        elif op == "<=":
            return value1 <= value2
        elif op == "in":
            return value1 in value2
        elif op == "contains":
            return value2 in value1
        elif op == "exists":
            return value1 is not None
        elif op == "not exists":
            return value1 is None
        else:
            raise ValueError(f"Unsupported comparison operator: {op}")
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any], default: str = END) -> 'Branch':
        """Create a Branch from a configuration dictionary."""
        return cls(
            function=config.get("function"),
            key=config.get("key"),
            value=config.get("value"),
            comparison=config.get("comparison", "=="),
            destinations=config.get("destinations"),
            default=config.get("default", default)
        )
    
    @classmethod
    def from_condition(cls, key: str, value: Any, if_true: str = "continue", if_false: str = END) -> 'Branch':
        """Create a simple conditional branch."""
        return cls(
            key=key,
            value=value,
            destinations={True: if_true, False: if_false}
        )
    
    @classmethod
    def from_function(cls, function: Callable, destinations: Dict[Any, str], default: str = END) -> 'Branch':
        """Create a branch with a custom function."""
        return cls(
            function=function,
            destinations=destinations,
            default=default
        )
    
    @classmethod
    def multi_condition(cls, conditions: List['Branch'], operator: str = "and", default: str = END) -> 'Branch':
        """Create a composite branch with multiple conditions."""
        def evaluate_multi(state):
            results = [condition.evaluate(state) != condition.default for condition in conditions]
            if operator == "and":
                return all(results)
            elif operator == "or":
                return any(results)
            else:
                raise ValueError(f"Unsupported operator: {operator}")
        
        return cls(
            function=evaluate_multi,
            destinations={True: "continue", False: default},
            default=default
        )