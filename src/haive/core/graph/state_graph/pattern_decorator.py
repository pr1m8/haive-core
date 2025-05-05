import functools
from typing import Any, Callable, Dict, Optional

from ..models.function_ref import FunctionReference
from .pattern_registry import PatternDefinition, PatternRegistry


def register_pattern(name: str, pattern_type: str, **parameters):
    """
    Decorator to register a function as a graph pattern.

    Args:
        name: Pattern name
        pattern_type: Pattern type
        **parameters: Default pattern parameters

    Returns:
        Decorator function
    """

    def decorator(func: Callable):
        # Create pattern definition
        pattern = PatternDefinition(
            name=name,
            description=func.__doc__,
            pattern_type=pattern_type,
            apply_func=FunctionReference.from_callable(func, name=name),
            parameters=parameters,
        )

        # Register pattern
        registry = PatternRegistry.get_instance()
        registry.register(pattern)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
