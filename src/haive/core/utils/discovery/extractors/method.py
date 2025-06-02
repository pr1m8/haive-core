"""
Method extraction utilities.

Extracts method signatures, parameters, type hints, and other metadata from classes.
"""

import inspect
import logging
from typing import Callable, Dict, Optional, Type, get_type_hints

from ..models.component import MethodInfo, ParameterInfo

logger = logging.getLogger(__name__)


class MethodExtractor:
    """Extracts method information from classes and functions."""

    def extract_methods(self, cls: Type) -> Dict[str, MethodInfo]:
        """
        Extract all methods from a class.

        Args:
            cls: Class to analyze

        Returns:
            Dictionary of method name to MethodInfo
        """
        methods = {}

        # Try to extract methods from the class
        try:
            # Regular methods and functions
            for name, method in inspect.getmembers(
                cls, lambda x: inspect.isfunction(x) or inspect.ismethod(x)
            ):
                # Skip dunder methods except __init__
                if name.startswith("__") and name != "__init__":
                    continue

                method_info = self.extract_method(method)
                if method_info:
                    methods[name] = method_info
        except Exception as e:
            logger.warning(f"Error extracting methods from {cls.__name__}: {e}")

        return methods

    def extract_method(self, method: Callable) -> Optional[MethodInfo]:
        """
        Extract information about a single method.

        Args:
            method: Method to analyze

        Returns:
            MethodInfo object or None if extraction fails
        """
        try:
            # Get basic method info
            method_name = method.__name__

            # Get signature
            sig = inspect.signature(method)
            signature_str = str(sig)

            # Get docstring
            docstring = inspect.getdoc(method) or ""

            # Check if method is async
            is_async = inspect.iscoroutinefunction(method)

            # Extract parameters
            parameters = self._extract_parameters(method, sig)

            # Extract return type
            return_type = self._extract_return_type(method, sig)

            # Get source code if possible
            source_code = self._get_source_code(method)

            # Create method info
            method_info = MethodInfo(
                name=method_name,
                parameters=parameters,
                return_type=return_type,
                docstring=docstring,
                is_async=is_async,
                source_code=source_code,
                signature_str=signature_str,
            )

            return method_info

        except Exception as e:
            logger.debug(
                f"Error extracting method {getattr(method, '__name__', 'unknown')}: {e}"
            )
            return None

    def _extract_parameters(
        self, method: Callable, sig: inspect.Signature
    ) -> Dict[str, ParameterInfo]:
        """
        Extract parameter information from a method.

        Args:
            method: Method to analyze
            sig: Method signature

        Returns:
            Dictionary of parameter name to ParameterInfo
        """
        parameters = {}

        # Try to get type hints
        try:
            type_hints = get_type_hints(method)
        except (TypeError, ValueError):
            type_hints = {}

        # Extract parameter information
        for name, param in sig.parameters.items():
            # Skip self and cls parameters
            if name in ("self", "cls") and param.annotation == inspect.Parameter.empty:
                continue

            # Handle default values
            default = ...
            if param.default is not inspect.Parameter.empty:
                default = param.default

            # Handle type annotations
            type_hint = "Any"
            if param.annotation is not inspect.Parameter.empty:
                type_hint = str(param.annotation)
            elif name in type_hints:
                type_hint = str(type_hints[name])

            # Extract description from docstring if available
            description = None

            # Create parameter info
            parameters[name] = ParameterInfo(
                name=name,
                type_hint=type_hint,
                default_value=default,
                is_required=param.default is inspect.Parameter.empty
                and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD),
                description=description,
            )

        return parameters

    def _extract_return_type(self, method: Callable, sig: inspect.Signature) -> str:
        """
        Extract return type from a method.

        Args:
            method: Method to analyze
            sig: Method signature

        Returns:
            Return type as string
        """
        # Try to get from signature return annotation
        if sig.return_annotation is not inspect.Parameter.empty:
            return str(sig.return_annotation)

        # Try to get from type hints
        try:
            type_hints = get_type_hints(method)
            if "return" in type_hints:
                return str(type_hints["return"])
        except (TypeError, ValueError):
            pass

        # Default to Any
        return "Any"

    def _get_source_code(self, method: Callable) -> Optional[str]:
        """
        Get source code of a method if possible.

        Args:
            method: Method to get source code for

        Returns:
            Source code as string or None if unavailable
        """
        try:
            return inspect.getsource(method)
        except (OSError, TypeError):
            # Try getting source from __wrapped__ if available
            if hasattr(method, "__wrapped__"):
                try:
                    return inspect.getsource(method.__wrapped__)
                except (OSError, TypeError):
                    pass
        return None
