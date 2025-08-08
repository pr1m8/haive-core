#!/usr/bin/env python3
"""Enhanced naming utilities for complex type annotations and tool names.

This module provides advanced utilities for sanitizing complex Python type annotations
and generic classes into OpenAI-compliant tool names with descriptive transformations.

Key Features:
- Handles nested generics: List[Dict[str, Task]] -> list_dict_str_task_nested_generic
- Provides transformation descriptions for debugging
- Handles Union, Optional, and complex type annotations
- Maintains type hierarchy information in the name
- Supports custom naming strategies

Examples:
    Complex type handling::

        from haive.core.utils.enhanced_naming import enhanced_sanitize_tool_name

        # Nested generics
        result, desc = enhanced_sanitize_tool_name("List[Dict[str, Task]]")
        # Returns: ("list_dict_str_task_nested_generic", "3-level nested generic with 4 type parameters")

        # Union types
        result, desc = enhanced_sanitize_tool_name("Union[str, List[Task]]")
        # Returns: ("union_str_list_task_generic", "Union type with 2 alternatives")

        # Optional types
        result, desc = enhanced_sanitize_tool_name("Optional[Plan[Task]]")
        # Returns: ("optional_plan_task_generic", "Optional type wrapping Plan[Task] generic")
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Import base functionality
from haive.core.utils.naming import (
    _convert_to_snake_case,
    _ensure_openai_compliance,
    _final_cleanup,
    sanitize_tool_name,
)


@dataclass
class NamingTransformation:
    """Metadata about a naming transformation."""

    original_name: str
    final_name: str
    transformation_type: str
    complexity_level: int
    type_parameters: List[str]
    nesting_depth: int
    description: str
    warnings: List[str]


class EnhancedGenericParser:
    """Advanced parser for complex generic type annotations."""

    # Enhanced patterns for different type constructs
    PATTERNS = {
        "simple_generic": re.compile(r"^(\w+)\[([^\[\]]+)\]$"),
        "nested_generic": re.compile(r"^(\w+)\[(.*)\]$"),
        "union_type": re.compile(r"^Union\[(.*)\]$"),
        "optional_type": re.compile(r"^Optional\[(.*)\]$"),
        "literal_type": re.compile(r"^Literal\[(.*)\]$"),
    }

    def parse_complex_type(self, type_name: str) -> NamingTransformation:
        """Parse complex type annotations into naming transformation."""
        original = type_name.strip()
        warnings = []

        # Detect the type of generic
        transformation_type, parsed_data = self._identify_type_pattern(original)

        if transformation_type == "nested_generic":
            result = self._handle_nested_generic(original, parsed_data, warnings)
        elif transformation_type == "union_type":
            result = self._handle_union_type(original, parsed_data, warnings)
        elif transformation_type == "optional_type":
            result = self._handle_optional_type(original, parsed_data, warnings)
        elif transformation_type == "simple_generic":
            result = self._handle_simple_generic(original, parsed_data, warnings)
        else:
            # Fall back to basic sanitization
            result = self._handle_unknown_type(original, warnings)

        return result

    def _identify_type_pattern(self, type_name: str) -> Tuple[str, Dict[str, Any]]:
        """Identify which pattern the type matches."""
        # Check for Union types first
        if union_match := self.PATTERNS["union_type"].match(type_name):
            return "union_type", {"content": union_match.group(1)}

        # Check for Optional types
        if optional_match := self.PATTERNS["optional_type"].match(type_name):
            return "optional_type", {"content": optional_match.group(1)}

        # Check for complex nested generics
        if nested_match := self.PATTERNS["nested_generic"].match(type_name):
            base_class = nested_match.group(1)
            content = nested_match.group(2)

            # Check if content contains nested brackets (indicating complexity)
            bracket_count = content.count("[") + content.count("]")
            if bracket_count > 0:
                return "nested_generic", {"base": base_class, "content": content}
            else:
                return "simple_generic", {"base": base_class, "content": content}

        return "unknown", {"content": type_name}

    def _handle_nested_generic(
        self, original: str, data: Dict, warnings: List[str]
    ) -> NamingTransformation:
        """Handle complex nested generics like List[Dict[str, Task]]."""
        base_class = data["base"]
        content = data["content"]

        # Recursively parse the nested content
        type_parameters = self._extract_nested_parameters(content)
        nesting_depth = self._calculate_nesting_depth(content)

        if nesting_depth > 3:
            warnings.append(
                f"Very deep nesting ({nesting_depth} levels) may be hard to understand"
            )

        # Build descriptive name
        param_parts = []
        for param in type_parameters:
            # Recursively handle each parameter
            if "[" in param and "]" in param:
                # This parameter is itself generic
                sub_result = self.parse_complex_type(param)
                param_parts.append(sub_result.final_name.replace("_generic", ""))
            else:
                # Simple parameter
                param_parts.append(_convert_to_snake_case(param.strip()))

        # Combine into final name
        base_snake = _convert_to_snake_case(base_class)
        combined_params = "_".join(param_parts)

        if nesting_depth > 1:
            final_name = f"{base_snake}_{combined_params}_nested_generic"
            description = f"{nesting_depth}-level nested generic with {len(type_parameters)} type parameters"
        else:
            final_name = f"{base_snake}_{combined_params}_generic"
            description = f"Generic type with {len(type_parameters)} parameters"

        return NamingTransformation(
            original_name=original,
            final_name=_ensure_openai_compliance(final_name),
            transformation_type="nested_generic",
            complexity_level=nesting_depth,
            type_parameters=type_parameters,
            nesting_depth=nesting_depth,
            description=description,
            warnings=warnings,
        )

    def _handle_union_type(
        self, original: str, data: Dict, warnings: List[str]
    ) -> NamingTransformation:
        """Handle Union types like Union[str, List[Task]]."""
        content = data["content"]

        # Split union alternatives
        alternatives = self._split_union_alternatives(content)

        if len(alternatives) > 4:
            warnings.append(
                f"Union with {len(alternatives)} alternatives may be too complex"
            )

        # Process each alternative
        alt_parts = []
        for alt in alternatives:
            alt = alt.strip()
            if "[" in alt and "]" in alt:
                # Generic alternative
                sub_result = self.parse_complex_type(alt)
                alt_parts.append(sub_result.final_name.replace("_generic", ""))
            else:
                # Simple alternative
                alt_parts.append(_convert_to_snake_case(alt))

        final_name = f"union_{'_'.join(alt_parts)}_generic"
        description = f"Union type with {len(alternatives)} alternatives"

        return NamingTransformation(
            original_name=original,
            final_name=_ensure_openai_compliance(final_name),
            transformation_type="union_type",
            complexity_level=1,
            type_parameters=alternatives,
            nesting_depth=1,
            description=description,
            warnings=warnings,
        )

    def _handle_optional_type(
        self, original: str, data: Dict, warnings: List[str]
    ) -> NamingTransformation:
        """Handle Optional types like Optional[Plan[Task]]."""
        content = data["content"].strip()

        # Process the wrapped type
        if "[" in content and "]" in content:
            # Generic wrapped type
            sub_result = self.parse_complex_type(content)
            wrapped_name = sub_result.final_name.replace("_generic", "")
            description = f"Optional type wrapping {sub_result.description}"
        else:
            # Simple wrapped type
            wrapped_name = _convert_to_snake_case(content)
            description = f"Optional type wrapping {content}"

        final_name = f"optional_{wrapped_name}_generic"

        return NamingTransformation(
            original_name=original,
            final_name=_ensure_openai_compliance(final_name),
            transformation_type="optional_type",
            complexity_level=1,
            type_parameters=[content],
            nesting_depth=1,
            description=description,
            warnings=warnings,
        )

    def _handle_simple_generic(
        self, original: str, data: Dict, warnings: List[str]
    ) -> NamingTransformation:
        """Handle simple generics like Plan[Task] or Plan[Task, Status]."""
        base_class = data["base"]
        content = data["content"]

        # Split multiple parameters
        parameters = [p.strip() for p in content.split(",")]

        # Convert to snake case
        base_snake = _convert_to_snake_case(base_class)
        param_parts = [_convert_to_snake_case(p) for p in parameters]

        final_name = f"{base_snake}_{'_'.join(param_parts)}_generic"
        description = f"Simple generic with {len(parameters)} type parameter{'s' if len(parameters) > 1 else ''}"

        return NamingTransformation(
            original_name=original,
            final_name=_ensure_openai_compliance(final_name),
            transformation_type="simple_generic",
            complexity_level=1,
            type_parameters=parameters,
            nesting_depth=1,
            description=description,
            warnings=warnings,
        )

    def _handle_unknown_type(
        self, original: str, warnings: List[str]
    ) -> NamingTransformation:
        """Handle unknown/complex types by falling back to basic sanitization."""
        warnings.append("Unknown type pattern, using basic sanitization")

        # Use the original sanitize_tool_name as fallback
        final_name = sanitize_tool_name(original)

        return NamingTransformation(
            original_name=original,
            final_name=final_name,
            transformation_type="unknown",
            complexity_level=0,
            type_parameters=[],
            nesting_depth=0,
            description="Unknown type pattern, basic sanitization applied",
            warnings=warnings,
        )

    def _extract_nested_parameters(self, content: str) -> List[str]:
        """Extract parameters from nested generic content."""
        parameters = []
        current_param = ""
        bracket_depth = 0

        for char in content:
            if char == "[":
                bracket_depth += 1
                current_param += char
            elif char == "]":
                bracket_depth -= 1
                current_param += char
            elif char == "," and bracket_depth == 0:
                # Top-level comma, new parameter
                if current_param.strip():
                    parameters.append(current_param.strip())
                current_param = ""
            else:
                current_param += char

        # Add final parameter
        if current_param.strip():
            parameters.append(current_param.strip())

        return parameters

    def _calculate_nesting_depth(self, content: str) -> int:
        """Calculate the maximum nesting depth in the type."""
        max_depth = 0
        current_depth = 0

        for char in content:
            if char == "[":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == "]":
                current_depth -= 1

        return max_depth + 1  # +1 for the outer level

    def _split_union_alternatives(self, content: str) -> List[str]:
        """Split Union alternatives respecting nested brackets."""
        alternatives = []
        current_alt = ""
        bracket_depth = 0

        for char in content:
            if char == "[":
                bracket_depth += 1
                current_alt += char
            elif char == "]":
                bracket_depth -= 1
                current_alt += char
            elif char == "," and bracket_depth == 0:
                # Top-level comma, new alternative
                if current_alt.strip():
                    alternatives.append(current_alt.strip())
                current_alt = ""
            else:
                current_alt += char

        # Add final alternative
        if current_alt.strip():
            alternatives.append(current_alt.strip())

        return alternatives


def enhanced_sanitize_tool_name(
    raw_name: str, include_metadata: bool = True, max_complexity: int = 5
) -> Tuple[str, Optional[NamingTransformation]]:
    """Enhanced tool name sanitization with metadata and complex type support.

    Args:
        raw_name: Raw tool name from __name__ or type annotation
        include_metadata: Whether to return transformation metadata
        max_complexity: Maximum complexity level to allow (warns if exceeded)

    Returns:
        Tuple of (sanitized_name, transformation_metadata)

    Examples:
        >>> name, meta = enhanced_sanitize_tool_name("List[Dict[str, Task]]")
        >>> print(name)
        'list_dict_str_task_nested_generic'
        >>> print(meta.description)
        '3-level nested generic with 4 type parameters'

        >>> name, meta = enhanced_sanitize_tool_name("Union[str, Optional[Plan[Task]]]")
        >>> print(name)
        'union_str_optional_plan_task_generic'
        >>> print(meta.complexity_level)
        2
    """
    if not raw_name or not isinstance(raw_name, str):
        simple_result = sanitize_tool_name(raw_name)
        if include_metadata:
            return simple_result, None
        return simple_result, None

    parser = EnhancedGenericParser()

    # Check if this looks like a complex type
    if any(
        pattern in raw_name
        for pattern in ["[", "Union", "Optional", "Literal", "Dict", "List"]
    ):
        # Use enhanced parsing
        transformation = parser.parse_complex_type(raw_name)

        # Check complexity warnings
        if transformation.complexity_level > max_complexity:
            transformation.warnings.append(
                f"Complexity level {transformation.complexity_level} exceeds recommended maximum {max_complexity}"
            )

        # Apply final cleanup
        final_name = _final_cleanup(transformation.final_name)
        transformation.final_name = final_name

        if include_metadata:
            return final_name, transformation
        return final_name, None
    else:
        # Use simple sanitization
        simple_result = sanitize_tool_name(raw_name)

        if include_metadata:
            # Create minimal metadata for simple case
            metadata = NamingTransformation(
                original_name=raw_name,
                final_name=simple_result,
                transformation_type="simple",
                complexity_level=0,
                type_parameters=[],
                nesting_depth=0,
                description="Simple name sanitization",
                warnings=[],
            )
            return simple_result, metadata
        return simple_result, None


def analyze_naming_complexity(type_names: List[str]) -> Dict[str, Any]:
    """Analyze naming complexity across multiple type names.

    Args:
        type_names: List of type names to analyze

    Returns:
        Dictionary with complexity analysis results

    Examples:
        >>> results = analyze_naming_complexity([
        ...     "Plan[Task]",
        ...     "List[Dict[str, Task]]",
        ...     "Union[str, Optional[Plan[Task]]]"
        ... ])
        >>> print(f"Average complexity: {results['average_complexity']}")
        >>> print(f"Most complex: {results['most_complex']['name']}")
    """
    results = {
        "total_count": len(type_names),
        "transformations": [],
        "complexity_distribution": {},
        "total_warnings": 0,
        "most_complex": None,
        "average_complexity": 0.0,
        "recommendations": [],
    }

    max_complexity = 0
    total_complexity = 0

    for name in type_names:
        _, transformation = enhanced_sanitize_tool_name(name, include_metadata=True)

        if transformation:
            results["transformations"].append(transformation)

            # Track complexity
            complexity = transformation.complexity_level
            total_complexity += complexity

            if complexity > max_complexity:
                max_complexity = complexity
                results["most_complex"] = {
                    "name": name,
                    "complexity": complexity,
                    "description": transformation.description,
                }

            # Count complexity levels
            level_key = f"level_{complexity}"
            results["complexity_distribution"][level_key] = (
                results["complexity_distribution"].get(level_key, 0) + 1
            )

            # Count warnings
            results["total_warnings"] += len(transformation.warnings)

    # Calculate averages
    if results["total_count"] > 0:
        results["average_complexity"] = total_complexity / results["total_count"]

    # Generate recommendations
    if results["average_complexity"] > 2:
        results["recommendations"].append(
            "Consider simplifying type annotations for better tool naming"
        )

    if results["total_warnings"] > 0:
        results["recommendations"].append(
            f"Review {results['total_warnings']} naming warnings"
        )

    if max_complexity > 4:
        results["recommendations"].append(
            "Some types are very complex - consider breaking them down"
        )

    return results


# Convenience functions for common use cases
def sanitize_pydantic_model_name_enhanced(
    model,
) -> Tuple[str, Optional[NamingTransformation]]:
    """Enhanced Pydantic model name sanitization with metadata."""
    if hasattr(model, "__name__"):
        raw_name = model.__name__
    elif hasattr(model, "model_config") and hasattr(model.model_config, "title"):
        raw_name = model.model_config.title
    else:
        raw_name = str(model)

    return enhanced_sanitize_tool_name(raw_name)


def get_naming_suggestions_enhanced(
    raw_name: str, count: int = 5
) -> List[Dict[str, Any]]:
    """Get multiple enhanced naming suggestions with metadata."""
    suggestions = []

    # Base suggestion
    base_name, base_meta = enhanced_sanitize_tool_name(raw_name)
    suggestions.append(
        {
            "name": base_name,
            "strategy": "enhanced_default",
            "metadata": base_meta,
            "description": (
                base_meta.description if base_meta else "Enhanced default sanitization"
            ),
        }
    )

    # Alternative strategies
    if count > 1:
        # Try with different complexity tolerance
        alt_name, alt_meta = enhanced_sanitize_tool_name(raw_name, max_complexity=10)
        if alt_name != base_name:
            suggestions.append(
                {
                    "name": alt_name,
                    "strategy": "high_complexity",
                    "metadata": alt_meta,
                    "description": "Allow higher complexity transformations",
                }
            )

    # Add tool suffix versions
    remaining_count = count - len(suggestions)
    for i in range(remaining_count):
        suffix_name = f"{base_name}_tool_{i+1}" if i > 0 else f"{base_name}_tool"
        suggestions.append(
            {
                "name": suffix_name,
                "strategy": f"tool_suffix_{i+1}",
                "metadata": None,
                "description": f"Base name with tool suffix variant {i+1}",
            }
        )

    return suggestions[:count]
