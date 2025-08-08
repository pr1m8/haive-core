r"""Robust naming utilities for OpenAI compliance and tool naming.

This module provides comprehensive utilities for sanitizing and transforming
class names, especially generic classes, to be compatible with OpenAI's
function calling requirements and follow consistent naming conventions.

Key Features:
- Handles generic classes (e.g., Plan[Task] -> plan_task_generic)
- Converts CamelCase to snake_case consistently
- Ensures OpenAI API compliance (pattern: ^[a-zA-Z0-9_\\.-]+$)
- Handles edge cases like acronyms, numbers, and special characters
- Provides reverse mapping capabilities for debugging

Example:
    Basic usage::

        from haive.core.utils.naming import sanitize_tool_name

        # Generic class handling
        name = sanitize_tool_name("Plan[Task]")
        # Returns: "plan_task_generic"

        # CamelCase conversion
        name = sanitize_tool_name("MyComplexModel")
        # Returns: "my_complex_model"

        # OpenAI compliance
        name = sanitize_tool_name("Invalid-Name[With]Brackets!")
        # Returns: "invalid_name_with_brackets"
"""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# OpenAI function name pattern: ^[a-zA-Z0-9_\\.-]+$
OPENAI_VALID_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

# Common acronyms that should be treated as single words
COMMON_ACRONYMS = {
    "HTML",
    "XML",
    "JSON",
    "API",
    "URL",
    "HTTP",
    "HTTPS",
    "SQL",
    "CSV",
    "PDF",
    "UUID",
    "JWT",
    "OAuth",
    "SMTP",
    "FTP",
    "SSH",
    "AWS",
    "GCP",
    "AI",
    "ML",
    "NLP",
    "LLM",
    "GPT",
    "CPU",
    "GPU",
}


def sanitize_tool_name(raw_name: str, preserve_acronyms: bool = True) -> str:
    """Sanitize tool names for OpenAI compliance and readability.

    This is the main function for converting any class name into an OpenAI-compliant
    tool name. It handles generic classes, CamelCase conversion, and ensures all
    characters are valid for OpenAI's function calling API.

    Args:
        raw_name: Raw tool name from __name__ or other source
        preserve_acronyms: Whether to treat known acronyms as single words

    Returns:
        Sanitized snake_case name that's OpenAI-compliant

    Examples:
        >>> sanitize_tool_name("Plan[Task]")
        'plan_task_generic'

        >>> sanitize_tool_name("HTTPSParser")
        'https_parser'

        >>> sanitize_tool_name("MyModel[String]")
        'my_model_string_generic'

        >>> sanitize_tool_name("Invalid-Name!")
        'invalid_name'
    """
    if not raw_name or not isinstance(raw_name, str):
        return "unnamed_tool"

    logger.debug(f"Sanitizing tool name: '{raw_name}'")

    # Step 1: Handle generic classes like Plan[Task] -> Plan_Task_generic
    processed_name = _handle_generic_classes(raw_name)
    logger.debug(f"After generic handling: '{processed_name}'")

    # Step 2: Convert CamelCase to snake_case
    snake_name = _convert_to_snake_case(processed_name, preserve_acronyms)
    logger.debug(f"After snake_case conversion: '{snake_name}'")

    # Step 3: Ensure OpenAI compliance
    compliant_name = _ensure_openai_compliance(snake_name)
    logger.debug(f"After compliance check: '{compliant_name}'")

    # Step 4: Final cleanup
    final_name = _final_cleanup(compliant_name)
    logger.debug(f"Final sanitized name: '{final_name}'")

    return final_name


def create_openai_compliant_name(raw_name: str, suffix: str = None) -> str:
    """Create an OpenAI-compliant name with optional suffix.

    This is a higher-level function that creates compliant names and can
    add suffixes for disambiguation.

    Args:
        raw_name: Raw name to process
        suffix: Optional suffix to add (e.g., 'tool', 'generic')

    Returns:
        OpenAI-compliant name with optional suffix

    Examples:
        >>> create_openai_compliant_name("Plan[Task]", "tool")
        'plan_task_generic_tool'

        >>> create_openai_compliant_name("MyModel")
        'my_model'
    """
    base_name = sanitize_tool_name(raw_name)

    if suffix:
        sanitized_suffix = sanitize_tool_name(suffix)
        return f"{base_name}_{sanitized_suffix}"

    return base_name


def _handle_generic_classes(name: str) -> str:
    """Handle generic class names like Plan[Task] -> Plan_Task_generic.

    This function specifically handles Python generic type syntax and converts
    it to a readable format that indicates the generic nature.
    """
    # Pattern for generic classes: ClassName[TypeParam] with optional additional text
    # This handles cases like Plan[Task]WithExtra! -> Plan_Task_WithExtra_generic
    generic_pattern = re.compile(r"^(\w+)\[(\w+(?:,\s*\w+)*)\](.*)$")
    match = generic_pattern.match(name.strip())

    if match:
        base_class = match.group(1)
        type_params = match.group(2)
        additional_text = match.group(3)

        # Handle multiple type parameters: Plan[Task,Status] -> Plan_Task_Status_generic
        type_parts = [param.strip() for param in type_params.split(",")]
        combined_types = "_".join(type_parts)

        # Include additional text if present
        if additional_text:
            # Clean additional text of special characters but preserve letters/numbers
            clean_additional = re.sub(r"[^a-zA-Z0-9]", "", additional_text)
            if clean_additional:
                result = f"{base_class}_{combined_types}_{clean_additional}_generic"
            else:
                result = f"{base_class}_{combined_types}_generic"
        else:
            result = f"{base_class}_{combined_types}_generic"

        logger.debug(f"Generic class detected: {name} -> {result}")
        return result

    return name


def _convert_to_snake_case(name: str, preserve_acronyms: bool = True) -> str:
    """Convert CamelCase to snake_case with smart acronym handling.

    This function handles various CamelCase patterns including acronyms
    and ensures proper snake_case conversion.
    """
    if preserve_acronyms:
        # Enhanced acronym handling - process each acronym individually
        result = name

        # Process from longest to shortest to avoid substring issues
        sorted_acronyms = sorted(COMMON_ACRONYMS, key=len, reverse=True)

        for acronym in sorted_acronyms:
            # Look for the acronym in the text
            # Handle cases like HTTPSParser -> HTTPS + Parser
            # Also handle cases like XMLToJSON -> XML + To + JSON
            pattern = re.compile(f"({acronym})([A-Z]|$)", re.IGNORECASE)

            def replace_acronym(match):
                found_acronym = match.group(1)
                following = match.group(2) if match.group(2) else ""
                # Convert to lowercase and add underscore if there's following text
                if following and following != "":
                    return f"{found_acronym.lower()}_{following}"
                else:
                    return found_acronym.lower()

            result = pattern.sub(replace_acronym, result)

        # Now handle remaining CamelCase patterns
        # Handle sequences of capitals followed by lowercase
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", result)
        # Handle lowercase followed by uppercase
        result = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", result)

        return result.lower()
    else:
        # Simple conversion without acronym protection
        # Handle sequences of capitals
        snake = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
        # Handle lowercase to uppercase transitions
        snake = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", snake)
        return snake.lower()


def _ensure_openai_compliance(name: str) -> str:
    r"""Ensure the name complies with OpenAI's function name requirements.

    OpenAI requires function names to match: ^[a-zA-Z0-9_\\.-]+$
    """
    # Remove any invalid characters, keeping only alphanumeric, underscore, dot, dash
    compliant = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

    # Ensure it doesn't start with a number
    if compliant and compliant[0].isdigit():
        compliant = f"tool_{compliant}"

    return compliant


def _final_cleanup(name: str) -> str:
    """Perform final cleanup on the name.

    This removes excessive underscores, ensures proper length,
    and handles edge cases.
    """
    # Clean up multiple underscores
    cleaned = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    cleaned = cleaned.strip("_")

    # Ensure minimum length
    if not cleaned:
        return "unnamed_tool"

    # Ensure maximum reasonable length (OpenAI doesn't specify, but be reasonable)
    if len(cleaned) > 64:
        logger.warning(
            f"Tool name '{cleaned}' is very long ({len(cleaned)} chars), truncating"
        )
        cleaned = cleaned[:61] + "..."  # Keep some indication it was truncated

    return cleaned


def validate_tool_name(name: str) -> Tuple[bool, List[str]]:
    """Validate a tool name against OpenAI requirements.

    Args:
        name: Tool name to validate

    Returns:
        Tuple of (is_valid, list_of_issues)

    Examples:
        >>> validate_tool_name("valid_tool_name")
        (True, [])

        >>> validate_tool_name("Invalid[Name]!")
        (False, ['Contains invalid characters: [, ], !', 'Contains brackets which are not allowed'])
    """
    issues = []

    if not name:
        return False, ["Tool name is empty"]

    if not isinstance(name, str):
        return False, ["Tool name must be a string"]

    # Check OpenAI pattern
    if not OPENAI_VALID_PATTERN.match(name):
        invalid_chars = [c for c in name if not re.match(r"[a-zA-Z0-9_.-]", c)]
        if invalid_chars:
            unique_invalid = list(set(invalid_chars))
            issues.append(f"Contains invalid characters: {', '.join(unique_invalid)}")

    # Check for brackets specifically (common issue with generics)
    if "[" in name or "]" in name:
        issues.append("Contains brackets which are not allowed")

    # Check length
    if len(name) > 64:
        issues.append(f"Name is too long ({len(name)} chars, recommended max: 64)")

    # Check starts with number
    if name[0].isdigit():
        issues.append("Name cannot start with a number")

    return len(issues) == 0, issues


def get_name_suggestions(raw_name: str, count: int = 3) -> List[str]:
    """Get multiple naming suggestions for a raw name.

    Provides different variations of sanitized names to choose from.

    Args:
        raw_name: Original name to generate suggestions for
        count: Number of suggestions to generate

    Returns:
        List of suggested names

    Examples:
        >>> get_name_suggestions("Plan[Task]")
        ['plan_task_generic', 'plan_task_tool', 'task_plan_generic']
    """
    suggestions = []

    # Base sanitized name
    base = sanitize_tool_name(raw_name)
    suggestions.append(base)

    if count > 1:
        # With 'tool' suffix
        tool_version = create_openai_compliant_name(raw_name, "tool")
        if tool_version != base:
            suggestions.append(tool_version)

    if count > 2 and "[" in raw_name and "]" in raw_name:
        # For generics, try reversing the order
        generic_pattern = re.compile(r"^(\w+)\[(\w+)\]$")
        match = generic_pattern.match(raw_name.strip())
        if match:
            base_class, type_param = match.groups()
            reversed_name = f"{type_param}_{base_class}_generic"
            reversed_sanitized = sanitize_tool_name(reversed_name)
            if reversed_sanitized not in suggestions:
                suggestions.append(reversed_sanitized)

    # Pad with variations if needed
    while len(suggestions) < count:
        variant = f"{base}_{len(suggestions)}"
        suggestions.append(variant)

    return suggestions[:count]


def create_name_mapping(original_names: List[str]) -> Dict[str, str]:
    """Create a mapping from original names to sanitized names.

    Useful for batch processing and maintaining mappings for debugging.

    Args:
        original_names: List of original names to process

    Returns:
        Dictionary mapping original -> sanitized names

    Examples:
        >>> create_name_mapping(["Plan[Task]", "MyModel", "HTTPParser"])
        {
            'Plan[Task]': 'plan_task_generic',
            'MyModel': 'my_model',
            'HTTPParser': 'http_parser'
        }
    """
    mapping = {}
    used_names = set()

    for original in original_names:
        sanitized = sanitize_tool_name(original)

        # Handle collisions
        base_sanitized = sanitized
        counter = 1
        while sanitized in used_names:
            sanitized = f"{base_sanitized}_{counter}"
            counter += 1

        mapping[original] = sanitized
        used_names.add(sanitized)

        logger.debug(f"Mapped '{original}' -> '{sanitized}'")

    return mapping


# Convenience functions for common patterns
def sanitize_class_name(cls) -> str:
    """Sanitize a class object's name for tool usage.

    Args:
        cls: Class object or class name string

    Returns:
        Sanitized name suitable for OpenAI tools
    """
    if hasattr(cls, "__name__"):
        return sanitize_tool_name(cls.__name__)
    else:
        return sanitize_tool_name(str(cls))


def sanitize_pydantic_model_name(model) -> str:
    """Sanitize a Pydantic model's name for tool usage.

    Specifically handles Pydantic model naming patterns.

    Args:
        model: Pydantic model class

    Returns:
        Sanitized name suitable for OpenAI tools
    """
    # Get the model name
    if hasattr(model, "__name__"):
        raw_name = model.__name__
    elif hasattr(model, "model_config") and hasattr(model.model_config, "title"):
        raw_name = model.model_config.title
    else:
        raw_name = str(model)

    return sanitize_tool_name(raw_name)
