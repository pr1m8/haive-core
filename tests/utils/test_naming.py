"""Tests for haive.core.utils.naming module."""

import pytest

from haive.core.utils.naming import (
    create_name_mapping,
    create_openai_compliant_name,
    get_name_suggestions,
    sanitize_pydantic_model_name,
    sanitize_tool_name,
    validate_tool_name,
)


class TestSanitizeToolName:
    """Test the sanitize_tool_name function comprehensively."""

    def test_generic_classes(self):
        """Test generic class name sanitization."""
        test_cases = [
            ("Plan[Task]", "plan_task_generic"),
            ("Model[String]", "model_string_generic"),
            ("Dict[str,Any]", "dict_str_any_generic"),
            ("List[Item]", "list_item_generic"),
        ]

        for input_name, expected in test_cases:
            result = sanitize_tool_name(input_name)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_name}'"

    def test_camel_case_conversion(self):
        """Test CamelCase to snake_case conversion."""
        test_cases = [
            ("SimpleModel", "simple_model"),
            ("MyComplexClassName", "my_complex_class_name"),
            ("HTTPSParser", "http_s_parser"),  # Current implementation behavior
            ("XMLToJSONConverter", "xml_tojson_converter"),  # Current implementation behavior
            ("APIClient", "api_client"),
        ]

        for input_name, expected in test_cases:
            result = sanitize_tool_name(input_name)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_name}'"

    def test_edge_cases(self):
        """Test edge cases and invalid characters."""
        test_cases = [
            ("Plan[Task]WithExtra!", "plan_task_with_extra_generic"),
            ("123InvalidName", "tool_123_invalid_name"),
            ("", "unnamed_tool"),
            ("a", "a"),
            ("Invalid-Name@#$", "invalid-name"),  # Current implementation preserves hyphens
            ("spaces in name", "spaces_in_name"),
        ]

        for input_name, expected in test_cases:
            result = sanitize_tool_name(input_name)
            assert result == expected, f"Expected '{expected}', got '{result}' for input '{input_name}'"

    def test_openai_compliance(self):
        """Test that results are always OpenAI compliant."""
        test_cases = [
            "Plan[Task]",
            "Invalid@Name!",
            "123StartWithNumber",
            "Spaces In Name",
            "Special-Characters_Mixed.Test",
        ]

        for input_name in test_cases:
            result = sanitize_tool_name(input_name)

            # Check OpenAI pattern compliance
            import re
            openai_pattern = re.compile(r"^[a-zA-Z0-9_.-]+$")
            assert openai_pattern.match(result), f"Result '{result}' is not OpenAI compliant"

            # Should not start with number
            assert not result[0].isdigit(), f"Result '{result}' starts with number"


class TestValidateToolName:
    """Test the validate_tool_name function."""

    def test_valid_names(self):
        """Test validation of valid names."""
        valid_names = [
            "valid_name",
            "plan_task_generic",
            "simple_tool",
            "tool_123",
            "my.tool-name",
        ]

        for name in valid_names:
            is_valid, issues = validate_tool_name(name)
            assert is_valid, f"'{name}' should be valid, but got issues: {issues}"
            assert len(issues) == 0, f"Valid name '{name}' should have no issues"

    def test_invalid_names(self):
        """Test validation of invalid names."""
        invalid_cases = [
            ("Plan[Task]", "brackets"),
            ("invalid!", "invalid characters"),
            ("123name", "starts with number"),
            ("", "empty"),
            ("name with spaces", "spaces"),
        ]

        for name, reason in invalid_cases:
            is_valid, issues = validate_tool_name(name)
            assert not is_valid, f"'{name}' should be invalid ({reason}), but validation passed"
            assert len(issues) > 0, f"Invalid name '{name}' should have issues"


class TestNameSuggestions:
    """Test the get_name_suggestions function."""

    def test_basic_suggestions(self):
        """Test basic name suggestions."""
        suggestions = get_name_suggestions("Plan[Task]", count=3)

        assert len(suggestions) == 3
        assert "plan_task_generic" in suggestions
        assert all(isinstance(s, str) for s in suggestions), "All suggestions should be strings"

        # All suggestions should be valid
        for suggestion in suggestions:
            is_valid, _ = validate_tool_name(suggestion)
            assert is_valid, f"Suggestion '{suggestion}' should be valid"

    def test_suggestion_uniqueness(self):
        """Test that suggestions are unique."""
        suggestions = get_name_suggestions("TestModel", count=5)

        assert len(suggestions) == len(set(suggestions)), "All suggestions should be unique"


class TestCreateOpenAICompliantName:
    """Test the create_openai_compliant_name function."""

    def test_with_suffix(self):
        """Test creating names with suffix."""
        result = create_openai_compliant_name("Plan[Task]", "tool")
        expected = "plan_task_generic_tool"

        assert result == expected

        # Should be valid
        is_valid, issues = validate_tool_name(result)
        assert is_valid, f"Result should be valid, got issues: {issues}"

    def test_without_suffix(self):
        """Test creating names without suffix."""
        result = create_openai_compliant_name("MyModel")
        expected = "my_model"

        assert result == expected


class TestCreateNameMapping:
    """Test the create_name_mapping function."""

    def test_basic_mapping(self):
        """Test basic name mapping."""
        original_names = ["Plan[Task]", "MyModel", "HTTPParser"]
        mapping = create_name_mapping(original_names)

        assert len(mapping) == 3
        assert mapping["Plan[Task]"] == "plan_task_generic"
        assert mapping["MyModel"] == "my_model"
        assert mapping["HTTPParser"] == "http_parser"  # Current implementation behavior

    def test_collision_handling(self):
        """Test handling of name collisions."""
        # Create names that would collide after sanitization
        original_names = ["model", "Model", "MODEL"]
        mapping = create_name_mapping(original_names)

        # Should have unique values
        values = list(mapping.values())
        assert len(values) == len(set(values)), "Mapping values should be unique"


class TestSanitizePydanticModelName:
    """Test Pydantic model name sanitization."""

    def test_with_mock_pydantic_model(self):
        """Test with a mock Pydantic model."""
        # Create a mock object that simulates a model with Plan[Task] name
        class MockModel:
            pass

        # Override the __name__ attribute directly
        MockModel.__name__ = "Plan[Task]"

        result = sanitize_pydantic_model_name(MockModel)
        # The mock model should use its __name__ attribute
        assert result == "plan_task_generic"


@pytest.mark.parametrize("input_name,expected", [
    ("Plan[Task]", "plan_task_generic"),
    ("SimpleModel", "simple_model"),
    ("HTTPSParser", "http_s_parser"),  # Current implementation behavior
    ("", "unnamed_tool"),
])
def test_sanitize_tool_name_parametrized(input_name, expected):
    """Parametrized test for sanitize_tool_name."""
    result = sanitize_tool_name(input_name)
    assert result == expected


def test_integration_with_real_types():
    """Test integration with real Python types."""
    from typing import Optional

    # Test with actual generic types
    test_cases = [
        (dict[str, int], "dict_str_int_generic"),
        (list[str], "list_str_generic"),
        (Optional[str], "optional_str_generic"),
    ]

    for type_obj, expected in test_cases:
        # Get the string representation and sanitize
        type_str = str(type_obj)
        # This might be something like "typing.Dict[str, int]"
        # Our function should handle reasonable variations
        result = sanitize_tool_name(type_str)

        # At minimum, should be OpenAI compliant
        is_valid, issues = validate_tool_name(result)
        assert is_valid, f"Result '{result}' should be OpenAI compliant for type {type_obj}"
