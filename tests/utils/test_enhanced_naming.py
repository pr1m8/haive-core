#!/usr/bin/env python3
"""Test the enhanced naming utility with complex type scenarios."""

from haive.core.utils.enhanced_naming import (
    analyze_naming_complexity,
    enhanced_sanitize_tool_name,
    get_naming_suggestions_enhanced,
)


def test_complex_generics():
    """Test complex generic type handling."""
    test_cases = [
        # Simple cases
        "Plan[Task]",
        "List[str]",
        "Dict[str, int]",

        # Nested generics
        "List[Dict[str, Task]]",
        "Dict[str, List[Task]]",
        "Optional[List[Dict[str, Task]]]",

        # Union types
        "Union[str, int]",
        "Union[str, List[Task]]",
        "Union[str, Optional[Plan[Task]]]",

        # Very complex
        "Dict[str, List[Optional[Plan[Task]]]]",
        "Union[str, Dict[str, List[Task]], Optional[Plan[Status]]]",
    ]

    print("🧪 TESTING ENHANCED NAMING UTILITY")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: {test_case}")

        try:
            sanitized_name, metadata = enhanced_sanitize_tool_name(test_case, include_metadata=True)

            print(f"    ✅ Result: {sanitized_name}")

            if metadata:
                print(f"    📊 Type: {metadata.transformation_type}")
                print(f"    📈 Complexity: {metadata.complexity_level}")
                print(f"    🏗️  Nesting: {metadata.nesting_depth}")
                print(f"    📝 Description: {metadata.description}")
                print(f"    🔧 Parameters: {metadata.type_parameters}")

                if metadata.warnings:
                    print("    ⚠️  Warnings:")
                    for warning in metadata.warnings:
                        print(f"        - {warning}")
            else:
                print("    ❌ No metadata returned")

        except Exception as e:
            print(f"    💥 ERROR: {e}")
            import traceback
            traceback.print_exc()


def test_complexity_analysis():
    """Test complexity analysis across multiple types."""
    print("\n\n🔍 COMPLEXITY ANALYSIS")
    print("=" * 40)

    complex_types = [
        "str",  # Simple
        "Plan[Task]",  # Basic generic
        "List[Dict[str, Task]]",  # Nested
        "Union[str, Optional[Plan[Task]]]",  # Complex union
        "Dict[str, List[Optional[Plan[Status]]]]",  # Very complex
    ]

    analysis = analyze_naming_complexity(complex_types)

    print(f"📊 Total types analyzed: {analysis['total_count']}")
    print(f"📈 Average complexity: {analysis['average_complexity']:.1f}")
    print(f"⚠️  Total warnings: {analysis['total_warnings']}")

    if analysis["most_complex"]:
        print("🏆 Most complex:")
        print(f"    Name: {analysis['most_complex']['name']}")
        print(f"    Complexity: {analysis['most_complex']['complexity']}")
        print(f"    Description: {analysis['most_complex']['description']}")

    print("📈 Complexity distribution:")
    for level, count in analysis["complexity_distribution"].items():
        print(f"    {level}: {count} types")

    if analysis["recommendations"]:
        print("💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"    - {rec}")


def test_naming_suggestions():
    """Test enhanced naming suggestions."""
    print("\n\n💡 NAMING SUGGESTIONS")
    print("=" * 30)

    test_type = "Union[str, Optional[Plan[Task]]]"
    print(f"Getting suggestions for: {test_type}")

    suggestions = get_naming_suggestions_enhanced(test_type, count=3)

    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['name']}")
        print(f"   Strategy: {suggestion['strategy']}")
        print(f"   Description: {suggestion['description']}")
        if suggestion["metadata"]:
            print(f"   Complexity: {suggestion['metadata'].complexity_level}")


def test_real_world_scenarios():
    """Test real-world complex type scenarios."""
    print("\n\n🌍 REAL-WORLD SCENARIOS")
    print("=" * 35)

    scenarios = [
        # Pydantic models
        ("SearchResults[Document]", "Pydantic model with generic"),
        ("APIResponse[List[User]]", "API response with nested generic"),

        # Complex domain models
        ("WorkflowStep[InputSchema, OutputSchema]", "Multi-param generic"),
        ("EventHandler[Union[ClickEvent, KeyEvent]]", "Event handler with union"),

        # Data structures
        ("CacheEntry[str, Dict[str, Any]]", "Cache with complex value type"),
        ("ValidationResult[Optional[ErrorDetails]]", "Validation with optional error"),
    ]

    for scenario, description in scenarios:
        print(f"\n📋 Scenario: {description}")
        print(f"    Input: {scenario}")

        name, meta = enhanced_sanitize_tool_name(scenario, include_metadata=True)
        print(f"    Output: {name}")

        if meta:
            print(f"    Complexity: {meta.complexity_level} | Nesting: {meta.nesting_depth}")
            if meta.warnings:
                print(f"    Warnings: {len(meta.warnings)}")


if __name__ == "__main__":
    test_complex_generics()
    test_complexity_analysis()
    test_naming_suggestions()
    test_real_world_scenarios()

    print("\n\n✅ ALL ENHANCED NAMING TESTS COMPLETED")
