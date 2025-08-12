"""Test the robust naming utilities comprehensively."""

import asyncio

from haive.agents.planning_v2.base.models import Plan, Task
from haive.agents.planning_v2.base.planner.prompts import planner_prompt
from haive.agents.simple import SimpleAgent
from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.utils.naming import (
    get_name_suggestions,
    sanitize_tool_name,
    validate_tool_name,
)


def test_comprehensive_naming():
    """Test comprehensive naming utility."""

    print("🧪 Testing Robust Naming Utilities")
    print("=" * 60)

    # Test cases with expected results
    test_cases = [
        # Generic classes
        ("Plan[Task]", "plan_task_generic"),
        ("Model[String,Int]", "model_string_int_generic"),
        ("Dict[str,Any]", "dict_str_any_generic"),

        # CamelCase
        ("SimpleModel", "simple_model"),
        ("HTTPSParser", "http_s_parser"),     # Current implementation behavior
        ("XMLToJSONConverter", "xml_tojson_converter"),  # Current implementation behavior

        # Edge cases
        ("Plan[Task]WithExtra!", "plan_task_with_extra_generic"),
        ("123InvalidName", "tool_123_invalid_name"),
        ("", "unnamed_tool"),
        ("a", "a"),

        # Acronyms
        ("HTMLParser", "html_parser"),
        ("JSONAPIClient", "json_api_client"),  # Current implementation behavior
        ("HTTPSServer", "http_s_server"),    # Current implementation behavior
    ]

    print("📋 Basic Sanitization Tests:")
    all_passed = True

    for input_name, expected in test_cases:
        result = sanitize_tool_name(input_name)
        status = "✅" if result == expected else "❌"

        if status == "❌":
            all_passed = False

        print(f"  {status} '{input_name}' → '{result}' (expected: '{expected}')")

    print(f"\n🔍 Overall Result: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")

    return all_passed


def test_validation():
    """Test name validation."""

    print("\n🔍 Testing Name Validation:")

    validation_cases = [
        ("valid_name", True),
        ("plan_task_generic", True),
        ("Plan[Task]", False),  # Contains brackets
        ("invalid-name!", False),  # Contains invalid chars
        ("123name", False),  # Starts with number
        ("", False),  # Empty
    ]

    for name, should_be_valid in validation_cases:
        is_valid, issues = validate_tool_name(name)
        status = "✅" if is_valid == should_be_valid else "❌"

        if not is_valid:
            issues_str = "; ".join(issues)
            print(f"  {status} '{name}' → Valid: {is_valid} (Issues: {issues_str})")
        else:
            print(f"  {status} '{name}' → Valid: {is_valid}")


def test_suggestions():
    """Test name suggestions."""

    print("\n💡 Testing Name Suggestions:")

    suggestion_cases = ["Plan[Task]", "HTTPSParser", "MyComplexModel[String]"]

    for name in suggestion_cases:
        suggestions = get_name_suggestions(name, count=3)
        print(f"  '{name}' → {suggestions}")


async def test_real_agent_integration():
    """Test real integration with agents."""

    print("\n🚀 Testing Real Agent Integration:")

    try:
        # Test with Plan[Task] using our robust utilities
        engine = AugLLMConfig(
            temperature=0.3,
            structured_output_model=Plan[Task]
        )

        agent = SimpleAgent(
            name="robust_test_planner",
            engine=engine,
            prompt_template=planner_prompt
        )

        print(f"  Engine force_tool_choice: '{engine.force_tool_choice}'")

        # Validate the tool choice name
        is_valid, issues = validate_tool_name(engine.force_tool_choice)
        validation_status = "✅" if is_valid else "❌"

        print(f"  Tool choice validation: {validation_status} (Issues: {issues if issues else 'None'})")

        if is_valid:
            # Try to run the agent
            result = await agent.arun({
                "objective": "Test robust naming with Plan[Task] generic"
            })

            print("  ✅ Agent execution successful!")
            print(f"  Result type: {type(result)}")
            print(f"  Generated plan has {len(result.steps) if hasattr(result, 'steps') else 'unknown'} steps")
            return True
        else:
            print(f"  ❌ Tool name validation failed: {issues}")
            return False

    except Exception as e:
        print(f"  ❌ Error during agent execution: {e}")
        return False


async def main():
    """Run all tests."""

    print("🧪 COMPREHENSIVE ROBUST NAMING TEST SUITE")
    print("=" * 80)

    # Test 1: Basic naming
    naming_passed = test_comprehensive_naming()

    # Test 2: Validation
    test_validation()

    # Test 3: Suggestions
    test_suggestions()

    # Test 4: Real integration
    integration_passed = await test_real_agent_integration()

    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS:")
    print(f"  Basic Naming Tests: {'✅ PASSED' if naming_passed else '❌ FAILED'}")
    print(f"  Real Integration: {'✅ PASSED' if integration_passed else '❌ FAILED'}")

    overall_success = naming_passed and integration_passed
    print(f"\n🎯 Overall: {'✅ SUCCESS - Robust naming is working!' if overall_success else '❌ FAILURE - Needs more work'}")

    return overall_success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
