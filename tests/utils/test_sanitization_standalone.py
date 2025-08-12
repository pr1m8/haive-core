"""Test the sanitization function directly."""

from haive.core.common.mixins.tool_route_mixin import sanitize_tool_name

# Test cases
test_cases = [
    ("Plan[Task]", "plan_task_generic"),
    ("MyModel[String]", "my_model_string_generic"),
    ("SimpleClass", "simple_class"),
    ("HTMLParser", "html_parser"),
    ("CamelCaseExample", "camel_case_example"),
]

print("Testing sanitize_tool_name function:")
print("=" * 50)

for input_name, expected in test_cases:
    result = sanitize_tool_name(input_name)
    status = "✅" if result == expected else "❌"
    print(f"{status} {input_name} → {result} (expected: {expected})")

print("\n" + "=" * 50)
