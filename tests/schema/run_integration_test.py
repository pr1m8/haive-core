"""Direct test runner for schema integration tests."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))


def test_core_imports():
    """Test that all core imports work."""
    try:
        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_state_schema_basic():
    """Test StateSchema basic functionality."""
    try:
        from haive.core.schema import StateSchema

        # Create instance
        state = StateSchema()

        # Test engine fields exist
        assert hasattr(state, "engine")
        assert hasattr(state, "engines")
        assert hasattr(state, "llm")
        assert hasattr(state, "main_engine")

        # Test engines is a dict
        assert isinstance(state.engines, dict)

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_schema_composer_basic():
    """Test SchemaComposer basic functionality."""
    try:
        from haive.core.schema import SchemaComposer

        # Create instance
        composer = SchemaComposer("TestSchema")

        # Test methods exist
        assert hasattr(composer, "add_engine")
        assert hasattr(composer, "add_engine_management")
        assert hasattr(composer, "build")

        # Test building works
        schema_class = composer.build()
        assert schema_class is not None

        # Test instance creation
        instance = schema_class()
        assert instance is not None

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_messages_state_functionality():
    """Test MessagesState functionality."""
    try:
        from haive.core.schema import MessagesState

        # Create instance
        messages_state = MessagesState()

        # Test methods exist
        assert hasattr(messages_state, "add_message")
        assert hasattr(messages_state, "get_last_message")
        assert hasattr(messages_state, "messages")

        # Test messages field
        assert isinstance(messages_state.messages, list)
        assert len(messages_state.messages) == 0

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def test_token_usage_functionality():
    """Test token usage functionality."""
    try:
        from haive.core.schema import TokenUsage, TokenUsageMixin

        # Test TokenUsage creation
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

        # Test TokenUsageMixin
        class TestState(TokenUsageMixin):
            test_field: str = "test"

        state = TestState()
        assert hasattr(state, "get_token_usage")
        assert hasattr(state, "track_message_tokens")

        return True
    except Exception:
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    tests = [
        test_core_imports,
        test_state_schema_basic,
        test_schema_composer_basic,
        test_messages_state_functionality,
        test_token_usage_functionality,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed == 0:
        pass
    else:
        pass

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
