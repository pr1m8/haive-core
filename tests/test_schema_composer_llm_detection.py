"""Test that SchemaComposer properly detects LLM engines and uses LLMState."""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.schema.prebuilt.llm_state import LLMState
from haive.core.schema.schema_composer import SchemaComposer


def test_aug_llm_config_uses_llm_state():
    """Test that AugLLMConfig results in LLMState being used as base class."""
    # Create AugLLMConfig
    config = AugLLMConfig(temperature=0.7, name="test_llm")

    # Create schema from components
    schema = SchemaComposer.from_components([config], name="TestSchema")

    # Check the MRO to see which base class was used
    mro_names = [cls.__name__ for cls in schema.__mro__]
    print(f"Schema MRO: {mro_names}")

    # LLMState should be in the MRO
    assert "LLMState" in mro_names, f"Expected LLMState in MRO but got: {mro_names}"

    # MessagesState should NOT be directly in MRO (LLMState inherits from it)
    # But it will appear later in the chain through LLMState
    llm_index = mro_names.index("LLMState")
    messages_index = (
        mro_names.index("MessagesState") if "MessagesState" in mro_names else -1
    )

    if messages_index != -1:
        assert (
            llm_index < messages_index
        ), "LLMState should come before MessagesState in MRO"

    # Schema should have all required fields
    assert "engine" in schema.model_fields
    assert "engines" in schema.model_fields
    assert "messages" in schema.model_fields
    assert (
        "token_usage" in schema.model_fields
    ), "LLMState should provide token_usage field"

    # Should be able to create an instance
    # Need to provide the engine since it's required
    instance = schema(engine=config)
    assert hasattr(instance, "token_usage")
    assert hasattr(instance, "engine")
    assert hasattr(instance, "messages")


def test_schema_composer_detects_llm_engine_early():
    """Test that the detection logic properly identifies LLM engines."""
    composer = SchemaComposer(name="TestSchema")

    # Create AugLLMConfig
    config = AugLLMConfig(temperature=0.7, name="test_llm")

    # Call detection directly
    composer._detect_base_class_requirements([config])

    # Check that has_llm_engine was detected
    assert (
        composer.detected_base_class == LLMState
    ), f"Expected LLMState but got {composer.detected_base_class}"

    # Check that engine was added to tracking
    assert "llm" in composer.engines_by_type
    assert "test_llm" in composer.engines_by_type["llm"]
