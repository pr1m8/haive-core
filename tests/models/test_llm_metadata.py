"""
Tests for model metadata functionality in LLM configurations.

This module tests the metadata features added to the LLM configuration classes
with enhanced logging and clarity for better test diagnostics.
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
from pydantic import SecretStr

# Setup logging for better test diagnostics
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
test_logger = logging.getLogger("llm_metadata_tests")

from haive.core.models.llm.base import (
    AnthropicLLMConfig,
    GeminiLLMConfig,
    LLMConfig,
    OpenAILLMConfig,
)
from haive.core.models.llm.provider_types import LLMProvider

# Import the modules to test - using the correct path
from haive.core.models.metadata import (
    add_metadata_methods,
    get_context_window,
    get_model_metadata,
    get_token_pricing,
    model_supports_feature,
)


# Print test banner for better visibility
def print_test_banner(test_name):
    """Print a visible banner around test name for easier test log reading."""
    banner = "=" * 80
    test_logger.info(f"\n{banner}\nRUNNING TEST: {test_name}\n{banner}")


# Use a test cache directory to avoid interfering with real cache
@pytest.fixture(scope="function")
def temp_cache_dir():
    """Create a temporary directory for cache files."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        cache_dir = Path(tmpdirname)
        test_logger.info(f"Created temporary cache directory: {cache_dir}")

        # Patch the cache file path
        with patch(
            "haive.core.models.metadata._METADATA_CACHE_FILE",
            cache_dir / "model_metadata.json",
        ):
            # Reset the global cache for each test
            from haive.core.models.metadata import (
                _METADATA_LAST_UPDATED,
                _MODEL_METADATA_CACHE,
            )

            _MODEL_METADATA_CACHE.clear()
            _METADATA_LAST_UPDATED = None
            test_logger.info("Reset metadata cache and last updated timestamp")
            yield cache_dir


# Sample metadata that matches what's actually being downloaded
@pytest.fixture
def sample_metadata():
    """Sample metadata that matches real values."""
    metadata = {
        "gpt-4": {
            "max_tokens": 4096,
            "max_input_tokens": 8192,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_prompt_caching": True,
            "supports_system_messages": True,
            "supports_tool_choice": True,
        },
        "gpt-4-turbo": {
            "max_tokens": 4096,
            "max_input_tokens": 128000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.00001,
            "output_cost_per_token": 0.00003,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_prompt_caching": True,
            "supports_system_messages": True,
            "supports_tool_choice": True,
        },
        "claude-3-opus-20240229": {
            "max_tokens": 4096,
            "max_input_tokens": 200000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.000015,
            "output_cost_per_token": 0.000075,
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_system_messages": True,
        },
        "gemini-1.5-pro": {
            "max_tokens": 4096,
            "max_input_tokens": 1000000,
            "max_output_tokens": 4096,
            "input_cost_per_token": 0.000005,
            "output_cost_per_token": 0.000005,
            "litellm_provider": "gemini",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_system_messages": True,
        },
    }

    test_logger.info(f"Created sample metadata with {len(metadata)} models")
    return metadata


@pytest.fixture
def setup_metadata_cache(temp_cache_dir, sample_metadata):
    """Set up the metadata cache with sample data."""
    cache_file = temp_cache_dir / "model_metadata.json"
    with open(cache_file, "w") as f:
        json.dump(sample_metadata, f)

    test_logger.info(f"Saved sample metadata to cache file: {cache_file}")

    # Force a reload from the cache
    from haive.core.models.metadata import _load_metadata_from_cache

    loaded_metadata = _load_metadata_from_cache()
    test_logger.info(f"Loaded metadata from cache with {len(loaded_metadata)} models")

    return sample_metadata


# Add unittest.mock import here
from unittest.mock import MagicMock, patch


def test_get_model_metadata(setup_metadata_cache):
    """Test retrieving model metadata."""
    print_test_banner("test_get_model_metadata")

    # Use real function (no mocking)
    metadata = get_model_metadata("gpt-4")

    # Print the actual metadata for clarity
    test_logger.info(f"Retrieved metadata for gpt-4: {json.dumps(metadata, indent=2)}")

    # Verify we get expected values
    assert (
        metadata["max_tokens"] == 4096
    ), f"Expected max_tokens to be 4096, got {metadata.get('max_tokens')}"
    assert (
        metadata["input_cost_per_token"] == 0.00003
    ), f"Expected input_cost to be 0.00003, got {metadata.get('input_cost_per_token')}"
    assert (
        metadata["litellm_provider"] == "openai"
    ), f"Expected litellm_provider to be 'openai', got {metadata.get('litellm_provider')}"

    # Test with provider
    metadata = get_model_metadata("gpt-4", "openai")
    test_logger.info(
        f"Retrieved metadata for gpt-4 with provider 'openai': {metadata.get('litellm_provider')}"
    )
    assert metadata["litellm_provider"] == "openai"

    # Test non-existent model
    metadata = get_model_metadata("non-existent-model")
    test_logger.info(f"Retrieved metadata for non-existent model: {metadata}")
    assert metadata == {}


def test_get_context_window(setup_metadata_cache):
    """Test getting context window size."""
    print_test_banner("test_get_context_window")

    # Test model with max_tokens
    context_window = get_context_window("gpt-4")
    test_logger.info(f"Context window for gpt-4: {context_window}")
    assert (
        context_window == 4096
    ), f"Expected context window of 4096, got {context_window}"

    # Test model with input/output tokens
    context_window = get_context_window("gpt-4-turbo")
    test_logger.info(f"Context window for gpt-4-turbo: {context_window}")
    assert (
        context_window == 4096
    ), f"Expected context window of 4096, got {context_window}"

    # Test non-existent model
    context_window = get_context_window("non-existent-model")
    test_logger.info(f"Context window for non-existent model: {context_window}")
    assert context_window == 0, f"Expected context window of 0, got {context_window}"


def test_get_token_pricing(setup_metadata_cache):
    """Test getting token pricing."""
    print_test_banner("test_get_token_pricing")

    # Test normal model
    input_cost, output_cost = get_token_pricing("gpt-4")
    test_logger.info(
        f"Token pricing for gpt-4: input=${input_cost}, output=${output_cost}"
    )
    assert input_cost == 0.00003, f"Expected input cost of 0.00003, got {input_cost}"
    assert output_cost == 0.00006, f"Expected output cost of 0.00006, got {output_cost}"

    # Test with provider
    input_cost, output_cost = get_token_pricing("claude-3-opus-20240229", "anthropic")
    test_logger.info(
        f"Token pricing for claude-3-opus with provider 'anthropic': input=${input_cost}, output=${output_cost}"
    )
    assert input_cost == 0.000015, f"Expected input cost of 0.000015, got {input_cost}"
    assert (
        output_cost == 0.000075
    ), f"Expected output cost of 0.000075, got {output_cost}"

    # Test non-existent model
    input_cost, output_cost = get_token_pricing("non-existent-model")
    test_logger.info(
        f"Token pricing for non-existent model: input=${input_cost}, output=${output_cost}"
    )
    assert input_cost == 0.0, f"Expected input cost of 0.0, got {input_cost}"
    assert output_cost == 0.0, f"Expected output cost of 0.0, got {output_cost}"


def test_model_supports_feature(setup_metadata_cache):
    """Test checking feature support."""
    print_test_banner("test_model_supports_feature")

    # Test supported feature
    result = model_supports_feature("gpt-4", "function_calling")
    test_logger.info(f"gpt-4 supports function_calling: {result}")
    assert result is True, "Expected gpt-4 to support function_calling"

    # Test unsupported feature
    # Note: If the real metadata differs, adjust this test
    result = model_supports_feature("gpt-4", "vision")
    test_logger.info(f"gpt-4 supports vision: {result}")
    assert result is False, "Expected gpt-4 to not support vision"

    # Test model with vision support
    result = model_supports_feature("gpt-4-turbo", "vision")
    test_logger.info(f"gpt-4-turbo supports vision: {result}")
    assert result is True, "Expected gpt-4-turbo to support vision"

    # Test non-existent feature
    result = model_supports_feature("gpt-4", "non_existent_feature")
    test_logger.info(f"gpt-4 supports non_existent_feature: {result}")
    assert result is False, "Expected gpt-4 to not support non_existent_feature"

    # Test non-existent model
    result = model_supports_feature("non-existent-model", "vision")
    test_logger.info(f"non-existent-model supports vision: {result}")
    assert result is False, "Expected non-existent-model to not support vision"


# Tests for the LLMConfig integration
def test_add_metadata_methods():
    """Test adding metadata methods to a class."""
    print_test_banner("test_add_metadata_methods")

    # Create a simple class
    class TestConfig:
        provider = "test"
        model = "test-model"

    # Add metadata methods
    add_metadata_methods(TestConfig)

    # Check if methods were added
    for method in ["get_context_window", "get_token_pricing", "supports_feature"]:
        assert hasattr(TestConfig, method), f"Method {method} not added to class"
        test_logger.info(f"Method '{method}' successfully added to class")

    # Check property getters
    for prop in [
        "supports_vision",
        "supports_function_calling",
        "supports_system_messages",
    ]:
        prop_descriptor = getattr(TestConfig, prop, None)
        assert prop_descriptor is not None, f"Property {prop} not added to class"
        assert isinstance(prop_descriptor, property), f"{prop} is not a property"
        test_logger.info(f"Property '{prop}' successfully added to class")


# For the LLMConfig tests, we'll need to patch the model_post_init method to avoid
# it calling metadata methods during initialization
@pytest.fixture
def patch_model_post_init():
    """Patch the model_post_init method to avoid calling metadata during initialization."""
    with patch.object(LLMConfig, "model_post_init", lambda self, _: None):
        test_logger.info("Patched LLMConfig.model_post_init to do nothing during tests")
        yield


def test_llm_config_get_context_window(patch_model_post_init, setup_metadata_cache):
    """Test the get_context_window method on LLMConfig."""
    print_test_banner("test_llm_config_get_context_window")

    # Create config without triggering model_post_init
    config = OpenAILLMConfig(model="gpt-4-turbo", api_key=SecretStr("test-key"))
    test_logger.info(f"Created OpenAILLMConfig for model: {config.model}")

    # Call method directly
    context_window = config.get_context_window()
    test_logger.info(f"Retrieved context window for {config.model}: {context_window}")

    # Verify result matches our test data
    assert (
        context_window == 4096
    ), f"Expected context window of 4096, got {context_window}"


def test_llm_config_get_token_pricing(patch_model_post_init, setup_metadata_cache):
    """Test the get_token_pricing method on LLMConfig."""
    print_test_banner("test_llm_config_get_token_pricing")

    # Create config
    config = OpenAILLMConfig(model="gpt-4-turbo", api_key=SecretStr("test-key"))
    test_logger.info(f"Created OpenAILLMConfig for model: {config.model}")

    # Test method
    input_cost, output_cost = config.get_token_pricing()
    test_logger.info(
        f"Token pricing for {config.model}: input=${input_cost}, output=${output_cost}"
    )

    # Verify result matches our test data
    assert input_cost == 0.00001, f"Expected input cost of 0.00001, got {input_cost}"
    assert output_cost == 0.00003, f"Expected output cost of 0.00003, got {output_cost}"


def test_llm_config_supports_feature(patch_model_post_init, setup_metadata_cache):
    """Test the supports_feature method on LLMConfig."""
    print_test_banner("test_llm_config_supports_feature")

    # Create config
    config = AnthropicLLMConfig(
        model="claude-3-opus-20240229", api_key=SecretStr("test-key")
    )
    test_logger.info(f"Created AnthropicLLMConfig for model: {config.model}")

    # Test method
    result = config.supports_feature("vision")
    test_logger.info(f"{config.model} supports vision: {result}")

    # Verify result matches our test data
    assert result is True, f"Expected {config.model} to support vision"


def test_llm_config_property_getters(patch_model_post_init, setup_metadata_cache):
    """Test the property getters for feature support."""
    print_test_banner("test_llm_config_property_getters")

    # Create config
    config = GeminiLLMConfig(model="gemini-1.5-pro", api_key=SecretStr("test-key"))
    test_logger.info(f"Created GeminiLLMConfig for model: {config.model}")

    # Test properties
    vision_support = config.supports_vision
    test_logger.info(f"{config.model} supports vision: {vision_support}")
    assert vision_support is True, f"Expected {config.model} to support vision"

    function_support = config.supports_function_calling
    test_logger.info(f"{config.model} supports function_calling: {function_support}")
    assert (
        function_support is True
    ), f"Expected {config.model} to support function_calling"

    system_support = config.supports_system_messages
    test_logger.info(f"{config.model} supports system_messages: {system_support}")
    assert system_support is True, f"Expected {config.model} to support system_messages"


# Integration tests
def test_model_post_init_hook(setup_metadata_cache):
    """Test the model_post_init hook loads metadata."""
    print_test_banner("test_model_post_init_hook")

    with patch("haive.core.models.llm.base.logger") as mock_logger:
        # Create config (this will trigger model_post_init)
        config = OpenAILLMConfig(model="gpt-4", api_key=SecretStr("test-key"))
        test_logger.info(f"Created OpenAILLMConfig for model: {config.model}")

        # Verify log calls were made
        call_count = mock_logger.debug.call_count
        test_logger.info(
            f"Logger.debug was called {call_count} times during initialization"
        )
        assert call_count >= 1, f"Expected at least 1 debug log call, got {call_count}"

        # We should be able to get context window
        assert hasattr(
            config, "get_context_window"
        ), "Config missing get_context_window method"
        window = config.get_context_window()
        test_logger.info(f"Context window for {config.model}: {window}")
        assert window == 4096, f"Expected context window of 4096, got {window}"


def test_different_provider_configs(patch_model_post_init, setup_metadata_cache):
    """Test metadata works with different provider configs."""
    print_test_banner("test_different_provider_configs")

    # Create configs
    openai_config = OpenAILLMConfig(model="gpt-4", api_key=SecretStr("test"))
    anthropic_config = AnthropicLLMConfig(
        model="claude-3-opus-20240229", api_key=SecretStr("test")
    )
    gemini_config = GeminiLLMConfig(model="gemini-1.5-pro", api_key=SecretStr("test"))

    test_logger.info(
        f"Created configs for models: {openai_config.model}, {anthropic_config.model}, {gemini_config.model}"
    )

    # Test context windows
    openai_window = openai_config.get_context_window()
    test_logger.info(f"Context window for {openai_config.model}: {openai_window}")
    assert (
        openai_window == 4096
    ), f"Expected context window of 4096 for {openai_config.model}, got {openai_window}"

    anthropic_window = anthropic_config.get_context_window()
    test_logger.info(f"Context window for {anthropic_config.model}: {anthropic_window}")
    assert (
        anthropic_window == 4096
    ), f"Expected context window of 4096 for {anthropic_config.model}, got {anthropic_window}"

    gemini_window = gemini_config.get_context_window()
    test_logger.info(f"Context window for {gemini_config.model}: {gemini_window}")
    assert (
        gemini_window == 4096
    ), f"Expected context window of 4096 for {gemini_config.model}, got {gemini_window}"

    # Test pricing
    openai_pricing = openai_config.get_token_pricing()
    test_logger.info(
        f"Token pricing for {openai_config.model}: input=${openai_pricing[0]}, output=${openai_pricing[1]}"
    )
    assert openai_pricing == (
        0.00003,
        0.00006,
    ), f"Expected (0.00003, 0.00006) for {openai_config.model}, got {openai_pricing}"

    anthropic_pricing = anthropic_config.get_token_pricing()
    test_logger.info(
        f"Token pricing for {anthropic_config.model}: input=${anthropic_pricing[0]}, output=${anthropic_pricing[1]}"
    )
    assert anthropic_pricing == (
        0.000015,
        0.000075,
    ), f"Expected (0.000015, 0.000075) for {anthropic_config.model}, got {anthropic_pricing}"

    gemini_pricing = gemini_config.get_token_pricing()
    test_logger.info(
        f"Token pricing for {gemini_config.model}: input=${gemini_pricing[0]}, output=${gemini_pricing[1]}"
    )
    assert gemini_pricing == (
        0.000005,
        0.000005,
    ), f"Expected (0.000005, 0.000005) for {gemini_config.model}, got {gemini_pricing}"

    # Test feature support
    openai_vision = openai_config.supports_vision
    test_logger.info(f"{openai_config.model} supports vision: {openai_vision}")
    assert (
        openai_vision is False
    ), f"Expected {openai_config.model} to not support vision"

    openai_function = openai_config.supports_function_calling
    test_logger.info(
        f"{openai_config.model} supports function_calling: {openai_function}"
    )
    assert (
        openai_function is True
    ), f"Expected {openai_config.model} to support function_calling"

    anthropic_vision = anthropic_config.supports_vision
    test_logger.info(f"{anthropic_config.model} supports vision: {anthropic_vision}")
    assert (
        anthropic_vision is True
    ), f"Expected {anthropic_config.model} to support vision"

    gemini_vision = gemini_config.supports_vision
    test_logger.info(f"{gemini_config.model} supports vision: {gemini_vision}")
    assert gemini_vision is True, f"Expected {gemini_config.model} to support vision"


# Error handling tests
def test_metadata_error_handling():
    """Test error handling in metadata functions."""
    print_test_banner("test_metadata_error_handling")

    # Reset module cache to ensure clean state
    from haive.core.models.metadata import _METADATA_LAST_UPDATED, _MODEL_METADATA_CACHE

    _MODEL_METADATA_CACHE.clear()
    _METADATA_LAST_UPDATED = None
    test_logger.info("Reset metadata cache and last updated timestamp")

    # Use a non-existent cache path to simulate missing file
    with patch(
        "haive.core.models.metadata._METADATA_CACHE_FILE",
        Path("/nonexistent/path/metadata.json"),
    ):
        # Also patch download to fail
        with patch("haive.core.models.metadata._download_metadata", return_value={}):
            test_logger.info(
                "Patched metadata file path and download function to simulate failure"
            )

            # Test functions should return defaults
            metadata = get_model_metadata("gpt-4")
            test_logger.info(
                f"Retrieved metadata for gpt-4 with simulated failure: {metadata}"
            )
            assert metadata == {}, f"Expected empty dict, got {metadata}"

            context_window = get_context_window("gpt-4")
            test_logger.info(
                f"Retrieved context window for gpt-4 with simulated failure: {context_window}"
            )
            assert context_window == 0, f"Expected 0, got {context_window}"

            pricing = get_token_pricing("gpt-4")
            test_logger.info(
                f"Retrieved token pricing for gpt-4 with simulated failure: {pricing}"
            )
            assert pricing == (0.0, 0.0), f"Expected (0.0, 0.0), got {pricing}"

            feature_support = model_supports_feature("gpt-4", "vision")
            test_logger.info(
                f"Checked if gpt-4 supports vision with simulated failure: {feature_support}"
            )
            assert feature_support is False, f"Expected False, got {feature_support}"


def test_download_fallback_to_cache(temp_cache_dir):
    """Test fallback to cache when download fails."""
    print_test_banner("test_download_fallback_to_cache")

    cache_file = temp_cache_dir / "model_metadata.json"

    # Create cache file with test data
    cache_data = {
        "gpt-4": {
            "max_tokens": 4096,
            "input_cost_per_token": 0.00003,
            "output_cost_per_token": 0.00006,
        }
    }

    with open(cache_file, "w") as f:
        json.dump(cache_data, f)

    test_logger.info(
        f"Created test cache file with data: {json.dumps(cache_data, indent=2)}"
    )

    # Mock download to fail
    with patch(
        "haive.core.models.metadata._download_metadata",
        side_effect=Exception("Download failed"),
    ):
        test_logger.info("Patched download function to fail with exception")

        # Reset module cache
        from haive.core.models.metadata import (
            _METADATA_LAST_UPDATED,
            _MODEL_METADATA_CACHE,
        )

        _MODEL_METADATA_CACHE.clear()
        _METADATA_LAST_UPDATED = None
        test_logger.info("Reset metadata cache and last updated timestamp")

        # Should fall back to cache
        metadata = get_model_metadata("gpt-4")
        test_logger.info(
            f"Retrieved metadata for gpt-4 with download failure (fallback to cache): {metadata}"
        )
        assert (
            metadata["max_tokens"] == 4096
        ), f"Expected max_tokens of 4096, got {metadata.get('max_tokens')}"


if __name__ == "__main__":
    pytest.main(["-v"])
