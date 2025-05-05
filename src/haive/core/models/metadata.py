"""
Model metadata utilities for LLM configurations.

This module provides utilities for downloading, caching, and accessing
model metadata from LiteLLM's model_prices_and_context_window.json.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Singleton metadata cache
_MODEL_METADATA_CACHE = {}
_METADATA_LAST_UPDATED = None
_METADATA_CACHE_TTL = timedelta(hours=24)
_METADATA_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_METADATA_CACHE_FILE = Path.home() / ".haive" / "cache" / "model_metadata.json"


def _ensure_cache_dir():
    """Ensure the cache directory exists."""
    cache_dir = _METADATA_CACHE_FILE.parent
    cache_dir.mkdir(parents=True, exist_ok=True)


def _load_metadata_from_cache() -> Dict[str, Any]:
    """Load metadata from cache file."""
    try:
        if _METADATA_CACHE_FILE.exists():
            with open(_METADATA_CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metadata from cache: {e}")
    return {}


def _save_metadata_to_cache(metadata: Dict[str, Any]) -> None:
    """Save metadata to cache file."""
    try:
        _ensure_cache_dir()
        with open(_METADATA_CACHE_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save metadata to cache: {e}")


def _download_metadata() -> Dict[str, Any]:
    """Download metadata from LiteLLM's repository."""
    try:
        response = requests.get(_METADATA_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        logger.warning(f"Failed to download metadata: {e}")
        return {}


def get_model_metadata(
    model_name: str, provider: Optional[str] = None, force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get metadata for a specific model.

    Args:
        model_name: Name of the model (e.g., "gpt-4", "claude-3-opus")
        provider: Optional provider prefix (e.g., "azure", "anthropic")
        force_refresh: Force download fresh metadata

    Returns:
        Model metadata dictionary or empty dict if not found
    """
    global _MODEL_METADATA_CACHE, _METADATA_LAST_UPDATED

    # Check if we need to refresh metadata
    if (
        force_refresh
        or not _MODEL_METADATA_CACHE
        or (
            _METADATA_LAST_UPDATED is None
            or datetime.now() - _METADATA_LAST_UPDATED > _METADATA_CACHE_TTL
        )
    ):
        # Try to load from cache first
        if not force_refresh:
            cache_data = _load_metadata_from_cache()
            if cache_data:
                _MODEL_METADATA_CACHE = cache_data
                _METADATA_LAST_UPDATED = datetime.now()

        # Download if still needed
        if force_refresh or not _MODEL_METADATA_CACHE:
            fresh_data = _download_metadata()
            if fresh_data:
                _MODEL_METADATA_CACHE = fresh_data
                _METADATA_LAST_UPDATED = datetime.now()
                _save_metadata_to_cache(fresh_data)

    # No metadata available
    if not _MODEL_METADATA_CACHE:
        return {}

    # Try exact match with provider prefix
    if provider:
        provider_model_name = f"{provider}/{model_name}"
        if provider_model_name in _MODEL_METADATA_CACHE:
            return _MODEL_METADATA_CACHE[provider_model_name]

    # Try exact match without provider
    if model_name in _MODEL_METADATA_CACHE:
        return _MODEL_METADATA_CACHE[model_name]

    # Try matching parts
    normalized_name = model_name.lower().strip()

    # Try with provider prefix
    if provider:
        for key in _MODEL_METADATA_CACHE.keys():
            if key == "sample_spec":
                continue
            if key.startswith(f"{provider}/") and normalized_name in key.lower():
                return _MODEL_METADATA_CACHE[key]

    # Try without provider
    for key in _MODEL_METADATA_CACHE.keys():
        if key == "sample_spec":
            continue
        if normalized_name in key.lower():
            # If provider specified, prioritize keys with that provider
            if (
                provider
                and ("litellm_provider" in _MODEL_METADATA_CACHE[key])
                and (
                    _MODEL_METADATA_CACHE[key]["litellm_provider"].lower()
                    == provider.lower()
                )
            ):
                return _MODEL_METADATA_CACHE[key]
            # Otherwise just return first match
            if "litellm_provider" in _MODEL_METADATA_CACHE[key]:
                return _MODEL_METADATA_CACHE[key]

    return {}


def get_context_window(model_name: str, provider: Optional[str] = None) -> int:
    """
    Get the context window size for a model.

    Args:
        model_name: Name of the model
        provider: Optional provider name

    Returns:
        Context window size or 0 if not found
    """
    model_info = get_model_metadata(model_name, provider)
    if not model_info:
        return 0

    # Return max_tokens if specified
    if "max_tokens" in model_info:
        return model_info["max_tokens"]

    # Otherwise calculate from input + output tokens
    input_tokens = model_info.get("max_input_tokens", 0)
    output_tokens = model_info.get("max_output_tokens", 0)
    return input_tokens + output_tokens


def get_token_pricing(
    model_name: str, provider: Optional[str] = None
) -> Tuple[float, float]:
    """
    Get input and output token pricing for a model.

    Args:
        model_name: Name of the model
        provider: Optional provider name

    Returns:
        Tuple of (input_cost_per_token, output_cost_per_token)
    """
    model_info = get_model_metadata(model_name, provider)
    if not model_info:
        return (0.0, 0.0)

    input_cost = model_info.get("input_cost_per_token", 0.0)
    output_cost = model_info.get("output_cost_per_token", 0.0)
    return (input_cost, output_cost)


def model_supports_feature(
    model_name: str, feature: str, provider: Optional[str] = None
) -> bool:
    """
    Check if a model supports a specific feature.

    Args:
        model_name: Name of the model
        feature: Feature to check (e.g., "vision", "function_calling")
        provider: Optional provider name

    Returns:
        True if the model supports the feature, False otherwise
    """
    model_info = get_model_metadata(model_name, provider)
    if not model_info:
        return False

    # Handle different feature naming conventions
    feature_key = f"supports_{feature}"
    if feature_key in model_info:
        return bool(model_info[feature_key])

    return False


class ModelMetadataMixin:
    """
    Mixin to add model metadata methods to LLMConfig classes.
    """

    def get_context_window(self) -> int:
        """Get the context window size for this model."""
        provider_value = getattr(self, "provider", None)
        if hasattr(provider_value, "value"):
            provider_value = provider_value.value
        else:
            provider_value = str(provider_value) if provider_value else None

        model_name = getattr(self, "model", "")
        return get_context_window(model_name, provider_value)

    def get_token_pricing(self) -> Tuple[float, float]:
        """Get token pricing for this model."""
        provider_value = getattr(self, "provider", None)
        if hasattr(provider_value, "value"):
            provider_value = provider_value.value
        else:
            provider_value = str(provider_value) if provider_value else None

        model_name = getattr(self, "model", "")
        return get_token_pricing(model_name, provider_value)

    def supports_feature(self, feature: str) -> bool:
        """Check if this model supports a specific feature."""
        provider_value = getattr(self, "provider", None)
        if hasattr(provider_value, "value"):
            provider_value = provider_value.value
        else:
            provider_value = str(provider_value) if provider_value else None

        model_name = getattr(self, "model", "")
        return model_supports_feature(model_name, feature, provider_value)

    @property
    def supports_vision(self) -> bool:
        """Check if model supports vision/image inputs."""
        return self.supports_feature("vision")

    @property
    def supports_function_calling(self) -> bool:
        """Check if model supports function calling."""
        return self.supports_feature("function_calling")

    @property
    def supports_system_messages(self) -> bool:
        """Check if model supports system messages."""
        return self.supports_feature("system_messages")


def add_metadata_methods(llm_config_class):
    """
    Add metadata methods to an existing LLMConfig class.

    Args:
        llm_config_class: The LLMConfig class to modify
    """
    # Add metadata methods
    llm_config_class.get_context_window = ModelMetadataMixin.get_context_window
    llm_config_class.get_token_pricing = ModelMetadataMixin.get_token_pricing
    llm_config_class.supports_feature = ModelMetadataMixin.supports_feature

    # Add property getters
    setattr(llm_config_class, "supports_vision", ModelMetadataMixin.supports_vision)
    setattr(
        llm_config_class,
        "supports_function_calling",
        ModelMetadataMixin.supports_function_calling,
    )
    setattr(
        llm_config_class,
        "supports_system_messages",
        ModelMetadataMixin.supports_system_messages,
    )

    logger.debug(f"Added metadata methods to {llm_config_class.__name__}")
    return llm_config_class
