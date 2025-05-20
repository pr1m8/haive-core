"""
Secure configuration mixin for API credentials.

This module provides a mixin for secure handling of API credentials
with environment variable fallbacks and validation logic.
"""

import logging
import os
from typing import Optional

from pydantic import SecretStr, field_validator

logger = logging.getLogger(__name__)


class SecureConfigMixin:
    """
    A mixin to provide secure and flexible configuration for API keys.

    This mixin enables:
    1. Dynamic API key resolution from multiple sources
    2. Secure storage using SecretStr
    3. Environment variable fallbacks based on provider type
    4. Validation and error reporting
    """

    @field_validator("api_key", mode="after")
    @classmethod
    def _validate_api_key(cls, v, values):
        """
        Dynamically set the API key with robust fallback mechanism.

        1. Use explicitly provided value
        2. Try environment variable based on provider
        3. Fall back to default/empty
        """
        # If a value is already set and not empty, return it
        if v is not None and v != "":
            # If it's not a SecretStr, convert it
            if not isinstance(v, SecretStr):
                return SecretStr(str(v))
            return v

        # Determine the environment variable based on provider
        provider = values.get("provider")
        if not provider:
            return SecretStr("")

        logger.debug(f"Validating API key for provider: {provider}")

        # Create mapping for both enum values and string values
        env_key_map = {
            # Using .value to handle enum objects
            "azure": "AZURE_OPENAI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "mistralai": "MISTRAL_API_KEY",
            "groq": "GROQ_API_KEY",
            "cohere": "COHERE_API_KEY",
            "together_ai": "TOGETHER_AI_API_KEY",
            "fireworks_ai": "FIREWORKS_AI_API_KEY",
            "perplexity": "PERPLEXITY_API_KEY",
            "huggingface": "HUGGING_FACE_API_KEY",
            "ai21": "AI21_API_KEY",
            "aleph_alpha": "ALEPH_ALPHA_API_KEY",
            "gooseai": "GOOSEAI_API_KEY",
            "mosaicml": "MOSAICML_API_KEY",
            "nlp_cloud": "NLP_CLOUD_API_KEY",
            "openlm": "OPENLM_API_KEY",
            "petals": "PETALS_API_KEY",
            "replicate": "REPLICATE_API_KEY",
            "vertex_ai": "VERTEX_AI_API_KEY",
        }

        # Get provider value - handle both enum and string
        provider_value = provider.value if hasattr(provider, "value") else str(provider)

        # Try to get API key from environment variables
        env_key = env_key_map.get(provider_value.lower())

        if env_key:
            logger.debug(f"Looking for environment variable: {env_key}")
            api_key = os.getenv(env_key)
            if api_key and api_key.strip():
                logger.debug(
                    f"Found API key for {provider_value} in {env_key} (length: {len(api_key)})"
                )
                return SecretStr(api_key)
            else:
                logger.warning(f"Environment variable {env_key} not found or empty")
        else:
            logger.warning(
                f"No environment mapping found for provider: {provider_value}"
            )

        # If no key found, return an empty SecretStr
        logger.debug(
            f"No API key found for {provider_value}, returning empty SecretStr"
        )
        return SecretStr("")

    def get_api_key(self) -> Optional[str]:
        """
        Safely retrieve the API key with improved error handling.

        Returns:
            Optional[str]: The API key value, or None if not set
        """
        try:
            # Check if api_key attribute exists and is not None
            if not hasattr(self, "api_key") or self.api_key is None:
                logger.warning(
                    f"No API key attribute found for {getattr(self, 'provider', 'unknown provider')}"
                )
                return None

            # Try to get the secret value
            key_value = self.api_key.get_secret_value()

            # Check if the key is actually empty
            if not key_value or key_value.strip() == "":
                logger.warning(
                    f"API key for {getattr(self, 'provider', 'unknown provider')} is empty"
                )
                return None

            return key_value
        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")

            # If we're in a testing environment, use fake keys for development
            if (
                os.getenv("ENVIRONMENT") == "development"
                or os.getenv("TESTING") == "true"
            ):
                provider_value = getattr(self, "provider", "")
                provider_str = (
                    provider_value.value
                    if hasattr(provider_value, "value")
                    else str(provider_value)
                )
                logger.debug(
                    f"Using fake test key for {provider_str} in development environment"
                )
                return f"test_key_{provider_str}"

            return None
