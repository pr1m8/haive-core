"""Mistral AI Provider Module.

This module implements the Mistral AI language model provider for the Haive framework,
supporting Mistral's family of high-performance open and commercial language models.

The provider handles API key management, model configuration, and safe imports of
the langchain-mistralai package dependencies.

Examples:
    Basic usage:

    .. code-block:: python

        from haive.core.models.llm.providers.mistral import MistralProvider

        provider = MistralProvider(
            model="mistral-large-latest",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With function calling::

        provider = MistralProvider(
            model="mistral-large-latest",
            temperature=0.1,
            max_tokens=2000
        )

.. autosummary::
   :toctree: generated/

   MistralProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class MistralProvider(BaseLLMProvider):
    """Mistral AI language model provider configuration.

    This provider supports Mistral's family of models including Mistral Large,
    Mistral Medium, Mistral Small, and the open Mixtral models.

    Attributes:
        provider (LLMProvider): Always LLMProvider.MISTRALAI
        model (str): The Mistral model to use
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        random_seed (int): Seed for reproducible generation
        safe_mode (bool): Enable content filtering

    Examples:
        Large model for complex tasks:

        .. code-block:: python

            provider = MistralProvider(
                model="mistral-large-latest",
                temperature=0.7,
                max_tokens=2000
            )

        Small model for fast inference::

            provider = MistralProvider(
                model="mistral-small-latest",
                temperature=0.1,
                max_tokens=500
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.MISTRALAI, description="Provider identifier"
    )

    # Mistral model parameters
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Sampling temperature (0.0-1.0 for Mistral)",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    random_seed: int | None = Field(
        default=None, description="Seed for reproducible generation"
    )
    safe_mode: bool = Field(default=False, description="Enable content filtering")

    def _get_chat_class(self) -> type[Any]:
        """Get the Mistral chat class."""
        try:
            from langchain_mistralai import ChatMistralAI

            return ChatMistralAI
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Mistral AI requires langchain-mistralai. Install with: pip install langchain-mistralai",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Mistral model."""
        return "mistral-large-latest"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-mistralai"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Mistral-specific initialization parameters."""
        params = {
            "model": self.model,
            **kwargs,
        }

        # Add model parameters if specified
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.random_seed is not None:
            params["random_seed"] = self.random_seed
        if self.safe_mode is not None:
            params["safe_mode"] = self.safe_mode

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["mistral_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "MISTRAL_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available Mistral models."""
        return [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "mixtral-8x7b-instruct",
            "mistral-7b-instruct",
            "codestral-latest",
        ]
