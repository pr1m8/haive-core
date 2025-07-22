"""Fireworks AI Provider Module.

This module implements the Fireworks AI language model provider for the Haive framework,
supporting fast inference for open-source models through Fireworks' optimized infrastructure.

The provider handles API key management, model configuration, and safe imports of
the langchain-fireworks package dependencies.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.fireworks import FireworksProvider

        provider = FireworksProvider(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With streaming::

        provider = FireworksProvider(
            model="accounts/fireworks/models/llama-v2-70b-chat",
            temperature=0.1,
            stream=True,
            top_p=0.9
        )

.. autosummary::
   :toctree: generated/

   FireworksProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class FireworksProvider(BaseLLMProvider):
    """Fireworks AI language model provider configuration.

    This provider supports high-speed inference for open-source models through
    Fireworks' optimized infrastructure, including Mixtral, Llama, and others.

    Attributes:
        provider (LLMProvider): Always LLMProvider.FIREWORKS_AI
        model (str): The Fireworks model to use
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        stream (bool): Enable streaming responses
        stop (list): Stop sequences for generation

    Examples:
        Mixtral for reasoning::

            provider = FireworksProvider(
                model="accounts/fireworks/models/mixtral-8x7b-instruct",
                temperature=0.3,
                max_tokens=2000
            )

        Llama 2 with streaming::

            provider = FireworksProvider(
                model="accounts/fireworks/models/llama-v2-70b-chat",
                temperature=0.7,
                stream=True,
                top_p=0.9
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.FIREWORKS_AI, description="Provider identifier"
    )

    # Fireworks model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    top_k: int | None = Field(
        default=None, ge=1, description="Top-k sampling parameter"
    )
    stream: bool = Field(default=False, description="Enable streaming responses")
    stop: list[str] | None = Field(
        default=None, description="Stop sequences for generation"
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the Fireworks chat class."""
        try:
            from langchain_fireworks import ChatFireworks

            return ChatFireworks
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Fireworks AI requires langchain-fireworks. Install with: pip install langchain-fireworks",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Fireworks model."""
        return "accounts/fireworks/models/mixtral-8x7b-instruct"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-fireworks"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Fireworks-specific initialization parameters."""
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
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.stream is not None:
            params["streaming"] = self.stream
        if self.stop is not None:
            params["stop"] = self.stop

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["fireworks_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "FIREWORKS_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available Fireworks models."""
        return [
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/llama-v2-70b-chat",
            "accounts/fireworks/models/llama-v2-13b-chat",
            "accounts/fireworks/models/llama-v2-7b-chat",
            "accounts/fireworks/models/codellama-34b-instruct",
            "accounts/fireworks/models/yi-34b-200k-capybara",
        ]
