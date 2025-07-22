"""xAI Provider Module.

This module implements the xAI language model provider for the Haive framework,
supporting Grok models developed by Elon Musk's xAI company.

The provider handles API key management, model configuration, and safe imports of
the langchain-xai package dependencies.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.xai import XAIProvider

        provider = XAIProvider(
            model="grok-beta",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom parameters::

        provider = XAIProvider(
            model="grok-1",
            temperature=0.1,
            top_p=0.9,
            stream=True
        )

.. autosummary::
   :toctree: generated/

   XAIProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class XAIProvider(BaseLLMProvider):
    """xAI language model provider configuration.

    This provider supports xAI's Grok family of models known for their
    real-time information access and conversational capabilities.

    Attributes:
        provider (LLMProvider): Always LLMProvider.XAI
        model (str): The xAI model to use
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        stream (bool): Enable streaming responses
        stop (list): Stop sequences for generation

    Examples:
        Grok Beta for general conversation::

            provider = XAIProvider(
                model="grok-beta",
                temperature=0.7,
                max_tokens=2000
            )

        Grok with streaming::

            provider = XAIProvider(
                model="grok-1",
                temperature=0.1,
                stream=True,
                top_p=0.9
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.XAI, description="Provider identifier"
    )

    # xAI model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    stream: bool = Field(default=False, description="Enable streaming responses")
    stop: list[str] | None = Field(
        default=None, description="Stop sequences for generation"
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the xAI chat class."""
        try:
            from langchain_xai import ChatXAI

            return ChatXAI
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="xAI requires langchain-xai. Install with: pip install langchain-xai",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default xAI model."""
        return "grok-beta"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-xai"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get xAI-specific initialization parameters."""
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
        if self.stream is not None:
            params["streaming"] = self.stream
        if self.stop is not None:
            params["stop"] = self.stop

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["xai_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "XAI_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available xAI models."""
        return ["grok-beta", "grok-1", "grok-vision-beta"]
