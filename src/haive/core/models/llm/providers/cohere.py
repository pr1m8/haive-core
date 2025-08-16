"""Cohere Provider Module.

This module implements the Cohere language model provider for the Haive framework,
supporting Cohere's Command models with advanced generation and reasoning capabilities.

The provider handles API key management, model configuration, and safe imports of
the langchain-cohere package dependencies.

Examples:
    Basic usage:

    .. code-block:: python

        from haive.core.models.llm.providers.cohere import CohereProvider

        provider = CohereProvider(
            model="command-r-plus",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom parameters::

        provider = CohereProvider(
            model="command-r",
            temperature=0.1,
            k=40,
            p=0.9,
            frequency_penalty=0.1
        )

.. autosummary::
   :toctree: generated/

   CohereProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class CohereProvider(BaseLLMProvider):
    """Cohere language model provider configuration.

    This provider supports Cohere's Command models including Command R+, Command R,
    and other models optimized for reasoning, generation, and multilingual tasks.

    Attributes:
        provider (LLMProvider): Always LLMProvider.COHERE
        model (str): The Cohere model to use
        temperature (float): Sampling temperature (0.0-5.0)
        max_tokens (int): Maximum tokens in response
        k (int): Top-k sampling parameter
        p (float): Top-p nucleus sampling parameter
        frequency_penalty (float): Frequency penalty parameter
        presence_penalty (float): Presence penalty parameter
        stop_sequences (list): Stop sequences for generation

    Examples:
        High-performance reasoning::

            provider = CohereProvider(
                model="command-r-plus",
                temperature=0.3,
                max_tokens=2000,
                k=40
            )

        Creative writing::

            provider = CohereProvider(
                model="command-r",
                temperature=0.9,
                p=0.95,
                frequency_penalty=0.2
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.COHERE, description="Provider identifier"
    )

    # Cohere model parameters
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Sampling temperature (0.0-5.0 for Cohere)",
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    k: int | None = Field(
        default=None, ge=0, le=500, description="Top-k sampling parameter"
    )
    p: float | None = Field(
        default=None, ge=0, le=1, description="Top-p nucleus sampling parameter"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=0, le=1, description="Frequency penalty parameter"
    )
    presence_penalty: float | None = Field(
        default=None, ge=0, le=1, description="Presence penalty parameter"
    )
    stop_sequences: list[str] | None = Field(
        default=None, description="Stop sequences for generation"
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the Cohere chat class."""
        try:
            from langchain_cohere import ChatCohere

            return ChatCohere
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Cohere requires langchain-cohere. Install with: pip install langchain-cohere",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Cohere model."""
        return "command-r-plus"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-cohere"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Cohere-specific initialization parameters."""
        params = {
            "model": self.model,
            **kwargs,
        }

        # Add model parameters if specified
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.k is not None:
            params["k"] = self.k
        if self.p is not None:
            params["p"] = self.p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop_sequences is not None:
            params["stop"] = self.stop_sequences

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["cohere_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "COHERE_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available Cohere models."""
        return [
            "command-r-plus",
            "command-r",
            "command-nightly",
            "command-light",
            "command-light-nightly",
        ]
