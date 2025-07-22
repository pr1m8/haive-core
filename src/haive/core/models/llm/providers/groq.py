"""Groq Provider Module.

This module implements the Groq language model provider for the Haive framework,
supporting ultra-fast inference with Groq's Language Processing Units (LPUs).

The provider handles API key management, model configuration, and safe imports of
the langchain-groq package dependencies for high-speed LLM inference.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.groq import GroqProvider

        provider = GroqProvider(
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With streaming for real-time responses::

        provider = GroqProvider(
            model="llama2-70b-4096",
            streaming=True,
            temperature=0.1
        )

.. autosummary::
   :toctree: generated/

   GroqProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class GroqProvider(BaseLLMProvider):
    """Groq language model provider configuration.

    This provider supports Groq's high-speed LLM inference including Mixtral,
    Llama 2, and other optimized models running on Language Processing Units.

    Attributes:
        provider (LLMProvider): Always LLMProvider.GROQ
        model (str): The Groq model to use
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        stream (bool): Enable streaming responses
        stop (list): Stop sequences for generation

    Examples:
        High-speed inference::

            provider = GroqProvider(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                max_tokens=2000
            )

        Streaming responses::

            provider = GroqProvider(
                model="llama2-70b-4096",
                stream=True,
                temperature=0.1
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.GROQ, description="Provider identifier"
    )

    # Groq model parameters
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
        """Get the Groq chat class."""
        try:
            from langchain_groq import ChatGroq

            return ChatGroq
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Groq requires langchain-groq. Install with: pip install langchain-groq",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Groq model."""
        return "mixtral-8x7b-32768"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-groq"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Groq-specific initialization parameters."""
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
            params["groq_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available Groq models."""
        return [
            "mixtral-8x7b-32768",
            "llama2-70b-4096",
            "gemma-7b-it",
            "llama3-70b-8192",
            "llama3-8b-8192",
        ]
