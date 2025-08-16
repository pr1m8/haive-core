"""NVIDIA AI Endpoints Provider Module.

This module implements the NVIDIA AI Endpoints language model provider for the Haive framework,
supporting NVIDIA's optimized models through their AI Foundation API.

The provider handles API key management, model configuration, and safe imports of
the langchain-nvidia-ai-endpoints package dependencies.

Examples:
    Basic usage:

    .. code-block:: python

        from haive.core.models.llm.providers.nvidia import NVIDIAProvider

        provider = NVIDIAProvider(
            model="meta/llama3-70b-instruct",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With streaming::

        provider = NVIDIAProvider(
            model="microsoft/phi-3-medium-4k-instruct",
            temperature=0.1,
            stream=True
        )

.. autosummary::
   :toctree: generated/

   NVIDIAProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class NVIDIAProvider(BaseLLMProvider):
    """NVIDIA AI Endpoints language model provider configuration.

    This provider supports NVIDIA's optimized models including Llama, Mixtral,
    and other high-performance models through NVIDIA's AI Foundation API.

    Attributes:
        provider (LLMProvider): Always LLMProvider.NVIDIA
        model (str): The NVIDIA model to use
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        stream (bool): Enable streaming responses
        stop (list): Stop sequences for generation

    Examples:
        Llama 3 for reasoning:

        .. code-block:: python

            provider = NVIDIAProvider(
                model="meta/llama3-70b-instruct",
                temperature=0.3,
                max_tokens=2000
            )

        Mixtral for fast inference::

            provider = NVIDIAProvider(
                model="mistralai/mixtral-8x22b-instruct-v0.1",
                temperature=0.7,
                stream=True
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.NVIDIA, description="Provider identifier"
    )

    # NVIDIA model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=1, description="Sampling temperature"
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
        """Get the NVIDIA chat class."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            return ChatNVIDIA
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="NVIDIA AI Endpoints requires langchain-nvidia-ai-endpoints. Install with: pip install langchain-nvidia-ai-endpoints",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default NVIDIA model."""
        return "meta/llama3-70b-instruct"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-nvidia-ai-endpoints"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get NVIDIA-specific initialization parameters."""
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
            params["nvidia_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "NVIDIA_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available NVIDIA models."""
        return [
            "meta/llama3-70b-instruct",
            "meta/llama3-8b-instruct",
            "mistralai/mixtral-8x22b-instruct-v0.1",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "microsoft/phi-3-medium-4k-instruct",
            "microsoft/phi-3-mini-4k-instruct",
        ]
