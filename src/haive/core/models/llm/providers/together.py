"""Together AI Provider Module.

This module implements the Together AI language model provider for the Haive framework,
supporting a wide variety of open-source models through Together's inference platform.

The provider handles API key management, model configuration, and safe imports of
the langchain-together package dependencies.

Examples:
    Basic usage:

    .. code-block:: python

        from haive.core.models.llm.providers.together import TogetherProvider

        provider = TogetherProvider(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom parameters::

        provider = TogetherProvider(
            model="meta-llama/Llama-2-70b-chat-hf",
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1
        )

.. autosummary::
   :toctree: generated/

   TogetherProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class TogetherProvider(BaseLLMProvider):
    """Together AI language model provider configuration.

    This provider supports a wide variety of open-source models through Together's
    inference platform, including Llama, Mixtral, CodeLlama, and many others.

    Attributes:
        provider (LLMProvider): Always LLMProvider.TOGETHER_AI
        model (str): The Together model to use (full model path)
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Repetition penalty parameter
        stop (list): Stop sequences for generation

    Examples:
        Mixtral model for reasoning:

        .. code-block:: python

            provider = TogetherProvider(
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                temperature=0.3,
                max_tokens=2000
            )

        Llama 2 for conversation::

            provider = TogetherProvider(
                model="meta-llama/Llama-2-70b-chat-hf",
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.TOGETHER_AI, description="Provider identifier"
    )

    # Together model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=1, description="Sampling temperature"
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
    repetition_penalty: float | None = Field(
        default=None, ge=0, le=2, description="Repetition penalty parameter"
    )
    stop: list[str] | None = Field(
        default=None, description="Stop sequences for generation"
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the Together chat class."""
        try:
            from langchain_together import ChatTogether

            return ChatTogether
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Together AI requires langchain-together. Install with: pip install langchain-together",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Together model."""
        return "mistralai/Mixtral-8x7B-Instruct-v0.1"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-together"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Together-specific initialization parameters."""
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
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.stop is not None:
            params["stop"] = self.stop

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["together_api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "TOGETHER_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get popular Together models."""
        return [
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        ]
