"""Replicate Provider Module.

This module implements the Replicate language model provider for the Haive framework,
supporting a wide variety of open-source models hosted on Replicate's platform.

The provider handles API key management, model configuration, and safe imports of
the langchain-community package dependencies for Replicate integration.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.replicate import ReplicateProvider

        provider = ReplicateProvider(
            model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom parameters::

        provider = ReplicateProvider(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            temperature=0.1,
            top_p=0.9,
            top_k=50
        )

.. autosummary::
   :toctree: generated/

   ReplicateProvider
"""

from typing import Any

from pydantic import Field, field_validator

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class ReplicateProvider(BaseLLMProvider):
    """Replicate language model provider configuration.

    This provider supports a wide variety of open-source models hosted on Replicate,
    including Llama, Mixtral, CodeLlama, and many others with flexible versioning.

    Attributes:
        provider (LLMProvider): Always LLMProvider.REPLICATE
        model (str): The Replicate model to use (owner/name:version format)
        temperature (float): Sampling temperature (0.0-5.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Repetition penalty parameter
        stop_sequences (list): Stop sequences for generation

    Examples:
        Llama 2 70B model::

            provider = ReplicateProvider(
                model="meta/llama-2-70b-chat",
                temperature=0.7,
                max_tokens=2000
            )

        Mixtral with specific version::

            provider = ReplicateProvider(
                model="mistralai/mixtral-8x7b-instruct-v0.1:7b3212fbaf88310047672c7764d9f2cce7493d0d80666d899b72af8c0662df7a",
                temperature=0.1,
                top_p=0.9
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.REPLICATE, description="Provider identifier"
    )

    # Replicate model parameters
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=5,
        description="Sampling temperature (0.0-5.0 for Replicate)",
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
    stop_sequences: list[str] | None = Field(
        default=None, description="Stop sequences for generation"
    )

    @field_validator("model")
    @classmethod
    def validate_model_format(cls, v: str) -> str:
        """Validate Replicate model format."""
        if "/" not in v:
            raise ValueError(
                "Replicate model must be in format 'owner/name' or 'owner/name:version'"
            )
        return v

    def _get_chat_class(self) -> type[Any]:
        """Get the Replicate chat class."""
        try:
            from langchain_community.chat_models import ChatReplicate

            return ChatReplicate
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Replicate requires langchain-community. Install with: pip install langchain-community replicate",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Replicate model."""
        return "meta/llama-2-70b-chat"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-community"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Replicate-specific initialization parameters."""
        params = {
            "model": self.model,
            **kwargs,
        }

        # Replicate uses model_kwargs for parameters
        model_kwargs = {}
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            model_kwargs["max_new_tokens"] = self.max_tokens
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            model_kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            model_kwargs["repetition_penalty"] = self.repetition_penalty
        if self.stop_sequences is not None:
            model_kwargs["stop"] = self.stop_sequences

        if model_kwargs:
            params["model_kwargs"] = model_kwargs

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["replicate_api_token"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "REPLICATE_API_TOKEN"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get popular Replicate models."""
        return [
            "meta/llama-2-70b-chat",
            "meta/llama-2-13b-chat",
            "meta/llama-2-7b-chat",
            "mistralai/mixtral-8x7b-instruct-v0.1",
            "meta/codellama-70b-instruct",
            "togethercomputer/redpajama-incite-7b-chat",
            "replicate/vicuna-13b",
        ]
