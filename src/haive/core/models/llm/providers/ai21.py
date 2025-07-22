"""AI21 Labs Provider Module.

This module implements the AI21 Labs language model provider for the Haive framework,
supporting Jurassic models known for their strong performance on various NLP tasks.

The provider handles API key management, model configuration, and safe imports of
the langchain-ai21 package dependencies.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.ai21 import AI21Provider

        provider = AI21Provider(
            model="j2-ultra",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom parameters::

        provider = AI21Provider(
            model="j2-grande-instruct",
            temperature=0.1,
            top_p=0.9,
            frequency_penalty=0.2
        )

.. autosummary::
   :toctree: generated/

   AI21Provider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class AI21Provider(BaseLLMProvider):
    """AI21 Labs language model provider configuration.

    This provider supports AI21's Jurassic family of models including J2-Ultra,
    J2-Mid, and instruction-tuned variants optimized for various tasks.

    Attributes:
        provider (LLMProvider): Always LLMProvider.AI21
        model (str): The AI21 model to use
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        top_k_return (int): Number of top tokens to consider
        frequency_penalty (dict): Frequency penalty settings
        presence_penalty (dict): Presence penalty settings
        count_penalty (dict): Count penalty settings

    Examples:
        Ultra model for complex tasks::

            provider = AI21Provider(
                model="j2-ultra",
                temperature=0.7,
                max_tokens=2000
            )

        Instruct model with penalties::

            provider = AI21Provider(
                model="j2-grande-instruct",
                temperature=0.1,
                frequency_penalty={"scale": 0.2, "apply_to_whitespaces": False}
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.AI21, description="Provider identifier"
    )

    # AI21 model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    top_k_return: int | None = Field(
        default=None,
        ge=1,
        description="Number of top tokens to return probabilities for",
    )
    frequency_penalty: dict[str, Any] | None = Field(
        default=None, description="Frequency penalty settings"
    )
    presence_penalty: dict[str, Any] | None = Field(
        default=None, description="Presence penalty settings"
    )
    count_penalty: dict[str, Any] | None = Field(
        default=None, description="Count penalty settings"
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the AI21 chat class."""
        try:
            from langchain_ai21 import ChatAI21

            return ChatAI21
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="AI21 requires langchain-ai21. Install with: pip install langchain-ai21",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default AI21 model."""
        return "j2-ultra"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-ai21"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get AI21-specific initialization parameters."""
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
        if self.top_k_return is not None:
            params["top_k_return"] = self.top_k_return
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.count_penalty is not None:
            params["count_penalty"] = self.count_penalty

        # Add API key
        api_key = self.get_api_key()
        if api_key:
            params["api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "AI21_API_KEY"

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available AI21 models."""
        return [
            "j2-ultra",
            "j2-mid",
            "j2-light",
            "j2-grande",
            "j2-grande-instruct",
            "j2-jumbo-instruct",
        ]
