"""HuggingFace Provider Module.

This module implements the HuggingFace language model provider for the Haive framework,
supporting both HuggingFace Hub hosted models and local transformer models.

The provider handles API key management (for Hub), model configuration, and safe imports
of the langchain-huggingface package dependencies.

Examples:
    Hub-hosted model::

        from haive.core.models.llm.providers.huggingface import HuggingFaceProvider

        provider = HuggingFaceProvider(
            model="microsoft/DialoGPT-medium",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    Local transformer model::

        provider = HuggingFaceProvider(
            model="gpt2",
            device_map="auto",
            load_in_8bit=True,
            temperature=0.8
        )

.. autosummary::
   :toctree: generated/

   HuggingFaceProvider
"""

from typing import Any

from pydantic import Field

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace language model provider configuration.

    This provider supports both HuggingFace Hub hosted models and local
    transformer models, providing access to thousands of open-source models.

    Attributes:
        provider (LLMProvider): Always LLMProvider.HUGGINGFACE
        model (str): The HuggingFace model to use
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        top_k (int): Top-k sampling parameter
        repetition_penalty (float): Repetition penalty parameter
        device_map (str): Device mapping for model loading
        load_in_8bit (bool): Use 8-bit quantization
        load_in_4bit (bool): Use 4-bit quantization
        trust_remote_code (bool): Trust remote code execution

    Examples:
        Popular conversational model:

        .. code-block:: python

            provider = HuggingFaceProvider(
                model="microsoft/DialoGPT-medium",
                temperature=0.7,
                max_tokens=1000
            )

        Local model with quantization::

            provider = HuggingFaceProvider(
                model="meta-llama/Llama-2-7b-chat-hf",
                load_in_8bit=True,
                device_map="auto",
                temperature=0.1
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.HUGGINGFACE, description="Provider identifier"
    )

    # HuggingFace model parameters
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
    repetition_penalty: float | None = Field(
        default=None, ge=0, le=2, description="Repetition penalty parameter"
    )

    # Model loading parameters
    device_map: str | None = Field(
        default=None,
        description="Device mapping for model loading ('auto', 'cpu', 'cuda', etc.)",
    )
    load_in_8bit: bool = Field(
        default=False, description="Use 8-bit quantization for memory efficiency"
    )
    load_in_4bit: bool = Field(
        default=False, description="Use 4-bit quantization for memory efficiency"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code execution (required for some models)",
    )

    def _get_chat_class(self) -> type[Any]:
        """Get the HuggingFace chat class."""
        try:
            from langchain_huggingface import ChatHuggingFace

            return ChatHuggingFace
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="HuggingFace requires langchain-huggingface. Install with: pip install langchain-huggingface",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default HuggingFace model."""
        return "microsoft/DialoGPT-medium"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-huggingface"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get HuggingFace-specific initialization parameters."""
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

        # Model loading parameters
        if self.device_map is not None:
            params["device_map"] = self.device_map
        if self.load_in_8bit:
            params["load_in_8bit"] = self.load_in_8bit
        if self.load_in_4bit:
            params["load_in_4bit"] = self.load_in_4bit
        if self.trust_remote_code:
            params["trust_remote_code"] = self.trust_remote_code

        # Add API key if available (for Hub access)
        api_key = self.get_api_key()
        if api_key:
            params["huggingfacehub_api_token"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "HUGGINGFACEHUB_API_TOKEN"

    def _requires_api_key(self) -> bool:
        """HuggingFace doesn't require API key for public models."""
        return False

    @classmethod
    def get_models(cls) -> list[str]:
        """Get popular HuggingFace models."""
        return [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "facebook/blenderbot-400M-distill",
            "facebook/opt-1.3b",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
        ]
