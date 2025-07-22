"""Azure OpenAI Provider Module.

This module implements the Azure OpenAI language model provider for the Haive framework,
supporting GPT models deployed on Microsoft Azure with enhanced security and compliance.

The provider handles Azure-specific authentication, endpoint configuration, and model
deployment access through the langchain-openai package dependencies.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.azure import AzureOpenAIProvider

        provider = AzureOpenAIProvider(
            deployment_name="gpt-4-deployment",
            azure_endpoint="https://myresource.openai.azure.com/",
            api_version="2024-02-15-preview",
            temperature=0.7
        )
        llm = provider.instantiate()

    With Azure AD authentication::

        provider = AzureOpenAIProvider(
            deployment_name="gpt-35-turbo",
            azure_endpoint="https://myresource.openai.azure.com/",
            use_azure_ad=True
        )

.. autosummary::
   :toctree: generated/

   AzureOpenAIProvider
"""

from typing import Any

from pydantic import Field, field_validator

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI language model provider configuration.

    This provider supports all OpenAI models deployed on Microsoft Azure,
    including GPT-4, GPT-3.5-turbo, and others with enterprise-grade security.

    Attributes:
        provider (LLMProvider): Always LLMProvider.AZURE
        deployment_name (str): Azure deployment name for the model
        azure_endpoint (str): Azure OpenAI resource endpoint URL
        api_version (str): Azure OpenAI API version
        use_azure_ad (bool): Whether to use Azure AD authentication
        temperature (float): Sampling temperature (0.0-2.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter
        frequency_penalty (float): Frequency penalty parameter
        presence_penalty (float): Presence penalty parameter

    Examples:
        Standard deployment::

            provider = AzureOpenAIProvider(
                deployment_name="gpt-4",
                azure_endpoint="https://myresource.openai.azure.com/",
                temperature=0.7,
                max_tokens=1000
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.AZURE, description="Provider identifier"
    )

    # Azure-specific parameters
    deployment_name: str = Field(..., description="Azure deployment name for the model")
    azure_endpoint: str = Field(..., description="Azure OpenAI resource endpoint URL")
    api_version: str = Field(
        default="2024-02-15-preview", description="Azure OpenAI API version"
    )
    use_azure_ad: bool = Field(
        default=False, description="Use Azure AD authentication instead of API key"
    )

    # OpenAI model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )
    frequency_penalty: float | None = Field(
        default=None, ge=-2, le=2, description="Frequency penalty parameter"
    )
    presence_penalty: float | None = Field(
        default=None, ge=-2, le=2, description="Presence penalty parameter"
    )

    @field_validator("azure_endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """Validate Azure endpoint format."""
        if not v.startswith("https://"):
            raise ValueError("Azure endpoint must use HTTPS")
        if not v.endswith("/"):
            v += "/"
        return v

    def _get_chat_class(self) -> type[Any]:
        """Get the Azure OpenAI chat class."""
        try:
            from langchain_openai import AzureChatOpenAI

            return AzureChatOpenAI
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="Azure OpenAI requires langchain-openai. Install with: pip install langchain-openai",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default model (deployment name)."""
        return "gpt-35-turbo"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-openai"

    def _get_env_key_name(self) -> str:
        """Get the environment variable name for API key."""
        return "AZURE_OPENAI_API_KEY"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Azure-specific initialization parameters."""
        params = {
            "deployment_name": self.deployment_name,
            "azure_endpoint": self.azure_endpoint,
            "api_version": self.api_version,
            **kwargs,
        }

        # Add model parameters if specified
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty

        # Handle Azure AD authentication
        if not self.use_azure_ad:
            api_key = self.get_api_key()
            if api_key:
                params["api_key"] = api_key

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _requires_api_key(self) -> bool:
        """Azure OpenAI requires API key unless using Azure AD."""
        return not self.use_azure_ad
