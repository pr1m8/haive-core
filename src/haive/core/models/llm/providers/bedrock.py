"""AWS Bedrock Provider Module.

This module implements the AWS Bedrock language model provider for the Haive framework,
supporting Amazon's managed LLM service with models from Anthropic, AI21, Cohere, and others.

The provider handles AWS credentials, region configuration, and safe imports of
the langchain-aws package dependencies.

Examples:
    Basic usage::

        from haive.core.models.llm.providers.bedrock import BedrockProvider

        provider = BedrockProvider(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            temperature=0.7,
            max_tokens=1000
        )
        llm = provider.instantiate()

    With custom AWS configuration::

        provider = BedrockProvider(
            model="ai21.j2-ultra-v1",
            region_name="us-west-2",
            aws_access_key_id="...",
            aws_secret_access_key="...",
            temperature=0.1
        )

.. autosummary::
   :toctree: generated/

   BedrockProvider
"""

from typing import Any

from pydantic import Field, field_validator

from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.llm.providers.base import BaseLLMProvider, ProviderImportError


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock language model provider configuration.

    This provider supports AWS Bedrock's managed LLM service including models
    from Anthropic Claude, AI21 Jurassic, Cohere Command, and Amazon Titan.

    Attributes:
        provider (LLMProvider): Always LLMProvider.BEDROCK
        model (str): The Bedrock model ID to use
        region_name (str): AWS region for Bedrock service
        aws_access_key_id (str): AWS access key ID
        aws_secret_access_key (str): AWS secret access key
        aws_session_token (str): AWS session token (for temporary credentials)
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        top_p (float): Nucleus sampling parameter

    Examples:
        Claude 3 on Bedrock::

            provider = BedrockProvider(
                model="anthropic.claude-3-sonnet-20240229-v1:0",
                region_name="us-east-1",
                temperature=0.7
            )

        AI21 Jurassic model::

            provider = BedrockProvider(
                model="ai21.j2-ultra-v1",
                region_name="us-west-2",
                temperature=0.1,
                max_tokens=2000
            )
    """

    provider: LLMProvider = Field(
        default=LLMProvider.BEDROCK, description="Provider identifier"
    )

    # AWS configuration
    region_name: str = Field(
        default="us-east-1", description="AWS region for Bedrock service"
    )
    aws_access_key_id: str | None = Field(
        default=None, description="AWS access key ID (optional if using IAM roles)"
    )
    aws_secret_access_key: str | None = Field(
        default=None, description="AWS secret access key (optional if using IAM roles)"
    )
    aws_session_token: str | None = Field(
        default=None, description="AWS session token for temporary credentials"
    )

    # Model parameters
    temperature: float | None = Field(
        default=None, ge=0, le=1, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Maximum tokens in response"
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter"
    )

    @field_validator("model")
    @classmethod
    def validate_model_id(cls, v: str) -> str:
        """Validate Bedrock model ID format."""
        if not v or "." not in v:
            raise ValueError("Bedrock model must be in format 'provider.model-name'")
        return v

    def _get_chat_class(self) -> type[Any]:
        """Get the Bedrock chat class."""
        try:
            from langchain_aws import ChatBedrock

            return ChatBedrock
        except ImportError as e:
            raise ProviderImportError(
                provider=self.provider.value,
                package=self._get_import_package(),
                message="AWS Bedrock requires langchain-aws. Install with: pip install langchain-aws",
            ) from e

    def _get_default_model(self) -> str:
        """Get the default Bedrock model."""
        return "anthropic.claude-3-sonnet-20240229-v1:0"

    def _get_import_package(self) -> str:
        """Get the required package name."""
        return "langchain-aws"

    def _get_initialization_params(self, **kwargs) -> dict[str, Any]:
        """Get Bedrock-specific initialization parameters."""
        params = {
            "model_id": self.model,
            "region_name": self.region_name,
            **kwargs,
        }

        # Add AWS credentials if provided
        if self.aws_access_key_id:
            params["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            params["aws_secret_access_key"] = self.aws_secret_access_key
        if self.aws_session_token:
            params["aws_session_token"] = self.aws_session_token

        # Add model parameters if specified
        model_kwargs = {}
        if self.temperature is not None:
            model_kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            model_kwargs["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            model_kwargs["top_p"] = self.top_p

        if model_kwargs:
            params["model_kwargs"] = model_kwargs

        # Add extra params
        params.update(self.extra_params or {})

        return params

    def _requires_api_key(self) -> bool:
        """Bedrock uses AWS credentials instead of API key."""
        return False

    @classmethod
    def get_models(cls) -> list[str]:
        """Get available Bedrock models."""
        return [
            # Anthropic Claude models
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-v2:1",
            "anthropic.claude-v2",
            "anthropic.claude-instant-v1",
            # AI21 Jurassic models
            "ai21.j2-ultra-v1",
            "ai21.j2-mid-v1",
            # Cohere Command models
            "cohere.command-text-v14",
            "cohere.command-light-text-v14",
            # Amazon Titan models
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
        ]
