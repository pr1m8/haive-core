"""LLM provider implementations.

This module provides specific implementations for different LLM providers,
including OpenAI, Azure, and others. Each provider has its own configuration
and engine class that builds on the base LLM structure.

Key Components:
    - OpenAI configuration and engine
    - Azure configuration and engine
    - Provider-specific utilities and helpers
"""

from __future__ import annotations

from typing import Optional, TypeVar

from pydantic import Field

from haive.core.models.llm.engine import EngineRegistry, LLMEngine, LLMEngineConfig

# Type variables for provider-specific types
TOpenAIConfig = TypeVar("TOpenAIConfig", bound="OpenAIConfig")
TAzureConfig = TypeVar("TAzureConfig", bound="AzureConfig")


class OpenAIConfig(LLMEngineConfig[TOpenAIConfig]):
    """Configuration for OpenAI LLM engines.

    This class provides configuration specific to OpenAI's API,
    including model selection and API parameters.

    Attributes:
        model (str): OpenAI model identifier
        api_key (str): OpenAI API key
        organization (Optional[str]): OpenAI organization ID

    Example:
        >>> config = OpenAIConfig(
        ...     name="gpt4",
        ...     model="gpt-4",
        ...     api_key="sk-..."
        ... )
    """

    model: str = Field(..., description="OpenAI model identifier")
    api_key: str = Field(..., description="OpenAI API key")
    organization: Optional[str] = Field(None, description="OpenAI organization ID")

    def instantiate(self, **kwargs) -> "OpenAIEngine":
        """Create an OpenAI engine instance.

        Args:
            **kwargs: Additional instantiation parameters

        Returns:
            OpenAIEngine: New OpenAI engine instance
        """
        return OpenAIEngine(self)


class OpenAIEngine(LLMEngine[OpenAIConfig]):
    """OpenAI LLM engine implementation.

    This class implements the LLM engine interface for OpenAI's API,
    providing text generation and streaming capabilities.

    Attributes:
        config (OpenAIConfig): Configuration for this OpenAI engine

    Example:
        >>> engine = OpenAIEngine(config)
        >>> response = engine.generate("Hello, world!")
    """

    def __init__(self, config: OpenAIConfig):
        """Initialize the OpenAI engine.

        Args:
            config (OpenAIConfig): Configuration for this OpenAI engine
        """
        super().__init__(config)
        self._model = None

    def initialize(self) -> None:
        """Initialize the OpenAI model.

        This method creates the ChatOpenAI instance with the configured
        parameters.
        """
        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            model=self.config.model,
            openai_api_key=self.config.api_key,
            openai_organization=self.config.organization,
            **self.config.get_generation_config(),
        )

    def cleanup(self) -> None:
        """Clean up OpenAI resources.

        Currently no cleanup is needed for OpenAI.
        """
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI's API.

        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        if self._model is None:
            self.initialize()
        response = self._model.invoke(prompt)
        return response.content

    def generate_stream(self, prompt: str, **kwargs):
        """Generate text using OpenAI's API in streaming mode.

        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks
        """
        if self._model is None:
            self.initialize()
        for chunk in self._model.stream(prompt):
            yield chunk.content


class AzureConfig(LLMEngineConfig[TAzureConfig]):
    """Configuration for Azure LLM engines.

    This class provides configuration specific to Azure's OpenAI API,
    including deployment settings and API parameters.

    Attributes:
        deployment_name (str): Azure deployment name
        api_key (str): Azure API key
        api_base (str): Azure API base URL
        api_version (str): Azure API version

    Example:
        >>> config = AzureConfig(
        ...     name="azure-gpt4",
        ...     deployment_name="gpt-4",
        ...     api_key="...",
        ...     api_base="https://..."
        ... )
    """

    deployment_name: str = Field(..., description="Azure deployment name")
    api_key: str = Field(..., description="Azure API key")
    api_base: str = Field(..., description="Azure API base URL")
    api_version: str = Field(default="2023-05-15", description="Azure API version")

    def instantiate(self, **kwargs) -> "AzureEngine":
        """Create an Azure engine instance.

        Args:
            **kwargs: Additional instantiation parameters

        Returns:
            AzureEngine: New Azure engine instance
        """
        return AzureEngine(self)


class AzureEngine(LLMEngine[AzureConfig]):
    """Azure LLM engine implementation.

    This class implements the LLM engine interface for Azure's OpenAI API,
    providing text generation and streaming capabilities.

    Attributes:
        config (AzureConfig): Configuration for this Azure engine

    Example:
        >>> engine = AzureEngine(config)
        >>> response = engine.generate("Hello, world!")
    """

    def __init__(self, config: AzureConfig):
        """Initialize the Azure engine.

        Args:
            config (AzureConfig): Configuration for this Azure engine
        """
        super().__init__(config)
        self._model = None

    def initialize(self) -> None:
        """Initialize the Azure model.

        This method creates the AzureChatOpenAI instance with the configured
        parameters.
        """
        from langchain_openai import AzureChatOpenAI

        self._model = AzureChatOpenAI(
            deployment_name=self.config.deployment_name,
            openai_api_key=self.config.api_key,
            openai_api_base=self.config.api_base,
            openai_api_version=self.config.api_version,
            **self.config.get_generation_config(),
        )

    def cleanup(self) -> None:
        """Clean up Azure resources.

        Currently no cleanup is needed for Azure.
        """
        pass

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Azure's API.

        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters

        Returns:
            str: Generated text
        """
        if self._model is None:
            self.initialize()
        response = self._model.invoke(prompt)
        return response.content

    def generate_stream(self, prompt: str, **kwargs):
        """Generate text using Azure's API in streaming mode.

        Args:
            prompt (str): Input prompt
            **kwargs: Additional generation parameters

        Yields:
            str: Generated text chunks
        """
        if self._model is None:
            self.initialize()
        for chunk in self._model.stream(prompt):
            yield chunk.content


# Register the provider-specific engine types
EngineRegistry.register(OpenAIConfig, OpenAIEngine)
EngineRegistry.register(AzureConfig, AzureEngine)
