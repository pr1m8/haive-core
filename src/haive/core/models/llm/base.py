"""
Base LLM configuration with model metadata support.

This module provides base classes and implementations for LLM providers
with support for model metadata, context windows, and capabilities.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

# Import the mixins
from haive.core.common.mixins.secure_config import SecureConfigMixin
from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.metadata_mixin import ModelMetadataMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
# Try to import rich for enhanced debugging
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.debug("dotenv module not available, skipping .env file loading")

# Set up LangChain cache if needed
try:
    # Define a relative cache path
    from pathlib import Path

    from langchain.globals import set_llm_cache
    from langchain_community.cache import SQLiteCache

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CACHE_DIR = BASE_DIR / "lc_cache"
    CACHE_FILE = CACHE_DIR / ".langchain_cache.db"

    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Set cache
    set_llm_cache(SQLiteCache(database_path=str(CACHE_FILE)))
except ImportError:
    logger.debug("LangChain cache modules not available, skipping cache setup")


class LLMConfig(BaseModel, SecureConfigMixin, ModelMetadataMixin):
    """
    Base configuration for Language Model providers with security and metadata support.

    This class provides:
    1. Secure API key handling with environment variable fallbacks
    2. Model metadata access (context windows, capabilities, pricing)
    3. Common configuration parameters
    4. Graph transformation utilities
    """

    provider: LLMProvider = Field(description="The provider of the LLM.")
    model: str = Field(..., description="The model to be used, e.g., gpt-4.")
    name: Optional[str] = Field(
        default=None, description="Friendly display name for this model."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(""), description="API key for LLM provider."
    )
    cache_enabled: bool = Field(
        default=True, description="Enable or disable response caching."
    )
    cache_ttl: Optional[int] = Field(
        default=300, description="Time-to-live for cache (in seconds)."
    )
    extra_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Optional extra parameters."
    )
    debug: bool = Field(default=False, description="Enable detailed debug output.")

    model_config = {"arbitrary_types_allowed": True}
    model_alias: Optional[str] = Field(default=None, description="Alias for the model.")

    @model_validator(mode="after")
    def set_default_name(self) -> "LLMConfig":
        """
        Set a default name for the model if not provided.
        """
        if self.name is None:
            # Default to model ID if no name provided
            self.name = self.model
        return self

    @model_validator(mode="after")
    def load_model_metadata(self) -> "LLMConfig":
        """
        Load and validate model metadata after initialization.
        """
        logger.debug(f"Loading metadata for {self.model} from {self.provider}")

        # Check model capabilities after initialization
        try:
            # Get metadata for validation and logging
            context_window = self.get_context_window()
            logger.debug(f"Model {self.model} context window: {context_window}")

            pricing = self.get_token_pricing()
            logger.debug(f"Model {self.model} pricing: {pricing}")

            # Display rich debug info if enabled
            if self.debug and RICH_AVAILABLE:
                self._display_debug_info()

        except Exception as e:
            logger.warning(f"Error loading model metadata: {e}")

        return self

    def _display_debug_info(self) -> None:
        """Display rich debug information about the model metadata."""
        if not RICH_AVAILABLE:
            logger.debug("Rich library not available for enhanced debugging")
            return

        raw_metadata = self._get_model_metadata()

        # Display metadata tree
        tree = Tree(f"[bold cyan]{self.model} Metadata[/bold cyan]")

        def _add_dict_to_tree(tree_node, data):
            if isinstance(data, dict):
                for key, value in sorted(data.items()):
                    if isinstance(value, dict):
                        branch = tree_node.add(f"[yellow]{key}[/yellow]")
                        _add_dict_to_tree(branch, value)
                    elif isinstance(value, list):
                        branch = tree_node.add(
                            f"[yellow]{key}[/yellow] (list, {len(value)} items)"
                        )
                        for i, item in enumerate(value):
                            if isinstance(item, (dict, list)):
                                sub_branch = branch.add(f"[blue]Item {i}[/blue]")
                                _add_dict_to_tree(sub_branch, item)
                            else:
                                branch.add(f"[blue]Item {i}:[/blue] {item}")
                    else:
                        tree_node.add(f"[green]{key}[/green]: {value}")

        _add_dict_to_tree(tree, raw_metadata)
        console.print(tree)

        # Display capabilities panel
        capabilities = {
            "vision": self.supports_vision,
            "function_calling": self.supports_function_calling,
            "parallel_function_calling": self.supports_parallel_function_calling,
            "system_messages": self.supports_system_messages,
            "tool_choice": self.supports_tool_choice,
            "response_schema": self.supports_response_schema,
            "web_search": self.supports_web_search,
            "pdf_input": self.supports_pdf_input,
            "audio_input": self.supports_audio_input,
            "audio_output": self.supports_audio_output,
            "prompt_caching": self.supports_prompt_caching,
            "native_streaming": self.supports_native_streaming,
            "reasoning": self.supports_reasoning,
        }

        capability_text = "\n".join(
            [
                f"  {'✓' if supported else '✗'} {capability.replace('_', ' ').title()}"
                for capability, supported in capabilities.items()
            ]
        )

        context_window = self.get_context_window()
        max_input = self.get_max_input_tokens()
        max_output = self.get_max_output_tokens()
        input_cost, output_cost = self.get_token_pricing()

        info_text = (
            f"Context Window: {context_window} tokens\n"
            f"Max Input: {max_input} tokens\n"
            f"Max Output: {max_output} tokens\n"
            f"Input cost per token: ${input_cost}\n"
            f"Output cost per token: ${output_cost}\n\n"
            f"Capabilities:\n{capability_text}"
        )

        # Check for deprecation
        deprecation_date = self.get_deprecation_date()
        if deprecation_date:
            info_text += f"\n\n⚠️ Model will be deprecated on: {deprecation_date}"

        console.print(Panel(info_text, title=f"[bold]{self.model} Summary[/bold]"))

    def format_metadata_for_display(self) -> Dict[str, Any]:
        """
        Format metadata for structured display or comparison.

        Returns:
            Dictionary with formatted metadata
        """
        raw_metadata = self._get_model_metadata()
        context_window = self.get_context_window()
        max_input = self.get_max_input_tokens()
        max_output = self.get_max_output_tokens()
        input_cost, output_cost = self.get_token_pricing()
        deprecation_date = self.get_deprecation_date()

        capabilities = {
            "vision": self.supports_vision,
            "function_calling": self.supports_function_calling,
            "parallel_function_calling": self.supports_parallel_function_calling,
            "system_messages": self.supports_system_messages,
            "tool_choice": self.supports_tool_choice,
            "response_schema": self.supports_response_schema,
            "web_search": self.supports_web_search,
            "pdf_input": self.supports_pdf_input,
            "audio_input": self.supports_audio_input,
            "audio_output": self.supports_audio_output,
            "prompt_caching": self.supports_prompt_caching,
            "native_streaming": self.supports_native_streaming,
            "reasoning": self.supports_reasoning,
        }

        return {
            "name": self.name,
            "provider": (
                self.provider.value
                if hasattr(self.provider, "value")
                else str(self.provider)
            ),
            "model": self.model,
            "context_window": context_window,
            "max_input": max_input,
            "max_output": max_output,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "capabilities": capabilities,
            "raw_metadata": raw_metadata,
            "deprecation_date": deprecation_date,
        }

    def instantiate(self, **kwargs) -> Any:
        """
        Abstract method to be implemented by subclasses.

        Raises:
            NotImplementedError: If not overridden by a subclass
        """
        raise NotImplementedError("This method should be overridden in subclasses.")

    def create_graph_transformer(self) -> Any:
        """Creates an LLMGraphTransformer instance using the LLM."""
        from langchain_experimental.graph_transformers import LLMGraphTransformer

        llm = self.instantiate()
        return LLMGraphTransformer(llm=llm)


class AzureLLMConfig(LLMConfig):
    """Configuration specific to Azure OpenAI."""

    provider: LLMProvider = LLMProvider.AZURE
    model: str = Field(default="gpt-4o", description="Azure deployment name (model).")
    api_version: str = Field(
        default_factory=lambda: os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        ),
        description="Azure API version.",
    )
    api_base: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        description="Azure API base URL.",
    )
    api_type: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_TYPE", "azure"),
        description="API type for Azure.",
    )
    # Direct loading of API key from environment
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
        description="API key for Azure OpenAI.",
    )

    @field_validator("api_version")
    @classmethod
    def load_api_version(cls, v: str) -> str:
        """Load API version from environment if not provided."""
        if not v:
            env_value = os.getenv("AZURE_OPENAI_API_VERSION")
            if env_value:
                return env_value
        return v

    @field_validator("api_base")
    @classmethod
    def load_api_base(cls, v: str) -> str:
        """Load API base from environment if not provided."""
        if not v:
            env_value = os.getenv("AZURE_OPENAI_ENDPOINT")
            if env_value:
                return env_value
        return v

    @field_validator("api_type")
    @classmethod
    def load_api_type(cls, v: str) -> str:
        """Load API type from environment if not provided."""
        if not v:
            env_value = os.getenv("OPENAI_API_TYPE")
            if env_value:
                return env_value
        return v

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("AZURE_OPENAI_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Azure OpenAI Chat model with robust error handling.
        """
        from langchain_openai import AzureChatOpenAI

        # Debug output
        logger.debug("Attempting to instantiate Azure OpenAI model:")
        logger.debug(f"- Model/deployment: {self.model}")
        logger.debug(f"- API version: {self.api_version}")
        logger.debug(f"- API base: {self.api_base}")
        logger.debug(f"- API type: {self.api_type}")
        logger.debug(f"- API key available: {'Yes' if self.get_api_key() else 'No'}")

        # Validate required parameters
        if not self.get_api_key():
            raise ValueError(
                "Azure OpenAI API key is required. "
                "Please set AZURE_OPENAI_API_KEY environment variable or provide an API key."
            )

        if not self.api_base:
            raise ValueError(
                "Azure OpenAI endpoint is required. "
                "Please set AZURE_OPENAI_ENDPOINT environment variable."
            )

        try:
            # Use the new parameter names
            return AzureChatOpenAI(
                deployment_name=self.model,
                api_key=self.get_api_key(),  # Changed from openai_api_key
                api_version=self.api_version,  # Changed from openai_api_version
                azure_endpoint=self.api_base,  # Changed from openai_api_base
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to instantiate Azure OpenAI model: {str(e)}")
            raise RuntimeError(
                f"Failed to instantiate Azure OpenAI model: {str(e)}"
            ) from e


class OpenAILLMConfig(LLMConfig):
    """Configuration for OpenAI models."""

    provider: LLMProvider = LLMProvider.OPENAI
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")),
        description="API key for OpenAI.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("OPENAI_API_KEY", "")
            return SecretStr(env_value)
        return v

    @classmethod
    def get_models(cls) -> List[str]:
        """Get all available OpenAI models."""
        from openai import OpenAI

        client = OpenAI()
        return client.models.list().data

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate OpenAI Chat model.
        """
        from langchain_openai import OpenAIChat

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "OpenAI API key is required. "
                "Please set OPENAI_API_KEY environment variable or provide an API key."
            )

        try:
            return OpenAIChat(
                model_name=self.model,
                openai_api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate OpenAI model: {str(e)}") from e


class AnthropicLLMConfig(LLMConfig):
    """Configuration for Anthropic models."""

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
        description="Anthropic model name.",
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        description="API key for Anthropic.",
    )

    @field_validator("model")
    @classmethod
    def load_model(cls, v: str) -> str:
        """Load model from environment if not provided."""
        if not v:
            env_value = os.getenv("ANTHROPIC_MODEL")
            if env_value:
                return env_value
        return v

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("ANTHROPIC_API_KEY", "")
            return SecretStr(env_value)
        return v

    @classmethod
    def get_models(cls) -> List[str]:
        """Get all available Anthropic models."""
        from anthropic import Anthropic

        client = Anthropic()
        return client.models.list().data

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Anthropic Chat model.
        """
        from langchain_anthropic import ChatAnthropic

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Anthropic API key is required. "
                "Please set ANTHROPIC_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatAnthropic(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Anthropic model: {str(e)}"
            ) from e


class GeminiLLMConfig(LLMConfig):
    """Configuration for Google Gemini models."""

    provider: LLMProvider = LLMProvider.GEMINI
    model: str = Field(default="gemini-1.5-pro", description="Gemini model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GEMINI_API_KEY", "")),
        description="API key for Google Gemini.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("GEMINI_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Google Gemini Chat model.
        """
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Google Gemini API key is required. "
                "Please set GEMINI_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatGoogleGenerativeAI(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Gemini model: {str(e)}") from e


class DeepSeekLLMConfig(LLMConfig):
    """Configuration for DeepSeek models."""

    provider: LLMProvider = LLMProvider.DEEPSEEK
    model: str = Field(default="deepseek-chat", description="DeepSeek model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("DEEPSEEK_API_KEY", "")),
        description="API key for DeepSeek.",
    )

    @classmethod
    def get_models(cls) -> List[str]:
        """Get all available DeepSeek models."""
        from deepseek import DeepSeekAPI

        client = DeepSeekAPI(api_key=os.getenv("DEEPSEEK_API_KEY", ""))
        return client.get_models()

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate DeepSeek Chat model.
        """
        from langchain_deepseek import ChatDeepSeek

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "DeepSeek API key is required. "
                "Please set DEEPSEEK_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatDeepSeek(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate DeepSeek model: {str(e)}") from e


class MistralLLMConfig(LLMConfig):
    """Configuration for Mistral models."""

    provider: LLMProvider = LLMProvider.MISTRALAI
    model: str = Field(
        default="mistral-large-latest", description="Mistral model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MISTRAL_API_KEY", "")),
        description="API key for Mistral.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("MISTRAL_API_KEY", "")
            return SecretStr(env_value)
        return v

    @classmethod
    def get_models(cls) -> List[str]:
        """Get all available Mistral models."""
        from mistralai import Mistral

        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""))
        return client.models.list()

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Mistral Chat model.
        """
        from langchain_mistralai import ChatMistralAI

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Mistral API key is required. "
                "Please set MISTRAL_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatMistralAI(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Mistral model: {str(e)}") from e


class GroqLLMConfig(LLMConfig):
    """Configuration for Groq models."""

    provider: LLMProvider = LLMProvider.GROQ
    model: str = Field(default="llama3-70b-8192", description="Groq model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")),
        description="API key for Groq.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("GROQ_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Groq Chat model.
        """
        from langchain_groq import ChatGroq

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Groq API key is required. "
                "Please set GROQ_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatGroq(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Groq model: {str(e)}") from e


class CohereLLMConfig(LLMConfig):
    """Configuration for Cohere models."""

    provider: LLMProvider = LLMProvider.COHERE
    model: str = Field(default="command", description="Cohere model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("COHERE_API_KEY", "")),
        description="API key for Cohere.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("COHERE_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Cohere Chat model.
        """
        from langchain_cohere import ChatCohere

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Cohere API key is required. "
                "Please set COHERE_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatCohere(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Cohere model: {str(e)}") from e


class TogetherAILLMConfig(LLMConfig):
    """Configuration for Together AI models."""

    provider: LLMProvider = LLMProvider.TOGETHER_AI
    model: str = Field(
        default="meta-llama/Llama-3-70b-chat-hf", description="Together AI model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("TOGETHER_AI_API_KEY", "")),
        description="API key for Together AI.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("TOGETHER_AI_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Together AI Chat model.
        """
        from langchain_together import ChatTogether

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Together AI API key is required. "
                "Please set TOGETHER_AI_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatTogether(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Together AI model: {str(e)}"
            ) from e


class FireworksAILLMConfig(LLMConfig):
    """Configuration for Fireworks AI models."""

    provider: LLMProvider = LLMProvider.FIREWORKS_AI
    model: str = Field(
        default="fireworks/llama-v3-70b-chat", description="Fireworks AI model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("FIREWORKS_AI_API_KEY", "")),
        description="API key for Fireworks AI.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("FIREWORKS_AI_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Fireworks AI Chat model.
        """
        from langchain_fireworks import ChatFireworks

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Fireworks AI API key is required. "
                "Please set FIREWORKS_AI_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatFireworks(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Fireworks AI model: {str(e)}"
            ) from e


class PerplexityLLMConfig(LLMConfig):
    """Configuration for Perplexity AI models."""

    provider: LLMProvider = LLMProvider.PERPLEXITY
    model: str = Field(
        default="sonar-medium-online", description="Perplexity model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("PERPLEXITY_API_KEY", "")),
        description="API key for Perplexity AI.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("PERPLEXITY_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Perplexity AI Chat model.
        """
        from langchain_community.chat_models import ChatPerplexity

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Perplexity API key is required. "
                "Please set PERPLEXITY_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatPerplexity(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Perplexity model: {str(e)}"
            ) from e


class HuggingFaceLLMConfig(LLMConfig):
    """Configuration for Hugging Face models."""

    provider: LLMProvider = LLMProvider.HUGGINGFACE
    model: str = Field(..., description="Model ID on Hugging Face Hub.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("HUGGING_FACE_API_KEY", "")),
        description="API key for Hugging Face.",
    )
    endpoint_url: Optional[str] = Field(
        default=None, description="Optional Hugging Face Inference Endpoint URL"
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("HUGGING_FACE_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Hugging Face model.
        """
        try:
            from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        except ImportError:
            raise RuntimeError(
                "langchain-huggingface is not installed. "
                "Please install it with 'pip install langchain-huggingface'"
            )
        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Hugging Face API key is required. "
                "Please set HUGGING_FACE_API_KEY environment variable or provide an API key."
            )

        try:
            # Use HuggingFace Hub directly or Inference endpoint based on configuration
            if self.endpoint_url:
                from langchain_huggingface import ChatHuggingFace

                return ChatHuggingFace(
                    model=self.model,
                    api_key=self.get_api_key(),
                    endpoint_url=self.endpoint_url,
                    cache=self.cache_enabled,
                    **(self.extra_params or {}),
                    **kwargs,
                )
            else:
                from langchain_huggingface import HuggingFaceEndpoint

                return HuggingFaceEndpoint(
                    repo_id=self.model,
                    huggingfacehub_api_token=self.get_api_key(),
                    cache=self.cache_enabled,
                    **(self.extra_params or {}),
                    **kwargs,
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Hugging Face model: {str(e)}"
            ) from e


class AI21LLMConfig(LLMConfig):
    """Configuration for AI21 models."""

    provider: LLMProvider = LLMProvider.AI21
    model: str = Field(default="j2-ultra", description="AI21 model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AI21_API_KEY", "")),
        description="API key for AI21.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("AI21_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate AI21 Chat model.
        """
        try:
            from langchain_ai21 import ChatAI21
        except ImportError:
            raise RuntimeError(
                "langchain-ai21 is not installed. "
                "Please install it with 'pip install langchain-ai21'"
            )

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "AI21 API key is required. "
                "Please set AI21_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatAI21(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate AI21 model: {str(e)}") from e


class AlephAlphaLLMConfig(LLMConfig):
    """Configuration for Aleph Alpha models."""

    provider: LLMProvider = LLMProvider.ALEPH_ALPHA
    model: str = Field(default="luminous-base", description="Aleph Alpha model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ALEPH_ALPHA_API_KEY", "")),
        description="API key for Aleph Alpha.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("ALEPH_ALPHA_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Aleph Alpha Chat model.
        """
        from langchain_community.chat_models import ChatAlephAlpha

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Aleph Alpha API key is required. "
                "Please set ALEPH_ALPHA_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatAlephAlpha(
                model=self.model,
                aleph_alpha_api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Aleph Alpha model: {str(e)}"
            ) from e


class GooseAILLMConfig(LLMConfig):
    """Configuration for GooseAI models."""

    provider: LLMProvider = LLMProvider.GOOSEAI
    model: str = Field(default="gpt-neo-20b", description="GooseAI model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GOOSEAI_API_KEY", "")),
        description="API key for GooseAI.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("GOOSEAI_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate GooseAI Chat model.
        """
        try:
            from langchain_community.chat_models import ChatGooseAI
        except ImportError:
            raise RuntimeError(
                "langchain-community is not installed. "
                "Please install it with 'pip install langchain-community'"
            )

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "GooseAI API key is required. "
                "Please set GOOSEAI_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatGooseAI(
                model=self.model,
                gooseai_api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate GooseAI model: {str(e)}") from e


class MosaicMLLLMConfig(LLMConfig):
    """Configuration for MosaicML models."""

    provider: LLMProvider = LLMProvider.MOSAICML
    model: str = Field(default="mpt-7b", description="MosaicML model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MOSAICML_API_KEY", "")),
        description="API key for MosaicML.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("MOSAICML_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate MosaicML Chat model.
        """
        from langchain_community.chat_models import ChatMosaicML

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "MosaicML API key is required. "
                "Please set MOSAICML_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatMosaicML(
                model=self.model,
                mosaicml_api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate MosaicML model: {str(e)}") from e


class NLPCloudLLMConfig(LLMConfig):
    """Configuration for NLP Cloud models."""

    provider: LLMProvider = LLMProvider.NLP_CLOUD
    model: str = Field(
        default="finetuned-gpt-neox-20b", description="NLP Cloud model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("NLP_CLOUD_API_KEY", "")),
        description="API key for NLP Cloud.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("NLP_CLOUD_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate NLP Cloud Chat model.
        """
        try:
            from langchain_community.llms import ChatNLPCloud
        except ImportError:
            raise RuntimeError(
                "langchain-nlpcloud is not installed. "
                "Please install it with 'pip install langchain-nlpcloud'"
            )

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "NLP Cloud API key is required. "
                "Please set NLP_CLOUD_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatNLPCloud(
                model=self.model,
                api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate NLP Cloud model: {str(e)}"
            ) from e


class OpenLMLLMConfig(LLMConfig):
    """Configuration for OpenLM models."""

    provider: LLMProvider = LLMProvider.OPENLM
    model: str = Field(default="open-llama-3b", description="OpenLM model name.")
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("OPENLM_API_KEY", "")),
        description="API key for OpenLM.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("OPENLM_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate OpenLM Chat model.
        """
        from langchain_community.chat_models import ChatOpenLM

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "OpenLM API key is required. "
                "Please set OPENLM_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatOpenLM(
                model=self.model,
                openlm_api_key=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate OpenLM model: {str(e)}") from e


class PetalsLLMConfig(LLMConfig):
    """Configuration for Petals distributed models."""

    provider: LLMProvider = LLMProvider.PETALS
    model: str = Field(default="bigscience/bloom", description="Petals model name.")
    # No API key needed for Petals distributed inference

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Petals Chat model.
        """
        from langchain_community.chat_models import ChatPetals

        try:
            return ChatPetals(
                model=self.model,
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Petals model: {str(e)}") from e


class ReplicateLLMConfig(LLMConfig):
    """Configuration for Replicate models."""

    provider: LLMProvider = LLMProvider.REPLICATE
    model: str = Field(
        default="meta/llama-2-70b-chat",
        description="Replicate model name (org/model:version).",
    )
    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Replicate.",
    )

    @field_validator("api_key")
    @classmethod
    def load_api_key(cls, v: SecretStr) -> SecretStr:
        """Load API key from environment if not provided."""
        if v.get_secret_value() == "":
            env_value = os.getenv("REPLICATE_API_KEY", "")
            return SecretStr(env_value)
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Replicate Chat model.
        """
        from langchain_community.chat_models import ChatReplicate

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Replicate API key is required. "
                "Please set REPLICATE_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatReplicate(
                model=self.model,
                replicate_api_token=self.get_api_key(),
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Replicate model: {str(e)}"
            ) from e


class VertexAILLMConfig(LLMConfig):
    """Configuration for Google Vertex AI models."""

    provider: LLMProvider = LLMProvider.VERTEX_AI
    model: str = Field(default="gemini-1.5-pro", description="Vertex AI model name.")
    project: Optional[str] = Field(default="", description="Google Cloud Project ID.")
    location: str = Field(
        default="us-central1", description="Google Cloud region/location."
    )
    # No direct API key for Vertex AI - uses Google Cloud auth

    @field_validator("project")
    @classmethod
    def load_project(cls, v: str) -> str:
        """Load project from environment if not provided."""
        if not v:
            env_value = os.getenv("GOOGLE_CLOUD_PROJECT", "")
            return env_value
        return v

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Google Vertex AI Chat model.
        """
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError:
            raise RuntimeError(
                "langchain-google-vertexai is not installed. "
                "Please install it with 'pip install langchain-google-vertexai'"
            )

        # Validate project
        if not self.project:
            raise ValueError(
                "Google Cloud Project ID is required. "
                "Please set GOOGLE_CLOUD_PROJECT environment variable or provide a project ID."
            )

        try:
            return ChatVertexAI(
                model_name=self.model,
                project=self.project,
                location=self.location,
                cache=self.cache_enabled,
                **(self.extra_params or {}),
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate Vertex AI model: {str(e)}"
            ) from e


# TODO: CONVERT OT LIST AND ADD SUPP FOR GEMINI
"""
To convert a ModelList object from the Mistral AI Python client into a list of model names like ['mistral-ocr-2505'], use this code:

python
# Assuming `model_list` is your ModelList instance
model_names = [model.id for model in model_list.data]
This works because:

ModelList objects have a data attribute containing model entries

Each model entry has an id field with the model name string

List comprehension efficiently extracts these IDs

Example with full context:

python
from mistralai import Mistral

client = Mistral(api_key="your_api_key")
model_list = client.models.list()  # Returns ModelList object

# Convert to list of model names
model_names = [model.id for model in model_list.data]
print(model_names)  # Output: ['mistral-ocr-2505', 'mistral-small-2503', ...]
"""
# print(AnthropicLLMConfig.get_models()[:10])
# print(OpenAILLMConfig.get_models()[:10])
# print(MistralLLMConfig.get_models(),type(MistralLLMConfig.get_models()))
# print(DeepSeekLLMConfig.get_models()) -> deepseek-chat,deepseek-reasoner
