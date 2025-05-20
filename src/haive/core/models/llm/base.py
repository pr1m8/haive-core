"""
Base LLM configuration with model metadata support.

This module provides base classes and implementations for LLM providers
with support for model metadata, context windows, and capabilities.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, SecretStr, model_post_init

# Import the mixins
from haive.core.common.secure_config_mixin import SecureConfigMixin
from haive.core.models.llm.provider_types import LLMProvider
from haive.core.models.metadata_mixin import ModelMetadataMixin

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv(".env")
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
    api_key: SecretStr = Field(
        default=SecretStr(""), description="API key for LLM provider."
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

    model_config = {"arbitrary_types_allowed": True}

    @model_post_init
    def _post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to load model metadata.

        This automatically runs after the model is initialized.
        """
        logger.debug(f"Loading metadata for {self.model} from {self.provider}")

        # Check model capabilities after initialization
        try:
            context_window = self.get_context_window()
            logger.debug(f"Model {self.model} context window: {context_window}")

            pricing = self.get_token_pricing()
            logger.debug(f"Model {self.model} pricing: {pricing}")
        except Exception as e:
            logger.warning(f"Error loading model metadata: {e}")

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Azure OpenAI Chat model with robust error handling.
        """
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError:
            raise RuntimeError(
                "langchain-openai is not installed. "
                "Please install it with 'pip install langchain-openai'"
            )

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
                api_key=self.get_api_key(),
                api_version=self.api_version,
                azure_endpoint=self.api_base,
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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate OpenAI Chat model.
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise RuntimeError(
                "langchain-openai is not installed. "
                "Please install it with 'pip install langchain-openai'"
            )

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "OpenAI API key is required. "
                "Please set OPENAI_API_KEY environment variable or provide an API key."
            )

        try:
            return ChatOpenAI(
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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Anthropic Chat model.
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise RuntimeError(
                "langchain-anthropic is not installed. "
                "Please install it with 'pip install langchain-anthropic'"
            )

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
        default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")),
        description="API key for Google Gemini.",
    )

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Google Gemini Chat model.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError(
                "langchain-google-genai is not installed. "
                "Please install it with 'pip install langchain-google-genai'"
            )

        # Validate API key
        if not self.get_api_key():
            raise ValueError(
                "Google Gemini API key is required. "
                "Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable or provide an API key."
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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate DeepSeek Chat model.
        """
        try:
            from langchain_deepseek import ChatDeepSeek
        except ImportError:
            raise RuntimeError(
                "langchain-deepseek is not installed. "
                "Please install it with 'pip install langchain-deepseek'"
            )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Mistral Chat model.
        """
        try:
            from langchain_mistralai import ChatMistralAI
        except ImportError:
            raise RuntimeError(
                "langchain-mistralai is not installed. "
                "Please install it with 'pip install langchain-mistralai'"
            )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Groq Chat model.
        """
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise RuntimeError(
                "langchain-groq is not installed. "
                "Please install it with 'pip install langchain-groq'"
            )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Cohere Chat model.
        """
        try:
            from langchain_cohere import ChatCohere
        except ImportError:
            raise RuntimeError(
                "langchain-cohere is not installed. "
                "Please install it with 'pip install langchain-cohere'"
            )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Together AI Chat model.
        """
        try:
            from langchain_together import ChatTogether
        except ImportError:
            raise RuntimeError(
                "langchain-together is not installed. "
                "Please install it with 'pip install langchain-together'"
            )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Fireworks AI Chat model.
        """
        try:
            from langchain_fireworks import ChatFireworks
        except ImportError:
            raise RuntimeError(
                "langchain-fireworks is not installed. "
                "Please install it with 'pip install langchain-fireworks'"
            )

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
        description="API key for Perplexity.",
    )

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

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate NLP Cloud Chat model.
        """
        try:
            from langchain_nlpcloud import ChatNLPCloud
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
        default_factory=lambda: SecretStr(os.getenv("REPLICATE_API_KEY", "")),
        description="API key for Replicate.",
    )

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
    project: Optional[str] = Field(
        default_factory=lambda: os.getenv("GOOGLE_CLOUD_PROJECT", ""),
        description="Google Cloud Project ID.",
    )
    location: str = Field(
        default="us-central1", description="Google Cloud region/location."
    )
    # No direct API key for Vertex AI - uses Google Cloud auth

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
