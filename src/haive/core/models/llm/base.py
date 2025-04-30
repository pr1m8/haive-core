import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
from typing import Optional, Dict, Any
from pathlib import Path

# Load environment variables from .env file
load_dotenv('.env')

# Define a relative cache path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "lc_cache"
CACHE_FILE = CACHE_DIR / ".langchain_cache.db"

# Ensure cache directory exists
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set LangChain cache
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=str(CACHE_FILE)))

from haive.core.models.llm.provider_types import LLMProvider

class SecureConfigMixin:
    """
    A mixin to provide secure and flexible configuration for API keys.
    """
    @field_validator('api_key', mode='after')
    @classmethod
    def _validate_api_key(cls, v, values):
        """
        Dynamically set the API key with robust fallback mechanism.
        
        1. Use explicitly provided value
        2. Try environment variable based on provider
        3. Fall back to default/empty
        """
        import os
        from pydantic import SecretStr
        
        # If a value is already set and not empty, return it
        if v is not None and v != "":
            # If it's not a SecretStr, convert it
            if not isinstance(v, SecretStr):
                return SecretStr(str(v))
            return v
        
        # Determine the environment variable based on provider
        provider = values.get('provider')
        if not provider:
            return SecretStr("")
        
        print(f"Validating API key for provider: {provider}")
        
        # Create mapping for both enum values and string values
        env_key_map = {
            # Using .value to handle enum objects
            "azure": 'AZURE_OPENAI_API_KEY',
            "openai": 'OPENAI_API_KEY',
            "anthropic": 'ANTHROPIC_API_KEY',
            "gemini": 'GEMINI_API_KEY',
            "deepseek": 'DEEPSEEK_API_KEY',
            "mistralai": 'MISTRAL_API_KEY',
            "groq": 'GROQ_API_KEY',
            "cohere": 'COHERE_API_KEY',
            "together_ai": 'TOGETHER_AI_API_KEY',
            "fireworks_ai": 'FIREWORKS_AI_API_KEY',
            "perplexity": 'PERPLEXITY_API_KEY',
            "huggingface": 'HUGGING_FACE_API_KEY',
            "ai21": 'AI21_API_KEY',
            "aleph_alpha": 'ALEPH_ALPHA_API_KEY',
            "gooseai": 'GOOSEAI_API_KEY',
            "mosaicml": 'MOSAICML_API_KEY',
            "nlp_cloud": 'NLP_CLOUD_API_KEY',
            "openlm": 'OPENLM_API_KEY',
            "petals": 'PETALS_API_KEY',
            "replicate": 'REPLICATE_API_KEY',
        }
        
        # Get provider value - handle both enum and string
        provider_value = provider.value if hasattr(provider, 'value') else str(provider)
        
        # Try to get API key from environment variables
        env_key = env_key_map.get(provider_value.lower())
        
        if env_key:
            print(f"Looking for environment variable: {env_key}")
            api_key = os.getenv(env_key)
            if api_key and api_key.strip():
                print(f"Found API key for {provider_value} in {env_key} (length: {len(api_key)})")
                return SecretStr(api_key)
            else:
                print(f"WARNING: Environment variable {env_key} not found or empty")
        else:
            print(f"WARNING: No environment mapping found for provider: {provider_value}")
        
        # If no key found, return an empty SecretStr
        print(f"No API key found for {provider_value}, returning empty SecretStr")
        return SecretStr("")

    def get_api_key(self) -> Optional[str]:
        """
        Safely retrieve the API key with improved error handling.
        
        Returns:
            Optional[str]: The API key value, or None if not set
        """
        try:
            # Check if api_key attribute exists and is not None
            if not hasattr(self, 'api_key') or self.api_key is None:
                print(f"WARNING: No API key attribute found for {getattr(self, 'provider', 'unknown provider')}")
                return None
                
            # Try to get the secret value
            key_value = self.api_key.get_secret_value()
            
            # Check if the key is actually empty
            if not key_value or key_value.strip() == "":
                print(f"WARNING: API key for {getattr(self, 'provider', 'unknown provider')} is empty")
                return None
                
            return key_value
        except Exception as e:
            print(f"Error retrieving API key: {e}")
            
            # If we're in a testing environment, use fake keys for development
            if os.getenv('ENVIRONMENT') == 'development' or os.getenv('TESTING') == 'true':
                provider_value = getattr(self, 'provider', '')
                provider_str = provider_value.value if hasattr(provider_value, 'value') else str(provider_value)
                print(f"Using fake test key for {provider_str} in development environment")
                return f"test_key_{provider_str}"
                
            return None

class LLMConfig(BaseModel, SecureConfigMixin):
    """
    Base configuration for Language Model providers with secure key handling.
    """
    provider: LLMProvider = Field(description="The provider of the LLM.")
    model: str = Field(..., description="The model to be used, e.g., gpt-4.")
    api_key: SecretStr = Field(default=SecretStr(""), description="API key for LLM provider.")
    cache_enabled: bool = Field(default=True, description="Enable or disable response caching.")
    cache_ttl: Optional[int] = Field(default=300, description="Time-to-live for cache (in seconds).")
    extra_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional extra parameters.")

    def instantiate(self, **kwargs) -> Any:
        """
        Abstract method to be implemented by subclasses.
        
        Raises:
            NotImplementedError: If not overridden by a subclass
        """
        raise NotImplementedError("This method should be overridden in subclasses.")
class AzureLLMConfig(LLMConfig):
    """Configuration specific to Azure OpenAI."""
    provider: LLMProvider = LLMProvider.AZURE
    model: str = Field(
        default="gpt-4o", 
        description="Azure deployment name (model)."
    )
    api_version: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"), 
        description="Azure API version."
    )
    api_base: str = Field(
        default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT", ""), 
        description="Azure API base URL."
    )
    api_type: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_TYPE", "azure"), 
        description="API type for Azure."
    )
    # Direct loading of API key from environment
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("AZURE_OPENAI_API_KEY", "")),
        description="API key for Azure OpenAI."
    )

    def instantiate(self, **kwargs) -> Any:
        """
        Instantiate Azure OpenAI Chat model with robust error handling.
        """
        from langchain_openai import AzureChatOpenAI
        
        # Debug output (can keep this)
        print(f"Debug - Attempting to instantiate Azure OpenAI model:")
        print(f"- Model/deployment: {self.model}")
        print(f"- API version: {self.api_version}")
        print(f"- API base: {self.api_base}")
        print(f"- API type: {self.api_type}")
        print(f"- API key available: {'Yes' if self.get_api_key() else 'No'}")
        
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            print(f"Failed to instantiate Azure OpenAI model: {str(e)}")
            raise RuntimeError(f"Failed to instantiate Azure OpenAI model: {str(e)}") from e
class OpenAILLMConfig(LLMConfig):
    """Configuration for OpenAI models."""
    provider: LLMProvider = LLMProvider.OPENAI

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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate OpenAI model: {str(e)}") from e

class AnthropicLLMConfig(LLMConfig):
    """Configuration for Anthropic models."""
    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229"),
        description="Anthropic model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", "")),
        description="API key for Anthropic."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Anthropic model: {str(e)}") from e

class GeminiLLMConfig(LLMConfig):
    """Configuration for Google Gemini models."""
    provider: LLMProvider = LLMProvider.GEMINI
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GOOGLE_API_KEY", "")),
        description="API key for Google Gemini."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Gemini model: {str(e)}") from e

class DeepSeekLLMConfig(LLMConfig):
    """Configuration for DeepSeek models."""
    provider: LLMProvider = LLMProvider.DEEPSEEK
    model: str = Field(
        default="deepseek-chat",
        description="DeepSeek model name."
    )
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("DEEPSEEK_API_KEY", "")),
        description="API key for DeepSeek."
    )

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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate DeepSeek model: {str(e)}") from e
class MistralLLMConfig(LLMConfig):
    """Configuration for Mistral models."""
    provider: LLMProvider = LLMProvider.MISTRALAI
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("MISTRAL_API_KEY", "")),
        description="API key for Mistral."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Mistral model: {str(e)}") from e

class GroqLLMConfig(LLMConfig):
    """Configuration for Groq models."""
    provider: LLMProvider = LLMProvider.GROQ
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")),
        description="API key for Groq."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Groq model: {str(e)}") from e

class CohereLLMConfig(LLMConfig):
    """Configuration for Cohere models."""
    provider: LLMProvider = LLMProvider.COHERE
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("COHERE_API_KEY", "")),
        description="API key for Cohere."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Cohere model: {str(e)}") from e

class TogetherAILLMConfig(LLMConfig):
    """Configuration for Together AI models."""
    provider: LLMProvider = LLMProvider.TOGETHER_AI
    api_key: SecretStr = Field(
        default_factory=lambda: SecretStr(os.getenv("TOGETHER_AI_API_KEY", "")),
        description="API key for Together AI."
    )
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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Together AI model: {str(e)}") from e

class FireworksAILLMConfig(LLMConfig):
    """Configuration for Fireworks AI models."""
    provider: LLMProvider = LLMProvider.FIREWORKS_AI

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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Fireworks AI model: {str(e)}") from e

class PerplexityLLMConfig(LLMConfig):
    """Configuration for Perplexity AI models."""
    provider: LLMProvider = LLMProvider.PERPLEXITY

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
                **(self.extra_params or {}),
                **kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate Perplexity model: {str(e)}") from e

# Example usage and testing
if __name__ == "__main__":
    # Demonstration of configuration and instantiation for different providers
    providers = [
        AzureLLMConfig(model="gpt-4o"),
        OpenAILLMConfig(model="gpt-3.5-turbo"),
        AnthropicLLMConfig(model="claude-3-opus-20240229"),
        GeminiLLMConfig(model="gemini-pro"),
        DeepSeekLLMConfig(model="deepseek-chat"),
        MistralLLMConfig(model="mistral-large-latest"),
        GroqLLMConfig(model="llama-3-70b-chat"),
        CohereLLMConfig(model="command-r"),
        TogetherAILLMConfig(model="mistralai/Mistral-7B-Instruct-v0.2"),
        FireworksAILLMConfig(model="accounts/fireworks/models/llama-v2-7b-chat"),
        PerplexityLLMConfig(model="llama-3-sonar-large-32k-chat")
    ]

    for provider_config in providers:
        try:
            print(f"\nTesting {provider_config.provider} configuration:")
            llm = provider_config.instantiate_llm()
            print(f"{provider_config.provider} LLM instantiated successfully!")
        except Exception as e:
            print(f"Error initializing {provider_config.provider} LLM: {e}")


