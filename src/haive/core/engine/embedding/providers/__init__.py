"""Embedding provider configurations.

This module contains all the embedding provider configurations for the Haive framework.
Each provider is implemented as a separate class that extends BaseEmbeddingConfig.

Available Providers:
    - OpenAI: OpenAI embedding models (text-embedding-3-large, etc.)
    - Azure OpenAI: Azure-hosted OpenAI embedding models
    - HuggingFace: HuggingFace Hub and local transformer models
    - Cohere: Cohere embedding models (embed-english-v3.0, etc.)
    - Google Vertex AI: Google Cloud Vertex AI embedding models
    - Ollama: Locally hosted Ollama embedding models
    - Fake: Fake embeddings for testing

Examples:
    Basic usage::

        from haive.core.engine.embedding.providers import OpenAIEmbeddingConfig

        config = OpenAIEmbeddingConfig(
            name="my_embeddings",
            model="text-embedding-3-large",
            api_key="sk-..."
        )

        embeddings = config.instantiate()

    Discovering providers::

        from haive.core.engine.embedding.base import BaseEmbeddingConfig

        # List all registered providers
        providers = BaseEmbeddingConfig.list_registered_types()
        print(f"Available providers: {list(providers.keys())}")

        # Get a specific provider
        provider_class = BaseEmbeddingConfig.get_config_class("OpenAI")
"""

# Lazy import all provider configurations to avoid registration overhead
_PROVIDER_CLASSES = {
    "AzureOpenAIEmbeddingConfig": "AzureOpenAIEmbeddingConfig",
    "CohereEmbeddingConfig": "CohereEmbeddingConfig",
    "FakeEmbeddingConfig": "FakeEmbeddingConfig",
    "GoogleVertexAIEmbeddingConfig": "GoogleVertexAIEmbeddingConfig",
    "HuggingFaceEmbeddingConfig": "HuggingFaceEmbeddingConfig",
    "OllamaEmbeddingConfig": "OllamaEmbeddingConfig",
    "OpenAIEmbeddingConfig": "OpenAIEmbeddingConfig",
}


def __getattr__(name: str):
    """Lazy load provider configurations to avoid import-time registration."""
    if name in _PROVIDER_CLASSES:
        module_name = _PROVIDER_CLASSES[name]
        module = __import__(
            f"haive.core.engine.embedding.providers.{module_name}", fromlist=[name]
        )
        config_class = getattr(module, name)
        # Cache the class in globals for subsequent access
        globals()[name] = config_class
        return config_class
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Export all configurations
__all__ = [
    "AzureOpenAIEmbeddingConfig",
    "CohereEmbeddingConfig",
    "FakeEmbeddingConfig",
    "GoogleVertexAIEmbeddingConfig",
    "HuggingFaceEmbeddingConfig",
    "OllamaEmbeddingConfig",
    "OpenAIEmbeddingConfig",
]


# Provider information for discovery - lazy class loading
def _get_provider_info():
    """Generate provider info with lazy class loading."""
    return {
        "OpenAI": {
            "class_name": "OpenAIEmbeddingConfig",
            "description": "OpenAI embedding models",
            "requires": ["langchain-openai"],
            "auth_required": True,
            "popular_models": ["text-embedding-3-large", "text-embedding-3-small"],
        },
        "AzureOpenAI": {
            "class_name": "AzureOpenAIEmbeddingConfig",
            "description": "Azure OpenAI embedding models",
            "requires": ["langchain-openai"],
            "auth_required": True,
            "popular_models": ["text-embedding-3-large", "text-embedding-3-small"],
        },
        "HuggingFace": {
            "class_name": "HuggingFaceEmbeddingConfig",
            "description": "HuggingFace Hub and local transformer models",
            "requires": ["langchain-huggingface", "sentence-transformers"],
            "auth_required": False,
            "popular_models": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-large-en-v1.5",
            ],
        },
        "Cohere": {
            "class_name": "CohereEmbeddingConfig",
            "description": "Cohere embedding models",
            "requires": ["langchain-cohere"],
            "auth_required": True,
            "popular_models": ["embed-english-v3.0", "embed-multilingual-v3.0"],
        },
        "GoogleVertexAI": {
            "class_name": "GoogleVertexAIEmbeddingConfig",
            "description": "Google Vertex AI embedding models",
            "requires": ["langchain-google-vertexai"],
            "auth_required": True,
            "popular_models": ["text-embedding-004", "text-multilingual-embedding-002"],
        },
        "Ollama": {
            "class_name": "OllamaEmbeddingConfig",
            "description": "Locally hosted Ollama embedding models",
            "requires": ["langchain-ollama"],
            "auth_required": False,
            "popular_models": ["nomic-embed-text", "mxbai-embed-large"],
        },
        "Fake": {
            "class_name": "FakeEmbeddingConfig",
            "description": "Fake embeddings for testing",
            "requires": ["langchain-community"],
            "auth_required": False,
            "popular_models": ["fake-model"],
        },
    }


PROVIDER_INFO = _get_provider_info()


def get_provider_info(provider_name: str | None = None) -> dict:
    """Get information about embedding providers.

    Args:
        provider_name: Optional provider name to get info for

    Returns:
        Provider information dictionary

    Examples:
        Get all provider info::

            info = get_provider_info()
            for name, details in info.items():
                print(f"{name}: {details['description']}")

        Get specific provider info::

            info = get_provider_info("OpenAI")
            print(f"Requires: {info['requires']}")
    """
    if provider_name:
        return PROVIDER_INFO.get(provider_name, {})
    return PROVIDER_INFO


def list_providers() -> list[str]:
    """List all available embedding providers.

    Returns:
        List of provider names

    Examples:
        List providers::

            providers = list_providers()
            print(f"Available providers: {providers}")
    """
    return list(PROVIDER_INFO.keys())


def get_providers_by_requirement(
    auth_required: bool | None = None, local_only: bool | None = None
) -> list[str]:
    """Get providers filtered by requirements.

    Args:
        auth_required: Filter by authentication requirement
        local_only: Filter for local-only providers

    Returns:
        List of provider names matching criteria

    Examples:
        Get providers that don't require auth::

            providers = get_providers_by_requirement(auth_required=False)

        Get local providers::

            providers = get_providers_by_requirement(local_only=True)
    """
    providers = []

    for name, info in PROVIDER_INFO.items():
        if auth_required is not None and info["auth_required"] != auth_required:
            continue
        if local_only is not None:
            is_local = name in ["Ollama", "HuggingFace", "Fake"]
            if is_local != local_only:
                continue
        providers.append(name)

    return providers


def get_installation_requirements(provider_name: str) -> list[str]:
    """Get installation requirements for a provider.

    Args:
        provider_name: Name of the provider

    Returns:
        List of required packages

    Examples:
        Get requirements::

            reqs = get_installation_requirements("OpenAI")
            print(f"Install: pip install {' '.join(reqs)}")
    """
    info = PROVIDER_INFO.get(provider_name, {})
    return info.get("requires", [])
