# src/haive/core/models/provider_types.py

from enum import Enum


class LLMProvider(str, Enum):
    """Types of LLM providers supported by the system."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    COHERE = "cohere"
    AI21 = "ai21"
    ALEPH_ALPHA = "aleph_alpha"
    GOOSEAI = "gooseai"
    MOSAICML = "mosaicml"
    NLP_CLOUD = "nlp_cloud"
    OPENLM = "openlm"
    PETALS = "petals"
    REPLICATE = "replicate"
    TOGETHER_AI = "together_ai"
    FIREWORKS_AI = "fireworks_ai"
    PERPLEXITY = "perplexity"
    MISTRALAI = "mistralai"
    GROQ = "groq"
    VERTEX_AI = "vertex_ai"
