from enum import Enum


class LLMProvider(str, Enum):
    """Enumeration of supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGING_FACE = "huggingface"
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
    MISTRALAI = "mistralai"
    GROQ = "groq"
    FIREWORKS_AI = "fireworks_ai"
    PERPLEXITY = "perplexity"

    # Override string representation for better compatibility
    def __str__(self):
        return self.value

    # Enable direct string comparison
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
