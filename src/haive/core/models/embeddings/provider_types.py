from enum import Enum
class EmbeddingProvider(str, Enum):
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    COHERE = "cohere"
