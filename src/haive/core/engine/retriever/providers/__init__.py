"""Retriever provider implementations for the Haive framework.

This package contains implementations of various retriever providers that extend
the core retriever functionality. These implementations are automatically registered
with the BaseRetrieverConfig registry system through the @BaseRetrieverConfig.register
decorator, making them available for use through the common retriever interface.

Available retriever categories:
- Vector-based: VectorStore, MultiVector, SelfQuery, ParentDocument, TimeWeighted
- Sparse/Traditional: BM25, TFIDF, KNN, SVM
- Hybrid/Advanced: Ensemble, ContextualCompression, MultiQuery
- Cloud Services: Amazon Knowledge Bases, Azure AI Search, Google Vertex AI
- Specialty: Arxiv, PubMed, Wikipedia, WebResearch, Tavily

Features:
- Automatic registration through decorators
- Dynamic loading and discovery
- Consistent configuration interface
- Support for metadata filtering and reranking
- Integration with various vector stores and LLMs

Examples:
    Using a vector store retriever::

        from haive.core.engine.retriever import VectorStoreRetrieverConfig

        config = VectorStoreRetrieverConfig(
            vectorstore=my_vectorstore,
            search_kwargs={"k": 5}
        )
        retriever = config.instantiate()

    Using an ensemble retriever::

        from haive.core.engine.retriever import EnsembleRetrieverConfig

        config = EnsembleRetrieverConfig(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3]
        )
        ensemble = config.instantiate()

See Also:
    - BaseRetrieverConfig: Base class for all retriever configurations
    - RetrieverType: Enumeration of available retriever types
    - Retriever documentation: Complete usage guides and examples
"""

# Import all retriever provider configurations
from .AmazonKnowledgeBasesRetrieverConfig import AmazonKnowledgeBasesRetrieverConfig
from .ArceeRetrieverConfig import ArceeRetrieverConfig
from .ArxivRetrieverConfig import ArxivRetrieverConfig
from .AskNewsRetrieverConfig import AskNewsRetrieverConfig
from .AzureAISearchRetrieverConfig import AzureAISearchRetrieverConfig
from .BedrockRetrieverConfig import BedrockRetrieverConfig
from .BM25RetrieverConfig import BM25RetrieverConfig
from .ChatGPTPluginRetrieverConfig import ChatGPTPluginRetrieverConfig
from .CohereRagRetrieverConfig import CohereRagRetrieverConfig
from .ContextualCompressionRetrieverConfig import ContextualCompressionRetrieverConfig
from .DocArrayRetrieverConfig import DocArrayRetrieverConfig
from .ElasticsearchRetrieverConfig import ElasticsearchRetrieverConfig
from .EnsembleRetrieverConfig import EnsembleRetrieverConfig
from .GoogleDocumentAIWarehouseRetrieverConfig import (
    GoogleDocumentAIWarehouseRetrieverConfig,
)
from .GoogleVertexAISearchRetrieverConfig import GoogleVertexAISearchRetrieverConfig
from .KendraRetrieverConfig import KendraRetrieverConfig
from .KNNRetrieverConfig import KNNRetrieverConfig
from .LlamaIndexGraphRetrieverConfig import LlamaIndexGraphRetrieverConfig
from .LlamaIndexRetrieverConfig import LlamaIndexRetrieverConfig
from .MergerRetrieverConfig import MergerRetrieverConfig
from .MetalRetrieverConfig import MetalRetrieverConfig
from .MilvusRetrieverConfig import MilvusRetrieverConfig
from .MultiQueryRetrieverConfig import MultiQueryRetrieverConfig
from .MultiVectorRetrieverConfig import MultiVectorRetrieverConfig
from .NeuralDBRetrieverConfig import NeuralDBRetrieverConfig
from .ParentDocumentRetrieverConfig import ParentDocumentRetrieverConfig
from .PineconeHybridSearchRetrieverConfig import PineconeHybridSearchRetrieverConfig
from .PubMedRetrieverConfig import PubMedRetrieverConfig
from .QdrantSparseVectorRetrieverConfig import QdrantSparseVectorRetrieverConfig
from .RemoteLangChainRetrieverConfig import RemoteLangChainRetrieverConfig
from .RePhraseQueryRetrieverConfig import RePhraseQueryRetrieverConfig
from .SelfQueryRetrieverConfig import SelfQueryRetrieverConfig
from .SVMRetrieverConfig import SVMRetrieverConfig
from .TavilySearchAPIRetrieverConfig import TavilySearchAPIRetrieverConfig
from .TFIDFRetrieverConfig import TFIDFRetrieverConfig
from .TimeWeightedVectorStoreRetrieverConfig import (
    TimeWeightedVectorStoreRetrieverConfig,
)
from .VespaRetrieverConfig import VespaRetrieverConfig
from .WeaviateHybridSearchRetrieverConfig import WeaviateHybridSearchRetrieverConfig
from .WebResearchRetrieverConfig import WebResearchRetrieverConfig
from .WikipediaRetrieverConfig import WikipediaRetrieverConfig
from .YouRetrieverConfig import YouRetrieverConfig
from .ZepCloudRetrieverConfig import ZepCloudRetrieverConfig
from .ZepRetrieverConfig import ZepRetrieverConfig

__all__ = [
    # Vector-based retrievers
    "MultiVectorRetrieverConfig",
    "SelfQueryRetrieverConfig",
    "ParentDocumentRetrieverConfig",
    "TimeWeightedVectorStoreRetrieverConfig",
    # Sparse/Traditional retrievers
    "BM25RetrieverConfig",
    "TFIDFRetrieverConfig",
    "KNNRetrieverConfig",
    "SVMRetrieverConfig",
    # Hybrid/Advanced retrievers
    "EnsembleRetrieverConfig",
    "ContextualCompressionRetrieverConfig",
    "MultiQueryRetrieverConfig",
    "MergerRetrieverConfig",
    "RePhraseQueryRetrieverConfig",
    # Cloud service retrievers
    "AmazonKnowledgeBasesRetrieverConfig",
    "AzureAISearchRetrieverConfig",
    "GoogleVertexAISearchRetrieverConfig",
    "GoogleDocumentAIWarehouseRetrieverConfig",
    "BedrockRetrieverConfig",
    "KendraRetrieverConfig",
    # Search platform retrievers
    "ElasticsearchRetrieverConfig",
    "PineconeHybridSearchRetrieverConfig",
    "WeaviateHybridSearchRetrieverConfig",
    "QdrantSparseVectorRetrieverConfig",
    "VespaRetrieverConfig",
    "MilvusRetrieverConfig",
    "DocArrayRetrieverConfig",
    # Specialty/Domain retrievers
    "ArxivRetrieverConfig",
    "PubMedRetrieverConfig",
    "WikipediaRetrieverConfig",
    "WebResearchRetrieverConfig",
    "TavilySearchAPIRetrieverConfig",
    "AskNewsRetrieverConfig",
    "YouRetrieverConfig",
    # Integration retrievers
    "ChatGPTPluginRetrieverConfig",
    "CohereRagRetrieverConfig",
    "LlamaIndexRetrieverConfig",
    "LlamaIndexGraphRetrieverConfig",
    "RemoteLangChainRetrieverConfig",
    "MetalRetrieverConfig",
    "NeuralDBRetrieverConfig",
    "ArceeRetrieverConfig",
    "ZepRetrieverConfig",
    "ZepCloudRetrieverConfig",
]
