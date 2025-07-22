"""Engine module for the Haive framework.

This module provides the core engine system that powers all AI agents and workflows
in Haive. It includes LLM configurations, retriever systems, vector stores, and
the base engine infrastructure.

The engine system is designed to be modular and extensible, allowing for easy
integration of new AI models, retrieval systems, and vector storage backends.

Modules:
    aug_llm: Augmented LLM configurations and factories for creating LLM engines
    base: Base engine classes, protocols, and registry system
    retriever: Retriever implementations for RAG (Retrieval-Augmented Generation)
    vectorstore: Vector store integrations for semantic search and storage
    document: Document processing and transformation utilities
    agent: Agent-specific engine components

Key Classes:
    AugLLMConfig: Configuration for augmented LLM engines with enhanced capabilities
    Engine: Base class for all engine implementations
    EngineRegistry: Registry for managing and discovering engine types
    BaseRetrieverConfig: Configuration for retriever engines
    VectorStoreConfig: Configuration for vector store engines

Factory Functions:
    create_retriever: Create a retriever engine from configuration
    create_vectorstore: Create a vector store engine from configuration
    create_retriever_from_documents: Create a retriever with pre-loaded documents

Examples:
    Creating an LLM engine::

        from haive.core.engine import AugLLMConfig

        config = AugLLMConfig(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

    Creating a vector store::

        from haive.core.engine import create_vectorstore, VectorStoreConfig

        config = VectorStoreConfig(
            type="chroma",
            collection_name="knowledge_base"
        )
        vectorstore = create_vectorstore(config)

    Creating a retriever::

        from haive.core.engine import create_retriever

        retriever = create_retriever(
            vectorstore=vectorstore,
            search_kwargs={"k": 5}
        )
"""

from haive.core.engine.agent import (
    AGENT_REGISTRY,
    Agent,
    AgentConfig,
    AgentProtocol,
    PatternConfig,
    PatternManager,
    PersistentAgentProtocol,
    StreamingAgentProtocol,
)
from haive.core.engine.aug_llm import (
    AugLLMConfig,
    AugLLMFactory,
    MCPAugLLMConfig,
    compose_runnable,
    merge_configs,
)
from haive.core.engine.base import (
    Engine,
    EngineRegistry,
    EngineType,
    InvokableEngine,
    NonInvokableEngine,
)
from haive.core.engine.document import (  # Core engine; Factory functions; Configuration models; Enums; Path analysis; Loaders; Registry; Processors; Factory and enhanced loaders; Enhanced source system; Strategy system; Specific loaders
    AutoLoaderFactory,
    BaseDocumentLoader,
    ChunkingProcessor,
    ChunkingStrategy,
    CloudProvider,
    ContentNormalizer,
    CredentialManager,
    DatabaseType,
    DocumentChunk,
    DocumentEngine,
    DocumentEngineConfig,
    DocumentFormat,
    DocumentInput,
    DocumentLoaderRegistry,
    DocumentOutput,
    DocumentProcessor,
    DocumentSourceType,
    EnhancedSource,
    FileCategory,
    FormatDetector,
    LoaderCapability,
    LoaderPreference,
    LoaderPriority,
    LoaderStrategy,
    MetadataExtractor,
    MongoDBSource,
    PathAnalysisResult,
    PathType,
    PostgreSQLSource,
    ProcessedDocument,
    ProcessingStrategy,
    SimpleDocumentLoader,
    TextDocumentLoader,
    analyze_path_comprehensive,
    analyze_source,
    create_directory_document_engine,
    create_document_engine,
    create_document_loader,
    create_file_document_engine,
    create_loader,
    create_web_document_engine,
    get_default_registry,
    get_loader,
    load_documents,
    register_loader,
)
from haive.core.engine.embedding import (
    BaseEmbeddingConfig,
    EmbeddingType,
    create_embedding_config,
)
from haive.core.engine.output_parser import (
    OutputParserEngine,
    OutputParserType,
)
from haive.core.engine.prompt_template import (
    PromptTemplateEngine,
)
from haive.core.engine.retriever import (
    BaseRetrieverConfig,
    RetrieverType,
    VectorStoreRetrieverConfig,
)
from haive.core.engine.tool import (
    ToolEngine,
)
from haive.core.engine.vectorstore import (
    VectorStoreConfig,
    create_retriever,
    create_retriever_from_documents,
    create_vectorstore,
    create_vs_config_from_documents,
    create_vs_from_documents,
)

__all__ = [
    # Agent Components
    "Agent",
    "AgentConfig",
    "PatternConfig",
    "PatternManager",
    "AgentProtocol",
    "StreamingAgentProtocol",
    "PersistentAgentProtocol",
    "AGENT_REGISTRY",
    # LLM Components
    "AugLLMConfig",
    "AugLLMFactory",
    "MCPAugLLMConfig",
    "compose_runnable",
    "merge_configs",
    # Base Engine Classes
    "Engine",
    "EngineRegistry",
    "EngineType",
    "InvokableEngine",
    "NonInvokableEngine",
    # Document Components - Core
    "DocumentEngine",
    "create_document_engine",
    "load_documents",
    # Document Components - Factory Functions
    "create_file_document_engine",
    "create_web_document_engine",
    "create_directory_document_engine",
    # Document Components - Configuration
    "DocumentEngineConfig",
    "DocumentInput",
    "DocumentOutput",
    "ProcessedDocument",
    "DocumentChunk",
    # Document Components - Enums
    "DocumentFormat",
    "DocumentSourceType",
    "LoaderPreference",
    "ProcessingStrategy",
    "ChunkingStrategy",
    # Document Components - Path Analysis
    "analyze_path_comprehensive",
    "PathAnalysisResult",
    "PathType",
    "FileCategory",
    "DatabaseType",
    "CloudProvider",
    # Document Components - Loaders
    "BaseDocumentLoader",
    "SimpleDocumentLoader",
    "TextDocumentLoader",
    "DocumentLoaderRegistry",
    "get_default_registry",
    "register_loader",
    "get_loader",
    "create_loader",
    # Document Components - Processors
    "DocumentProcessor",
    "ChunkingProcessor",
    "ContentNormalizer",
    "FormatDetector",
    "MetadataExtractor",
    # Document Components - Advanced
    "AutoLoaderFactory",
    "create_document_loader",
    "analyze_source",
    "CredentialManager",
    "EnhancedSource",
    "LoaderStrategy",
    "LoaderCapability",
    "LoaderPriority",
    "MongoDBSource",
    "PostgreSQLSource",
    # Embedding Components
    "BaseEmbeddingConfig",
    "EmbeddingType",
    "create_embedding_config",
    # Output Parser Components
    "OutputParserEngine",
    "OutputParserType",
    # Prompt Template Components
    "PromptTemplateEngine",
    # Retriever Components
    "BaseRetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
    # Tool Components
    "ToolEngine",
    # Vector Store Components
    "VectorStoreConfig",
    "create_retriever",
    "create_retriever_from_documents",
    "create_vectorstore",
    "create_vs_config_from_documents",
    "create_vs_from_documents",
]
