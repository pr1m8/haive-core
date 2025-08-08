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

# Agent imports are lazy-loaded to avoid expensive schema_composer initialization (17+ seconds)
# from haive.core.engine.agent import (...)
from haive.core.engine.aug_llm import (  # Temporarily commented out to fix circular import; MCPAugLLMConfig,
    AugLLMConfig,
    AugLLMFactory,
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

# Document imports are lazy-loaded to avoid expensive initialization
# from haive.core.engine.document import (...)
# Embedding imports are lazy-loaded to avoid numpy/pandas imports
# from haive.core.engine.embedding import (...)
from haive.core.engine.output_parser import OutputParserEngine, OutputParserType

# Prompt template imports are lazy-loaded to avoid circular import with schema_composer
# from haive.core.engine.prompt_template import (...)
# Retriever imports are lazy-loaded to avoid expensive initialization
# from haive.core.engine.retriever import (...)
from haive.core.engine.tool import ToolEngine

# Vectorstore imports are lazy-loaded to avoid pandas imports
# from haive.core.engine.vectorstore import (...)

# ========================================================================
# LAZY LOADING IMPLEMENTATION - Document components loaded on demand
# ========================================================================

# Component names for lazy loading
# Agent components - Heavy due to schema_composer (17+ seconds)
_AGENT_COMPONENTS = {
    "AGENT_REGISTRY",
    # "Agent",  # Moved to haive-agents package
    "AgentConfig",
    "AgentProtocol",
    "PatternConfig",
    "PatternManager",
    "PersistentAgentProtocol",
    "StreamingAgentProtocol",
}

_DOCUMENT_COMPONENTS = {
    # Core engine components
    "DocumentEngine",
    "create_document_engine",
    "load_documents",
    # Factory functions
    "create_file_document_engine",
    "create_web_document_engine",
    "create_directory_document_engine",
    # Configuration models
    "DocumentEngineConfig",
    "DocumentInput",
    "DocumentOutput",
    "ProcessedDocument",
    "DocumentChunk",
    # Enums
    "DocumentFormat",
    "DocumentSourceType",
    "LoaderPreference",
    "ProcessingStrategy",
    "ChunkingStrategy",
    # Path analysis
    "analyze_path_comprehensive",
    "PathAnalysisResult",
    "PathType",
    "FileCategory",
    "DatabaseType",
    "CloudProvider",
    # Loaders
    "BaseDocumentLoader",
    "SimpleDocumentLoader",
    "TextDocumentLoader",
    "DocumentLoaderRegistry",
    "get_default_registry",
    "register_loader",
    "get_loader",
    "create_loader",
    # Processors
    "DocumentProcessor",
    "ChunkingProcessor",
    "ContentNormalizer",
    "FormatDetector",
    "MetadataExtractor",
    # Advanced components
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
}

_RETRIEVER_COMPONENTS = {
    # Core retriever components
    "BaseRetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
}

_PROMPT_COMPONENTS = {
    # Prompt template components - lazy to avoid circular import
    "PromptTemplateEngine"
}

_EMBEDDING_COMPONENTS = {
    # Embedding components - lazy to avoid numpy/pandas imports
    "BaseEmbeddingConfig",
    "EmbeddingType",
    "create_embedding_config",
}

_VECTORSTORE_COMPONENTS = {
    # Vectorstore components - lazy to avoid pandas imports
    "VectorStoreConfig",
    "create_retriever",
    "create_retriever_from_documents",
    "create_vectorstore",
    "create_vs_config_from_documents",
    "create_vs_from_documents",
}


# Import core components directly - lazy loading was causing too many issues
from haive.core.engine.embedding import (
    BaseEmbeddingConfig,
    EmbeddingType,
    create_embedding_config,
)

# Note: Heavy components (agent, document, retriever, vectorstore) are available
# via explicit imports but not auto-imported to avoid startup cost


__all__ = [
    # Core LLM Components
    "AugLLMConfig",
    # Base Engine Classes
    "Engine",
    "InvokableEngine",
    "EngineType",
    "EngineRegistry",
    # Embedding Components
    "BaseEmbeddingConfig",
    "EmbeddingType",
    "create_embedding_config",
    # Output Parser Components
    "OutputParserEngine",
    "OutputParserType",
    # Note: Heavy components (agent, document, retriever, vectorstore)
    # are available via explicit submodule imports:
    # from haive.core.engine.agent import ...
    # from haive.core.engine.document import ...
    # from haive.core.engine.retriever import ...
    # from haive.core.engine.vectorstore import ...
]
