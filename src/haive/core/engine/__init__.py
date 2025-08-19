"""⚡ Haive Engine System - The Universal AI Component Interface

**THE BEATING HEART OF INTELLIGENT AI ARCHITECTURES**

Welcome to the Engine System - a revolutionary abstraction layer that unifies all AI 
components under a single, elegant interface. This isn't just another wrapper around
language models; it's a comprehensive orchestration platform that enables seamless
integration, composition, and enhancement of any AI capability imaginable.

🧬 ENGINE ARCHITECTURE OVERVIEW
-------------------------------

The Engine System represents a paradigm shift in how AI components are built, composed,
and orchestrated. Every AI capability - from language models to vector stores, from
tools to retrievers - is unified under the Engine abstraction, enabling:

**1. Universal Interface** 🌐
   - **InvokableEngine**: Synchronous and asynchronous execution for any AI component
   - **Streaming Support**: Real-time token streaming for responsive experiences
   - **Batch Processing**: Efficient parallel execution of multiple requests
   - **Error Recovery**: Automatic retries and graceful degradation

**2. Enhanced Language Models** 🧠
   - **AugLLM**: Supercharged LLMs with tool use, structured output, and memory
   - **Multi-Provider Support**: OpenAI, Anthropic, Google, Hugging Face, and more
   - **Dynamic Configuration**: Runtime model switching without code changes
   - **Cost Optimization**: Automatic model selection based on task complexity

**3. Retrieval Systems** 📚
   - **Vector Search**: Semantic retrieval with multiple embedding models
   - **Hybrid Search**: Combine semantic, keyword, and metadata filtering
   - **Reranking**: Advanced relevance scoring with cross-encoders
   - **Incremental Indexing**: Update knowledge bases without full rebuilds

**4. Document Processing** 📄
   - **Universal Loaders**: Handle any document format automatically
   - **Intelligent Chunking**: Context-aware text splitting strategies
   - **Metadata Extraction**: Automatic extraction of document properties
   - **Format Detection**: Smart content type identification

**5. Tool Integration** 🛠️
   - **Type-Safe Tools**: Full validation and error handling
   - **Automatic Discovery**: Find and register tools dynamically
   - **Parallel Execution**: Run multiple tools simultaneously
   - **Result Aggregation**: Intelligent combination of tool outputs

**6. Vector Storage** 🗄️
   - **Multi-Backend Support**: Pinecone, Weaviate, Chroma, pgvector, and more
   - **Automatic Embeddings**: Generate and cache embeddings efficiently
   - **Similarity Search**: Fast k-NN search with filtering
   - **Persistence**: Durable storage with backup and recovery

🚀 QUICK START
--------------

```python
from haive.core.engine import AugLLMConfig, create_retriever, create_vectorstore

# 1. Create an enhanced LLM with tools and structured output
llm = AugLLMConfig(
    model="gpt-4",
    temperature=0.7,
    tools=["web_search", "calculator", "code_executor"],
    structured_output_model=AnalysisResult,
    system_message="You are a helpful AI assistant with tool access."
)

# 2. Create a vector store for knowledge management
vectorstore = create_vectorstore(
    type="pinecone",
    index_name="knowledge_base",
    embedding_model="text-embedding-3-large"
)

# 3. Create a retriever for RAG workflows
retriever = create_retriever(
    vectorstore=vectorstore,
    search_type="similarity",
    search_kwargs={"k": 5, "score_threshold": 0.7}
)

# 4. Compose engines for complex workflows
from haive.core.engine import compose_runnable

rag_chain = compose_runnable([
    retriever,
    llm.with_context_from_retriever()
])

# 5. Execute with streaming
async for chunk in rag_chain.astream("What are the latest AI breakthroughs?"):
    print(chunk, end="", flush=True)
```

🎯 KEY INNOVATIONS
------------------

**1. Unified Execution Model** 🔄
   Every engine supports the same interface:
   ```python
   # Synchronous
   result = engine.invoke(input_data)
   
   # Asynchronous
   result = await engine.ainvoke(input_data)
   
   # Streaming
   for chunk in engine.stream(input_data):
       process(chunk)
   
   # Batch processing
   results = engine.batch([input1, input2, input3])
   ```

**2. Dynamic Composition** 🧩
   Engines can be composed like building blocks:
   ```python
   # Chain engines together
   pipeline = retriever | reranker | llm | output_parser
   
   # Parallel execution
   parallel = retriever & web_search & database_query
   
   # Conditional routing
   router = conditional_engine(
       condition=lambda x: x.get("type") == "technical",
       if_true=technical_llm,
       if_false=general_llm
   )
   ```

**3. Intelligent Caching** 💾
   Automatic result caching with semantic similarity:
   ```python
   cached_llm = llm.with_caching(
       cache_type="semantic",
       similarity_threshold=0.95,
       ttl=3600
   )
   ```

**4. Observability Built-In** 📊
   Every engine emits detailed telemetry:
   ```python
   # Automatic metrics collection
   llm.metrics.latency_p95  # 95th percentile latency
   llm.metrics.token_usage  # Token consumption
   llm.metrics.error_rate   # Error percentage
   ```

📚 ENGINE MODULES
-----------------

**Core Modules**:
- `aug_llm/`: Enhanced language model configurations and factories
- `base/`: Base engine classes, protocols, and registry system
- `tool/`: Tool creation, discovery, and execution engine
- `output_parser/`: Structured output parsing and validation

**Data Processing**:
- `document/`: Universal document loading and processing
- `embedding/`: Embedding generation for multiple providers
- `vectorstore/`: Vector database integrations
- `retriever/`: Retrieval strategies and implementations

**Specialized Engines**:
- `agent/`: Agent-specific engine components (heavy, lazy-loaded)
- `prompt_template/`: Dynamic prompt generation engines

🏗️ ARCHITECTURE PATTERNS
------------------------

**1. Provider Abstraction**
```python
# Switch providers without changing code
config = AugLLMConfig(
    model="gpt-4",  # or "claude-3", "gemini-pro", "llama-2"
    provider="openai"  # auto-detected from model
)
```

**2. Engine Registry**
```python
# Register custom engines
@EngineRegistry.register("my_custom_engine")
class MyCustomEngine(InvokableEngine):
    async def ainvoke(self, input_data):
        # Custom implementation
        return process(input_data)

# Use anywhere
engine = EngineRegistry.create("my_custom_engine", config)
```

**3. Middleware Pattern**
```python
# Add capabilities to any engine
enhanced = base_engine.pipe(
    add_retry(max_attempts=3),
    add_rate_limiting(requests_per_minute=100),
    add_caching(ttl=3600),
    add_logging(level="DEBUG")
)
```

🎨 ADVANCED FEATURES
--------------------

**1. Multi-Modal Support** 🖼️
```python
vision_llm = AugLLMConfig(
    model="gpt-4-vision",
    accept_types=["text", "image", "video"]
)

result = vision_llm.invoke({
    "text": "What's in this image?",
    "image": image_data
})
```

**2. Function Calling** 📞
```python
llm_with_tools = AugLLMConfig(
    model="gpt-4",
    tools=[weather_tool, calculator_tool],
    tool_choice="auto"  # or "required", "none", specific tool
)
```

**3. Structured Output** 📋
```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    key_points: List[str]

llm = AugLLMConfig(
    model="gpt-4",
    structured_output_model=Analysis
)

result: Analysis = llm.invoke("Analyze this text...")
```

**4. Streaming with Callbacks** 🌊
```python
async def on_token(token: str):
    print(token, end="", flush=True)

async def on_complete(result: dict):
    print(f"\nTokens used: {result['usage']['total_tokens']}")

llm = AugLLMConfig(
    streaming=True,
    callbacks={
        "on_llm_new_token": on_token,
        "on_llm_end": on_complete
    }
)
```

🚨 PERFORMANCE OPTIMIZATIONS
----------------------------

The Engine System is optimized for production workloads:

- **Lazy Loading**: Heavy components loaded only when needed
- **Connection Pooling**: Reuse connections to external services
- **Batch Processing**: Efficient handling of multiple requests
- **Automatic Retries**: Exponential backoff for transient failures
- **Resource Management**: Automatic cleanup and garbage collection

💡 BEST PRACTICES
-----------------

1. **Use Type Hints**: Engines validate inputs based on type annotations
2. **Handle Errors**: Always wrap engine calls in try-except blocks
3. **Monitor Usage**: Track token consumption and API costs
4. **Cache Wisely**: Use semantic caching for expensive operations
5. **Compose Engines**: Build complex behaviors from simple components

🔗 SEE ALSO
-----------

- `haive.core.graph`: Build visual workflows with engines
- `haive.core.schema`: State management for engine outputs
- `haive.core.tools`: Create custom tools as engines
- `haive.agents`: Pre-built agents using the engine system

---

**The Engine System: Where AI Components Become Intelligent Building Blocks** ⚡
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
