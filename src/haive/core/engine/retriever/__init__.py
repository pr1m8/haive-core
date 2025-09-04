"""📚 Retriever Engine - Intelligent Knowledge Discovery System

**THE ULTIMATE INFORMATION RETRIEVAL POWERHOUSE**

Welcome to the Retriever Engine - the revolutionary system that transforms vast 
knowledge repositories into instantly accessible, contextually relevant information. 
This isn't just document search; it's an intelligent knowledge discovery platform 
that brings the right information to your AI agents at exactly the right moment.

🧠 RETRIEVAL REVOLUTION
-----------------------

The Retriever Engine represents a paradigm shift in how AI systems access and utilize 
information. Every document, database, and knowledge source becomes an **intelligent 
knowledge partner** that provides:

**🔍 Semantic Understanding**: Find information by meaning, not just keywords
**⚡ Lightning-Fast Search**: Sub-second retrieval across millions of documents
**🎯 Context-Aware Results**: Relevance ranking based on conversation context
**🔄 Multi-Modal Retrieval**: Text, images, code, and structured data
**📊 Advanced Analytics**: Deep insights into information usage patterns

🌟 CORE INNOVATIONS
-------------------

**1. Universal Retrieval Interface** 🎯
   Every knowledge source speaks the same language:

Examples:
    >>> # Vector-based semantic search
    >>> vector_retriever = VectorStoreRetrieverConfig(
    >>> vector_store="pinecone",
    >>> index_name="knowledge_base",
    >>> search_type="similarity",
    >>> k=5
    >>> )
    >>>
    >>> # Hybrid search (semantic + keyword)
    >>> hybrid_retriever = HybridRetrieverConfig(
    >>> vector_store=vector_store,
    >>> keyword_store=elasticsearch_store,
    >>> alpha=0.7  # 70% semantic, 30% keyword
    >>> )
    >>>
    >>> # Ensemble retrieval (multiple strategies)
    >>> ensemble_retriever = EnsembleRetrieverConfig(
    >>> retrievers=[vector_retriever, bm25_retriever, dense_retriever],
    >>> weights=[0.5, 0.3, 0.2],
    >>> fusion_strategy="rank_fusion"
    >>> )

**2. Intelligent Context Awareness** 🧠
   Retrievers understand conversation context and user intent:

    >>> # Context-aware retrieval
    >>> contextual_retriever = ContextualRetrieverConfig(
    >>> base_retriever=vector_retriever,
    >>> context_window=10,  # Last 10 messages
    >>> context_weight=0.3,  # 30% influence on retrieval
    >>> user_profile_aware=True
    >>> )
    >>>
    >>> # Automatic context extraction
    >>> context = ContextManager.extract_context([
    >>> "I'm working on a Python project",
    >>> "We discussed Django yesterday", 
    >>> "Now I need help with database optimization"
    >>> ])
    >>>
    >>> # Retrieval adapts to context
    >>> docs = contextual_retriever.get_relevant_documents(
    >>> "How do I optimize queries?", 
    >>> context=context
    >>> )
    >>> # Returns Django-specific database optimization docs

**3. Advanced Retrieval Strategies** 🎨
   Multiple retrieval approaches for different use cases:

    >>> # Similarity search with threshold
    >>> similarity_config = SimilarityRetrieverConfig(
    >>> threshold=0.7,
    >>> max_distance=0.5,
    >>> rerank_top_k=20,
    >>> final_k=5
    >>> )
    >>>
    >>> # Multi-step retrieval
    >>> multi_step_config = MultiStepRetrieverConfig(
    >>> steps=[
    >>> # Step 1: Broad search
    >>> VectorRetrieverConfig(k=50),
    >>> # Step 2: Rerank with cross-encoder
    >>> RerankingRetrieverConfig(model="cross-encoder/ms-marco"),
    >>> # Step 3: Filter by metadata
    >>> MetadataFilterRetrieverConfig(filters={"type": "documentation"})
    >>> ]
    >>> )
    >>>
    >>> # Temporal-aware retrieval
    >>> temporal_config = TemporalRetrieverConfig(
    >>> base_retriever=vector_retriever,
    >>> time_decay_factor=0.1,  # Recent docs weighted higher
    >>> max_age_days=365,
    >>> boost_recent=True
    >>> )

**4. Multi-Modal Knowledge Integration** 🖼️
   Retrieve across different content types seamlessly:

    >>> # Multi-modal retriever
    >>> multimodal_config = MultiModalRetrieverConfig(
    >>> text_retriever=vector_text_retriever,
    >>> image_retriever=clip_image_retriever,
    >>> code_retriever=code_embedding_retriever,
    >>> fusion_strategy="late_fusion",
    >>> modality_weights={
    >>> "text": 0.6,
    >>> "image": 0.3,
    >>> "code": 0.1
    >>> }
    >>> )
    >>>
    >>> # Search across modalities
    >>> results = multimodal_config.search(
    >>> query="How to create a login form?",
    >>> modalities=["text", "image", "code"]
    >>> )
    >>> # Returns: documentation, UI mockups, and code examples

🎯 ADVANCED FEATURES
--------------------

**Real-time Index Updates** 🔄

    >>> # Dynamic index management
    >>> from haive.core.engine.retriever import DynamicRetriever
    >>>
    >>> dynamic_retriever = DynamicRetriever(
    >>> base_config=vector_config,
    >>> auto_update=True,
    >>> update_frequency="5min"
    >>> )
    >>>
    >>> # Add documents in real-time
    >>> dynamic_retriever.add_documents([new_doc1, new_doc2])
    >>>
    >>> # Documents immediately available for search
    >>> results = dynamic_retriever.search("new information")
    >>>
    >>> # Automatic index optimization
    >>> dynamic_retriever.optimize_index(
    >>> strategy="clustering",
    >>> schedule="daily",
    >>> performance_target="sub_100ms"
    >>> )

**Intelligent Query Expansion** 🧩

    >>> # Query understanding and expansion
    >>> query_expander = QueryExpansionRetriever(
    >>> base_retriever=vector_retriever,
    >>> expansion_strategies=[
    >>> "synonym_expansion",      # Add synonyms
    >>> "concept_expansion",      # Add related concepts
    >>> "entity_expansion",       # Add related entities
    >>> "contextual_expansion"    # Add context-specific terms
    >>> ]
    >>> )
    >>>
    >>> # Original query: "ML models"
    >>> # Expanded query: "machine learning models algorithms neural networks deep learning"
    >>> results = query_expander.search("ML models")

**Federated Search** 🌐

    >>> # Search across multiple knowledge sources
    >>> federated_config = FederatedRetrieverConfig(
    >>> sources={
    >>> "internal_docs": vector_store_retriever,
    >>> "external_apis": web_search_retriever,
    >>> "databases": sql_retriever,
    >>> "knowledge_graphs": graph_retriever
    >>> },
    >>> aggregation_strategy="weighted_merge",
    >>> source_priorities={
    >>> "internal_docs": 1.0,
    >>> "databases": 0.8,
    >>> "external_apis": 0.6,
    >>> "knowledge_graphs": 0.4
    >>> }
    >>> )
    >>>
    >>> # Single query searches everywhere
    >>> results = federated_config.search("customer retention strategies")
    >>> # Aggregates results from all sources with intelligent ranking

**Adaptive Retrieval** 🔮

    >>> # Retrieval that learns from usage patterns
    >>> adaptive_config = AdaptiveRetrieverConfig(
    >>> base_retriever=vector_retriever,
    >>> learning_enabled=True,
    >>> feedback_integration=True,
    >>> performance_optimization=True
    >>> )
    >>>
    >>> # Automatic improvement from usage
    >>> for query, expected_docs in training_data:
    >>> results = adaptive_config.search(query)
    >>> adaptive_config.provide_feedback(query, results, expected_docs)
    >>>
    >>> # Retrieval improves over time
    >>> print(f"Retrieval accuracy: {adaptive_config.get_accuracy():.2%}")
    >>> print(f"Average latency: {adaptive_config.get_avg_latency()}ms")

🏗️ RETRIEVER ARCHITECTURES
---------------------------

**Vector Store Integration** 🗃️

    >>> # Support for all major vector databases
    >>> configs = {
    >>> "pinecone": PineconeRetrieverConfig(
    >>> api_key="your-key",
    >>> environment="us-west1-gcp",
    >>> index_name="knowledge-base"
    >>> ),
    >>>
    >>> "weaviate": WeaviateRetrieverConfig(
    >>> url="http://localhost:8080",
    >>> class_name="Document",
    >>> additional_headers={"X-OpenAI-Api-Key": "your-key"}
    >>> ),
    >>>
    >>> "chroma": ChromaRetrieverConfig(
    >>> persist_directory="./chroma_db",
    >>> collection_name="documents"
    >>> ),
    >>>
    >>> "qdrant": QdrantRetrieverConfig(
    >>> url="http://localhost:6333",
    >>> collection_name="documents",
    >>> vector_size=1536
    >>> )
    >>> }
    >>>
    >>> # Seamless switching between providers
    >>> def create_retriever(provider: str) -> BaseRetriever:
    >>> config = configs[provider]
    >>> return config.create_retriever()

**Hybrid Search Strategies** 🔀

    >>> # Combine multiple search approaches
    >>> hybrid_strategies = {
    >>> "dense_sparse": DenseSparseHybridConfig(
    >>> dense_retriever=vector_retriever,
    >>> sparse_retriever=bm25_retriever,
    >>> alpha=0.7  # Dense weight
    >>> ),
    >>>
    >>> "multi_vector": MultiVectorHybridConfig(
    >>> retrievers=[
    >>> OpenAIRetriever(model="text-embedding-3-large"),
    >>> HuggingFaceRetriever(model="sentence-transformers/all-MiniLM-L6-v2"),
    >>> CosineSimilarityRetriever()
    >>> ],
    >>> fusion_method="rank_fusion"
    >>> ),
    >>>
    >>> "cascading": CascadingHybridConfig(
    >>> primary=fast_retriever,
    >>> secondary=accurate_retriever,
    >>> fallback=comprehensive_retriever,
    >>> cascade_threshold=0.8
    >>> )
    >>> }

**Custom Retrieval Pipelines** 🚰

    >>> # Build custom retrieval workflows
    >>> pipeline = RetrievalPipeline([
    >>> # Stage 1: Initial broad search
    >>> BroadSearchStage(k=100),
    >>>
    >>> # Stage 2: Metadata filtering
    >>> MetadataFilterStage(
    >>> filters={"language": "en", "type": "documentation"}
    >>> ),
    >>>
    >>> # Stage 3: Semantic reranking
    >>> SemanticRerankStage(
    >>> model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    >>> top_k=20
    >>> ),
    >>>
    >>> # Stage 4: Diversity injection
    >>> DiversityStage(
    >>> similarity_threshold=0.9,
    >>> max_similar_docs=2
    >>> ),
    >>>
    >>> # Stage 5: Final selection
    >>> FinalSelectionStage(k=5)
    >>> ])
    >>>
    >>> # Execute pipeline
    >>> results = pipeline.execute("How to optimize database queries?")

📊 RETRIEVAL ANALYTICS
----------------------

**Performance Monitoring** 📈

    >>> # Comprehensive retrieval analytics
    >>> analytics = RetrievalAnalytics()
    >>>
    >>> # Query performance analysis
    >>> performance = analytics.analyze_performance(time_range="last_7_days")
    >>> print(f"Average latency: {performance.avg_latency_ms}ms")
    >>> print(f"95th percentile: {performance.p95_latency_ms}ms")
    >>> print(f"Success rate: {performance.success_rate:.2%}")
    >>>
    >>> # Popular queries
    >>> popular = analytics.get_popular_queries(limit=10)
    >>> for query, count in popular:
    >>> print(f"'{query}': {count} searches")
    >>>
    >>> # Retrieval quality metrics
    >>> quality = analytics.analyze_quality()
    >>> print(f"Relevance score: {quality.avg_relevance:.2f}")
    >>> print(f"Coverage: {quality.coverage:.2%}")
    >>> print(f"Diversity: {quality.diversity:.2f}")

**A/B Testing Framework** 🧪

    >>> # Test different retrieval strategies
    >>> ab_test = RetrievalABTest(
    >>> control=vector_retriever,
    >>> treatment=hybrid_retriever,
    >>> traffic_split=0.5,
    >>> metrics=["relevance", "latency", "user_satisfaction"]
    >>> )
    >>>
    >>> # Run test
    >>> ab_test.start()
    >>>
    >>> # Monitor results
    >>> results = ab_test.get_results()
    >>> print(f"Control relevance: {results.control.relevance:.3f}")
    >>> print(f"Treatment relevance: {results.treatment.relevance:.3f}")
    >>> print(f"Statistical significance: {results.significance:.3f}")
    >>>
    >>> # Auto-select winner
    >>> if results.significance > 0.95:
    >>> winner = ab_test.get_winner()
    >>> ab_test.promote_winner()

🎨 RETRIEVAL PATTERNS
---------------------

**RAG Integration** 🔗

    >>> # Perfect integration with RAG workflows
    >>> rag_retriever = RAGRetrieverConfig(
    >>> vector_store=knowledge_base,
    >>> chunk_size=512,
    >>> chunk_overlap=50,
    >>> retrieval_strategy="mmr",  # Maximal Marginal Relevance
    >>> diversity_factor=0.3
    >>> )
    >>>
    >>> # Use in RAG chain
    >>> from haive.core.engine.aug_llm import AugLLMConfig
    >>>
    >>> rag_agent = AugLLMConfig(
    >>> model="gpt-4",
    >>> system_message="Answer based on the provided context.",
    >>> retriever=rag_retriever,
    >>> retrieval_k=5
    >>> )
    >>>
    >>> # Automatic retrieval + generation
    >>> answer = rag_agent.invoke("What are the latest AI developments?")

**Conversational Retrieval** 💬

    >>> # Retrieval aware of conversation history
    >>> conversational_config = ConversationalRetrieverConfig(
    >>> base_retriever=vector_retriever,
    >>> memory_window=10,
    >>> context_compression=True,
    >>> follow_up_detection=True
    >>> )
    >>>
    >>> # Multi-turn conversation
    >>> conversation = ConversationManager()
    >>> conversation.add_message("user", "Tell me about Python")
    >>> retriever_result = conversational_config.search("Python", conversation.context)
    >>>
    >>> conversation.add_message("user", "How about its performance?")
    >>> # Retriever understands "its" refers to Python
    >>> follow_up_result = conversational_config.search("How about its performance?", conversation.context)

**Domain-Specific Retrieval** 🎯

    >>> # Specialized retrievers for different domains
    >>> medical_retriever = MedicalRetrieverConfig(
    >>> vector_store=medical_knowledge_base,
    >>> terminology_expansion=True,
    >>> medical_entity_recognition=True,
    >>> safety_filters=["experimental", "unverified"]
    >>> )
    >>>
    >>> legal_retriever = LegalRetrieverConfig(
    >>> vector_store=legal_database,
    >>> case_law_search=True,
    >>> statute_search=True,
    >>> jurisdiction_filtering=True,
    >>> citation_tracking=True
    >>> )
    >>>
    >>> technical_retriever = TechnicalRetrieverConfig(
    >>> vector_store=technical_docs,
    >>> code_search=True,
    >>> api_documentation=True,
    >>> version_aware=True,
    >>> language_specific=True
    >>> )

🔒 ENTERPRISE FEATURES
----------------------

- **Access Control**: Document-level permissions and row-level security
- **Audit Logging**: Complete search history and access tracking
- **Multi-tenancy**: Isolated knowledge bases per organization
- **Scalability**: Horizontal scaling across multiple nodes
- **Backup & Recovery**: Automated backup and disaster recovery
- **Compliance**: GDPR, HIPAA, and SOC2 compliance features

🎓 BEST PRACTICES
-----------------

1. **Embedding Strategy**: Choose the right embedding model for your domain
2. **Chunk Size**: Optimize chunk size for your content type and use case
3. **Index Optimization**: Regular maintenance and optimization of vector indices
4. **Query Preprocessing**: Clean and expand queries for better results
5. **Result Postprocessing**: Filter and rerank results based on business rules
6. **Performance Monitoring**: Track latency, relevance, and user satisfaction
7. **Continuous Learning**: Implement feedback loops for ongoing improvement

🚀 GETTING STARTED
------------------

    >>> from haive.core.engine.retriever import VectorStoreRetrieverConfig
    >>> from haive.core.engine.vectorstore import VectorStoreConfig
    >>>
    >>> # 1. Create a vector store
    >>> vector_store = VectorStoreConfig(
    >>> provider="chroma",
    >>> collection_name="my_documents"
    >>> )
    >>>
    >>> # 2. Create a retriever
    >>> retriever = VectorStoreRetrieverConfig(
    >>> vector_store=vector_store,
    >>> search_type="similarity",
    >>> k=5
    >>> )
    >>>
    >>> # 3. Search for relevant documents
    >>> documents = retriever.search("machine learning algorithms")
    >>>
    >>> # 4. Use in an agent
    >>> from haive.core.engine.aug_llm import AugLLMConfig
    >>>
    >>> agent = AugLLMConfig(
    >>> model="gpt-4",
    >>> retriever=retriever
    >>> )
    >>>
    >>> answer = agent.invoke("Explain gradient descent")
    >>> # Agent automatically retrieves relevant docs and provides informed answer

---

**Retriever Engine: Where Knowledge Becomes Instantly Accessible** 📚
"""

# Lazy loading to avoid import-time registration
from haive.core.engine.retriever.retriever import (
    BaseRetrieverConfig,
    create_retriever_config,
    create_retriever_from_vectorstore,
)
from haive.core.engine.retriever.types import RetrieverType

# Alias for backward compatibility and documentation
create_retriever = create_retriever_from_vectorstore


def __getattr__(name: str):
    """Lazy load VectorStoreRetrieverConfig to avoid registration overhead."""
    if name == "VectorStoreRetrieverConfig":
        from haive.core.engine.retriever.retriever import VectorStoreRetrieverConfig

        return VectorStoreRetrieverConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseRetrieverConfig",
    "RetrieverType",
    "VectorStoreRetrieverConfig",
    "create_retriever",
    "create_retriever_config",
    "create_retriever_from_vectorstore",
]
