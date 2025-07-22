"""Vector Store Provider Discovery and Management.

This module provides utilities for discovering, comparing, and configuring
vector store providers within the Haive framework. It offers comprehensive
information about all available vector store backends.

Examples:
    Basic discovery::

        from haive.core.engine.vectorstore.discovery import get_vectorstore_providers

        providers = get_vectorstore_providers()
        print(f"Available: {list(providers.keys())}")

    Get provider recommendations::

        from haive.core.engine.vectorstore.discovery import recommend_vectorstore

        # For development
        dev_stores = recommend_vectorstore("development")
        print(f"For development: {dev_stores}")

        # For production
        prod_stores = recommend_vectorstore("production")
        print(f"For production: {prod_stores}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class VectorStoreType(str, Enum):
    """Categories of vector stores."""

    LOCAL = "local"  # FAISS, Annoy, InMemory
    CLOUD = "cloud"  # Pinecone, Weaviate Cloud
    HYBRID = "hybrid"  # Chroma, Qdrant, Weaviate
    DATABASE = "database"  # PGVector, ClickHouse
    SEARCH_ENGINE = "search"  # Elasticsearch, OpenSearch


class CostTier(str, Enum):
    """Cost structure for vector stores."""

    FREE = "free"  # FAISS, InMemory, Chroma (local)
    FREEMIUM = "freemium"  # Pinecone, Qdrant Cloud
    PAID = "paid"  # Most cloud offerings
    ENTERPRISE = "enterprise"  # Enterprise-only features


@dataclass
class VectorStoreInfo:
    """Comprehensive information about a vector store provider."""

    name: str
    description: str
    type: VectorStoreType
    cost: CostTier
    auth_required: bool
    setup_complexity: str  # "easy", "medium", "complex"
    performance_tier: str  # "basic", "good", "excellent"

    # Capabilities
    supports_metadata_filtering: bool = True
    supports_hybrid_search: bool = False
    supports_multi_tenancy: bool = False
    supports_real_time_updates: bool = True

    # Technical specs
    max_dimensions: Optional[int] = None
    index_types: List[str] = None
    distance_metrics: List[str] = None

    # Requirements
    python_packages: List[str] = None
    external_dependencies: List[str] = None

    # Usage info
    popular_use_cases: List[str] = None
    best_for: str = ""

    def __post_init__(self):
        """Set defaults for mutable fields."""
        if self.index_types is None:
            self.index_types = ["flat", "hnsw"]
        if self.distance_metrics is None:
            self.distance_metrics = ["cosine", "euclidean", "dot_product"]
        if self.python_packages is None:
            self.python_packages = []
        if self.external_dependencies is None:
            self.external_dependencies = []
        if self.popular_use_cases is None:
            self.popular_use_cases = ["semantic_search", "rag", "similarity_matching"]


def get_vectorstore_providers() -> Dict[str, VectorStoreInfo]:
    """Get comprehensive information about all vector store providers.

    Returns:
        Dictionary mapping provider name to VectorStoreInfo

    Examples:
        Get all providers::

            providers = get_vectorstore_providers()
            for name, info in providers.items():
                print(f"{name}: {info.description}")
    """
    return {
        # === LOCAL STORES ===
        "FAISS": VectorStoreInfo(
            name="FAISS",
            description="Facebook AI Similarity Search - High-performance local vector search",
            type=VectorStoreType.LOCAL,
            cost=CostTier.FREE,
            auth_required=False,
            setup_complexity="easy",
            performance_tier="excellent",
            supports_metadata_filtering=False,
            supports_hybrid_search=False,
            supports_multi_tenancy=False,
            max_dimensions=None,  # No limit
            index_types=["flat", "ivf", "hnsw", "pq"],
            python_packages=["faiss-cpu", "faiss-gpu"],
            best_for="High-performance local search, research, prototyping",
            popular_use_cases=["research", "prototyping", "high_throughput_search"],
        ),
        "InMemory": VectorStoreInfo(
            name="InMemory",
            description="Simple in-memory vector storage for testing and development",
            type=VectorStoreType.LOCAL,
            cost=CostTier.FREE,
            auth_required=False,
            setup_complexity="easy",
            performance_tier="basic",
            supports_metadata_filtering=True,
            max_dimensions=10000,  # Practical memory limit
            python_packages=[],  # Built-in
            best_for="Testing, development, small datasets",
            popular_use_cases=["testing", "development", "demos"],
        ),
        "Annoy": VectorStoreInfo(
            name="Annoy",
            description="Approximate Nearest Neighbors library by Spotify",
            type=VectorStoreType.LOCAL,
            cost=CostTier.FREE,
            auth_required=False,
            setup_complexity="easy",
            performance_tier="good",
            supports_metadata_filtering=False,
            supports_real_time_updates=False,  # Read-only after build
            index_types=["angular", "euclidean"],
            python_packages=["annoy"],
            best_for="Read-heavy workloads, music/content recommendation",
            popular_use_cases=["recommendation_systems", "content_similarity"],
        ),
        # === HYBRID STORES ===
        "Chroma": VectorStoreInfo(
            name="Chroma",
            description="Open-source embedding database with local and server modes",
            type=VectorStoreType.HYBRID,
            cost=CostTier.FREE,
            auth_required=False,
            setup_complexity="easy",
            performance_tier="good",
            supports_metadata_filtering=True,
            supports_hybrid_search=False,
            supports_multi_tenancy=True,
            python_packages=["chromadb"],
            external_dependencies=["docker (optional)"],
            best_for="Development to production, easy scaling, rich metadata",
            popular_use_cases=["rag", "semantic_search", "document_analysis"],
        ),
        "Qdrant": VectorStoreInfo(
            name="Qdrant",
            description="Vector database with advanced filtering and geo-search",
            type=VectorStoreType.HYBRID,
            cost=CostTier.FREEMIUM,
            auth_required=False,  # For local deployment
            setup_complexity="medium",
            performance_tier="excellent",
            supports_metadata_filtering=True,
            supports_hybrid_search=True,
            supports_multi_tenancy=True,
            index_types=["hnsw"],
            distance_metrics=["cosine", "euclidean", "dot_product", "manhattan"],
            python_packages=["qdrant-client"],
            external_dependencies=["docker", "qdrant-server"],
            best_for="Advanced filtering, geo-search, high-performance applications",
            popular_use_cases=[
                "location_based_search",
                "advanced_rag",
                "recommendation",
            ],
        ),
        "Weaviate": VectorStoreInfo(
            name="Weaviate",
            description="Open-source vector database with GraphQL API",
            type=VectorStoreType.HYBRID,
            cost=CostTier.FREEMIUM,
            auth_required=False,  # For local
            setup_complexity="medium",
            performance_tier="excellent",
            supports_metadata_filtering=True,
            supports_hybrid_search=True,
            supports_multi_tenancy=True,
            python_packages=["weaviate-client"],
            external_dependencies=["docker", "weaviate-server"],
            best_for="GraphQL integration, hybrid search, complex schemas",
            popular_use_cases=[
                "knowledge_graphs",
                "hybrid_search",
                "enterprise_search",
            ],
        ),
        # === CLOUD STORES ===
        "Pinecone": VectorStoreInfo(
            name="Pinecone",
            description="Fully managed vector database service",
            type=VectorStoreType.CLOUD,
            cost=CostTier.FREEMIUM,
            auth_required=True,
            setup_complexity="easy",
            performance_tier="excellent",
            supports_metadata_filtering=True,
            supports_hybrid_search=False,
            supports_multi_tenancy=True,
            supports_real_time_updates=True,
            python_packages=["pinecone-client"],
            best_for="Production apps, zero-ops, auto-scaling",
            popular_use_cases=[
                "production_rag",
                "recommendation_systems",
                "similarity_search",
            ],
        ),
        # === DATABASE EXTENSIONS ===
        "PGVector": VectorStoreInfo(
            name="PGVector",
            description="PostgreSQL extension for vector similarity search",
            type=VectorStoreType.DATABASE,
            cost=CostTier.FREE,
            auth_required=True,  # Database credentials
            setup_complexity="medium",
            performance_tier="good",
            supports_metadata_filtering=True,
            supports_hybrid_search=False,
            supports_multi_tenancy=True,
            index_types=["ivfflat", "hnsw"],
            python_packages=["psycopg2-binary", "pgvector"],
            external_dependencies=["postgresql", "pgvector-extension"],
            best_for="Existing PostgreSQL users, ACID compliance, complex queries",
            popular_use_cases=["enterprise_rag", "data_warehousing", "analytics"],
        ),
        "ClickHouse": VectorStoreInfo(
            name="ClickHouse",
            description="Column-oriented database with vector search capabilities",
            type=VectorStoreType.DATABASE,
            cost=CostTier.FREE,
            auth_required=True,
            setup_complexity="complex",
            performance_tier="excellent",
            supports_metadata_filtering=True,
            supports_hybrid_search=False,
            max_dimensions=65536,
            python_packages=["clickhouse-connect"],
            external_dependencies=["clickhouse-server"],
            best_for="Analytics workloads, large-scale data, time-series",
            popular_use_cases=["analytics", "time_series", "large_scale_search"],
        ),
        # === SEARCH ENGINES ===
        "Elasticsearch": VectorStoreInfo(
            name="Elasticsearch",
            description="Search engine with vector search capabilities",
            type=VectorStoreType.SEARCH_ENGINE,
            cost=CostTier.FREEMIUM,
            auth_required=False,  # For local
            setup_complexity="medium",
            performance_tier="excellent",
            supports_metadata_filtering=True,
            supports_hybrid_search=True,
            supports_multi_tenancy=True,
            python_packages=["elasticsearch"],
            external_dependencies=["elasticsearch-server"],
            best_for="Full-text + vector search, existing Elasticsearch users",
            popular_use_cases=["hybrid_search", "enterprise_search", "log_analysis"],
        ),
    }


def filter_vectorstores(
    type_filter: Optional[VectorStoreType] = None,
    cost_filter: Optional[CostTier] = None,
    auth_required: Optional[bool] = None,
    supports_metadata: Optional[bool] = None,
    supports_hybrid: Optional[bool] = None,
    setup_complexity: Optional[str] = None,
) -> Dict[str, VectorStoreInfo]:
    """Filter vector stores by criteria.

    Args:
        type_filter: Filter by store type
        cost_filter: Filter by cost tier
        auth_required: Filter by auth requirement
        supports_metadata: Filter by metadata support
        supports_hybrid: Filter by hybrid search support
        setup_complexity: Filter by setup complexity

    Returns:
        Filtered dictionary of vector stores

    Examples:
        Get free, easy-setup stores::

            stores = filter_vectorstores(
                cost_filter=CostTier.FREE,
                setup_complexity="easy"
            )

        Get hybrid search capable stores::

            stores = filter_vectorstores(supports_hybrid=True)
    """
    all_stores = get_vectorstore_providers()
    filtered = {}

    for name, info in all_stores.items():
        # Apply filters
        if type_filter and info.type != type_filter:
            continue
        if cost_filter and info.cost != cost_filter:
            continue
        if auth_required is not None and info.auth_required != auth_required:
            continue
        if (
            supports_metadata is not None
            and info.supports_metadata_filtering != supports_metadata
        ):
            continue
        if (
            supports_hybrid is not None
            and info.supports_hybrid_search != supports_hybrid
        ):
            continue
        if setup_complexity and info.setup_complexity != setup_complexity:
            continue

        filtered[name] = info

    return filtered


def recommend_vectorstore(use_case: str) -> List[str]:
    """Get vector store recommendations for specific use cases.

    Args:
        use_case: Use case ("development", "production", "research",
                          "enterprise", "free_only", "local_only")

    Returns:
        List of recommended vector store names

    Examples:
        Get development recommendations::

            dev_stores = recommend_vectorstore("development")
            # Returns: ["Chroma", "InMemory", "FAISS"]

        Get production recommendations::

            prod_stores = recommend_vectorstore("production")
            # Returns: ["Pinecone", "Qdrant", "Weaviate"]
    """
    if use_case == "development":
        return ["Chroma", "InMemory", "FAISS"]

    elif use_case == "production":
        return ["Pinecone", "Qdrant", "Weaviate", "Chroma"]

    elif use_case == "research":
        return ["FAISS", "Annoy", "Chroma", "Qdrant"]

    elif use_case == "enterprise":
        return ["PGVector", "Elasticsearch", "Weaviate", "Qdrant"]

    elif use_case == "free_only":
        free_stores = filter_vectorstores(cost_filter=CostTier.FREE)
        return list(free_stores.keys())

    elif use_case == "local_only":
        local_stores = filter_vectorstores(type_filter=VectorStoreType.LOCAL)
        return list(local_stores.keys())

    elif use_case == "no_auth":
        no_auth_stores = filter_vectorstores(auth_required=False)
        return list(no_auth_stores.keys())

    else:
        # Default general recommendations
        return ["Chroma", "Pinecone", "FAISS", "Qdrant"]


def get_setup_instructions(provider_name: str) -> str:
    """Get setup instructions for a vector store provider.

    Args:
        provider_name: Name of the vector store provider

    Returns:
        Setup instructions as formatted string

    Examples:
        Get Chroma setup::

            instructions = get_setup_instructions("Chroma")
            print(instructions)
    """
    providers = get_vectorstore_providers()
    info = providers.get(provider_name)

    if not info:
        return f"Provider '{provider_name}' not found."

    instructions = f"# {info.name} Setup Instructions\n\n"
    instructions += f"Description: {info.description}\n"
    instructions += f"Type: {info.type.value}\n"
    instructions += f"Cost: {info.cost.value}\n"
    instructions += f"Setup Complexity: {info.setup_complexity}\n\n"

    if info.python_packages:
        instructions += "## Python Installation\n"
        instructions += f"pip install {' '.join(info.python_packages)}\n\n"

    if info.external_dependencies:
        instructions += "## External Dependencies\n"
        for dep in info.external_dependencies:
            instructions += f"- {dep}\n"
        instructions += "\n"

    instructions += f"## Best For\n{info.best_for}\n\n"

    if info.popular_use_cases:
        instructions += "## Popular Use Cases\n"
        for use_case in info.popular_use_cases:
            instructions += f"- {use_case}\n"

    return instructions


def compare_vectorstores(provider_names: List[str]) -> str:
    """Compare multiple vector store providers.

    Args:
        provider_names: List of provider names to compare

    Returns:
        Comparison table as formatted string

    Examples:
        Compare popular stores::

            comparison = compare_vectorstores(["Chroma", "Pinecone", "FAISS"])
            print(comparison)
    """
    providers = get_vectorstore_providers()

    # Filter to requested providers
    selected = {
        name: info for name, info in providers.items() if name in provider_names
    }

    if not selected:
        return "No valid providers found for comparison."

    # Build comparison table
    comparison = "Vector Store Comparison\n"
    comparison += "=" * 50 + "\n\n"

    # Header
    comparison += f"{'Provider':<15} {'Type':<10} {'Cost':<10} {'Auth':<6} {'Setup':<8} {'Perf':<8}\n"
    comparison += "-" * 65 + "\n"

    # Rows
    for name, info in selected.items():
        auth_str = "Yes" if info.auth_required else "No"
        comparison += (
            f"{name:<15} {info.type.value:<10} {info.cost.value:<10} {auth_str:<6} "
        )
        comparison += f"{info.setup_complexity:<8} {info.performance_tier:<8}\n"

    comparison += "\n"

    # Detailed capabilities
    comparison += "Capabilities:\n"
    comparison += "-" * 20 + "\n"

    for name, info in selected.items():
        comparison += f"\n{name}:\n"
        comparison += f"  Metadata Filtering: {'✓' if info.supports_metadata_filtering else '✗'}\n"
        comparison += (
            f"  Hybrid Search: {'✓' if info.supports_hybrid_search else '✗'}\n"
        )
        comparison += (
            f"  Multi-tenancy: {'✓' if info.supports_multi_tenancy else '✗'}\n"
        )
        comparison += (
            f"  Real-time Updates: {'✓' if info.supports_real_time_updates else '✗'}\n"
        )
        comparison += f"  Best For: {info.best_for}\n"

    return comparison
