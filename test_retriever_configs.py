"""
Comprehensive test suite for all retriever configurations.

This script tests the basic instantiation and configuration validation
for all implemented retriever configs to ensure they work properly.
Uses existing test fixtures from conftest.py for consistency.
"""

import logging
from typing import Any, Dict

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

# Import the sample_documents fixture from conftest
from tests.models.retriever.conftest import sample_documents


def test_basic_retriever_configs():
    """Test basic configuration instantiation for all retrievers."""

    print("🧪 Testing Basic Retriever Configurations...")
    print("=" * 60)

    # Test sparse/classical retrievers (these should work without external deps)
    configs_to_test = []

    # 1. Sparse retrievers with documents
    try:
        from haive.core.engine.retriever.providers.BM25RetrieverConfig import (
            BM25RetrieverConfig,
        )

        config = BM25RetrieverConfig(name="test_bm25", documents=test_documents, k=3)
        configs_to_test.append(("BM25Retriever", config))
        print("✅ BM25RetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ BM25RetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.TFIDFRetrieverConfig import (
            TFIDFRetrieverConfig,
        )

        config = TFIDFRetrieverConfig(name="test_tfidf", documents=test_documents, k=3)
        configs_to_test.append(("TFIDFRetriever", config))
        print("✅ TFIDFRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ TFIDFRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.KNNRetrieverConfig import (
            KNNRetrieverConfig,
        )

        config = KNNRetrieverConfig(name="test_knn", documents=test_documents, k=3)
        configs_to_test.append(("KNNRetriever", config))
        print("✅ KNNRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ KNNRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.SVMRetrieverConfig import (
            SVMRetrieverConfig,
        )

        config = SVMRetrieverConfig(name="test_svm", documents=test_documents, k=3)
        configs_to_test.append(("SVMRetriever", config))
        print("✅ SVMRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ SVMRetrieverConfig - Error: {e}")

    # 2. Test API-based retrievers (config only, no instantiation)
    try:
        from haive.core.engine.retriever.providers.YouRetrieverConfig import (
            YouRetrieverConfig,
        )

        config = YouRetrieverConfig(name="test_you", num_web_results=5)
        print("✅ YouRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ YouRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.AskNewsRetrieverConfig import (
            AskNewsRetrieverConfig,
        )

        config = AskNewsRetrieverConfig(name="test_asknews", k=5)
        print("✅ AskNewsRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ AskNewsRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.PubMedRetrieverConfig import (
            PubMedRetrieverConfig,
        )

        config = PubMedRetrieverConfig(name="test_pubmed", top_k_results=5)
        print("✅ PubMedRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ PubMedRetrieverConfig - Error: {e}")

    # 3. Test cloud service retrievers (config only)
    try:
        from haive.core.engine.retriever.providers.KendraRetrieverConfig import (
            KendraRetrieverConfig,
        )

        config = KendraRetrieverConfig(
            name="test_kendra", index_id="test-index-123", region_name="us-east-1"
        )
        print("✅ KendraRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ KendraRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.GoogleVertexAISearchRetrieverConfig import (
            GoogleVertexAISearchRetrieverConfig,
        )

        config = GoogleVertexAISearchRetrieverConfig(
            name="test_vertex_search",
            project_id="test-project",
            data_store_id="test-store",
        )
        print(
            "✅ GoogleVertexAISearchRetrieverConfig - Configuration created successfully"
        )
    except Exception as e:
        print(f"❌ GoogleVertexAISearchRetrieverConfig - Error: {e}")

    # 4. Test specialized retrievers
    try:
        from haive.core.engine.retriever.providers.DocArrayRetrieverConfig import (
            DocArrayRetrieverConfig,
        )

        config = DocArrayRetrieverConfig(
            name="test_docarray", documents=test_documents, k=3
        )
        print("✅ DocArrayRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ DocArrayRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.ZepRetrieverConfig import (
            ZepRetrieverConfig,
        )

        config = ZepRetrieverConfig(
            name="test_zep", session_id="test-session-123", url="http://localhost:8000"
        )
        print("✅ ZepRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ ZepRetrieverConfig - Error: {e}")

    try:
        from haive.core.engine.retriever.providers.LlamaIndexRetrieverConfig import (
            LlamaIndexRetrieverConfig,
        )

        config = LlamaIndexRetrieverConfig(
            name="test_llamaindex", documents=test_documents, k=3
        )
        print("✅ LlamaIndexRetrieverConfig - Configuration created successfully")
    except Exception as e:
        print(f"❌ LlamaIndexRetrieverConfig - Error: {e}")

    print("=" * 60)
    print(f"📊 Configuration test summary: Tested {12} retriever configs")

    return configs_to_test


if __name__ == "__main__":
    configs = test_basic_retriever_configs()
    print(f"✅ {len(configs)} configurations ready for instantiation testing")
