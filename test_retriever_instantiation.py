"""
Test actual retriever instantiation for retrievers that don't require external services.

This script tests the actual instantiation of retrievers that can work
without external dependencies or API keys.
"""

import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

logger = logging.getLogger(__name__)

# Sample documents for testing
sample_documents = [
    Document(
        page_content="Python is a versatile programming language used for web development, data science, and automation.",
        metadata={"source": "doc1", "topic": "programming", "length": "long"},
    ),
    Document(
        page_content="Machine learning algorithms learn patterns from data to make predictions.",
        metadata={"source": "doc2", "topic": "AI", "length": "medium"},
    ),
    Document(
        page_content="Vector databases store high-dimensional embeddings for similarity search.",
        metadata={"source": "doc3", "topic": "databases", "length": "medium"},
    ),
    Document(
        page_content="RAG combines retrieval and generation for better AI responses.",
        metadata={"source": "doc4", "topic": "AI", "length": "short"},
    ),
    Document(
        page_content="Natural language processing enables computers to understand human language.",
        metadata={"source": "doc5", "topic": "AI", "length": "long"},
    ),
]


def test_bm25_retriever_instantiation():
    """Test BM25 retriever instantiation and basic functionality."""
    print("\n🧪 Testing BM25 Retriever Instantiation...")

    try:
        from haive.core.engine.retriever.providers.BM25RetrieverConfig import (
            BM25RetrieverConfig,
        )

        # Create configuration
        config = BM25RetrieverConfig(
            name="test_bm25_instantiation",
            documents=sample_documents,
            k=3,
            k1=1.2,
            b=0.75,
        )

        # Try to instantiate
        retriever = config.instantiate()

        # Verify it's a retriever
        assert isinstance(
            retriever, BaseRetriever
        ), f"Expected BaseRetriever, got {type(retriever)}"

        # Test basic retrieval
        results = retriever.get_relevant_documents("machine learning patterns")
        assert isinstance(results, list), "Results should be a list"
        assert len(results) <= 3, f"Should return at most 3 results, got {len(results)}"

        for doc in results:
            assert isinstance(
                doc, Document
            ), f"Each result should be a Document, got {type(doc)}"

        print(f"✅ BM25 Retriever instantiated successfully!")
        print(
            f"   Retrieved {len(results)} documents for query 'machine learning patterns'"
        )
        for i, doc in enumerate(results):
            print(f"   {i+1}. {doc.page_content[:50]}...")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ BM25 Retriever instantiation failed: {e}")
        return False


def test_tfidf_retriever_instantiation():
    """Test TF-IDF retriever instantiation and basic functionality."""
    print("\n🧪 Testing TF-IDF Retriever Instantiation...")

    try:
        from haive.core.engine.retriever.providers.TFIDFRetrieverConfig import (
            TFIDFRetrieverConfig,
        )

        # Create configuration
        config = TFIDFRetrieverConfig(
            name="test_tfidf_instantiation",
            documents=sample_documents,
            k=2,
            tfidf_params={"max_features": 1000, "stop_words": "english"},
        )

        # Try to instantiate
        retriever = config.instantiate()

        # Verify it's a retriever
        assert isinstance(
            retriever, BaseRetriever
        ), f"Expected BaseRetriever, got {type(retriever)}"

        # Test basic retrieval
        results = retriever.get_relevant_documents("vector databases")
        assert isinstance(results, list), "Results should be a list"
        assert len(results) <= 2, f"Should return at most 2 results, got {len(results)}"

        print(f"✅ TF-IDF Retriever instantiated successfully!")
        print(f"   Retrieved {len(results)} documents for query 'vector databases'")
        for i, doc in enumerate(results):
            print(f"   {i+1}. {doc.page_content[:50]}...")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ TF-IDF Retriever instantiation failed: {e}")
        return False


def test_pubmed_retriever_instantiation():
    """Test PubMed retriever instantiation (without API call)."""
    print("\n🧪 Testing PubMed Retriever Instantiation...")

    try:
        from haive.core.engine.retriever.providers.PubMedRetrieverConfig import (
            PubMedRetrieverConfig,
        )

        # Create configuration
        config = PubMedRetrieverConfig(
            name="test_pubmed_instantiation",
            top_k_results=3,
            load_max_docs=10,
            email="test@example.com",
        )

        # Try to instantiate (this should work without API key for configuration)
        retriever = config.instantiate()

        # Verify it's a retriever
        assert isinstance(
            retriever, BaseRetriever
        ), f"Expected BaseRetriever, got {type(retriever)}"

        print(f"✅ PubMed Retriever instantiated successfully!")
        print(f"   Configuration: top_k={config.top_k_results}, email={config.email}")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ PubMed Retriever instantiation failed: {e}")
        return False


def test_docarray_retriever_instantiation():
    """Test DocArray retriever instantiation and basic functionality."""
    print("\n🧪 Testing DocArray Retriever Instantiation...")

    try:
        from haive.core.engine.retriever.providers.DocArrayRetrieverConfig import (
            DocArrayRetrieverConfig,
        )

        # Create configuration
        config = DocArrayRetrieverConfig(
            name="test_docarray_instantiation",
            documents=sample_documents,
            k=3,
            similarity_metric="cosine",
            normalize_embeddings=True,
        )

        # Try to instantiate
        retriever = config.instantiate()

        # Verify it's a retriever
        assert isinstance(
            retriever, BaseRetriever
        ), f"Expected BaseRetriever, got {type(retriever)}"

        print(f"✅ DocArray Retriever instantiated successfully!")
        print(f"   Configuration: k={config.k}, metric={config.similarity_metric}")

        return True

    except ImportError as e:
        print(f"❌ Import error (likely missing docarray): {e}")
        return False
    except Exception as e:
        print(f"❌ DocArray Retriever instantiation failed: {e}")
        return False


def test_error_handling():
    """Test error handling for invalid configurations."""
    print("\n⚠️ Testing Error Handling...")

    try:
        from haive.core.engine.retriever.providers.BM25RetrieverConfig import (
            BM25RetrieverConfig,
        )

        # Test with empty documents
        config = BM25RetrieverConfig(
            name="test_empty_docs",
            documents=[],  # Empty documents should cause error
            k=3,
        )

        try:
            retriever = config.instantiate()
            print("❌ Expected error for empty documents, but instantiation succeeded")
            return False
        except ValueError as e:
            print(f"✅ Correctly caught error for empty documents: {e}")
            return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_secure_config_mixin():
    """Test SecureConfigMixin functionality."""
    print("\n🔐 Testing SecureConfigMixin...")

    try:
        from haive.core.engine.retriever.providers.YouRetrieverConfig import (
            YouRetrieverConfig,
        )

        # Create config without API key
        config = YouRetrieverConfig(name="test_secure_config", num_web_results=5)

        # Test that get_api_key method exists and returns None when no key is set
        api_key = config.get_api_key()
        assert api_key is None, f"Expected None for missing API key, got {api_key}"

        print("✅ SecureConfigMixin working correctly")
        print("   - get_api_key() returns None when no key is set")

        return True

    except Exception as e:
        print(f"❌ SecureConfigMixin test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Testing Retriever Instantiation...")
    print("=" * 70)

    test_results = []

    # Test retrievers that should work without external dependencies
    test_results.append(("BM25", test_bm25_retriever_instantiation()))
    test_results.append(("TF-IDF", test_tfidf_retriever_instantiation()))
    test_results.append(("PubMed", test_pubmed_retriever_instantiation()))
    test_results.append(("DocArray", test_docarray_retriever_instantiation()))
    test_results.append(("Error Handling", test_error_handling()))
    test_results.append(("SecureConfigMixin", test_secure_config_mixin()))

    print("\n" + "=" * 70)
    print("📊 Test Results Summary:")

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:<20}: {status}")
        if result:
            passed += 1

    print(
        f"\n🎯 Overall Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)"
    )

    if passed == total:
        print("🎉 All instantiation tests passed!")
    else:
        print("⚠️ Some tests failed - check the output above for details")
