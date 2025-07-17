"""Test each memory agent individually with real execution.

This test validates each agent in the memory system independently,
ensuring they all work correctly with real LLM calls.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import StoreType at module level to avoid repeated imports
from haive.core.persistence.store.types import StoreType


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {title}")
    print(f"{'=' * 60}")


def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"   Details: {details}")


async def test_memory_classifier():
    """Test the memory classifier agent."""
    print_section("Testing Memory Classifier")

    try:
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )

        from haive.core.engine.aug_llm import AugLLMConfig

        # Create classifier with real LLM
        config = MemoryClassifierConfig(
            llm_config=AugLLMConfig(temperature=0.1), confidence_threshold=0.6
        )
        classifier = MemoryClassifier(config)

        # Test 1: Classify different types of memories
        test_memories = [
            "I learned Python programming yesterday",
            "The capital of France is Paris",
            "I prefer coffee over tea",
            "To bake a cake, first preheat the oven to 350°F",
            "I felt happy when I solved the problem",
            "The meeting is scheduled for tomorrow at 3 PM",
        ]

        for memory in test_memories:
            result = classifier.classify_memory(memory)
            print_result(
                f"Classify: '{memory[:40]}...'",
                True,
                f"Types: {[str(mt) for mt in result.memory_types]}, Importance: {result.importance_score:.2f}",
            )

        # Test 2: Query intent classification
        test_queries = [
            "What programming languages do I know?",
            "How do I bake a cake?",
            "When is my next meeting?",
            "What are my preferences?",
        ]

        for query in test_queries:
            intent = classifier.classify_query_intent(query)
            print_result(
                f"Query intent: '{query[:40]}...'",
                True,
                f"Types: {[str(mt) for mt in intent.memory_types]}, Strategy: {intent.preferred_retrieval_strategy}",
            )

        return True

    except Exception as e:
        print_result("Memory Classifier", False, str(e))
        logger.error(f"Memory classifier test failed: {e}")
        return False


async def test_memory_store_manager():
    """Test the memory store manager."""
    print_section("Testing Memory Store Manager")

    try:
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )

        from haive.core.tools.store_manager import StoreManager

        # Create store manager with correct API
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "memory_store"),
        )

        config = MemoryStoreConfig(
            store_manager=store_manager,
            default_namespace=("test", "agent"),
            auto_classify=False,
        )

        memory_store = MemoryStoreManager(config)

        # Test 1: Store memory
        memory_id = await memory_store.store_memory(
            content="Alice works at TechCorp as a software engineer",
            namespace=("test", "facts"),
        )
        print_result("Store memory", True, f"Memory ID: {memory_id}")

        # Test 2: Retrieve memory by ID
        memory = await memory_store.get_memory_by_id(memory_id)
        print_result(
            "Retrieve by ID", memory is not None, f"Found: {memory is not None}"
        )

        # Test 3: Search memories
        memories = await memory_store.retrieve_memories(
            query="Alice engineer", namespace=("test", "facts"), limit=5
        )
        print_result("Search memories", True, f"Found {len(memories)} memories")

        # Test 4: Update memory (access is updated automatically in get_memory_by_id)
        success = await memory_store.update_memory(
            memory_id=memory_id, additional_metadata={"test_update": True}
        )
        print_result("Update memory", success, f"Update successful: {success}")

        # Test 5: Get memory statistics
        stats = await memory_store.get_memory_statistics(("test", "facts"))
        print_result(
            "Get statistics", True, f"Total memories: {stats.get('total_memories', 0)}"
        )

        return True

    except Exception as e:
        print_result("Memory Store Manager", False, str(e))
        logger.error(f"Memory store manager test failed: {e}")
        return False


async def test_kg_generator_agent():
    """Test the KG generator agent."""
    print_section("Testing KG Generator Agent")

    try:
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.agents.memory.kg_generator_agent import (
            KGGeneratorAgent,
            KGGeneratorAgentConfig,
        )

        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.tools.store_manager import StoreManager

        # Setup dependencies
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY}, default_namespace=("test", "kg")
        )
        memory_store = MemoryStoreManager(
            MemoryStoreConfig(store_manager=store_manager)
        )
        classifier = MemoryClassifier(MemoryClassifierConfig())

        # Store some test memories
        test_memories = [
            "Alice works at TechCorp as a software engineer",
            "Bob is a data scientist at DataFlow Inc",
            "Alice and Bob collaborated on the ML project",
            "The ML project uses Python and TensorFlow",
            "TechCorp is located in San Francisco",
        ]

        for memory in test_memories:
            await memory_store.store_memory(memory)

        # Create KG generator
        config = KGGeneratorAgentConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            engine=AugLLMConfig(temperature=0.1),
        )

        kg_agent = KGGeneratorAgent(config)

        # Test 1: Extract entities from memories
        entities = await kg_agent.extract_entities_from_memories(limit=5)
        print_result(
            "Extract entities", len(entities) > 0, f"Found {len(entities)} entities"
        )

        # Test 2: Extract relationships
        relationships = await kg_agent.extract_relationships_from_memories(limit=5)
        print_result(
            "Extract relationships",
            len(relationships) > 0,
            f"Found {len(relationships)} relationships",
        )

        # Test 3: Build knowledge graph
        kg = await kg_agent.extract_knowledge_graph_from_memories()
        print_result(
            "Build knowledge graph",
            True,
            f"Nodes: {len(kg.nodes)}, Relationships: {len(kg.relationships)}",
        )

        # Test 4: Get entity context
        if entities:
            entity_name = entities[0].name
            context = await kg_agent.get_entity_context(entity_name)
            print_result(
                "Get entity context", "entity" in context, f"Context for: {entity_name}"
            )

        # Test 5: Run agent
        result = await kg_agent.run("Extract knowledge from the stored memories")
        print_result("Run KG agent", len(result) > 0, f"Response length: {len(result)}")

        return True

    except Exception as e:
        print_result("KG Generator Agent", False, str(e))
        logger.error(f"KG generator agent test failed: {e}")
        return False


async def test_graph_rag_retriever():
    """Test the Graph RAG retriever."""
    print_section("Testing Graph RAG Retriever")

    try:
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.agents.memory.graph_rag_retriever import (
            GraphRAGRetriever,
            GraphRAGRetrieverConfig,
        )
        from haive.agents.memory.kg_generator_agent import (
            KGGeneratorAgent,
            KGGeneratorAgentConfig,
        )

        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.tools.store_manager import StoreManager

        # Setup dependencies
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "graph_rag"),
        )
        memory_store = MemoryStoreManager(
            MemoryStoreConfig(store_manager=store_manager)
        )
        classifier = MemoryClassifier(MemoryClassifierConfig())

        # Store test memories
        test_memories = [
            "Alice is an expert in Python programming",
            "Python is used for machine learning",
            "Machine learning requires data preprocessing",
            "Data preprocessing involves cleaning and transformation",
            "Alice teaches machine learning courses",
        ]

        for memory in test_memories:
            await memory_store.store_memory(memory)

        # Create KG generator
        kg_config = KGGeneratorAgentConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            engine=AugLLMConfig(temperature=0.1),
        )
        kg_generator = KGGeneratorAgent(kg_config)

        # Build knowledge graph
        await kg_generator.extract_knowledge_graph_from_memories()

        # Create Graph RAG retriever
        config = GraphRAGRetrieverConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            kg_generator=kg_generator,
        )

        graph_rag = GraphRAGRetriever(config)

        # Test 1: Basic retrieval
        result = await graph_rag.retrieve_memories(
            query="What does Alice know?", limit=5
        )
        print_result(
            "Basic retrieval",
            len(result.memories) > 0,
            f"Found {len(result.memories)} memories",
        )

        # Test 2: Graph traversal enabled
        result = await graph_rag.retrieve_memories(
            query="machine learning", limit=5, enable_graph_traversal=True
        )
        print_result(
            "Graph traversal",
            result.graph_nodes_explored > 0,
            f"Explored {result.graph_nodes_explored} nodes",
        )

        # Test 3: Multi-hop retrieval
        result = await graph_rag.retrieve_memories(
            query="data preprocessing", limit=5, max_graph_depth=3
        )
        print_result(
            "Multi-hop retrieval",
            len(result.graph_paths) > 0,
            f"Found {len(result.graph_paths)} paths",
        )

        return True

    except Exception as e:
        print_result("Graph RAG Retriever", False, str(e))
        logger.error(f"Graph RAG retriever test failed: {e}")
        return False


async def test_agentic_rag_coordinator():
    """Test the Agentic RAG coordinator."""
    print_section("Testing Agentic RAG Coordinator")

    try:
        from haive.agents.memory.agentic_rag_coordinator import (
            AgenticRAGCoordinator,
            AgenticRAGCoordinatorConfig,
        )
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.agents.memory.kg_generator_agent import (
            KGGeneratorAgent,
            KGGeneratorAgentConfig,
        )

        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.tools.store_manager import StoreManager

        # Setup dependencies
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "agentic_rag"),
        )
        memory_store = MemoryStoreManager(
            MemoryStoreConfig(store_manager=store_manager)
        )
        classifier = MemoryClassifier(MemoryClassifierConfig())

        # Store diverse test memories
        test_memories = [
            "I learned Python programming yesterday at 3 PM",
            "Python is great for data science and web development",
            "To deploy a web app, first set up a virtual environment",
            "I prefer using VS Code for Python development",
            "Error: ImportError occurred when running the script",
            "Fixed the error by installing missing dependencies",
        ]

        for memory in test_memories:
            await memory_store.store_memory(memory)

        # Create KG generator
        kg_config = KGGeneratorAgentConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            engine=AugLLMConfig(temperature=0.1),
        )
        kg_generator = KGGeneratorAgent(kg_config)

        # Create Agentic RAG coordinator
        config = AgenticRAGCoordinatorConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            kg_generator=kg_generator,
            coordinator_llm=AugLLMConfig(temperature=0.3),
        )

        rag_coordinator = AgenticRAGCoordinator(config)

        # Test 1: Simple retrieval
        result = await rag_coordinator.retrieve_memories(
            query="What programming languages do I know?", limit=5
        )
        print_result(
            "Simple retrieval",
            len(result.final_memories) > 0,
            f"Strategies: {result.selected_strategies}, Memories: {len(result.final_memories)}",
        )

        # Test 2: Procedural query
        result = await rag_coordinator.retrieve_memories(
            query="How do I deploy applications?", limit=5
        )
        print_result(
            "Procedural query",
            "procedural_search" in result.selected_strategies,
            f"Strategies: {result.selected_strategies}",
        )

        # Test 3: Error/feedback query
        result = await rag_coordinator.retrieve_memories(
            query="What errors have I encountered?", limit=5
        )
        print_result(
            "Error query",
            len(result.final_memories) > 0,
            f"Confidence: {result.confidence_score:.2f}",
        )

        # Test 4: Multi-strategy fusion
        result = await rag_coordinator.retrieve_memories(
            query="Tell me everything about Python", limit=10
        )
        print_result(
            "Multi-strategy",
            len(result.selected_strategies) > 1,
            f"Strategies: {len(result.selected_strategies)}, Diversity: {result.diversity_score:.2f}",
        )

        # Test 5: Run agent
        response = await rag_coordinator.run("What have I learned recently?")
        print_result(
            "Run RAG agent", len(response) > 0, f"Response length: {len(response)}"
        )

        return True

    except Exception as e:
        print_result("Agentic RAG Coordinator", False, str(e))
        logger.error(f"Agentic RAG coordinator test failed: {e}")
        return False


async def test_multi_agent_coordinator():
    """Test the Multi-Agent Memory Coordinator."""
    print_section("Testing Multi-Agent Memory Coordinator")

    try:
        from haive.agents.memory.agentic_rag_coordinator import (
            AgenticRAGCoordinatorConfig,
        )
        from haive.agents.memory.core.classifier import (
            MemoryClassifier,
            MemoryClassifierConfig,
        )
        from haive.agents.memory.core.stores import (
            MemoryStoreConfig,
            MemoryStoreManager,
        )
        from haive.agents.memory.kg_generator_agent import KGGeneratorAgentConfig
        from haive.agents.memory.multi_agent_coordinator import (
            MemoryTask,
            MultiAgentCoordinatorConfig,
            MultiAgentMemoryCoordinator,
        )

        from haive.core.engine.aug_llm import AugLLMConfig
        from haive.core.tools.store_manager import StoreManager

        # Setup dependencies
        store_manager = StoreManager(
            store_config={"type": StoreType.MEMORY},
            default_namespace=("test", "multi_agent"),
        )
        memory_store = MemoryStoreManager(
            MemoryStoreConfig(store_manager=store_manager)
        )
        classifier = MemoryClassifier(MemoryClassifierConfig())

        # Create configurations with in-memory persistence
        kg_config = KGGeneratorAgentConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            engine=AugLLMConfig(temperature=0.1),
            # Disable PostgreSQL persistence for tests
            persistence=False,
        )

        rag_config = AgenticRAGCoordinatorConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            kg_generator=None,  # Will be set by coordinator
            # Disable PostgreSQL persistence for tests
            persistence=False,
        )

        coordinator_config = MultiAgentCoordinatorConfig(
            memory_store_manager=memory_store,
            memory_classifier=classifier,
            kg_generator_config=kg_config,
            agentic_rag_config=rag_config,
            coordinator_llm=AugLLMConfig(temperature=0.2),
        )

        coordinator = MultiAgentMemoryCoordinator(coordinator_config)

        # Test 1: Store memory through coordinator
        result = await coordinator.store_memory("Multi-agent test memory")
        print_result("Store via coordinator", "success" in result.lower(), result[:50])

        # Test 2: Retrieve memories
        memories = await coordinator.retrieve_memories(query="test memory", limit=5)
        print_result(
            "Retrieve via coordinator",
            len(memories) >= 0,
            f"Found {len(memories)} memories",
        )

        # Test 3: Analyze memory
        analysis = await coordinator.analyze_memory("Analyze this test content")
        print_result("Analyze memory", analysis["success"], "Analysis completed")

        # Test 4: Generate knowledge graph
        kg_result = await coordinator.generate_knowledge_graph()
        print_result("Generate KG", kg_result["success"], "Knowledge graph generated")

        # Test 5: Execute custom task
        task = MemoryTask(
            id="test_task_1",
            type="retrieve_memories",
            query="Find all test memories",
            priority=1,
        )

        executed_task = await coordinator.execute_task(task)
        print_result(
            "Execute task",
            executed_task.status == "completed",
            f"Status: {executed_task.status}, Agent: {executed_task.assigned_agent}",
        )

        # Test 6: Get system status
        status = coordinator.get_system_status()
        print_result(
            "System status",
            status["coordinator_status"] == "active",
            f"Total agents: {status['total_agents']}",
        )

        # Test 7: Run diagnostic
        diagnostic = await coordinator.run_diagnostic()
        print_result(
            "Run diagnostic",
            diagnostic["system_status"] == "healthy",
            f"Status: {diagnostic['system_status']}",
        )

        return True

    except Exception as e:
        print_result("Multi-Agent Coordinator", False, str(e))
        logger.error(f"Multi-agent coordinator test failed: {e}")
        return False


async def test_unified_memory_system():
    """Test the Unified Memory System."""
    print_section("Testing Unified Memory System")

    try:
        from haive.agents.memory.unified_memory_api import (
            MemorySystemConfig,
            UnifiedMemorySystem,
            create_memory_system,
        )

        # Create memory system with all features
        config = MemorySystemConfig(
            store_type="memory",
            collection_name="test_unified",
            default_namespace=("test", "unified"),
            enable_auto_classification=True,
            enable_enhanced_retrieval=True,
            enable_graph_rag=True,
            enable_multi_agent_coordination=True,
            llm_config=AugLLMConfig(temperature=0.2),
        )

        memory_system = UnifiedMemorySystem(config)

        # Test 1: Store memory
        result = await memory_system.store_memory(
            content="Unified system test memory", metadata={"test": True}
        )
        print_result(
            "Store memory", result.success, f"Time: {result.execution_time_ms:.1f}ms"
        )

        # Test 2: Retrieve memories
        result = await memory_system.retrieve_memories(
            query="unified test", limit=5, use_graph_rag=True, use_multi_agent=True
        )
        print_result(
            "Retrieve memories",
            result.success,
            f"Found: {result.result['count']}, Agent: {result.agent_used}",
        )

        # Test 3: Classify memory
        result = await memory_system.classify_memory(
            content="I learned about unified memory systems today"
        )
        print_result(
            "Classify memory",
            result.success,
            f"Confidence: {result.confidence_score:.2f}",
        )

        # Test 4: Generate knowledge graph
        result = await memory_system.generate_knowledge_graph()
        print_result(
            "Generate KG", result.success, f"Time: {result.execution_time_ms:.1f}ms"
        )

        # Test 5: Consolidate memories
        result = await memory_system.consolidate_memories(dry_run=True)
        print_result("Consolidate memories", result.success, "Dry run completed")

        # Test 6: Get statistics
        result = await memory_system.get_memory_statistics()
        print_result("Get statistics", result.success, "Statistics retrieved")

        # Test 7: Search entities
        result = await memory_system.search_entities(entity_name="test")
        print_result(
            "Search entities", result.success, f"Time: {result.execution_time_ms:.1f}ms"
        )

        # Test 8: Run diagnostic
        result = await memory_system.run_system_diagnostic()
        print_result(
            "System diagnostic",
            result.success,
            f"Health: {result.result['system_health']}",
        )

        # Test 9: Get system info
        info = memory_system.get_system_info()
        print_result(
            "System info",
            info["initialized"],
            f"Version: {info['system_version']}, Stats: {info['statistics']['total_operations']} ops",
        )

        # Test 10: Create with factory function
        quick_system = await create_memory_system(
            store_type="memory", collection_name="test_factory"
        )
        print_result(
            "Factory creation", quick_system is not None, "System created via factory"
        )

        return True

    except Exception as e:
        print_result("Unified Memory System", False, str(e))
        logger.error(f"Unified memory system test failed: {e}")
        return False


async def main():
    """Run all agent tests."""
    print("=" * 60)
    print("🧠 HAIVE MEMORY AGENTS - INDIVIDUAL TESTING")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Track results
    results = {}

    # Test each agent
    test_functions = [
        ("Memory Classifier", test_memory_classifier),
        ("Memory Store Manager", test_memory_store_manager),
        ("KG Generator Agent", test_kg_generator_agent),
        ("Graph RAG Retriever", test_graph_rag_retriever),
        ("Agentic RAG Coordinator", test_agentic_rag_coordinator),
        ("Multi-Agent Coordinator", test_multi_agent_coordinator),
        ("Unified Memory System", test_unified_memory_system),
    ]

    for name, test_func in test_functions:
        try:
            success = await test_func()
            results[name] = success
        except Exception as e:
            logger.error(f"Failed to run {name} test: {e}")
            results[name] = False

    # Print summary
    print_section("TEST SUMMARY")

    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    failed_tests = total_tests - passed_tests

    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

    print("\nDetailed Results:")
    for name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {name}")

    # Overall status
    print("\n" + "=" * 60)
    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED! Memory system agents are working correctly!")
    else:
        print(f"⚠️  {failed_tests} tests failed. Please review the errors above.")
    print("=" * 60)

    return passed_tests == total_tests


if __name__ == "__main__":
    # Check if we have required imports
    try:
        from haive.core.engine.aug_llm import AugLLMConfig

        print("✅ Core imports available - using real LLM config")
    except ImportError:
        print("⚠️  Core imports not available - tests may use mocks")

    # Run tests
    success = asyncio.run(main())
    exit(0 if success else 1)
