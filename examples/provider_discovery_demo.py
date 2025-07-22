#!/usr/bin/env python3
"""Demo of the Haive Provider Discovery System.

This script demonstrates how to use the comprehensive provider discovery
and management system to explore, filter, and configure providers across
all categories (LLMs, vector stores, embeddings, retrievers, document loaders).

Run with: poetry run python examples/provider_discovery_demo.py
"""

from haive.core import (
    discovery,
    get_free_providers,
    get_local_providers,
    list_all_providers,
)


def demo_basic_discovery():
    """Demonstrate basic provider discovery."""
    print("🔍 BASIC PROVIDER DISCOVERY")
    print("=" * 50)

    # Get all providers
    all_providers = list_all_providers()
    total = sum(len(providers) for providers in all_providers.values())
    print(f"Total providers available: {total}")
    print()

    # Show by category
    for category, providers in all_providers.items():
        print(f"{category.upper()}: {len(providers)} providers")
        print(f"  {', '.join(providers[:5])}" + ("..." if len(providers) > 5 else ""))
    print()


def demo_provider_details():
    """Demonstrate getting detailed provider information."""
    print("📋 DETAILED PROVIDER INFORMATION")
    print("=" * 50)

    # Show details for a few key providers
    examples = [
        ("llm", "OpenAI"),
        ("llm", "Ollama"),
        ("vectorstore", "Chroma"),
        ("vectorstore", "Pinecone"),
        ("embeddings", "HuggingFace"),
    ]

    for category, name in examples:
        info = discovery.get_provider_info(category, name)
        if info:
            print(f"{info.name} ({info.category.value})")
            print(f"  Description: {info.description}")
            print(f"  Cost: {info.cost_tier.value}")
            print(f"  Auth required: {info.auth_required}")
            print(f"  Deployment: {info.deployment_type.value}")
            if info.popular_models:
                print(f"  Popular models: {', '.join(info.popular_models[:3])}")
            if info.capabilities:
                print(f"  Capabilities: {', '.join(info.capabilities[:3])}")
            print()


def demo_filtering():
    """Demonstrate provider filtering capabilities."""
    print("🔧 PROVIDER FILTERING")
    print("=" * 50)

    # Get free providers
    free_providers = get_free_providers()
    print(f"Free providers: {len(free_providers)}")
    for _key, info in list(free_providers.items())[:5]:
        print(f"  {info.name} ({info.category.value}) - {info.description}")
    print()

    # Get local providers
    local_providers = get_local_providers()
    print(f"Local deployment providers: {len(local_providers)}")
    for _key, info in list(local_providers.items())[:5]:
        print(f"  {info.name} ({info.category.value}) - {info.deployment_type.value}")
    print()

    # Get cloud LLMs
    cloud_llms = discovery.filter_providers(category="llm", deployment_type="cloud")
    print(f"Cloud LLM providers: {len(cloud_llms)}")
    for _key, info in cloud_llms.items():
        print(f"  {info.name} - {info.description}")
    print()


def demo_recommendations():
    """Demonstrate provider recommendations."""
    print("💡 PROVIDER RECOMMENDATIONS")
    print("=" * 50)

    # Different use cases
    use_cases = ["development", "production", "local", "free"]
    categories = ["llm", "vectorstore", "embeddings"]

    for category in categories:
        print(f"{category.upper()} Recommendations:")
        for use_case in use_cases:
            recs = discovery.get_recommendations(category, use_case)
            if recs:
                print(f"  {use_case}: {', '.join(recs[:3])}")
        print()


def demo_configuration_templates():
    """Demonstrate configuration template generation."""
    print("⚙️  CONFIGURATION TEMPLATES")
    print("=" * 50)

    examples = [
        ("llm", "OpenAI"),
        ("vectorstore", "Chroma"),
        ("embeddings", "HuggingFace"),
    ]

    for category, provider in examples:
        print(f"{provider} ({category}) Configuration Template:")
        print("-" * 40)
        template = discovery.get_config_template(category, provider)
        # Show first few lines
        lines = template.split("\n")
        for line in lines[:10]:
            print(line)
        if len(lines) > 10:
            print("...")
        print()


def demo_provider_comparison():
    """Demonstrate provider comparison."""
    print("⚖️  PROVIDER COMPARISON")
    print("=" * 50)

    # Compare vector stores
    vector_stores = ["Chroma", "Pinecone", "FAISS"]
    comparison = discovery.compare_providers("vectorstore", vector_stores)

    print("Vector Store Comparison:")
    print(f"{'Provider':<12} {'Cost':<10} {'Auth':<6} {'Deployment':<12} {'Async':<5}")
    print("-" * 50)

    for name, info in comparison.items():
        print(
            f"{name:<12} {info.cost_tier.value:<10} {str(info.auth_required):<6} "
            f"{info.deployment_type.value:<12} {str(info.supports_async):<5}"
        )
    print()

    # Compare LLMs
    llms = ["OpenAI", "Anthropic", "Ollama"]
    llm_comparison = discovery.compare_providers("llm", llms)

    print("LLM Provider Comparison:")
    print(f"{'Provider':<12} {'Cost':<10} {'Max Context':<12} {'Streaming':<9}")
    print("-" * 50)

    for name, info in llm_comparison.items():
        max_ctx = str(info.max_context_length) if info.max_context_length else "N/A"
        print(
            f"{name:<12} {info.cost_tier.value:<10} {max_ctx:<12} "
            f"{str(info.supports_streaming):<9}"
        )
    print()


def demo_installation_requirements():
    """Demonstrate installation requirements."""
    print("📦 INSTALLATION REQUIREMENTS")
    print("=" * 50)

    examples = [
        ("llm", "OpenAI"),
        ("llm", "Anthropic"),
        ("vectorstore", "Pinecone"),
        ("embeddings", "HuggingFace"),
    ]

    for category, provider in examples:
        reqs = discovery.get_installation_requirements(category, provider)
        if reqs:
            print(f"{provider} ({category}):")
            print(f"  pip install {' '.join(reqs)}")
        else:
            print(f"{provider} ({category}): No additional requirements")
    print()


def main():
    """Run the complete provider discovery demo."""
    print("🚀 HAIVE PROVIDER DISCOVERY SYSTEM DEMO")
    print("=" * 60)
    print()

    demo_basic_discovery()
    demo_provider_details()
    demo_filtering()
    demo_recommendations()
    demo_configuration_templates()
    demo_provider_comparison()
    demo_installation_requirements()

    print("✅ Demo completed! The provider discovery system gives you comprehensive")
    print("   access to all Haive providers with rich metadata, filtering, and")
    print("   configuration assistance.")
    print()
    print("🎯 Quick usage:")
    print("   from haive.core import discovery")
    print("   providers = discovery.get_providers('vectorstore')")
    print("   free_providers = discovery.filter_providers(cost_tier='free')")
    print("   template = discovery.get_config_template('llm', 'OpenAI')")


if __name__ == "__main__":
    main()
