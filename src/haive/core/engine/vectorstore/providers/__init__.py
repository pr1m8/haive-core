"""
Vector store provider implementations for the Haive framework.

This package contains implementations of vector store providers that extend
the core vector store functionality. New providers can be added here to support
additional vector database backends or specialized vector store implementations.

To create a new provider:
1. Create a new file in this directory
2. Implement a class that extends langchain_core.vectorstores.VectorStore
3. Register the provider with VectorStoreProviderRegistry

Examples:
    ```python
    # In my_provider.py
    from langchain_core.vectorstores import VectorStore

    class MyCustomVectorStore(VectorStore):
        # Implementation of VectorStore methods
        pass

    # In your initialization code
    from haive.core.engine.vectorstore import VectorStoreProviderRegistry
    from haive.core.engine.vectorstore.providers.my_provider import MyCustomVectorStore

    VectorStoreProviderRegistry.register_provider("MyCustom", MyCustomVectorStore)
    ```
"""

# Import and register any built-in providers here
