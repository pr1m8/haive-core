.. haive-core documentation master file

==============================================
Haive Core - AI Agent Framework Foundation
==============================================

.. raw:: html

   <!-- Styles moved to _static/custom.css -->
   <style>/* Minimal styles only */
      /* Hero section with guaranteed contrast */
      .hero-section {
         text-align: center;
         padding: 3.5rem 2rem;
         margin: -2rem -2rem 3rem -2rem;
         background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 50%, #4c1d95 100%);
         color: white !important; /* Force white text */
         border-radius: 0;
         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
      }
      
      /* Ensure all hero children have white text on purple background */
      .hero-section h1,
      .hero-section h2,
      .hero-section p,
      .hero-section .subtitle,
      .hero-section .description {
         color: white !important;
         background: transparent !important; /* Transparent is safe here since parent has gradient */
      }
      
      .hero-section h1 {
         font-size: 3rem;
         font-weight: 800;
         margin-bottom: 1rem;
         letter-spacing: -0.025em;
      }
      .hero-section .subtitle {
         font-size: 1.5rem;
         font-weight: 300;
         opacity: 0.95;
         margin-bottom: 0.5rem;
      }
      .hero-section .description {
         font-size: 1.1rem;
         max-width: 800px;
         margin: 0 auto;
         opacity: 0.9;
         line-height: 1.6;
      }
      
      /* Ensure feature icons are visible */
      .feature-icon {
         font-size: 2.5rem;
         margin-bottom: 1rem;
         display: block;
         color: var(--color-brand-primary, #8b5cf6);
      }
      
      /* Code examples with proper contrast */
      .code-example {
         background: #f8f9fa !important; /* Light gray background */
         color: #1f2937 !important; /* Dark text */
         border-left: 4px solid #8b5cf6;
         margin: 1rem 0;
         padding: 0.5rem;
      }
      
      /* Fix for grid cards to ensure text visibility */
      .sd-card {
         background: var(--color-card-background, #ffffff) !important;
         color: var(--color-foreground-primary, #1f2937) !important;
      }
      
      /* Ensure all text outside hero has proper contrast */
      body .content h1:not(.hero-section h1),
      body .content h2:not(.hero-section h2),
      body .content h3:not(.hero-section h3),
      body .content p:not(.hero-section p) {
         color: var(--color-foreground-primary, #1f2937) !important;
         background: transparent; /* Safe since body has white background */
      }
      
      /* Sidebar text visibility */
      .sidebar {
         background: var(--color-sidebar-background, #faf5ff) !important;
      }
      .sidebar a {
         color: var(--color-sidebar-link-text, #374151) !important;
      }
      
      /* Tab content visibility */
      .tab-content {
         background: var(--color-background-primary, #ffffff) !important;
         color: var(--color-foreground-primary, #1f2937) !important;
      }
   </style>
   
   <!-- Hero section removed per request -->

.. grid:: 1 2 2 3
   :gutter: 3
   :class: landing-grid

   .. grid-item-card:: 
      :class-card: sd-border-0 sd-shadow-sm
      :class-title: sd-text-center sd-fs-5
      
      **⚡ Quick Start**
      ^^^
      Get up and running in under 5 minutes. Build your first agent with our streamlined API.
      +++
      .. button-ref:: getting_started
         :expand:
         :color: primary
         :outline:
         
         Start Building →

   .. grid-item-card::
      :class-card: sd-border-0 sd-shadow-sm
      :class-title: sd-text-center sd-fs-5
      
      **📘 Learn by Example**
      ^^^
      Explore practical examples and real-world patterns from simple agents to complex workflows.
      +++
      .. button-ref:: examples
         :expand:
         :color: info
         :outline:
         
         View Examples →

   .. grid-item-card::
      :class-card: sd-border-0 sd-shadow-sm
      :class-title: sd-text-center sd-fs-5
      
      **🔍 API Reference**
      ^^^
      Comprehensive API documentation with detailed class references and method signatures.
      +++
      .. button-link:: autoapi/haive/core/index.html
         :expand:
         :color: warning
         :outline:
         
         Browse API →

.. admonition:: 🎉 Latest Release: v0.1.0
   :class: tip

   **What's New in haive-core:**
   
   • **MetaStateSchema** - Revolutionary agent composition with type-safe state management
   • **Dynamic Graphs** - Runtime graph modification and node composition  
   • **Tool Orchestration** - Seamless tool discovery, registration, and validation
   • **Performance** - 2x faster state transitions with optimized reducers
   • **Developer Experience** - Enhanced debugging tools and error messages
   
   **Installation:** ``pip install git+https://github.com/pr1m8/haive-core.git``

Documentation Hub
-----------------

.. tab-set::

   .. tab-item:: 🎓 Learn

      .. grid:: 1 2 2 2
         :gutter: 2

         .. grid-item::

            **Fundamentals**

            .. toctree::
               :maxdepth: 1

               installation
               getting_started
               overview
               concepts

         .. grid-item::

            **Tutorials**

            .. toctree::
               :maxdepth: 1

               examples
               vectorstore_example_enhanced

      **Key Concepts to Master:**

      • **Engines** - The computational heart of agents (LLMs, retrievers, vector stores)
      • **Schemas** - Type-safe state management with Pydantic validation
      • **Graphs** - Workflow orchestration with nodes, edges, and conditional routing
      • **Tools** - External capabilities and function calling

   .. tab-item:: 🛠️ Build

      .. grid:: 1 2 2 2
         :gutter: 2

         .. grid-item::

            **Core Architecture**

            .. toctree::
               :maxdepth: 2

               engine_architecture
               schema_system
               graph_workflows

         .. grid-item::

            **Implementation**

            .. toctree::
               :maxdepth: 2

               configuration
               tool_integration
               persistence_layer
               common_utilities

      **Architecture Highlights:**

      • **Modular Design** - Compose complex systems from simple, reusable components
      • **Type Safety** - Full Pydantic integration for runtime validation
      • **Async-First** - Built on asyncio for high-performance concurrent operations
      • **Extensible** - Plugin architecture for custom engines and tools

   .. tab-item:: 📚 API Docs

      .. grid:: 1 1 2 2
         :gutter: 3

         .. grid-item::

            **🎮 Core Systems**
            
            .. toctree::
               :maxdepth: 2

               Engine Module <autoapi/haive/core/engine/index>
               Schema Module <autoapi/haive/core/schema/index>
               Graph Module <autoapi/haive/core/graph/index>

         .. grid-item::

            **🔧 Supporting Systems**
            
            .. toctree::
               :maxdepth: 2

               Models <autoapi/haive/core/models/index>
               Tools <autoapi/haive/core/tools/index>
               Persistence <autoapi/haive/core/persistence/index>

         .. grid-item::

            **🛠️ Utilities**
            
            .. toctree::
               :maxdepth: 2

               Common <autoapi/haive/core/common/index>
               Utils <autoapi/haive/core/utils/index>
               Types <autoapi/haive/core/types/index>

         .. grid-item::

            **📖 Guides**
            
            .. toctree::
               :maxdepth: 2

               API Overview <autoapi/haive/core/index>
               common_module_overview
               api_reference

   .. tab-item:: ℹ️ Resources

      .. toctree::
         :maxdepth: 1

         additional_resources
         changelog

      **Quick Links**
      
      * `Source Code & Issues <https://github.com/pr1m8/haive-core>`_
      * `Central Documentation Hub <https://pub-7f716b302a2948e19f08b49b71408039.r2.dev>`_
      * `Installation Guide <https://github.com/pr1m8/haive-core#installation>`_

      **License**

      MIT License - Free for commercial and personal use

Core Capabilities
-----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: **🎮 Engine System**
      :class-card: sd-border-1
      
      **Augmented LLM Engine**
      
      • Multi-provider support (OpenAI, Anthropic, Azure)
      • Structured output with Pydantic models
      • Token management and cost tracking
      • Streaming and async execution
      
      **Retriever & Vector Stores**
      
      • ChromaDB, FAISS, Pinecone integrations
      • Embedding model flexibility
      • Hybrid search capabilities
      • Document processing pipelines

   .. grid-item-card:: **📋 State Management**
      :class-card: sd-border-1
      
      **Type-Safe Schemas**
      
      • Pydantic-based validation
      • Automatic serialization/deserialization
      • State composition and inheritance
      • Custom field validators
      
      **MetaStateSchema**
      
      • Agent state embedding
      • Execution tracking
      • Recompilation management
      • Graph context preservation

   .. grid-item-card:: **🔄 Graph Workflows**
      :class-card: sd-border-1
      
      **StateGraph Architecture**
      
      • Node-based computation
      • Conditional branching
      • Parallel execution paths
      • Dynamic graph modification
      
      **Advanced Features**
      
      • Checkpointing and recovery
      • Graph visualization
      • Performance profiling
      • Error handling and retry

   .. grid-item-card:: **🔧 Tool Ecosystem**
      :class-card: sd-border-1
      
      **Built-in Tools**
      
      • File operations
      • Web scraping
      • API integrations
      • Database connectors
      
      **Tool Management**
      
      • Automatic discovery
      • Runtime registration
      • Validation framework
      • Human-in-the-loop support

Quick Examples
--------------

.. tab-set::

   .. tab-item:: Engine Configuration

      **Setting up an Augmented LLM Engine:**

      .. code-block:: python

         from haive.core.engine.aug_llm import AugLLMConfig
         from haive.core.schema.prebuilt.messages_state import MessagesState
         
         # Configure the engine with Azure OpenAI
         config = AugLLMConfig(
             model="gpt-4",
             temperature=0.7,
             max_tokens=2000,
             system_message="You are a helpful AI assistant.",
             provider="azure",  # or "openai", "anthropic"
             api_base="https://your-resource.openai.azure.com/"
         )
         
         # Initialize state management
         state = MessagesState()
         state.add_user_message("Explain quantum computing")
         
         # The engine is now ready for use in agents

   .. tab-item:: Graph Workflow

      **Building a multi-step workflow:**

      .. code-block:: python

         from haive.core.graph.state_graph import BaseGraph
         from haive.core.graph.node import create_node
         from haive.core.schema.prebuilt import MessagesState
         
         # Create a workflow graph
         graph = BaseGraph(state_schema=MessagesState)
         
         # Define processing nodes
         async def analyze_node(state: MessagesState):
             # Process and analyze input
             return {"analysis": "completed"}
         
         async def generate_node(state: MessagesState):
             # Generate response based on analysis
             return {"response": "generated"}
         
         # Build the graph
         graph.add_node("analyze", analyze_node)
         graph.add_node("generate", generate_node)
         graph.add_edge("analyze", "generate")
         graph.set_entry_point("analyze")
         
         # Compile and execute
         workflow = graph.compile()
         result = await workflow.ainvoke(state)

   .. tab-item:: Tool Integration

      **Creating and registering custom tools:**

      .. code-block:: python

         from langchain_core.tools import tool
         from haive.core.registry import get_registry
         from typing import Annotated
         
         @tool
         def calculate_compound_interest(
             principal: Annotated[float, "Initial amount"],
             rate: Annotated[float, "Annual interest rate (as decimal)"],
             time: Annotated[int, "Time period in years"]
         ) -> float:
             """Calculate compound interest with annual compounding."""
             amount = principal * (1 + rate) ** time
             return round(amount - principal, 2)
         
         # Register the tool globally
         registry = get_registry()
         registry.register_tool("compound_interest", calculate_compound_interest)
         
         # Tools are now available to all agents
         from haive.core.tools import get_available_tools
         tools = get_available_tools()

   .. tab-item:: Vector Store RAG

      **Setting up a RAG system with vector stores:**

      .. code-block:: python

         from haive.core.engine.vectorstore import VectorStoreConfig
         from haive.core.engine.embedding.providers import HuggingFaceEmbeddingConfig
         from haive.core.engine.document import DocumentProcessor
         
         # Configure embeddings
         embedding_config = HuggingFaceEmbeddingConfig(
             model_name="sentence-transformers/all-mpnet-base-v2",
             model_kwargs={"device": "cpu"},
             encode_kwargs={"normalize_embeddings": True}
         )
         
         # Setup vector store
         vector_config = VectorStoreConfig(
             provider="Chroma",
             embedding_config=embedding_config,
             collection_name="knowledge_base",
             persist_directory="./chroma_db"
         )
         
         # Initialize and populate
         vector_store = vector_config.create()
         
         # Process documents
         processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
         documents = processor.process_files(["./docs/*.pdf"])
         vector_store.add_documents(documents)
         
         # Ready for retrieval
         results = vector_store.similarity_search("What is haive?", k=5)

Architecture Overview
---------------------

.. mermaid::

   graph TB
      subgraph "Haive Core Architecture"
         A[Application Layer]
         A --> B[Engine Layer]
         A --> C[Schema Layer]
         A --> D[Graph Layer]
         
         B --> B1[AugLLM<br/>Engine]
         B --> B2[Retriever<br/>Engine]
         B --> B3[VectorStore<br/>Engine]
         
         C --> C1[State<br/>Schemas]
         C --> C2[Message<br/>Formats]
         C --> C3[Validation<br/>Rules]
         
         D --> D1[Nodes &<br/>Edges]
         D --> D2[Execution<br/>Runtime]
         D --> D3[State<br/>Management]
         
         E[Tool Registry]
         F[Persistence Layer]
         
         B1 & B2 & B3 --> E
         D --> F
      end
      
      style A fill:#8b5cf6,color:#ffffff,stroke:#6d28d9,stroke-width:2px
      style B fill:#6d28d9,color:#ffffff,stroke:#4c1d95,stroke-width:2px
      style C fill:#6d28d9,color:#ffffff,stroke:#4c1d95,stroke-width:2px
      style D fill:#6d28d9,color:#ffffff,stroke:#4c1d95,stroke-width:2px

Performance & Scalability
-------------------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item-card:: ⚡ **Fast**
      :text-align: center
      
      • Async/await throughout
      • Optimized state transitions
      • Efficient memory usage
      • Connection pooling

   .. grid-item-card:: 📈 **Scalable**
      :text-align: center
      
      • Distributed execution
      • Horizontal scaling
      • Queue-based processing
      • Load balancing

   .. grid-item-card:: 💪 **Reliable**
      :text-align: center
      
      • Automatic retries
      • Error recovery
      • State persistence
      • Health monitoring

Getting Help & Community
------------------------

.. grid:: 1 2 4 4
   :gutter: 2

   .. grid-item-card:: 📖 **Documentation**
      :text-align: center
      :class-card: sd-border-1
      
      Complete documentation hub
      
      +++
      
      `Central Docs → <https://pub-7f716b302a2948e19f08b49b71408039.r2.dev>`_

   .. grid-item-card:: 🐛 **Issues & Bugs**
      :text-align: center
      :class-card: sd-border-1
      
      Report bugs and request features
      
      +++
      
      `GitHub Issues → <https://github.com/pr1m8/haive-core/issues>`_

   .. grid-item-card:: 💡 **Discussions**
      :text-align: center
      :class-card: sd-border-1
      
      Community discussions and Q&A
      
      +++
      
      `GitHub Discussions → <https://github.com/pr1m8/haive-core/discussions>`_

   .. grid-item-card:: 📧 **Contact**
      :text-align: center
      :class-card: sd-border-1
      
      Direct support and inquiries
      
      +++
      
      `Email Support → <mailto:support@pr1m8.com>`_

Search & Navigation
-------------------

.. grid:: 1 3 3 3
   :gutter: 2

   .. grid-item::
   
      **🔍 Search Documentation**
      
      :ref:`search`

   .. grid-item::
   
      **📑 Full Index**
      
      :ref:`genindex`

   .. grid-item::
   
      **📦 Module Index**
      
      :ref:`modindex`

----

.. note::
   
   **Documentation Version:** This documentation is automatically generated from the latest source code 
   and is synchronized with version 0.1.0 of haive-core. All code examples are tested in our CI pipeline.

.. seealso::

   **Related Packages in the Haive Ecosystem:**
   
   • `haive-agents <https://github.com/pr1m8/haive-agents>`_ - Pre-built agent implementations
   • `haive-tools <https://github.com/pr1m8/haive-tools>`_ - Extended tool library
   • `haive-games <https://github.com/pr1m8/haive-games>`_ - Game environments for agent training