haive-core Documentation
========================

.. raw:: html

   <style>
   .hero-section {
      text-align: center;
      padding: 2rem 0;
      margin-bottom: 2rem;
   }
   .hero-section h2 {
      font-size: 2rem;
      margin-bottom: 1rem;
   }
   </style>

   <div class="hero-section">
      <h2>🚀 The Foundation of the Haive AI Agent Framework</h2>
      <p><strong>haive-core</strong> provides the essential infrastructure and patterns for building sophisticated AI agents with advanced capabilities.</p>
   </div>

.. tab-set::

   .. tab-item:: Quick Start

      Install haive-core:

      .. prompt:: bash $

         pip install haive-core

      Create your first agent:

      .. code-block:: python

         from haive.core.engine.aug_llm import AugLLMConfig
         from haive.core.schema.prebuilt.messages_state import MessagesState
         
         # Configure the LLM engine
         config = AugLLMConfig(
             model="gpt-4",
             temperature=0.7
         )

   .. tab-item:: Features

      * **Core Agent Engine** - Powerful agent infrastructure
      * **State Management** - Schema composition and validation
      * **Graph Workflows** - Orchestrate complex agent behaviors  
      * **Tool Integration** - Connect agents to external tools
      * **Vector Stores** - Embeddings and semantic search
      * **Persistence** - Checkpointing and state recovery

   .. tab-item:: Architecture

      .. mermaid::

         graph TD
             A[Agent] --> B[Engine]
             B --> C[Graph]
             C --> D[Nodes]
             D --> E[Tools]
             D --> F[Memory]
             D --> G[State]

.. toctree::
   :maxdepth: 2
   :caption: Documentation:
   :hidden:

   overview
   installation
   getting_started
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:

   configuration
   examples

.. toctree::
   :maxdepth: 2
   :caption: Project:
   :hidden:

   changelog

.. toctree::
   :maxdepth: 4
   :caption: API Reference:
   :titlesonly:
   :hidden:

   API Overview <autoapi/index>
   Engine <autoapi/core/engine/index>
   Schema <autoapi/core/schema/index>
   Graph <autoapi/core/graph/index>
   Persistence <autoapi/core/persistence/index>
   Tools <autoapi/core/tools/index>
   Common <autoapi/core/common/index>
   Config <autoapi/core/config/index>

Quick Navigation
----------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: 📚 **Overview**
      :link: overview
      :link-type: doc

      Get started with haive-core's comprehensive overview, architecture, and key features

   .. grid-item-card:: 🚀 **Quick Start**
      :link: getting_started
      :link-type: doc

      Jump right in with installation instructions and your first haive-core agent

   .. grid-item-card:: 🔍 **API Reference**
      :link: autoapi/index
      :link-type: doc

      Explore the complete API documentation with interactive examples

Key Features
------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: **🧠 Advanced Agent Engine**

      * Multi-provider LLM support (OpenAI, Anthropic, Google, AWS)
      * Structured output with Pydantic models
      * Tool integration and execution
      * Async and streaming support

   .. grid-item-card:: **📊 Graph-Based Workflows**

      * Visual workflow orchestration
      * State management with checkpointing
      * Conditional routing and branching
      * LangGraph integration

   .. grid-item-card:: **🗃️ Type-Safe State Management**

      * Pydantic-based state schemas
      * Schema composition and inheritance
      * Automatic validation and serialization
      * Meta-state patterns for complex agents

   .. grid-item-card:: **🛠️ Rich Tool Ecosystem**

      * Built-in tools (calculator, search, etc.)
      * Custom tool creation interface
      * Tool routing and validation
      * Agent-as-tool pattern

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`