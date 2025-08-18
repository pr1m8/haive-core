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
   :caption: Getting Started:

   installation
   getting_started
   concepts

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   configuration
   examples

.. toctree::
   :maxdepth: 2
   :caption: Project:

   changelog

.. toctree::
   :maxdepth: 3
   :caption: API Reference:

   autoapi/haive/index

Quick Start
-----------

Install haive-core::

    pip install haive-core

Basic usage::

    from haive.core.engine.aug_llm import AugLLMConfig
    from haive.core.schema.prebuilt.messages_state import MessagesState
    
    # Configure the LLM engine
    config = AugLLMConfig(
        model="gpt-4",
        temperature=0.7
    )

Features
--------

* Core agent engine and infrastructure
* State management and schema composition  
* Graph-based workflow orchestration
* Tool and retriever integration
* Vector stores and embeddings
* Persistence and checkpointing

API Reference
-------------

The complete API documentation is automatically generated from the source code.

See :doc:`API Reference <autoapi/index>` for detailed documentation of all modules, classes, and functions.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`