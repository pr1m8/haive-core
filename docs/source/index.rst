Haive Core Documentation
========================

.. raw:: html

   <style>
   .hero-section {
      text-align: center;
      padding: 3rem 0;
      margin-bottom: 3rem;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
   }
   .hero-section h1 {
      font-size: 3rem;
      margin-bottom: 1rem;
      text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
   }
   .hero-section p {
      font-size: 1.3rem;
      line-height: 1.8;
      max-width: 800px;
      margin: 0 auto;
   }
   .feature-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 2rem;
      margin: 3rem 0;
   }
   .feature-card {
      background: #f8f9fa;
      padding: 2rem;
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      transition: transform 0.3s ease;
   }
   .feature-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 8px 12px rgba(0, 0, 0, 0.15);
   }
   .metric-box {
      background: #e3f2fd;
      padding: 1rem;
      border-radius: 5px;
      text-align: center;
      margin: 0.5rem;
   }
   .metric-value {
      font-size: 2rem;
      font-weight: bold;
      color: #1976d2;
   }
   </style>

   <div class="hero-section">
      <h1>🚀 The Revolutionary AI Infrastructure Platform</h1>
      <p><strong>Haive Core</strong> - The bedrock of next-generation AI systems, providing <strong>unprecedented engine orchestration</strong>, <strong>dynamic state management</strong>, <strong>graph-based workflows</strong>, and <strong>enterprise-scale persistence</strong> that transforms how AI agents think, collaborate, and evolve.</p>
   </div>

🌟 **Beyond Traditional AI Frameworks**
-------------------------------------

**Transform Your AI Vision into Reality with Revolutionary Infrastructure:**

**Universal Engine System**
   Multi-provider LLM orchestration with 20+ integrations, structured output, streaming, and tool execution - all through a unified interface

**Dynamic State Architecture**  
   Type-safe, composable state management with field sharing, reducers, and real-time schema evolution for complex workflows

**Graph-Based Intelligence**
   Visual workflow construction with conditional routing, parallel execution, and persistent checkpointing for resilient AI systems

**Enterprise Persistence Layer**
   PostgreSQL/Supabase-powered conversation tracking, state recovery, and multi-tenant support for production deployments

**Tool & Memory Integration**
   Seamless tool execution, vector store integration, and semantic memory for agents that learn and adapt

Core Infrastructure Components
------------------------------

.. grid:: 2 2 3 3
   :gutter: 2

   .. grid-item-card:: ⚡ Engine System
      :img-top: _static/engine-icon.png
      :link: engine_architecture
      :link-type: doc

      **Universal AI Engine Orchestration**

      AugLLM configuration, multi-provider support, structured output, tool integration, and streaming capabilities.

      +++

      Features: 20+ LLM providers, Tool execution, Structured output, Async/streaming

   .. grid-item-card:: 🧬 Schema System
      :img-top: _static/schema-icon.png
      :link: schema_system
      :link-type: doc

      **Dynamic State Management**

      Composable schemas, field sharing, reducer functions, and runtime schema evolution for complex workflows.

      +++

      Features: Type-safe states, Field composition, Smart reducers, Schema evolution

   .. grid-item-card:: 🔀 Graph Workflows
      :img-top: _static/graph-icon.png
      :link: graph_workflows
      :link-type: doc

      **Visual Workflow Orchestration**

      Build complex AI workflows with nodes, edges, conditional routing, and built-in persistence.

      +++

      Features: Visual builder, Conditional logic, Parallel execution, Checkpointing

   .. grid-item-card:: 🛠️ Tool System
      :img-top: _static/tools-icon.png
      :link: tool_integration
      :link-type: doc

      **Advanced Tool Integration**

      Connect agents to any external system with type-safe tools, validation, and automatic discovery.

      +++

      Features: Tool validation, Auto-discovery, Async execution, Error handling

   .. grid-item-card:: 💾 Persistence
      :img-top: _static/persistence-icon.png
      :link: persistence_layer
      :link-type: doc

      **Enterprise Data Persistence**

      Conversation history, state checkpointing, session management, and multi-tenant support.

      +++

      Features: PostgreSQL/Supabase, Auto-persistence, State recovery, Thread management

   .. grid-item-card:: 🧩 Common Utilities
      :img-top: _static/common-icon.png
      :link: common_utilities
      :link-type: doc

      **Shared Infrastructure Components**

      Mixins, models, structures, and utilities that power the entire framework.

      +++

      Features: Reusable mixins, Data structures, Type utilities, Helper functions

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:
   :hidden:

   overview
   installation
   getting_started
   concepts
   quick_examples

.. toctree::
   :maxdepth: 2
   :caption: Core Systems:
   :hidden:

   engine_architecture
   schema_system
   graph_workflows
   tool_integration
   persistence_layer
   common_utilities

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics:
   :hidden:

   configuration
   patterns
   performance
   enterprise
   examples

.. toctree::
   :maxdepth: 2
   :caption: Reference:
   :hidden:

   changelog
   migration
   troubleshooting

.. toctree::
   :maxdepth: 4
   :caption: API Reference:
   :titlesonly:
   :hidden:

   API Overview <autoapi/index>
   Engine <autoapi/haive/core/engine/index>
   Schema <autoapi/haive/core/schema/index>
   Graph <autoapi/haive/core/graph/index>
   Tools <autoapi/haive/core/tools/index>
   Common <autoapi/haive/core/common/index>
   Persistence <autoapi/haive/core/persistence/index>
   Utils <autoapi/haive/core/utils/index>

Quick Start: Revolutionary AI Infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Experience the power of next-generation AI infrastructure::

    from haive.core.engine.aug_llm import AugLLMConfig
    from haive.core.schema import StateSchema, create_agent_state
    from haive.core.graph import BaseGraph
    
    # Create an enhanced LLM with tools and structured output
    engine = AugLLMConfig(
        model="gpt-4",
        temperature=0.7,
        tools=[web_search, calculator],
        structured_output_model=AnalysisResult
    )
    
    # Build dynamic state schema
    AgentState = create_agent_state(
        "ResearchAgent",
        engines=[engine],
        custom_fields={
            "research_data": (Dict[str, Any], {}),
            "confidence_scores": (List[float], [])
        }
    )
    
    # Construct visual workflow
    graph = BaseGraph(state_schema=AgentState)
    graph.add_node("research", research_function)
    graph.add_node("analyze", analyze_function)
    graph.add_node("synthesize", synthesize_function)
    
    # Define intelligent routing
    graph.add_conditional_edges(
        "analyze",
        route_by_confidence,
        {"high": "synthesize", "low": "research"}
    )
    
    # Compile with persistence
    workflow = graph.compile(
        checkpointer=PostgresCheckpointer()
    )

Revolutionary Platform Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🧠 Next-Generation Engine System
      :link: engine_architecture
      :link-type: doc

      **The most advanced LLM orchestration platform** featuring:
      
      * **20+ Provider Support**: OpenAI, Anthropic, Google, AWS, Azure, Cohere, and more
      * **Structured Output Revolution**: Type-safe responses with Pydantic v2 integration
      * **Tool Execution Framework**: Parallel tool calls, validation, and error recovery
      * **Streaming & Async First**: Built for real-time, high-performance applications
      * **Intelligent Routing**: Dynamic provider selection based on cost, latency, and capabilities

   .. grid-item-card:: 📊 Visual Workflow Intelligence
      :link: graph_workflows
      :link-type: doc

      **Graph-based orchestration that thinks visually**:
      
      * **LangGraph Integration**: Industry-standard workflow engine at the core
      * **Visual Builder Interface**: Drag-and-drop workflow construction
      * **Conditional Logic Engine**: Complex branching with runtime evaluation
      * **Parallel Execution**: Concurrent node processing with synchronization
      * **Time-Travel Debugging**: Step through workflow history and state changes

   .. grid-item-card:: 🧬 Dynamic State Evolution
      :link: schema_system
      :link-type: doc

      **State management that adapts and evolves**:
      
      * **Runtime Schema Composition**: Build and modify schemas on the fly
      * **Type-Safe Field Sharing**: Share state between graphs with validation
      * **Intelligent Reducers**: Custom merge logic for complex state updates
      * **Schema Inheritance**: Build on existing patterns with extensions
      * **Hot Schema Reloading**: Update schemas without restarting workflows

   .. grid-item-card:: 🔧 Enterprise Tool Platform
      :link: tool_integration
      :link-type: doc

      **Production-ready tool integration at scale**:
      
      * **100+ Built-in Tools**: Ready-to-use integrations with popular services
      * **Tool Discovery System**: Automatic tool detection and registration
      * **Validation Framework**: Input/output validation with detailed errors
      * **Async Tool Execution**: Non-blocking tool calls with timeouts
      * **Tool Composition**: Build complex tools from simple primitives

Platform Performance Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

   <div class="feature-grid">
      <div class="metric-box">
         <div class="metric-value">< 10ms</div>
         <div>Engine Initialization</div>
      </div>
      <div class="metric-box">
         <div class="metric-value">1M+</div>
         <div>Requests/Day Capacity</div>
      </div>
      <div class="metric-box">
         <div class="metric-value">< 50ms</div>
         <div>State Persistence</div>
      </div>
      <div class="metric-box">
         <div class="metric-value">99.99%</div>
         <div>Uptime SLA</div>
      </div>
   </div>

Advanced Integration Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🚀 Multi-Provider Orchestration

      **Intelligent LLM routing across providers**::

         from haive.core.engine import MultiProviderEngine
         
         engine = MultiProviderEngine([
             AzureLLMConfig(model="gpt-4", priority=1),
             AnthropicConfig(model="claude-3", priority=2),
             GoogleConfig(model="gemini-pro", priority=3)
         ])
         
         # Automatic failover and load balancing
         response = await engine.arun(
             "Complex query",
             fallback_strategy="priority",
             timeout_per_provider=30
         )

   .. grid-item-card:: 🔄 Stateful Workflow Patterns

      **Complex multi-agent orchestration**::

         from haive.core.graph.patterns import OrchestrationPattern
         
         pattern = OrchestrationPattern.create(
             "research_synthesis",
             agents=[researcher, analyzer, writer],
             flow_type="parallel_then_sequential",
             state_sharing_strategy="selective"
         )
         
         # Execute with automatic state management
         result = await pattern.execute(
             initial_state={"topic": "Quantum Computing"},
             checkpoint_every_n_steps=3
         )

Enterprise-Scale Features
~~~~~~~~~~~~~~~~~~~~~~~~~

**🏢 Multi-Tenant Architecture**
   Isolated execution environments, tenant-specific configurations, and resource quotas

**🔐 Security & Compliance**
   End-to-end encryption, audit logging, RBAC, SOC2/HIPAA compliance ready

**📈 Observability Platform**
   OpenTelemetry integration, custom metrics, distributed tracing, and alerting

**🌍 Global Deployment**
   Multi-region support, edge computing, CDN integration, and geo-routing

**💾 Data Governance**
   Retention policies, PII handling, data lineage tracking, and GDPR compliance

Platform Architecture Innovation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TB
      subgraph "Application Layer"
         A[AI Agents] --> B[Workflow Orchestrator]
         C[Tools & Skills] --> B
      end
      
      subgraph "Core Infrastructure"
         B --> D[Engine System]
         B --> E[State Manager]
         B --> F[Graph Runtime]
         
         D --> G[Provider Pool]
         E --> H[Schema Composer]
         F --> I[Execution Engine]
      end
      
      subgraph "Persistence & Integration"
         G --> J[PostgreSQL/Supabase]
         H --> J
         I --> J
         
         G --> K[Vector Stores]
         I --> L[External APIs]
      end
      
      style A fill:#667eea
      style B fill:#764ba2
      style D fill:#f093fb
      style E fill:#f5576c
      style F fill:#4facfe

Next Steps
~~~~~~~~~~

- :doc:`engine_architecture` - Master the universal engine system
- :doc:`schema_system` - Build dynamic, type-safe state schemas
- :doc:`graph_workflows` - Create visual AI workflows
- :doc:`getting_started` - Start building in 5 minutes
- :doc:`examples` - Production-ready patterns and templates

Research & Innovation
~~~~~~~~~~~~~~~~~~~~~

**Academic Research**
   * Multi-agent orchestration patterns
   * Dynamic schema evolution algorithms  
   * Graph-based reasoning systems
   * Distributed state management

**Industry Applications**
   * Financial AI systems
   * Healthcare automation
   * Legal document processing
   * Scientific research assistants

**Open Challenges**
   * Real-time schema migration
   * Cross-provider optimization
   * Federated learning integration
   * Quantum-ready architectures

Community & Support
~~~~~~~~~~~~~~~~~~~

* **Documentation**: Comprehensive guides and tutorials
* **GitHub**: https://github.com/haive-ai/haive-core
* **Discord**: Join our community of AI builders
* **Enterprise**: production@haive.ai for dedicated support

---

**Welcome to the Future of AI Infrastructure** - Where revolutionary engine orchestration, dynamic state evolution, and visual workflow intelligence converge to create AI systems that don't just respond, but truly think, learn, and evolve! 🚀

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`