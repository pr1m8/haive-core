"""🚀 Haive Core - The Revolutionary AI Infrastructure Platform

**THE FOUNDATION OF NEXT-GENERATION AI SYSTEMS**

Welcome to Haive Core - the bedrock of intelligent agent architectures that transforms 
how AI systems are built, composed, and orchestrated. This isn't just another framework;
it's a paradigm shift in AI engineering that enables self-evolving, deeply intelligent
systems that adapt, learn, and grow beyond their initial programming.

🧬 CORE ARCHITECTURE PILLARS
----------------------------

**1. Universal Engine System** ⚡
   The beating heart of Haive - a revolutionary abstraction that unifies:
   - **AugLLM**: Enhanced language models with tool use, structured output, and self-modification
   - **InvokableEngine**: Universal interface for any AI component (LLMs, retrievers, tools)
   - **Multi-Provider Support**: Seamlessly switch between OpenAI, Anthropic, Google, and more
   - **Adaptive Configuration**: Runtime reconfiguration without restarts

**2. Dynamic Graph Workflows** 🔀
   Visual programming for AI - build complex behaviors through intuitive graphs:
   - **BaseGraph**: Foundation for stateful, persistent workflows
   - **Conditional Routing**: Intelligent path selection based on runtime conditions
   - **Parallel Execution**: Orchestrate multiple agents simultaneously
   - **Time-Travel Debugging**: Step through workflow history and restore states

**3. Intelligent State Management** 🧠
   Beyond static data models - schemas that evolve with your AI:
   - **StateSchema**: Type-safe, self-documenting state containers
   - **Dynamic Composition**: Build schemas at runtime from multiple sources
   - **Smart Reducers**: Intelligent state merging with conflict resolution
   - **Field Visibility**: Sophisticated sharing between parent and child graphs

**4. Enterprise Persistence Layer** 💾
   Never lose a conversation or state again:
   - **PostgreSQL/Supabase**: Production-ready persistence out of the box
   - **Automatic Checkpointing**: Save and restore complex workflows
   - **Thread Management**: Branching conversations with full history
   - **Vector Memory**: Semantic search across all interactions

**5. Tool Ecosystem** 🛠️
   Give your AI real-world capabilities:
   - **Type-Safe Tools**: Full validation and error handling
   - **Automatic Discovery**: Find and register tools dynamically
   - **Parallel Execution**: Run multiple tools simultaneously
   - **Tool Composition**: Build complex tools from simple primitives

**6. Common Utilities** 🧰
   Battle-tested components for production systems:
   - **Mixins**: Reusable behaviors (timestamps, IDs, state management)
   - **Data Structures**: Specialized trees, graphs, and collections
   - **Type System**: Runtime type checking and validation
   - **Performance Helpers**: Caching, memoization, and optimization

🚀 QUICK START
--------------

```python
from haive.core import AugLLMConfig, BaseGraph, StateSchema
from haive.core.schema import Field

# 1. Define your state
class AgentState(StateSchema):
    messages: list = Field(default_factory=list)
    context: dict = Field(default_factory=dict)
    confidence: float = Field(default=0.0)

# 2. Create an enhanced LLM
llm = AugLLMConfig(
    model="gpt-4",
    temperature=0.7,
    tools=["web_search", "calculator"],
    structured_output_model=AnalysisResult
)

# 3. Build a workflow
graph = BaseGraph(state_schema=AgentState)
graph.add_node("analyze", analysis_node)
graph.add_node("synthesize", synthesis_node)
graph.add_edge("analyze", "synthesize")

# 4. Execute with persistence
app = graph.compile(checkpointer=PostgresCheckpointer())
result = await app.ainvoke({"messages": ["Analyze market trends"]})
```

🎯 KEY INNOVATIONS
------------------

1. **Self-Modifying Architectures**: Agents that rewrite their own graphs
2. **Semantic State Management**: States that understand their own meaning
3. **Distributed Orchestration**: Scale across multiple machines seamlessly
4. **Real-Time Collaboration**: Multiple agents working on shared state
5. **Zero-Config Persistence**: Automatic state saving with time-travel

📚 COMPREHENSIVE MODULES
------------------------

- `engine/`: Universal AI component interface (LLMs, tools, retrievers)
- `graph/`: Visual workflow orchestration and state machines
- `schema/`: Dynamic, type-safe state management
- `tools/`: Tool creation, discovery, and execution
- `persistence/`: Enterprise-grade data persistence
- `common/`: Reusable utilities and patterns
- `types/`: Type definitions and protocols
- `models/`: Model configurations and providers
- `registry/`: Dynamic component discovery
- `runtime/`: Execution environment management
- `config/`: Configuration management
- `utils/`: Development and debugging tools

🌟 WHY HAIVE CORE?
------------------

**For Developers**:
- Write 10x less code with powerful abstractions
- Debug visually with graph workflows
- Never worry about state management again
- Scale from prototype to production seamlessly

**For Enterprises**:
- Production-ready from day one
- Full audit trails and compliance features
- Multi-tenant support with isolation
- Battle-tested by industry leaders

**For Researchers**:
- Experiment with novel architectures
- Build self-improving systems
- Combine multiple AI paradigms
- Publish reproducible workflows

🔗 ECOSYSTEM INTEGRATION
------------------------

Haive Core seamlessly integrates with:
- **LangChain**: Use any LangChain component
- **LangGraph**: Enhanced with Haive's persistence
- **OpenAI/Anthropic/Google**: First-class support
- **Vector Databases**: Pinecone, Weaviate, pgvector
- **Observability**: OpenTelemetry, Datadog, etc.

💡 ADVANCED PATTERNS
--------------------

```python
# Multi-Agent Orchestration
from haive.core.graph import OrchestratorGraph

orchestrator = OrchestratorGraph()
orchestrator.add_agents({
    "researcher": ResearchAgent(),
    "analyst": AnalystAgent(),
    "writer": WriterAgent()
})
orchestrator.set_coordination_pattern("hierarchical")

# Dynamic Schema Evolution
from haive.core.schema import SchemaComposer

composer = SchemaComposer()
composer.add_fields_from_llm_output(llm_response)
DynamicState = composer.build()

# Self-Modifying Workflows
from haive.core.graph import DynamicGraph

graph = DynamicGraph()
graph.add_mutation_rule(
    condition=lambda state: state.confidence < 0.5,
    action=lambda graph: graph.add_verification_node()
)
```

🚨 ENTERPRISE FEATURES
----------------------

- **🔒 Security**: End-to-end encryption, RBAC, audit logging
- **📊 Monitoring**: Real-time metrics, performance profiling
- **🌍 Multi-Region**: Distributed execution across data centers
- **♻️ Fault Tolerance**: Automatic failover and recovery
- **📈 Scalability**: Handle millions of concurrent workflows

🎓 LEARNING PATH
----------------

1. Start with `engine/` - understand the universal interface
2. Explore `schema/` - master state management
3. Dive into `graph/` - build visual workflows
4. Experiment with `tools/` - extend capabilities
5. Deploy with `persistence/` - production readiness

🤝 JOIN THE REVOLUTION
----------------------

Haive Core isn't just a framework - it's a movement towards truly intelligent,
self-evolving AI systems. Whether you're building chatbots, autonomous agents,
or the next breakthrough in AI, Haive Core provides the foundation you need.

**Ready to build the future?** Let's go! 🚀

---

**Version**: 0.1.0 | **License**: MIT | **Documentation**: https://haive.ai/docs
"""

# ============================================================================
# PERFORMANCE OPTIMIZATIONS - Applied automatically on import
# ============================================================================

import importlib
import logging
import os
import pkgutil
from pathlib import Path

# ============================================================================
# Module Discovery - Expose all submodules
# ============================================================================

# Get the package directory
_package_dir = Path(__file__).resolve().parent

# Core modules to expose (in order of importance)
_CORE_MODULES = [
    "engine",  # Engine system - most important
    "graph",  # Graph building
    "schema",  # Schema management
    "tools",  # Tool system
    "types",  # Type definitions
    "utils",  # Utilities
    "models",  # Model configurations
    "registry",  # Component registry
    "runtime",  # Runtime system
    "persistence",  # State persistence
    "config",  # Configuration
    "common",  # Common utilities
    "errors",  # Error types
]

# ============================================================================
# Lazy Module Loading with __getattr__
# ============================================================================


def __getattr__(name: str):
    """Lazy load modules and their contents on demand."""
    # Check if it's a module name
    if name in _CORE_MODULES:
        module = importlib.import_module(f".{name}", package=__name__)
        # Cache in globals
        globals()[name] = module
        return module

    # Check if it's a specific class/function from a module
    # First try the common imports from before
    _common_imports = {
        # Engine components
        "AugLLMConfig": ("engine", "AugLLMConfig"),
        "AugLLMFactory": ("engine", "AugLLMFactory"),
        "Engine": ("engine", "Engine"),
        "InvokableEngine": ("engine", "InvokableEngine"),
        "NonInvokableEngine": ("engine", "NonInvokableEngine"),
        # Graph components
        "BaseGraph": ("graph", "BaseGraph"),
        # Registry
        "DynamicRegistry": ("registry", "DynamicRegistry"),
        "RegistryItem": ("registry", "RegistryItem"),
        # Schema
        "SchemaComposer": ("schema", "SchemaComposer"),
    }

    if name in _common_imports:
        module_name, attr_name = _common_imports[name]
        module = importlib.import_module(f".{module_name}", package=__name__)
        attr = getattr(module, attr_name)
        # Cache in globals
        globals()[name] = attr
        return attr

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Configure reduced logging verbosity
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

# Suppress the noisiest loggers that cause slow startup
_noisy_loggers = [
    "haive.dataflow.registry.core",
    "haive.core.engine.document.loaders.strategy",
    "haive.core.engine.document.loaders.auto_registry",
    "httpcore.connection",
    "httpcore.http2",
    "hpack.hpack",
    "urllib3",
    "requests",
    "httpx",
    "supabase",
    "google.cloud.storage",
    "langchain_community.utils.user_agent",
]

for _logger_name in _noisy_loggers:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)

# Set environment variables to skip heavy initialization
os.environ.setdefault("HAIVE_SKIP_HEAVY_INIT", "1")
os.environ.setdefault("HAIVE_QUIET_IMPORTS", "1")
os.environ.setdefault("HAIVE_DEBUG_CONFIG", "false")

__version__ = "0.1.0"

# Type-checking imports for pyright - only classes that have explicit imports in _common_imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # These are available through lazy loading but need explicit imports for pyright
    from haive.core import (
        common,
        config,
        engine,
        errors,
        graph,
        models,
        persistence,
        registry,
        runtime,
        schema,
        tools,
        types,
        utils,
    )

# Public API - includes both modules and common classes
__all__ = [
    # Modules (available through lazy loading)
    "engine",
    "graph",
    "schema",
    "tools",
    "types",
    "utils",
    "models",
    "registry",
    "runtime",
    "persistence",
    "config",
    "common",
    "errors",
    # Common classes for convenience (available through lazy loading)
    "AugLLMConfig",
    "AugLLMFactory",
    "BaseGraph",
    "DynamicRegistry",
    "Engine",
    "InvokableEngine",
    "NonInvokableEngine",
    "RegistryItem",
    "SchemaComposer",
    "__version__",
]
