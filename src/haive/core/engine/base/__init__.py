"""🏗️ Base Engine System - The Foundation of AI Component Architecture

**THE ARCHITECTURAL BEDROCK OF INTELLIGENT SYSTEMS**

Welcome to the Base Engine System - the fundamental abstraction layer that powers 
every AI component in the Haive framework. This isn't just another factory pattern; 
it's a revolutionary architecture that makes AI components configurable, composable, 
and universally interoperable.

🧬 ARCHITECTURAL PHILOSOPHY
---------------------------

The Base Engine System represents a paradigm shift in AI architecture design. Every 
AI capability - from language models to vector stores, from tools to retrievers - 
is built on the same foundational abstractions, enabling:

**🔧 Universal Configuration**: Every AI component follows the same configuration pattern
**⚡ Lazy Instantiation**: Components are created only when needed for optimal performance  
**🔄 Runtime Flexibility**: Switch implementations without changing code
**📦 Serializable State**: Configurations can be saved, loaded, and shared
**🌐 Cross-Platform Compatibility**: Run anywhere with consistent behavior

🎯 CORE ABSTRACTIONS
--------------------

**1. The Engine Pattern** 🚀
   The Engine is the heart of every AI component:

Examples:
    >>> # Engines are lightweight, serializable configurations
    >>> class MyEngine(Engine):
    >>> model_name: str = "gpt-4"
    >>> temperature: float = 0.7
    >>>
    >>> def create_runnable(self) -> Runnable:
    >>> \"\"\"Create the actual runtime component.\"\"\"
    >>> return ChatOpenAI(
    >>> model=self.model_name,
    >>> temperature=self.temperature
    >>> )
    >>>
    >>> # Configuration is separate from runtime
    >>> config = MyEngine(temperature=0.3)  # Lightweight
    >>> llm = config.create_runnable()       # Heavy runtime component

**2. Invokable Protocol** 🎯
   Universal interface for all AI components:

    >>> from haive.core.engine.base import Invokable, AsyncInvokable
    >>>
    >>> # Every component implements the same interface
    >>> def use_any_component(component: Invokable):
    >>> result = component.invoke("Hello, world!")
    >>> return result
    >>>
    >>> # Works with any Haive component
    >>> use_any_component(llm_engine.create_runnable())
    >>> use_any_component(retriever_engine.create_runnable())
    >>> use_any_component(tool_engine.create_runnable())

**3. Engine Registry** 📚
   Central registry for all AI components:

    >>> from haive.core.engine.base import EngineRegistry
    >>>
    >>> # Register custom engines
    >>> @EngineRegistry.register("my_custom_llm")
    >>> class CustomLLMEngine(InvokableEngine):
    >>> def create_runnable(self):
    >>> return MyCustomLLM()
    >>>
    >>> # Create from registry anywhere
    >>> engine = EngineRegistry.get("my_custom_llm")
    >>> component = engine.create_runnable()
    >>>
    >>> # List all available engines
    >>> available = EngineRegistry.list_engines()
    >>> print(f"Available engines: {available}")

🏗️ ARCHITECTURAL PATTERNS
--------------------------

**Configuration vs Runtime Separation** 🎭

    >>> # ✅ CORRECT: Lightweight configuration
    >>> class VectorStoreEngine(Engine):
    >>> provider: str = "pinecone"
    >>> index_name: str = "default"
    >>> dimension: int = 1536
    >>>
    >>> # Serializable, shareable, version-controlled
    >>> def to_dict(self) -> dict:
    >>> return {"provider": self.provider, "index_name": self.index_name}
    >>>
    >>> # ✅ CORRECT: Heavy runtime component created on demand
    >>> def create_runtime_component():
    >>> config = VectorStoreEngine(provider="weaviate")
    >>> vectorstore = config.create_runnable()  # Only create when needed
    >>> return vectorstore
    >>>
    >>> # ❌ WRONG: Mixing configuration with runtime
    >>> class BadVectorStore:
    >>> def __init__(self):
    >>> self.client = PineconeClient()  # Heavy initialization upfront
    >>> self.connection = connect_to_db()  # Cannot serialize this

**Factory Pattern with Validation** ✅

    >>> from haive.core.engine.base import ComponentFactory
    >>> from pydantic import Field, validator
    >>>
    >>> class LLMEngine(InvokableEngine):
    >>> \"\"\"Type-safe LLM configuration with validation.\"\"\"
    >>>
    >>> model: str = Field(..., description="Model name")
    >>> temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    >>> max_tokens: Optional[int] = Field(None, gt=0, description="Maximum tokens")
    >>>
    >>> @validator("model")
    >>> def validate_model(cls, v):
    >>> supported_models = ["gpt-4", "claude-3", "gemini-pro"]
    >>> if v not in supported_models:
    >>> raise ValueError(f"Model {v} not supported. Use: {supported_models}")
    >>> return v
    >>>
    >>> def create_runnable(self) -> ChatLLM:
    >>> \"\"\"Create validated runtime component.\"\"\"
    >>> return ComponentFactory.create_llm(
    >>> model=self.model,
    >>> temperature=self.temperature,
    >>> max_tokens=self.max_tokens
    >>> )

**Component References** 🔗

    >>> from haive.core.engine.base import ComponentRef
    >>>
    >>> class CompositeEngine(Engine):
    >>> \"\"\"Engine that composes multiple components.\"\"\"
    >>>
    >>> # References to other engines (lazy loading)
    >>> llm_ref: ComponentRef[LLMEngine] = ComponentRef("llm_engine")
    >>> retriever_ref: ComponentRef[RetrieverEngine] = ComponentRef("retriever_engine")
    >>>
    >>> def create_runnable(self) -> CompositeRunnable:
    >>> \"\"\"Create composite component from references.\"\"\"
    >>> llm = self.llm_ref.resolve().create_runnable()
    >>> retriever = self.retriever_ref.resolve().create_runnable()
    >>>
    >>> return CompositeRunnable(llm=llm, retriever=retriever)

🚀 ADVANCED FEATURES
--------------------

**Dynamic Engine Creation** 🔮

    >>> # Create engines dynamically from configuration
    >>> def create_engine_from_config(config: dict) -> Engine:
    >>> engine_type = config.pop("type")
    >>> engine_class = EngineRegistry.get_class(engine_type)
    >>> return engine_class(**config)
    >>>
    >>> # Load from JSON/YAML
    >>> config = {
    >>> "type": "llm_engine",
    >>> "model": "gpt-4",
    >>> "temperature": 0.7,
    >>> "tools": ["calculator", "web_search"]
    >>> }
    >>>
    >>> engine = create_engine_from_config(config)
    >>> component = engine.create_runnable()

**Engine Composition** 🧩

    >>> from haive.core.engine.base import compose_engines
    >>>
    >>> # Compose multiple engines into workflows
    >>> research_pipeline = compose_engines([
    >>> LLMEngine(model="gpt-4", system_message="You are a researcher"),
    >>> RetrieverEngine(provider="pinecone", index="knowledge_base"),
    >>> LLMEngine(model="claude-3", system_message="You are a synthesizer")
    >>> ])
    >>>
    >>> # Execute composed workflow
    >>> result = research_pipeline.invoke("Research quantum computing")

**Engine Middleware** 🔄

    >>> from haive.core.engine.base import EngineMiddleware
    >>>
    >>> # Add capabilities to any engine
    >>> class CachingMiddleware(EngineMiddleware):
    >>> def wrap_runnable(self, runnable: Runnable) -> Runnable:
    >>> return CachedRunnable(runnable, ttl=3600)
    >>>
    >>> class RetryMiddleware(EngineMiddleware):
    >>> def wrap_runnable(self, runnable: Runnable) -> Runnable:
    >>> return RetryRunnable(runnable, max_attempts=3)
    >>>
    >>> # Apply middleware to engines
    >>> enhanced_engine = base_engine.with_middleware([
    >>> CachingMiddleware(),
    >>> RetryMiddleware(),
    >>> ])

🎯 TYPE SYSTEM
--------------

**Engine Types** 📝

    >>> from haive.core.engine.base import EngineType
    >>>
    >>> # All supported engine types
    >>> engine_types = [
    >>> EngineType.LLM,           # Language models
    >>> EngineType.RETRIEVER,     # Document retrieval
    >>> EngineType.VECTORSTORE,   # Vector databases
    >>> EngineType.TOOL,          # Function calling tools
    >>> EngineType.EMBEDDING,     # Embedding models
    >>> EngineType.PARSER,        # Output parsers
    >>> EngineType.MEMORY,        # Memory systems
    >>> EngineType.AGENT,         # Intelligent agents
    >>> ]
    >>>
    >>> # Type-safe engine creation
    >>> def create_typed_engine(engine_type: EngineType, **kwargs) -> Engine:
    >>> return EngineRegistry.create(engine_type, **kwargs)

**Protocol Compliance** ✅

    >>> from typing import Protocol
    >>>
    >>> class AIComponent(Protocol):
    >>> \"\"\"Protocol for all AI components.\"\"\"
    >>>
    >>> def invoke(self, input_data: Any) -> Any: ...
    >>> async def ainvoke(self, input_data: Any) -> Any: ...
    >>> def stream(self, input_data: Any) -> Iterator[Any]: ...
    >>> async def astream(self, input_data: Any) -> AsyncIterator[Any]: ...
    >>>
    >>> # All Haive components implement this protocol automatically
    >>> def process_with_any_component(component: AIComponent, data: Any):
    >>> return component.invoke(data)

🔄 LIFECYCLE MANAGEMENT
-----------------------

**Engine Lifecycle** 🔄

    >>> class ManagedEngine(InvokableEngine):
    >>> \"\"\"Engine with full lifecycle management.\"\"\"
    >>>
    >>> def __init__(self, **kwargs):
    >>> super().__init__(**kwargs)
    >>> self._runnable = None
    >>> self._is_initialized = False
    >>>
    >>> def create_runnable(self) -> Runnable:
    >>> \"\"\"Lazy initialization with caching.\"\"\"
    >>> if not self._is_initialized:
    >>> self._runnable = self._initialize_component()
    >>> self._is_initialized = True
    >>> return self._runnable
    >>>
    >>> def __enter__(self):
    >>> \"\"\"Context manager support.\"\"\"
    >>> return self.create_runnable()
    >>>
    >>> def __exit__(self, exc_type, exc_val, exc_tb):
    >>> \"\"\"Cleanup resources.\"\"\"
    >>> if self._runnable and hasattr(self._runnable, "close"):
    >>> self._runnable.close()
    >>> self._is_initialized = False

**Resource Management** 🛡️

    >>> # Automatic resource cleanup
    >>> with ManagedEngine(model="gpt-4") as llm:
    >>> result = llm.invoke("Hello, world!")
    >>> # Automatically cleaned up after context
    >>>
    >>> # Batch processing with resource pooling
    >>> engine_pool = EnginePool([
    >>> LLMEngine(model="gpt-4"),
    >>> LLMEngine(model="claude-3"),
    >>> LLMEngine(model="gemini-pro")
    >>> ])
    >>>
    >>> async def process_batch(inputs: List[str]):
    >>> tasks = []
    >>> for input_data in inputs:
    >>> engine = await engine_pool.acquire()
    >>> task = engine.ainvoke(input_data)
    >>> tasks.append(task)
    >>>
    >>> results = await asyncio.gather(*tasks)
    >>> await engine_pool.release_all()
    >>> return results

🎨 CUSTOMIZATION PATTERNS
-------------------------

**Custom Engine Development** 🛠️

    >>> from haive.core.engine.base import InvokableEngine
    >>>
    >>> class MyCustomEngine(InvokableEngine):
    >>> \"\"\"Custom engine with unique capabilities.\"\"\"
    >>>
    >>> # Configuration fields with validation
    >>> api_key: str = Field(..., description="API key for external service")
    >>> endpoint: str = Field("https://api.example.com", description="Service endpoint")
    >>> timeout: float = Field(30.0, gt=0, description="Request timeout")
    >>>
    >>> def create_runnable(self) -> MyCustomRunnable:
    >>> \"\"\"Create runtime component with custom logic.\"\"\"
    >>> return MyCustomRunnable(
    >>> api_key=self.api_key,
    >>> endpoint=self.endpoint,
    >>> timeout=self.timeout
    >>> )
    >>>
    >>> # Optional: Custom validation
    >>> @validator("api_key")
    >>> def validate_api_key(cls, v):
    >>> if not v.startswith("sk-"):
    >>> raise ValueError("API key must start with 'sk-'")
    >>> return v
    >>>
    >>> # Register for automatic discovery
    >>> EngineRegistry.register("my_custom_engine", MyCustomEngine)

**Engine Inheritance** 🧬

    >>> class BaseLLMEngine(InvokableEngine):
    >>> \"\"\"Base class for all LLM engines.\"\"\"
    >>>
    >>> model: str = Field(..., description="Model identifier")
    >>> temperature: float = Field(0.7, ge=0.0, le=2.0)
    >>>
    >>> def _create_base_llm(self) -> BaseLLM:
    >>> \"\"\"Override in subclasses.\"\"\"
    >>> raise NotImplementedError
    >>>
    >>> class OpenAIEngine(BaseLLMEngine):
    >>> \"\"\"OpenAI-specific implementation.\"\"\"
    >>>
    >>> api_key: str = Field(..., description="OpenAI API key")
    >>>
    >>> def create_runnable(self) -> ChatOpenAI:
    >>> return ChatOpenAI(
    >>> model=self.model,
    >>> temperature=self.temperature,
    >>> api_key=self.api_key
    >>> )
    >>>
    >>> class AnthropicEngine(BaseLLMEngine):
    >>> \"\"\"Anthropic-specific implementation.\"\"\"
    >>>
    >>> api_key: str = Field(..., description="Anthropic API key")
    >>>
    >>> def create_runnable(self) -> ChatAnthropic:
    >>> return ChatAnthropic(
    >>> model=self.model,
    >>> temperature=self.temperature,
    >>> api_key=self.api_key
    >>> )

📊 MONITORING & OBSERVABILITY
------------------------------

**Engine Telemetry** 📡

    >>> class InstrumentedEngine(InvokableEngine):
    >>> \"\"\"Engine with built-in observability.\"\"\"
    >>>
    >>> def create_runnable(self) -> InstrumentedRunnable:
    >>> base_runnable = self._create_base_runnable()
    >>> return InstrumentedRunnable(
    >>> runnable=base_runnable,
    >>> metrics_collector=self._create_metrics_collector(),
    >>> tracer=self._create_tracer()
    >>> )
    >>>
    >>> def get_metrics(self) -> EngineMetrics:
    >>> \"\"\"Get real-time metrics for this engine.\"\"\"
    >>> return EngineMetrics(
    >>> invocation_count=self._metrics.total_invocations,
    >>> avg_latency=self._metrics.average_latency,
    >>> error_rate=self._metrics.error_rate,
    >>> last_used=self._metrics.last_invocation_time
    >>> )

🔒 ENTERPRISE FEATURES
----------------------

- **Multi-tenancy**: Isolated engine instances per tenant
- **Access Control**: Role-based permissions for engine types
- **Audit Logging**: Complete tracking of engine usage
- **Configuration Management**: Centralized config with versioning
- **Health Monitoring**: Real-time engine health checks
- **Failover**: Automatic switching to backup engines

🎓 BEST PRACTICES
-----------------

1. **Separate Configuration from Runtime**: Keep engines lightweight and serializable
2. **Use Type Hints**: Enable IDE support and runtime validation
3. **Implement Protocols**: Ensure compatibility with the broader ecosystem
4. **Lazy Loading**: Create expensive components only when needed
5. **Resource Management**: Always clean up connections and clients
6. **Error Handling**: Implement robust error recovery patterns
7. **Testing**: Use engine configurations for consistent test environments

🚀 GETTING STARTED
------------------

    >>> from haive.core.engine.base import InvokableEngine, EngineRegistry
    >>> from pydantic import Field
    >>>
    >>> # 1. Define your engine
    >>> class SimpleEngine(InvokableEngine):
    >>> message: str = Field(default="Hello")
    >>>
    >>> def create_runnable(self):
    >>> return lambda x: f"{self.message}, {x}!"
    >>>
    >>> # 2. Register it
    >>> EngineRegistry.register("simple", SimpleEngine)
    >>>
    >>> # 3. Use it anywhere
    >>> engine = EngineRegistry.get("simple")
    >>> component = engine.create_runnable()
    >>> result = component.invoke("World")  # "Hello, World!"

---

**Base Engine System: The Foundation Where AI Components Begin** 🏗️
"""

from haive.core.engine.base.base import Engine, InvokableEngine, NonInvokableEngine
from haive.core.engine.base.factory import ComponentFactory
from haive.core.engine.base.protocols import AsyncInvokable, Invokable
from haive.core.engine.base.reference import ComponentRef
from haive.core.engine.base.registry import EngineRegistry
from haive.core.engine.base.types import EngineType

__all__ = [
    "AsyncInvokable",
    # Factory and reference patterns
    "ComponentFactory",
    "ComponentRef",
    # Base engine classes
    "Engine",
    # Registry and type system
    "EngineRegistry",
    "EngineType",
    # Protocols
    "Invokable",
    "InvokableEngine",
    "NonInvokableEngine",
]
