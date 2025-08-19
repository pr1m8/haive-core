"""🧩 Mixins - Intelligent Component Superpowers System

**THE MOLECULAR BUILDING BLOCKS OF AI EXCELLENCE**

Welcome to Mixins - the revolutionary collection of intelligent, composable 
behaviors that transform ordinary classes into extraordinary AI components. 
This isn't just multiple inheritance; it's a sophisticated composition system 
where every mixin is a specialized capability that makes your components 
smarter, more reliable, and enterprise-ready by default.

⚡ REVOLUTIONARY MIXIN INTELLIGENCE
----------------------------------

Mixins represent a paradigm shift from manual feature implementation to 
**intelligent capability composition** where sophisticated behaviors are 
injected seamlessly into any class:

**🧠 Self-Configuring Behaviors**: Mixins that automatically adapt to their host class
**🔄 Zero-Conflict Composition**: Intelligent inheritance resolution and method chaining  
**⚡ Performance Optimization**: Built-in caching, lazy loading, and resource management
**📊 Enterprise-Grade Observability**: Automatic logging, metrics, and monitoring
**🎯 Type-Safe Integration**: Full Pydantic compatibility with intelligent field merging

🌟 CORE MIXIN CATEGORIES
------------------------

**1. Identity & Lifecycle Mixins** 🆔
   Fundamental behaviors for object identity and lifecycle management:
   ```python
   from haive.core.common.mixins import (
       IdentifierMixin, TimestampMixin, VersionMixin, MetadataMixin
   )
   
   class IntelligentAgent(
       IdentifierMixin,     # Unique IDs with collision detection
       TimestampMixin,      # Created/updated/accessed tracking
       VersionMixin,        # Semantic versioning with migrations
       MetadataMixin        # Rich metadata with indexing
   ):
       def __init__(self, name: str):
           super().__init__()
           self.name = name
           # Automatic capabilities:
           # - self.id: Unique identifier (UUID with prefix)
           # - self.created_at: ISO timestamp of creation
           # - self.version: Semantic version ("1.0.0")
           # - self.metadata: Indexed metadata storage
   
   # Enhanced instantiation
   agent = IntelligentAgent("research_assistant")
   assert agent.id.startswith("agent_")  # Automatic prefixing
   assert agent.created_at <= datetime.now()  # Timestamp validation
   assert agent.version == "1.0.0"  # Default version
   ```

**2. State Management Mixins** 🗄️
   Advanced state handling with intelligent persistence:
   ```python
   from haive.core.common.mixins import (
       StateMixin, StateInterfaceMixin, CheckpointerMixin
   )
   
   class StatefulProcessor(
       StateMixin,           # Core state management
       StateInterfaceMixin,  # Advanced state operations
       CheckpointerMixin     # Automatic checkpointing
   ):
       def __init__(self):
           super().__init__()
           # Automatic capabilities:
           # - State validation and serialization
           # - Automatic dirty tracking
           # - Checkpoint creation and restoration
           # - State migration support
       
       def process(self, data):
           # State automatically tracked
           self.state.update({"last_processed": data})
           
           # Automatic checkpoint creation
           if self.should_checkpoint():
               self.create_checkpoint("pre_processing")
           
           result = complex_processing(data)
           
           # State automatically persisted
           self.state.finalize_update()
           return result
   ```

For complete examples and advanced patterns, see the documentation.
"""

# Import main mixins
from haive.core.common.mixins.checkpointer_mixin import CheckpointerMixin
from haive.core.common.mixins.engine_mixin import EngineStateMixin as EngineMixin

# Import general mixins
from haive.core.common.mixins.general import (
    IdMixin,
    MetadataMixin,
    SerializationMixin,
    StateMixin,
    TimestampMixin,
    VersionMixin,
)
from haive.core.common.mixins.getter_mixin import GetterMixin
from haive.core.common.mixins.identifier import IdentifierMixin
from haive.core.common.mixins.mcp_mixin import MCPMixin
from haive.core.common.mixins.rich_logger_mixin import RichLoggerMixin
from haive.core.common.mixins.secure_config import SecureConfigMixin
from haive.core.common.mixins.state_interface_mixin import StateInterfaceMixin
from haive.core.common.mixins.structured_output_mixin import StructuredOutputMixin
from haive.core.common.mixins.tool_list_mixin import ToolListMixin
from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin

__all__ = [
    # Main mixins
    "CheckpointerMixin",
    "EngineMixin",
    "GetterMixin",
    # General mixins
    "IdMixin",
    "IdentifierMixin",
    "MCPMixin",
    "MetadataMixin",
    "RichLoggerMixin",
    "SecureConfigMixin",
    "SerializationMixin",
    "StateInterfaceMixin",
    "StateMixin",
    "StructuredOutputMixin",
    "TimestampMixin",
    "ToolListMixin",
    "ToolRouteMixin",
    "VersionMixin",
]
