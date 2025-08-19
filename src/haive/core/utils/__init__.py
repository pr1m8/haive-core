"""🛠️ Utils Module - Intelligent Utility Toolkit Revolution

**THE SWISS ARMY KNIFE OF AI DEVELOPMENT EXCELLENCE**

Welcome to the Utils Module - the revolutionary utility intelligence platform 
that transforms mundane utility functions into a sophisticated development 
acceleration system. This isn't just another utils library; it's an intelligent 
toolkit consciousness that anticipates developer needs, automates repetitive 
tasks, and enhances productivity through smart automation and adaptive helpers.

⚡ REVOLUTIONARY UTILITY INTELLIGENCE
------------------------------------

The Utils Module represents a paradigm shift from static utility functions to 
**intelligent, adaptive development accelerators** that evolve with your workflow:

**🧠 Intelligent Tool Discovery**: Auto-discovery and introspection of framework components
**🔄 Adaptive Pydantic Integration**: Smart model manipulation with type safety
**⚡ OpenAI-Compliant Naming**: Intelligent name sanitization and validation
**📊 Development Workflow Enhancement**: Productivity-boosting utilities and helpers
**🎯 Framework Integration**: Seamless integration utilities for all Haive components

🌟 CORE UTILITY INNOVATIONS
---------------------------

**1. Intelligent Pydantic Utilities** 🚀
   Revolutionary model manipulation that thinks and adapts:
   ```python
   from haive.core.utils import pydantic_utils
   from haive.core.utils.pydantic_utils import PydanticModelAnalyzer
   
   # Intelligent model serialization with context awareness
   class AgentConfig(BaseModel):
       name: str
       temperature: float
       tools: List[str]
       metadata: Dict[str, Any]
   
   config = AgentConfig(
       name="intelligent_agent",
       temperature=0.7,
       tools=["calculator", "web_search"],
       metadata={"version": "2.0", "type": "research"}
   )
   
   # Smart serialization with intelligent filtering
   serialized = pydantic_utils.model_to_dict(
       config,
       exclude_sensitive=True,
       include_computed=True,
       optimization_mode="api_ready"
   )
   
   # Intelligent model analysis and optimization
   analyzer = PydanticModelAnalyzer()
   analysis = analyzer.analyze_model_structure(AgentConfig)
   
   print(f"Model complexity: {analysis.complexity_score}")
   print(f"Serialization efficiency: {analysis.serialization_efficiency}")
   print(f"Optimization suggestions: {analysis.optimization_suggestions}")
   
   # Auto-optimize model for specific use cases
   optimized_model = analyzer.optimize_for_use_case(
       AgentConfig,
       use_case="high_frequency_api",
       performance_target="sub_millisecond"
   )
   
   # Intelligent model merging and transformation
   merged_config = pydantic_utils.intelligent_merge_models(
       base_model=config,
       override_model=production_overrides,
       merge_strategy="performance_optimized",
       conflict_resolution="weighted_priority"
   )
   ```

For complete examples and advanced patterns, see the documentation.
"""

# Import key naming utilities at package level for convenience
from haive.core.utils.naming import create_openai_compliant_name, sanitize_tool_name, validate_tool_name

__all__ = ["create_openai_compliant_name", "sanitize_tool_name", "validate_tool_name"]
