"""🏗️ Prebuilt Schema Collection - Production-Ready AI State Blueprints

**THE ULTIMATE STATE TEMPLATE LIBRARY**

Welcome to the Prebuilt Schema Collection - a comprehensive library of battle-tested, 
production-ready state schemas that accelerate AI agent development. These aren't just 
simple templates; they're sophisticated, fully-featured state architectures designed 
for real-world AI applications.

⚡ INSTANT AI DEVELOPMENT
-------------------------

The Prebuilt Collection represents years of AI development experience distilled into 
reusable, extensible patterns. Every schema provides **enterprise-grade features** out of the box:

**🔧 Production-Ready Architecture**: Battle-tested patterns used in thousands of deployments
**📊 Built-in Analytics**: Token usage tracking, performance metrics, and cost optimization  
**🔄 Multi-Agent Support**: Sophisticated coordination patterns for agent collaboration
**⚡ Streaming Integration**: Real-time conversation and tool execution capabilities
**🎯 Type-Safe Design**: Full Pydantic v2 validation with intelligent error handling

🌟 SCHEMA CATEGORIES
--------------------

**1. Conversation Management** 💬
   Foundation schemas for conversational AI agents:
   ```python
   from haive.core.schema.prebuilt import MessagesState, TokenAwareState
   
   # Basic conversation with LangChain integration
   class ChatAgent(Agent):
       state_schema = MessagesState
       
       async def process_message(self, user_input: str):
           # Automatic message history management
           self.state.messages.append(HumanMessage(content=user_input))
           
           # LLM processing with built-in conversation tracking
           response = await self.engine.ainvoke(self.state.dict())
           
           # Automatic response tracking
           self.state.messages.append(AIMessage(content=response))
           return response
   
   # Token-aware conversation with cost tracking
   class CostOptimizedAgent(Agent):
       state_schema = TokenAwareState
       
       def get_conversation_cost(self) -> float:
           return self.state.calculate_total_cost()
       
       def optimize_token_usage(self):
           # Automatic conversation trimming to stay within budget
           self.state.trim_to_budget(max_cost=10.0)
   ```

**2. Tool-Enabled Agents** 🛠️
   Advanced schemas for agents with external capabilities:
   ```python
   from haive.core.schema.prebuilt import ToolState
   
   class ToolEnabledAgent(Agent):
       state_schema = ToolState
       
       tools = [web_search, calculator, file_manager]
       
       async def execute_task(self, task: str):
           # Automatic tool discovery and execution
           result = await self.engine.ainvoke({
               "messages": self.state.messages,
               "available_tools": self.tools,
               "task": task
           })
           
           # Built-in tool execution tracking
           tool_calls = self.state.get_tool_execution_history()
           return result, tool_calls
   ```

**3. Multi-Agent Orchestration** 🤝
   Sophisticated schemas for agent collaboration:
   ```python
   from haive.core.schema.prebuilt import MultiAgentState, MetaStateSchema
   
   class TeamCoordinator(Agent):
       state_schema = MultiAgentState
       
       agents = {
           "researcher": ResearchAgent(),
           "analyst": AnalysisAgent(),
           "writer": WritingAgent()
       }
       
       async def coordinate_team(self, project: Project):
           # Sophisticated agent coordination
           research = await self.delegate_to("researcher", project.research_brief)
           analysis = await self.delegate_to("analyst", research)
           report = await self.delegate_to("writer", analysis)
           
           # Automatic state synchronization across agents
           self.state.synchronize_agent_states()
           return report
   
   # Meta-capable agent that can embed other agents
   class MetaAgent(Agent):
       state_schema = MetaStateSchema
       
       def embed_specialist(self, specialist: Agent):
           # Dynamic agent embedding with state tracking
           self.state.embed_agent(specialist)
           return self.state.get_agent_view(specialist.name)
   ```

**4. Query Processing & RAG** 📚
   Specialized schemas for information retrieval and processing:
   ```python
   from haive.core.schema.prebuilt import RAGState, QueryState
   
   class RAGAgent(Agent):
       state_schema = RAGState
       
       vector_store = PineconeVectorStore()
       
       async def answer_query(self, query: str):
           # Automatic query analysis and optimization
           self.state.analyze_query(query)
           
           # Intelligent retrieval with relevance scoring
           documents = await self.state.retrieve_documents(
               query=query,
               vector_store=self.vector_store,
               strategy=RetrievalStrategy.SEMANTIC_HYBRID
           )
           
           # Context-aware response generation
           response = await self.engine.ainvoke({
               "query": query,
               "context": documents,
               "conversation_history": self.state.messages
           })
           
           return response
   ```

🎯 ADVANCED FEATURES
--------------------

**Dynamic State Activation** 🔮
```python
from haive.core.schema.prebuilt import DynamicActivationState

class AdaptiveAgent(Agent):
    state_schema = DynamicActivationState
    
    async def process_input(self, input_data: Any):
        # Automatic capability detection and activation
        capabilities = self.state.detect_required_capabilities(input_data)
        
        # Dynamic schema evolution
        for capability in capabilities:
            self.state.activate_capability(capability)
        
        # Execute with expanded capabilities
        return await self.engine.ainvoke(input_data)
```

**Enhanced Multi-Agent Patterns** 🚀
```python
from haive.core.schema.prebuilt import EnhancedMultiAgentState

class EnhancedTeam(Agent):
    state_schema = EnhancedMultiAgentState
    
    def setup_collaboration_patterns(self):
        # Advanced coordination patterns
        self.state.configure_patterns([
            CollaborationPattern.HIERARCHICAL,
            CollaborationPattern.PEER_TO_PEER,
            CollaborationPattern.CONSENSUS_DRIVEN
        ])
        
        # Intelligent workload distribution
        self.state.enable_load_balancing(strategy="capability_based")
        
        # Real-time conflict resolution
        self.state.configure_conflict_resolution("semantic_merge")
```

**Token Economics & Cost Optimization** 💰
```python
from haive.core.schema.prebuilt import TokenUsage, TokenUsageMixin

class CostOptimizedState(ToolState, TokenUsageMixin):
    # Automatic cost tracking across all operations
    def get_detailed_costs(self) -> CostBreakdown:
        return CostBreakdown(
            llm_costs=self.calculate_llm_costs(),
            tool_costs=self.calculate_tool_costs(),
            storage_costs=self.calculate_storage_costs(),
            total=self.get_total_cost()
        )
    
    def optimize_for_budget(self, budget: float):
        # Intelligent budget management
        if self.get_total_cost() > budget * 0.8:
            self.enable_cost_optimization_mode()
            self.trim_conversation_history(keep_recent=10)
            self.use_cheaper_models_for_simple_tasks()
```

🏗️ STATE COMPOSITION PATTERNS
------------------------------

**Layered Architecture** 🏛️
```python
# Build complex states from simple components
class ComprehensiveAgentState(
    MessagesState,           # Conversation management
    ToolState,               # Tool capabilities  
    TokenUsageMixin,         # Cost tracking
    DynamicActivationState   # Adaptive capabilities
):
    # Automatic feature composition
    pass

# Usage with full feature set
agent = Agent(state_schema=ComprehensiveAgentState)
# Agent now has: messaging, tools, cost tracking, and dynamic capabilities
```

**Specialized Extensions** 🎨
```python
class CustomRAGState(RAGState):
    # Extend prebuilt patterns with domain-specific features
    domain_knowledge: DomainKnowledgeGraph = Field(default_factory=DomainKnowledgeGraph)
    specialized_retrievers: Dict[str, Retriever] = Field(default_factory=dict)
    
    def add_domain_expertise(self, domain: str, knowledge: KnowledgeBase):
        self.domain_knowledge.integrate(domain, knowledge)
        self.specialized_retrievers[domain] = knowledge.create_retriever()
```

📊 PRODUCTION FEATURES
----------------------

**Performance Monitoring** 📈
All prebuilt schemas include comprehensive monitoring:
- Token usage tracking and cost analysis
- Conversation length and quality metrics
- Tool execution performance and success rates
- Memory usage and optimization recommendations
- Real-time error tracking and recovery

**Enterprise Integration** 🏢
- Audit logging for compliance requirements
- Multi-tenant state isolation
- Role-based access control
- Data retention policy enforcement
- Backup and disaster recovery

**Scalability Features** ⚡
- Automatic state compression for large conversations
- Lazy loading of historical data
- Distributed state synchronization
- Load balancing across agent instances
- Horizontal scaling support

🎓 BEST PRACTICES
-----------------

1. **Start with Prebuilt**: Use existing schemas as foundation
2. **Compose Don't Duplicate**: Combine schemas using mixins
3. **Monitor Costs**: Always include token tracking in production
4. **Plan for Scale**: Use multi-agent patterns for complex workflows
5. **Validate Early**: Test state schemas with real data
6. **Document Extensions**: Clear documentation for custom fields
7. **Version Carefully**: Use migration strategies for schema changes

🚀 GETTING STARTED
------------------

```python
from haive.core.schema.prebuilt import (
    MessagesState, ToolState, RAGState, MultiAgentState,
    TokenAwareState, MetaStateSchema
)

# 1. Choose the right prebuilt schema for your use case
class MyAgent(Agent):
    # For simple conversation: MessagesState
    # For tool usage: ToolState  
    # For RAG: RAGState
    # For teams: MultiAgentState
    # For cost tracking: TokenAwareState
    # For meta-capabilities: MetaStateSchema
    state_schema = ToolState  # Example choice

# 2. Customize if needed
class MyCustomState(ToolState):
    custom_field: str = Field(default="")

# 3. Use with full features
agent = MyAgent()
await agent.process("Hello with tools and cost tracking!")
```

🌟 SCHEMA GALLERY
-----------------

**Available Schemas**:
- `MessagesState` - Basic conversation management
- `ToolState` - Tool-enabled agent state
- `RAGState` - Retrieval-augmented generation
- `QueryState` - Advanced query processing
- `MultiAgentState` - Agent coordination
- `MetaStateSchema` - Meta-agent capabilities
- `TokenAwareState` - Cost tracking and optimization
- `DynamicActivationState` - Adaptive capabilities
- `EnhancedMultiAgentState` - Advanced collaboration
- `LLMState` - Single-engine agent state

**Token & Cost Management**:
- `TokenUsage` - Token tracking utilities
- `TokenUsageMixin` - Add token tracking to any schema
- `calculate_token_cost()` - Cost calculation functions
- `aggregate_token_usage()` - Usage aggregation

---

**Prebuilt Schema Collection: Where Enterprise AI Development Starts** 🏗️"""

from haive.core.schema.multi_agent_state_schema import MultiAgentStateSchema

# from haive.core.schema.prebuilt.basic_agent_state import BasicAgentState
from haive.core.schema.prebuilt.dynamic_activation_state import DynamicActivationState
from haive.core.schema.prebuilt.enhanced_multi_agent_state import (
    EnhancedMultiAgentState,
)
from haive.core.schema.prebuilt.llm_state import LLMState

# Import messages module components
from haive.core.schema.prebuilt.messages import (
    MessagesStateWithTokenUsage,
    TokenUsage,
    TokenUsageMixin,
    aggregate_token_usage,
    calculate_token_cost,
    extract_token_usage_from_message,
)
from haive.core.schema.prebuilt.messages_state import MessagesState
from haive.core.schema.prebuilt.meta_state import MetaStateSchema
from haive.core.schema.prebuilt.multi_agent_state import MultiAgentState
from haive.core.schema.prebuilt.query_state import (
    QueryComplexity,
    QueryIntent,
    QueryMetrics,
    QueryProcessingConfig,
    QueryProcessingState,
    QueryResult,
    QueryState,
    QueryType,
    RetrievalStrategy,
)
from haive.core.schema.prebuilt.rag_state import RAGState
from haive.core.schema.prebuilt.tool_state import ToolState

# Document state components are imported lazily to avoid triggering document system auto-registry
# from haive.core.schema.prebuilt.document_state import (
#     DocumentEngineInputSchema,
#     DocumentEngineOutputSchema,
#     DocumentState,
# )


# Convenient aliases
TokenAwareState = MessagesStateWithTokenUsage  # Shorter name
TokenToolState = ToolState  # Makes it clear it has token tracking
AgentState = LLMState  # Generic agent state with single engine

# Lazy loading for document components to avoid auto-registry initialization
_DOCUMENT_COMPONENTS = {
    "DocumentState",
    "DocumentEngineInputSchema",
    "DocumentEngineOutputSchema",
}


def __getattr__(name: str):
    """Lazy loading for document state components."""
    if name in _DOCUMENT_COMPONENTS:
        return locals()[name]

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "AgentState",
    "DocumentEngineInputSchema",
    "DocumentEngineOutputSchema",
    # Document and query schemas - lazy loaded via __getattr__
    "DocumentState",
    # Core prebuilt schemas
    # "BasicAgentState",
    "DynamicActivationState",
    "EnhancedMultiAgentState",
    "LLMState",
    "MessagesState",
    "MessagesStateWithTokenUsage",
    "MetaStateSchema",
    "MultiAgentState",
    "MultiAgentStateSchema",
    "QueryComplexity",
    "QueryIntent",
    "QueryMetrics",
    "QueryProcessingConfig",
    "QueryProcessingState",
    "QueryResult",
    "QueryState",
    "QueryType",
    "RAGState",
    "RetrievalStrategy",
    # Aliases
    "TokenAwareState",
    "TokenToolState",
    # Token usage components
    "TokenUsage",
    "TokenUsageMixin",
    "ToolState",
    "aggregate_token_usage",
    "calculate_token_cost",
    # Token usage utilities
    "extract_token_usage_from_message",
]
