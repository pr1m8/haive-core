"""🚀 AugLLM - Augmented Language Model System

**THE ULTIMATE LLM ENHANCEMENT ENGINE**

Welcome to AugLLM - the revolutionary system that transforms ordinary language models 
into supercharged AI powerhouses. This isn't just another LLM wrapper; it's a 
comprehensive enhancement platform that adds tools, structured output, memory, 
and advanced reasoning capabilities to any language model.

🧠 AUGMENTATION REVOLUTION
--------------------------

AugLLM represents a paradigm shift in how we interact with language models. Every
LLM becomes a **versatile reasoning engine** capable of:

**🛠️ Tool Integration**: Connect any API, database, or function as a native capability
**📊 Structured Output**: Guarantee type-safe, validated responses every time  
**🧩 Dynamic Composition**: Chain, merge, and orchestrate multiple LLMs seamlessly
**⚡ Real-time Streaming**: Stream tokens, tool calls, and structured data in real-time
**🎯 Precision Control**: Fine-tune behavior with advanced configuration options

🌟 CORE INNOVATIONS
-------------------

**1. Universal LLM Enhancement** 🔮
   Transform any language model into an intelligent agent:
   ```python
   # Basic LLM
   basic_llm = ChatOpenAI(model="gpt-4")
   
   # Augmented LLM with superpowers
   aug_llm = AugLLMConfig(
       model="gpt-4",
       tools=[web_search, calculator, code_executor],
       structured_output_model=AnalysisResult,
       system_message="You are an expert analyst with tool access.",
       temperature=0.7,
       streaming=True
   )
   ```

**2. Intelligent Tool Orchestration** 🛠️
   Tools become natural extensions of language model capabilities:
   ```python
   @tool
   def stock_analyzer(symbol: str) -> StockAnalysis:
       \"\"\"Analyze stock performance and trends.\"\"\"
       return analyze_stock_data(symbol)
   
   @tool  
   def news_fetcher(query: str) -> List[NewsArticle]:
       \"\"\"Fetch latest news articles.\"\"\"
       return get_news(query)
   
   # LLM automatically uses tools when needed
   config = AugLLMConfig(
       model="gpt-4",
       tools=[stock_analyzer, news_fetcher],
       tool_choice="auto"  # Intelligent tool selection
   )
   ```

**3. Guaranteed Structured Output** 📋
   Never parse unstructured text again:
   ```python
   from pydantic import BaseModel, Field
   from typing import List, Optional
   
   class MarketAnalysis(BaseModel):
       \"\"\"Comprehensive market analysis structure.\"\"\"
       sentiment: str = Field(description="Market sentiment: bullish/bearish/neutral")
       confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
       key_factors: List[str] = Field(description="Main market drivers")
       price_prediction: Optional[float] = Field(description="Predicted price movement")
       risk_level: str = Field(description="Risk assessment: low/medium/high")
   
   # Guaranteed structured output every time
   analyst = AugLLMConfig(
       model="gpt-4",
       structured_output_model=MarketAnalysis,
       tools=[stock_analyzer, news_fetcher]
   )
   
   # Type-safe results
   result: MarketAnalysis = analyst.invoke("Analyze AAPL stock")
   print(f"Sentiment: {result.sentiment}, Confidence: {result.confidence}")
   ```

**4. Dynamic Chain Composition** 🔗
   Build complex workflows from simple components:
   ```python
   # Research pipeline
   researcher = AugLLMConfig(model="gpt-4", tools=[web_search, arxiv_search])
   analyst = AugLLMConfig(model="gpt-4", structured_output_model=Analysis)
   writer = AugLLMConfig(model="gpt-4", temperature=0.8)
   
   # Compose into intelligent pipeline
   research_pipeline = compose_runnable([
       researcher.with_system_message("Research the topic thoroughly"),
       analyst.with_system_message("Analyze the research findings"),
       writer.with_system_message("Write a comprehensive report")
   ])
   
   # Execute end-to-end
   report = research_pipeline.invoke("Latest advances in quantum computing")
   ```

🎯 ADVANCED FEATURES
--------------------

**Real-time Streaming** 🌊
```python
# Stream everything: tokens, tool calls, structured data
async def stream_analysis():
    config = AugLLMConfig(
        model="gpt-4",
        tools=[data_fetcher],
        structured_output_model=Analysis,
        streaming=True
    )
    
    async for chunk in config.astream("Analyze market trends"):
        if chunk.type == "token":
            print(chunk.content, end="", flush=True)
        elif chunk.type == "tool_call":
            print(f"\\n🛠️ Using tool: {chunk.tool_name}")
        elif chunk.type == "structured_output":
            print(f"\\n📊 Result: {chunk.data}")
```

**Multi-Provider Support** 🌐
```python
# Seamlessly switch between providers
configs = {
    "openai": AugLLMConfig(model="gpt-4", provider="openai"),
    "anthropic": AugLLMConfig(model="claude-3-opus", provider="anthropic"),
    "google": AugLLMConfig(model="gemini-pro", provider="google"),
    "local": AugLLMConfig(model="llama-2-70b", provider="ollama")
}

# Dynamic provider selection
def get_best_model(task_type: str) -> AugLLMConfig:
    if task_type == "reasoning":
        return configs["anthropic"]  # Claude excels at reasoning
    elif task_type == "coding":
        return configs["openai"]     # GPT-4 great for code
    else:
        return configs["google"]     # Gemini for general tasks
```

**Intelligent Caching** 💾
```python
# Semantic caching for expensive operations
cached_config = AugLLMConfig(
    model="gpt-4",
    tools=[expensive_api_tool],
    caching={
        "type": "semantic",
        "similarity_threshold": 0.95,
        "ttl": 3600,
        "max_entries": 1000
    }
)

# Automatic cache hits for similar queries
result1 = cached_config.invoke("What's the weather in NYC?")
result2 = cached_config.invoke("NYC weather today")  # Cache hit!
```

**Middleware Pipeline** 🔄
```python
# Add capabilities with middleware
enhanced_config = base_config.pipe(
    add_retry(max_attempts=3, backoff="exponential"),
    add_rate_limiting(requests_per_minute=60),
    add_cost_tracking(budget_limit=10.0),
    add_content_filtering(filter_nsfw=True),
    add_logging(level="INFO", include_tokens=True)
)
```

🏗️ CONFIGURATION MASTERY
-------------------------

**Production-Ready Setup** 🚀
```python
production_config = AugLLMConfig(
    # Model configuration
    model="gpt-4",
    temperature=0.3,
    max_tokens=2000,
    top_p=0.9,
    frequency_penalty=0.1,
    
    # System behavior
    system_message="You are a production AI assistant. Be precise and helpful.",
    
    # Tool integration
    tools=[validated_tool_set],
    tool_choice="auto",
    parallel_tool_calls=True,
    
    # Output control
    structured_output_model=ProductionResponse,
    output_validation=True,
    
    # Performance & reliability
    streaming=True,
    timeout=30.0,
    retry_config={
        "max_attempts": 3,
        "backoff_factor": 2.0,
        "retry_on": ["rate_limit", "timeout", "server_error"]
    },
    
    # Monitoring
    enable_metrics=True,
    log_requests=True,
    track_costs=True,
    
    # Security
    content_filter=ContentFilter(
        block_pii=True,
        block_harmful=True,
        scan_outputs=True
    )
)
```

**Development & Testing** 🧪
```python
# Quick prototyping configuration
dev_config = AugLLMConfig(
    model="gpt-3.5-turbo",  # Faster, cheaper for development
    temperature=0.7,
    debug=True,
    
    # Mock tools for testing
    tools=[mock_web_search, mock_database],
    
    # Validation in development
    structured_output_model=DevResponse,
    validate_inputs=True,
    validate_outputs=True,
    
    # Development features
    save_conversations=True,
    conversation_dir="./dev_conversations",
    enable_replay=True
)
```

🎨 COMPOSITION PATTERNS
-----------------------

**Sequential Processing** ➡️
```python
# Chain multiple LLMs for complex tasks
pipeline = compose_runnable([
    AugLLMConfig(model="gpt-4", system_message="Extract key information"),
    AugLLMConfig(model="claude-3", system_message="Analyze and synthesize"),
    AugLLMConfig(model="gpt-4", system_message="Generate final report")
])
```

**Parallel Processing** 🔀
```python
# Run multiple LLMs in parallel
from haive.core.engine.aug_llm.utils import parallel_invoke

results = parallel_invoke([
    AugLLMConfig(model="gpt-4", tools=[financial_tools]),
    AugLLMConfig(model="claude-3", tools=[research_tools]),
    AugLLMConfig(model="gemini-pro", tools=[analysis_tools])
], input_data)
```

**Conditional Routing** 🎯
```python
# Route to different LLMs based on input
def route_by_complexity(input_data: str) -> str:
    if len(input_data.split()) > 1000:
        return "complex"
    elif "technical" in input_data.lower():
        return "technical"
    else:
        return "simple"

router = ConditionalRouter(
    router_function=route_by_complexity,
    routes={
        "complex": AugLLMConfig(model="gpt-4", temperature=0.3),
        "technical": AugLLMConfig(model="claude-3", tools=[tech_tools]),
        "simple": AugLLMConfig(model="gpt-3.5-turbo", temperature=0.7)
    }
)
```

📊 MONITORING & OBSERVABILITY
------------------------------

Every AugLLM instance provides comprehensive metrics:

```python
# Access real-time metrics
config = AugLLMConfig(model="gpt-4", enable_metrics=True)

# After some usage
metrics = config.get_metrics()
print(f"Total requests: {metrics.total_requests}")
print(f"Average latency: {metrics.avg_latency_ms}ms")
print(f"Token usage: {metrics.total_tokens}")
print(f"Cost: ${metrics.total_cost:.4f}")
print(f"Error rate: {metrics.error_rate:.2%}")

# Export metrics for monitoring systems
config.export_metrics("prometheus")  # or "datadog", "cloudwatch"
```

🔒 ENTERPRISE FEATURES
----------------------

- **Security & Compliance**: Built-in PII detection and content filtering
- **Cost Management**: Automatic budget tracking and limits
- **Audit Logging**: Complete request/response logging for compliance
- **Multi-tenancy**: Isolated configurations per tenant
- **Load Balancing**: Automatic distribution across model endpoints
- **Failover**: Automatic switching to backup models

🎓 EXAMPLES GALLERY
-------------------

**Customer Support Agent**
```python
support_agent = AugLLMConfig(
    model="gpt-4",
    system_message="You are a helpful customer support agent.",
    tools=[ticket_system, knowledge_base, escalation_tool],
    structured_output_model=SupportResponse,
    temperature=0.6
)
```

**Data Analysis Pipeline**
```python
data_analyst = AugLLMConfig(
    model="claude-3",
    tools=[sql_executor, chart_generator, statistical_analyzer],
    structured_output_model=AnalysisReport,
    temperature=0.3
)
```

**Creative Writing Assistant**
```python
writer = AugLLMConfig(
    model="gpt-4",
    system_message="You are a creative writing assistant.",
    tools=[research_tool, style_analyzer],
    temperature=0.9,
    streaming=True
)
```

🚀 GETTING STARTED
------------------

```python
from haive.core.engine.aug_llm import AugLLMConfig, compose_runnable
from pydantic import BaseModel, Field

# 1. Define your output structure
class Response(BaseModel):
    answer: str = Field(description="The response to the user")
    confidence: float = Field(description="Confidence score 0-1")

# 2. Create configuration
config = AugLLMConfig(
    model="gpt-4",
    structured_output_model=Response,
    system_message="You are a helpful assistant."
)

# 3. Use it!
response: Response = config.invoke("What is machine learning?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
```

---

**AugLLM: Where Language Models Become Intelligent Agents** 🚀
"""

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.engine.aug_llm.factory import AugLLMFactory

# Temporarily commented out to fix circular import
# from haive.core.engine.aug_llm.mcp_config import (
#     MCPAugLLMConfig,
#     create_mcp_aug_llm_config,
# )
from haive.core.engine.aug_llm.utils import (
    chain_runnables,
    compose_runnable,
    compose_runnables_from_dict,
    create_runnables_dict,
    merge_configs,
)

__all__ = [
    "AugLLMConfig",
    "AugLLMFactory",
    # Temporarily commented out to fix circular import
    # "MCPAugLLMConfig",
    "chain_runnables",
    "compose_runnable",
    "compose_runnables_from_dict",
    # "create_mcp_aug_llm_config",
    "create_runnables_dict",
    "merge_configs",
]
