"""🛠️ Tool Engine - Universal AI Tool Integration System

**THE ULTIMATE TOOLKIT FOR INTELLIGENT AGENTS**

Welcome to the Tool Engine - the revolutionary system that transforms any function, 
API, or service into an intelligent agent capability. This isn't just function calling; 
it's a comprehensive tool ecosystem that makes AI agents capable of interacting with 
the entire digital world.

🚀 TOOL REVOLUTION
------------------

The Tool Engine represents a paradigm shift in how AI agents interact with external 
systems. Every function becomes an **intelligent capability** that agents can:

**🔌 Universal Integration**: Connect any API, database, service, or function instantly
**🧠 Intelligent Discovery**: Automatically analyze and categorize tool capabilities  
**⚡ Dynamic Execution**: Real-time tool selection and parallel execution
**🛡️ Type Safety**: Full validation and error handling for all tool interactions
**📊 Advanced Analytics**: Comprehensive tool usage metrics and optimization

🌟 CORE INNOVATIONS
-------------------

**1. Universal Tool Abstraction** 🎯
   Every tool follows the same interface, regardless of complexity:

Examples:
    >>> # Simple function becomes a tool
    >>> @tool
    >>> def calculator(expression: str) -> float:
    >>> \"\"\"Calculate mathematical expressions.\"\"\"
    >>> return eval(expression)
    >>>
    >>> # Complex API becomes a tool
    >>> @tool
    >>> def web_search(query: str, max_results: int = 5) -> List[SearchResult]:
    >>> \"\"\"Search the web for information.\"\"\"
    >>> return search_api.query(query, limit=max_results)
    >>>
    >>> # All tools work the same way in agents
    >>> agent = AugLLMConfig(
    >>> model="gpt-4",
    >>> tools=[calculator, web_search, database_query, email_sender]
    >>> )

**2. Intelligent Tool Analysis** 🔍
   Automatic discovery and categorization of tool capabilities:

    >>> from haive.core.engine.tool import ToolAnalyzer
    >>>
    >>> analyzer = ToolAnalyzer()
    >>>
    >>> # Analyze any tool automatically
    >>> analysis = analyzer.analyze_tool(my_complex_function)
    >>> print(f"Category: {analysis.category}")           # e.g., "data_processing"
    >>> print(f"Capabilities: {analysis.capabilities}")   # e.g., ["search", "filter"] 
    >>> print(f"Complexity: {analysis.complexity}")       # e.g., "medium"
    >>> print(f"Risk Level: {analysis.risk_level}")       # e.g., "low"
    >>>
    >>> # Automatic tool optimization suggestions
    >>> suggestions = analyzer.get_optimization_suggestions(my_function)
    >>> for suggestion in suggestions:
    >>> print(f"💡 {suggestion.description}")

**3. Advanced Tool Types** 🎨
   Rich type system for different tool behaviors:

    >>> from haive.core.engine.tool import (
    >>> ToolType, ToolCategory, ToolCapability, 
    >>> StateAwareTool, InterruptibleTool
    >>> )
    >>>
    >>> # State-aware tools that remember context
    >>> class DatabaseTool(StateAwareTool):
    >>> def __init__(self):
    >>> self.connection_pool = {}
    >>> self.transaction_state = {}
    >>>
    >>> def execute(self, query: str, context: ToolContext) -> QueryResult:
    >>> # Use context for connection management
    >>> conn = self.get_connection(context.session_id)
    >>> return conn.execute(query)
    >>>
    >>> # Interruptible tools for long-running operations
    >>> class DataProcessingTool(InterruptibleTool):
    >>> async def execute(self, data: LargeDataset) -> ProcessedData:
    >>> for chunk in data.chunks():
    >>> if self.should_interrupt():
    >>> return PartialResult(processed_so_far)
    >>> await self.process_chunk(chunk)

**4. Tool Composition & Orchestration** 🧩
   Build complex capabilities from simple tools:

    >>> from haive.core.engine.tool import ToolEngine, compose_tools
    >>>
    >>> # Compose tools into workflows
    >>> research_workflow = compose_tools([
    >>> web_search_tool,
    >>> content_extractor,
    >>> fact_checker,
    >>> summarizer
    >>> ])
    >>>
    >>> # Parallel tool execution
    >>> analysis_suite = ToolEngine.parallel_tools([
    >>> sentiment_analyzer,
    >>> entity_extractor,
    >>> keyword_extractor,
    >>> topic_classifier
    >>> ])
    >>>
    >>> # Conditional tool chains
    >>> smart_processor = ToolEngine.conditional_chain({
    >>> "text_input": text_processing_tools,
    >>> "image_input": image_processing_tools,
    >>> "audio_input": audio_processing_tools
    >>> })

🎯 ADVANCED FEATURES
--------------------

**Real-time Tool Discovery** 🔍

    >>> # Automatically discover tools from modules
    >>> from haive.core.engine.tool import discover_tools
    >>>
    >>> # Scan entire modules for tools
    >>> discovered = discover_tools([
    >>> "myproject.api_tools",
    >>> "myproject.data_tools", 
    >>> "myproject.ml_tools"
    >>> ])
    >>>
    >>> print(f"Found {len(discovered)} tools:")
    >>> for tool in discovered:
    >>> print(f"  🛠️ {tool.name}: {tool.description}")
    >>> print(f"     Categories: {', '.join(tool.categories)}")
    >>> print(f"     Capabilities: {', '.join(tool.capabilities)}")

**Tool Validation & Testing** ✅

    >>> # Comprehensive tool validation
    >>> validator = ToolValidator()
    >>>
    >>> validation_result = validator.validate_tool(my_tool)
    >>> if validation_result.is_valid:
    >>> print("✅ Tool is valid and ready to use")
    >>> else:
    >>> print("❌ Tool validation failed:")
    >>> for error in validation_result.errors:
    >>> print(f"  - {error.message}")
    >>>
    >>> # Automated tool testing
    >>> test_suite = ToolTestSuite()
    >>> test_suite.add_tool(my_tool)
    >>>
    >>> # Generate test cases automatically
    >>> test_cases = test_suite.generate_test_cases(my_tool)
    >>> results = test_suite.run_tests()
    >>>
    >>> for result in results:
    >>> if result.passed:
    >>> print(f"✅ {result.test_name}")
    >>> else:
    >>> print(f"❌ {result.test_name}: {result.error}")

**Tool Performance Optimization** 🚀

    >>> # Performance monitoring and optimization
    >>> optimizer = ToolOptimizer()
    >>>
    >>> # Analyze tool performance
    >>> performance = optimizer.analyze_performance(my_tool)
    >>> print(f"Average latency: {performance.avg_latency_ms}ms")
    >>> print(f"Success rate: {performance.success_rate:.2%}")
    >>> print(f"Memory usage: {performance.memory_usage_mb}MB")
    >>>
    >>> # Apply optimizations
    >>> optimized_tool = optimizer.optimize(
    >>> my_tool,
    >>> strategies=[
    >>> "caching",           # Cache frequent results
    >>> "connection_pooling", # Reuse connections
    >>> "batching",          # Batch similar requests
    >>> "async_execution"    # Use async when possible
    >>> ]
    >>> )
    >>>
    >>> # Performance improvement
    >>> improvement = optimizer.measure_improvement(my_tool, optimized_tool)
    >>> print(f"Performance improved by {improvement.latency_improvement:.1f}x")

**Tool Security & Sandboxing** 🔒

    >>> # Secure tool execution
    >>> from haive.core.engine.tool import SecureToolEngine, SandboxConfig
    >>>
    >>> # Configure security sandbox
    >>> sandbox_config = SandboxConfig(
    >>> max_execution_time=30.0,
    >>> max_memory_usage="500MB",
    >>> allowed_network_hosts=["api.example.com"],
    >>> blocked_filesystem_paths=["/etc", "/usr"],
    >>> enable_audit_logging=True
    >>> )
    >>>
    >>> # Create secure tool engine
    >>> secure_engine = SecureToolEngine(
    >>> tools=[potentially_unsafe_tool],
    >>> sandbox_config=sandbox_config
    >>> )
    >>>
    >>> # All tool execution is sandboxed
    >>> result = secure_engine.execute(
    >>> tool_name="data_processor",
    >>> input_data=untrusted_data,
    >>> context=execution_context
    >>> )
    >>>
    >>> # Review security audit log
    >>> audit_log = secure_engine.get_audit_log()
    >>> for entry in audit_log:
    >>> print(f"{entry.timestamp}: {entry.action} - {entry.result}")

🏗️ TOOL CATEGORIES & CAPABILITIES
----------------------------------

**Built-in Tool Categories** 📚

    >>> from haive.core.engine.tool import ToolCategory
    >>>
    >>> categories = [
    >>> ToolCategory.DATA_PROCESSING,    # Data manipulation and analysis
    >>> ToolCategory.WEB_INTERACTION,    # Web scraping, API calls
    >>> ToolCategory.FILE_OPERATIONS,    # File I/O, format conversion
    >>> ToolCategory.COMMUNICATION,      # Email, messaging, notifications
    >>> ToolCategory.COMPUTATION,        # Mathematical calculations
    >>> ToolCategory.SEARCH,             # Information retrieval
    >>> ToolCategory.VISUALIZATION,      # Charts, graphs, reports
    >>> ToolCategory.AI_MODELS,          # ML model inference
    >>> ToolCategory.DATABASE,           # Database operations
    >>> ToolCategory.SYSTEM_ADMIN,       # System management tasks
    >>> ]

**Tool Capabilities** 🎨

    >>> from haive.core.engine.tool import ToolCapability
    >>>
    >>> capabilities = [
    >>> ToolCapability.READ,             # Read data/information
    >>> ToolCapability.WRITE,            # Write/modify data
    >>> ToolCapability.SEARCH,           # Search/query capabilities
    >>> ToolCapability.TRANSFORM,        # Data transformation
    >>> ToolCapability.ANALYZE,          # Analysis and insights
    >>> ToolCapability.GENERATE,         # Content generation
    >>> ToolCapability.VALIDATE,         # Data validation
    >>> ToolCapability.MONITOR,          # Monitoring and alerting
    >>> ToolCapability.SCHEDULE,         # Task scheduling
    >>> ToolCapability.INTEGRATE,        # System integration
    >>> ]

🎨 TOOL DEVELOPMENT PATTERNS
----------------------------

**Simple Function Tool** 📝

    >>> from haive.core.engine.tool import tool
    >>>
    >>> @tool
    >>> def word_count(text: str) -> int:
    >>> \"\"\"Count words in text.\"\"\"
    >>> return len(text.split())
    >>>
    >>> # Automatically gets proper metadata
    >>> assert word_count.category == ToolCategory.DATA_PROCESSING
    >>> assert ToolCapability.ANALYZE in word_count.capabilities

**Stateful Tool Class** 🏪

    >>> from haive.core.engine.tool import StatefulTool
    >>>
    >>> class DatabaseTool(StatefulTool):
    >>> \"\"\"Database interaction tool with connection management.\"\"\"
    >>>
    >>> def __init__(self, connection_string: str):
    >>> super().__init__()
    >>> self.connection_string = connection_string
    >>> self._connection = None
    >>>
    >>> def connect(self):
    >>> \"\"\"Establish database connection.\"\"\"
    >>> if not self._connection:
    >>> self._connection = create_connection(self.connection_string)
    >>>
    >>> @tool_method
    >>> def query(self, sql: str) -> List[Dict]:
    >>> \"\"\"Execute SQL query.\"\"\"
    >>> self.connect()
    >>> return self._connection.execute(sql).fetchall()
    >>>
    >>> @tool_method  
    >>> def insert(self, table: str, data: Dict) -> bool:
    >>> \"\"\"Insert data into table.\"\"\"
    >>> self.connect()
    >>> return self._connection.insert(table, data)

**Async Tool Implementation** ⚡

    >>> from haive.core.engine.tool import async_tool
    >>>
    >>> @async_tool
    >>> async def fetch_url(url: str, timeout: float = 10.0) -> WebPage:
    >>> \"\"\"Fetch web page content asynchronously.\"\"\"
    >>> async with aiohttp.ClientSession() as session:
    >>> async with session.get(url, timeout=timeout) as response:
    >>> content = await response.text()
    >>> return WebPage(
    >>> url=url,
    >>> content=content,
    >>> status_code=response.status
    >>> )
    >>>
    >>> # Use in async agents
    >>> async def research_topic(topic: str):
    >>> urls = await web_search(topic)
    >>> pages = await asyncio.gather(*[
    >>> fetch_url(url) for url in urls[:5]
    >>> ])
    >>> return analyze_pages(pages)

**Tool with Complex Configuration** ⚙️

    >>> from haive.core.engine.tool import ConfigurableTool
    >>> from pydantic import BaseModel, Field
    >>>
    >>> class EmailConfig(BaseModel):
    >>> smtp_server: str = Field(..., description="SMTP server address")
    >>> port: int = Field(587, description="SMTP port")
    >>> username: str = Field(..., description="Email username")
    >>> password: str = Field(..., description="Email password")
    >>> use_tls: bool = Field(True, description="Use TLS encryption")
    >>>
    >>> class EmailTool(ConfigurableTool[EmailConfig]):
    >>> \"\"\"Email sending tool with full configuration.\"\"\"
    >>>
    >>> config_class = EmailConfig
    >>>
    >>> def send_email(self, to: str, subject: str, body: str) -> bool:
    >>> \"\"\"Send email message.\"\"\"
    >>> with smtplib.SMTP(self.config.smtp_server, self.config.port) as server:
    >>> if self.config.use_tls:
    >>> server.starttls()
    >>> server.login(self.config.username, self.config.password)
    >>> server.send_message(self._create_message(to, subject, body))
    >>> return True
    >>>
    >>> # Configure and use
    >>> email_tool = EmailTool(config=EmailConfig(
    >>> smtp_server="smtp.gmail.com",
    >>> username="agent@example.com",
    >>> password="app-password"
    >>> ))

📊 TOOL ANALYTICS & INSIGHTS
-----------------------------

**Usage Analytics** 📈

    >>> # Get comprehensive tool usage analytics
    >>> analytics = ToolAnalytics()
    >>>
    >>> # Tool usage statistics
    >>> usage_stats = analytics.get_usage_stats(time_range="last_7_days")
    >>> print(f"Most used tool: {usage_stats.top_tool}")
    >>> print(f"Total executions: {usage_stats.total_executions}")
    >>> print(f"Average success rate: {usage_stats.avg_success_rate:.2%}")
    >>>
    >>> # Performance insights
    >>> performance = analytics.get_performance_insights()
    >>> for insight in performance.slow_tools:
    >>> print(f"🐌 {insight.tool_name}: {insight.avg_latency}ms")
    >>> print(f"   Suggestion: {insight.optimization_suggestion}")
    >>>
    >>> # Error analysis
    >>> errors = analytics.get_error_analysis()
    >>> for error_pattern in errors.common_patterns:
    >>> print(f"❌ {error_pattern.error_type}: {error_pattern.frequency}")
    >>> print(f"   Tools affected: {', '.join(error_pattern.affected_tools)}")

**Tool Recommendation Engine** 🎯

    >>> # Intelligent tool recommendations
    >>> recommender = ToolRecommendationEngine()
    >>>
    >>> # Get tool suggestions for specific tasks
    >>> recommendations = recommender.recommend_tools(
    >>> task_description="I need to analyze customer feedback data",
    >>> context={
    >>> "data_type": "text",
    >>> "data_size": "large",
    >>> "output_format": "dashboard"
    >>> }
    >>> )
    >>>
    >>> for rec in recommendations:
    >>> print(f"🛠️ {rec.tool_name} (confidence: {rec.confidence:.2f})")
    >>> print(f"   Why: {rec.reasoning}")
    >>> print(f"   Usage: {rec.example_usage}")

🔒 ENTERPRISE FEATURES
----------------------

- **Tool Governance**: Approval workflows for new tools
- **Access Control**: Role-based tool permissions
- **Compliance**: Audit trails for all tool executions
- **Multi-tenancy**: Isolated tool environments per tenant
- **Cost Management**: Tool usage tracking and budgets
- **Quality Assurance**: Automated tool testing and validation

🎓 BEST PRACTICES
-----------------

1. **Clear Documentation**: Every tool needs comprehensive docstrings
2. **Type Safety**: Use proper type hints for all parameters and returns
3. **Error Handling**: Implement robust error handling and recovery
4. **Resource Management**: Clean up connections and temporary files
5. **Security**: Validate all inputs and sanitize outputs
6. **Performance**: Optimize for common use cases and cache when appropriate
7. **Testing**: Include unit tests and integration tests for all tools

🚀 GETTING STARTED
------------------

    >>> from haive.core.engine.tool import tool, ToolEngine
    >>>
    >>> # 1. Create a simple tool
    >>> @tool
    >>> def greet(name: str) -> str:
    >>> \"\"\"Greet someone by name.\"\"\"
    >>> return f"Hello, {name}!"
    >>>
    >>> # 2. Use in an agent
    >>> from haive.core.engine.aug_llm import AugLLMConfig
    >>>
    >>> agent = AugLLMConfig(
    >>> model="gpt-4",
    >>> tools=[greet]
    >>> )
    >>>
    >>> # 3. The agent can now greet people!
    >>> result = agent.invoke("Please greet Alice")
    >>> # Agent will call greet("Alice") and respond: "Hello, Alice!"

---

**Tool Engine: Where Functions Become Intelligent Capabilities** 🛠️
"""

from haive.core.engine.tool.analyzer import ToolAnalyzer
from haive.core.engine.tool.engine import ToolEngine
from haive.core.engine.tool.types import (
    InterruptibleTool,
    StateAwareTool,
    ToolCapability,
    ToolCategory,
    ToolLike,
    ToolProperties,
    ToolType,
)

__all__ = [
    # Main engine
    "ToolEngine",
    # Types
    "ToolLike",
    "ToolType",
    "ToolCategory",
    "ToolCapability",
    "ToolProperties",
    # Protocols
    "InterruptibleTool",
    "StateAwareTool",
    # Analyzer
    "ToolAnalyzer",
]


# Convenience exports for backward compatibility
def get_tool_type():
    """Get the universal ToolLike type."""
    return ToolLike


def get_tool_analyzer():
    """Get a tool analyzer instance."""
    return ToolAnalyzer()


def get_capability_enum():
    """Get the ToolCapability enum."""
    return ToolCapability


def get_category_enum():
    """Get the ToolCategory enum."""
    return ToolCategory
