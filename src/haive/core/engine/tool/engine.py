"""Rebuilt ToolEngine with universal typing and advanced features."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool, Tool, tool
from langchain_core.tools.base import BaseTool, BaseToolkit
from langgraph.prebuilt import ToolNode
from langgraph.types import RetryPolicy
from pydantic import BaseModel, ConfigDict, Field, field_validator

from haive.core.engine.base import InvokableEngine
from haive.core.engine.base.types import EngineType

from .analyzer import ToolAnalyzer
from .types import (
    ToolCapability,
    ToolCategory,
    ToolLike,
    ToolProperties,
    ToolType,
)

logger = logging.getLogger(__name__)


class ToolEngine(InvokableEngine[dict[str, Any], dict[str, Any]]):
    """Enhanced tool engine with universal typing and property analysis.

    This engine manages tools with comprehensive property analysis,
    capability-based routing, and state interaction tracking.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine_type: EngineType = EngineType.TOOL

    # Tool sources
    tools: Sequence[ToolLike] | None = Field(
        default=None, description="List of tools to manage"
    )
    toolkit: BaseToolkit | list[BaseToolkit] | None = Field(
        default=None, description="Toolkit(s) to use"
    )

    # Basic configuration
    retry_policy: RetryPolicy | None = None
    parallel: bool = Field(default=False)
    messages_key: str = Field(default="messages")
    tool_choice: Literal["auto", "required", "none"] = Field(default="auto")
    return_source: bool = Field(default=True)

    # Advanced options
    timeout: float | None = None
    max_iterations: int | None = None

    # Routing configuration with defaults
    auto_route: bool = Field(
        default=True,
        description="Automatically route to appropriate tool based on calls",
    )
    routing_strategy: Literal["auto", "capability", "category", "priority"] = Field(
        default="auto", description="Tool routing strategy"
    )

    # Tool analysis
    enable_analysis: bool = Field(
        default=True, description="Enable automatic tool analysis"
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize and analyze tools after creation."""
        super().model_post_init(__context)
        # Initialize private attributes
        self._analyzer = ToolAnalyzer()
        self._tool_properties: dict[str, ToolProperties] = {}
        self._tool_instances: dict[str, ToolLike] = {}

        if self.enable_analysis:
            self._analyze_all_tools()

    def _analyze_all_tools(self) -> None:
        """Analyze all configured tools."""
        all_tools = self._get_all_tools()

        for tool in all_tools:
            try:
                properties = self._analyzer.analyze(tool)
                self._tool_properties[properties.name] = properties
                self._tool_instances[properties.name] = tool

                logger.debug(
                    f"Analyzed tool '{properties.name}': "
                    f"type={properties.tool_type}, "
                    f"category={properties.category}, "
                    f"capabilities={properties.capabilities}"
                )
            except Exception as e:
                logger.warning(f"Failed to analyze tool: {e}")

    # Required abstract methods implementation

    def get_input_fields(self) -> dict[str, tuple[type, Any]]:
        """Define input fields for tool engine."""
        return {
            "messages": (
                list[BaseMessage],
                Field(default_factory=list, description="Input messages"),
            ),
            "state": (
                dict[str, Any],
                Field(default_factory=dict, description="Current state"),
            ),
            "tool_choice": (
                str | list[str] | None,
                Field(default=None, description="Specific tool(s) to use"),
            ),
            "required_capabilities": (
                list[ToolCapability] | None,
                Field(default=None, description="Required tool capabilities"),
            ),
        }

    def get_output_fields(self) -> dict[str, tuple[type, Any]]:
        """Define output fields for tool engine."""
        return {
            "messages": (
                list[BaseMessage],
                Field(
                    default_factory=list,
                    description="Output messages with tool results",
                ),
            ),
            "tool_results": (
                list[dict[str, Any]],
                Field(default_factory=list, description="Tool execution results"),
            ),
            "state": (
                dict[str, Any],
                Field(default_factory=dict, description="Updated state"),
            ),
            "execution_metadata": (
                dict[str, Any],
                Field(default_factory=dict, description="Execution metadata"),
            ),
        }

    def create_runnable(self, runnable_config: RunnableConfig | None = None) -> Any:
        """Create enhanced ToolNode with property awareness."""
        params = self.apply_runnable_config(runnable_config)

        # Get all tools
        all_tools = self._get_all_tools()

        # Apply routing strategy
        if self.routing_strategy != "auto":
            all_tools = self._apply_routing_strategy(all_tools, params)

        # Create ToolNode with supported parameters
        kwargs = {
            "tools": all_tools,
            "messages_key": self.messages_key,
        }

        # Note: ToolNode doesn't support retry_policy, timeout, or max_iterations directly
        # These would need to be implemented via custom error handling or wrapping

        return ToolNode(**kwargs)

    def _apply_routing_strategy(
        self, tools: list[ToolLike], params: dict[str, Any]
    ) -> list[ToolLike]:
        """Apply routing strategy to filter/order tools."""
        if self.routing_strategy == "capability":
            # Filter by required capabilities
            required = params.get("required_capabilities", [])
            if required:
                return self._filter_by_capabilities(tools, required)

        elif self.routing_strategy == "category":
            # Sort by category
            return sorted(
                tools,
                key=lambda t: self._tool_properties.get(
                    self._analyzer._get_tool_name(t),
                    ToolProperties(name="unknown", tool_type=ToolType.FUNCTION),
                ).category.value,
            )

        elif self.routing_strategy == "priority":
            # Sort by priority (interruptible first, then state-aware, etc.)
            def priority_key(tool):
                """Priority Key.

                Args:
                    tool: [TODO: Add description]
                """
                props = self._tool_properties.get(self._analyzer._get_tool_name(tool))
                if not props:
                    return 999
                score = 0
                if props.is_interruptible:
                    score -= 10
                if props.is_state_tool:
                    score -= 5
                if props.has_structured_output():
                    score -= 3
                return score

            return sorted(tools, key=priority_key)

        return tools

    # Tool factory methods

    @classmethod
    def create_retriever_tool(
        cls,
        retriever: BaseRetriever,
        name: str,
        description: str,
        *,
        document_prompt: BasePromptTemplate | None = None,
        document_separator: str = "\n\n",
        response_format: Literal["content", "content_and_artifact"] = "content",
    ) -> StructuredTool:
        """Create a tool to do retrieval of documents."""
        from langchain_core.tools import (
            create_retriever_tool as lc_create_retriever_tool,
        )

        # Create the base retriever tool
        retriever_tool = lc_create_retriever_tool(
            retriever=retriever,
            name=name,
            description=description,
            document_prompt=document_prompt,
            document_separator=document_separator,
            response_format=response_format,
        )

        # Add tool metadata
        retriever_tool.__tool_type__ = ToolType.RETRIEVER_TOOL
        retriever_tool.__tool_category__ = ToolCategory.RETRIEVAL
        retriever_tool.__tool_capabilities__ = {
            ToolCapability.RETRIEVER,
            ToolCapability.READS_STATE,
            ToolCapability.STRUCTURED_OUTPUT,
        }

        return retriever_tool

    @classmethod
    def create_structured_output_tool(
        cls,
        func: Callable[..., Any],
        name: str,
        description: str,
        output_model: type[BaseModel],
        *,
        infer_schema: bool = True,
    ) -> StructuredTool:
        """Create a tool that produces structured output."""

        # Create wrapper that validates output
        def structured_wrapper(*args, **kwargs):
            """Structured Wrapper."""
            result = func(*args, **kwargs)

            # Validate output against model
            if isinstance(result, output_model):
                return result
            elif isinstance(result, dict):
                return output_model(**result)
            else:
                # Try to construct from result
                return output_model(result=result)

        # Copy function metadata
        structured_wrapper.__name__ = func.__name__
        structured_wrapper.__doc__ = func.__doc__

        # Ensure function has docstring for @tool decorator
        if not structured_wrapper.__doc__:
            structured_wrapper.__doc__ = description

        # Create tool
        if infer_schema:
            structured_tool = tool(name)(structured_wrapper)
            if description != structured_wrapper.__doc__:
                structured_tool.description = description
        else:
            structured_tool = StructuredTool(
                name=name,
                description=description,
                func=structured_wrapper,
            )

        # Add metadata
        structured_tool.__tool_type__ = ToolType.STRUCTURED_TOOL
        structured_tool.__tool_capabilities__ = {
            ToolCapability.STRUCTURED_OUTPUT,
            ToolCapability.VALIDATED_OUTPUT,
        }
        object.__setattr__(structured_tool, "structured_output_model", output_model)

        return structured_tool

    @classmethod
    def create_state_tool(
        cls,
        func: Callable[..., Any] | StructuredTool | BaseTool,
        name: str | None = None,
        description: str | None = None,
        *,
        reads_state: bool = False,
        writes_state: bool = False,
        state_keys: list[str] | None = None,
    ) -> StructuredTool:
        """Create or wrap a tool with state interaction metadata."""
        # Handle wrapping existing tools
        if isinstance(func, (StructuredTool, BaseTool)):
            state_tool = func
            if name is None:
                name = func.name
            if description is None:
                description = func.description
        else:
            # Create new tool from function
            # Ensure function has a docstring (required by @tool decorator)
            if not func.__doc__:
                func.__doc__ = description or f"Tool function: {name or func.__name__}"

            if name:
                # Use decorator with name
                state_tool = tool(name)(func)
                if description and description != func.__doc__:
                    state_tool.description = description
            else:
                # Use plain decorator - requires docstring
                state_tool = tool(func)

        # Add state metadata
        capabilities = set(getattr(state_tool, "__tool_capabilities__", set()))

        if reads_state:
            capabilities.add(ToolCapability.READS_STATE)
            capabilities.add(ToolCapability.FROM_STATE)

        if writes_state:
            capabilities.add(ToolCapability.WRITES_STATE)
            capabilities.add(ToolCapability.TO_STATE)

        if reads_state or writes_state:
            capabilities.add(ToolCapability.STATE_AWARE)

        state_tool.__tool_capabilities__ = capabilities
        state_tool.__state_interaction__ = {
            "reads": reads_state,
            "writes": writes_state,
            "keys": state_keys or [],
        }

        # Implement StateAwareTool protocol
        object.__setattr__(state_tool, "reads_state", reads_state)
        object.__setattr__(state_tool, "writes_state", writes_state)
        object.__setattr__(state_tool, "state_dependencies", state_keys or [])

        return state_tool

    @classmethod
    def create_interruptible_tool(
        cls,
        func: Callable[..., Any] | StructuredTool | BaseTool,
        name: str | None = None,
        description: str | None = None,
        *,
        interrupt_message: str = "Tool execution interrupted",
    ) -> StructuredTool:
        """Create or wrap a tool with interruption capability."""
        # Handle wrapping existing tools
        if isinstance(func, (StructuredTool, BaseTool)):
            interruptible = func
            if name is None:
                name = func.name
            if description is None:
                description = func.description
        else:
            # Create new tool from function
            # Ensure function has a docstring (required by @tool decorator)
            if not func.__doc__:
                func.__doc__ = description or f"Tool function: {name or func.__name__}"

            if name:
                # Use decorator with name
                interruptible = tool(name)(func)
                if description and description != func.__doc__:
                    interruptible.description = description
            else:
                # Use plain decorator - requires docstring
                interruptible = tool(func)

        # Add interruption metadata
        capabilities = set(getattr(interruptible, "__tool_capabilities__", set()))
        capabilities.add(ToolCapability.INTERRUPTIBLE)

        interruptible.__tool_capabilities__ = capabilities
        interruptible.__interruptible__ = True
        interruptible.__interrupt_message__ = interrupt_message

        # Add interrupt method as attribute (not field)
        def interrupt():
            """Interrupt."""
            raise InterruptedError(interrupt_message)

        # Use object.__setattr__ to add attributes that aren't Pydantic fields
        object.__setattr__(interruptible, "interrupt", interrupt)
        object.__setattr__(interruptible, "is_interruptible", True)

        return interruptible

    @classmethod
    def augment_tool(
        cls,
        tool: StructuredTool | BaseTool | Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        make_interruptible: bool = False,
        interrupt_message: str = "Tool execution interrupted",
        reads_state: bool = False,
        writes_state: bool = False,
        state_keys: list[str] | None = None,
        structured_output_model: type[BaseModel] | None = None,
    ) -> StructuredTool:
        """Augment an existing tool with additional capabilities.

        This method adds capabilities to a tool without creating nested wrappers.
        All capabilities are added directly to the tool instance.
        """
        # Start with the base tool
        if not isinstance(tool, (StructuredTool, BaseTool)):
            # Convert function to tool first
            from langchain_core.tools import tool as create_tool

            enhanced = create_tool(tool)
        else:
            enhanced = tool

        # Collect all capabilities
        capabilities = set(getattr(enhanced, "__tool_capabilities__", set()))

        # Add interruption capability
        if make_interruptible:
            capabilities.add(ToolCapability.INTERRUPTIBLE)
            # Store custom attributes on the object directly (not as Pydantic fields)
            object.__setattr__(enhanced, "__interruptible__", True)
            object.__setattr__(enhanced, "__interrupt_message__", interrupt_message)
            object.__setattr__(enhanced, "is_interruptible", True)

            def interrupt():
                """Interrupt."""
                raise InterruptedError(interrupt_message)

            object.__setattr__(enhanced, "interrupt", interrupt)

        # Add state interaction capabilities
        if reads_state or writes_state:
            if reads_state:
                capabilities.add(ToolCapability.READS_STATE)
                capabilities.add(ToolCapability.FROM_STATE)
            if writes_state:
                capabilities.add(ToolCapability.WRITES_STATE)
                capabilities.add(ToolCapability.TO_STATE)
            capabilities.add(ToolCapability.STATE_AWARE)

            object.__setattr__(
                enhanced,
                "__state_interaction__",
                {
                    "reads": reads_state,
                    "writes": writes_state,
                    "keys": state_keys or [],
                },
            )
            object.__setattr__(enhanced, "reads_state", reads_state)
            object.__setattr__(enhanced, "writes_state", writes_state)
            object.__setattr__(enhanced, "state_dependencies", state_keys or [])

        # Add structured output capability
        if structured_output_model:
            capabilities.add(ToolCapability.STRUCTURED_OUTPUT)
            capabilities.add(ToolCapability.VALIDATED_OUTPUT)
            object.__setattr__(
                enhanced, "structured_output_model", structured_output_model
            )
            object.__setattr__(
                enhanced, "__structured_output_model__", structured_output_model
            )

        # Update capabilities
        object.__setattr__(enhanced, "__tool_capabilities__", capabilities)

        # Update name/description if provided
        if name:
            enhanced.name = name
        if description:
            enhanced.description = description

        return enhanced

    @classmethod
    def create_store_tools_suite(
        cls,
        store_manager: Any,  # StoreManager from haive.core.tools.store_manager
        namespace: tuple[str, ...] | None = None,
        include_tools: list[str] | None = None,
    ) -> list[StructuredTool]:
        """Create a suite of store/memory management tools.

        This creates tools for memory operations like store, search, retrieve,
        update, and delete. These tools integrate with the StoreManager for
        persistent memory management.

        Args:
            store_manager: The StoreManager instance to use
            namespace: Optional namespace for memory isolation
            include_tools: List of tools to include (defaults to all)
                          Options: ["store", "search", "retrieve", "update", "delete"]

        Returns:
            List of store tools with proper metadata
        """
        from haive.core.tools.store_tools import create_memory_tools_suite

        # Create the tools using the existing factory
        tools = create_memory_tools_suite(
            store_manager=store_manager,
            namespace=namespace,
            include_tools=include_tools,
        )

        # Enhance each tool with proper metadata
        enhanced_tools = []
        for tool in tools:
            # Add store tool metadata
            capabilities = {
                ToolCapability.STORE,
                ToolCapability.STRUCTURED_OUTPUT,
                ToolCapability.VALIDATED_OUTPUT,
            }

            # Specific capabilities based on tool name
            if "search" in tool.name:
                capabilities.add(ToolCapability.RETRIEVER)
                capabilities.add(ToolCapability.READS_STATE)
            elif "store" in tool.name or "update" in tool.name:
                capabilities.add(ToolCapability.WRITES_STATE)
                capabilities.add(ToolCapability.TO_STATE)
            elif "retrieve" in tool.name:
                capabilities.add(ToolCapability.READS_STATE)
                capabilities.add(ToolCapability.FROM_STATE)
            elif "delete" in tool.name:
                capabilities.add(ToolCapability.WRITES_STATE)

            tool.__tool_type__ = ToolType.STORE_TOOL
            tool.__tool_category__ = ToolCategory.MEMORY
            tool.__tool_capabilities__ = capabilities

            enhanced_tools.append(tool)

        return enhanced_tools

    @classmethod
    def create_retriever_tool_from_config(
        cls,
        retriever_config: Any,  # BaseRetrieverConfig from haive.core.engine.retriever
        name: str,
        description: str,
        *,
        document_prompt: BasePromptTemplate | None = None,
        document_separator: str = "\n\n",
        response_format: Literal["content", "content_and_artifact"] = "content",
    ) -> StructuredTool:
        """Create a retriever tool from a Haive retriever configuration.

        This is a convenience method that creates a retriever from the config
        and then converts it to a tool with proper metadata.

        Args:
            retriever_config: The BaseRetrieverConfig to use
            name: Tool name
            description: Tool description
            document_prompt: Optional prompt to format documents
            document_separator: Separator between documents
            response_format: Format for tool response

        Returns:
            A StructuredTool configured for retrieval
        """
        # Create retriever from config
        retriever = retriever_config.create_runnable()

        # Use the existing create_retriever_tool method
        return cls.create_retriever_tool(
            retriever=retriever,
            name=name,
            description=description,
            document_prompt=document_prompt,
            document_separator=document_separator,
            response_format=response_format,
        )

    @classmethod
    def create_human_interrupt_tool(
        cls,
        name: str = "human_interrupt",
        description: str = "Request human intervention or approval",
        *,
        allow_ignore: bool = True,
        allow_respond: bool = True,
        allow_edit: bool = False,
        allow_accept: bool = True,
        interrupt_message: str = "Human input requested",
    ) -> StructuredTool:
        """Create a tool that triggers human interrupt workflow.

        This creates a tool that can pause graph execution and request
        human intervention, following langgraph's interrupt patterns.

        Args:
            name: Tool name
            description: Tool description
            allow_ignore: Whether human can skip/ignore
            allow_respond: Whether human can provide text response
            allow_edit: Whether human can edit the request
            allow_accept: Whether human can accept as-is
            interrupt_message: Default interrupt message

        Returns:
            A StructuredTool that triggers human interrupts
        """

        class HumanInterruptInput(BaseModel):
            """Input schema for human interrupt requests."""

            action: str = Field(description="The action being requested")
            context: dict[str, Any] = Field(
                default_factory=dict, description="Context for the request"
            )
            description: str | None = Field(
                default=None, description="Detailed description of what's needed"
            )

        def request_human_input(
            action: str,
            context: dict[str, Any] = None,
            description: str | None = None,
        ) -> dict[str, Any]:
            """Request human intervention.

            This function triggers a human interrupt in the graph execution,
            allowing for human-in-the-loop workflows.
            """
            # Import inside function to avoid circular imports
            from langgraph.types import interrupt

            # Create interrupt configuration
            interrupt_config = {
                "allow_ignore": allow_ignore,
                "allow_respond": allow_respond,
                "allow_edit": allow_edit,
                "allow_accept": allow_accept,
            }

            # Create the interrupt request
            interrupt_request = {
                "action_request": {"action": action, "args": context or {}},
                "config": interrupt_config,
                "description": description or interrupt_message,
            }

            # Trigger the interrupt and get response
            response = interrupt([interrupt_request])[0]

            return {
                "type": response.get("type", "unknown"),
                "response": response.get("args"),
                "action": action,
                "approved": response.get("type") == "accept",
            }

        # Create the tool
        interrupt_tool = StructuredTool(
            name=name,
            description=description,
            func=request_human_input,
            args_schema=HumanInterruptInput,
        )

        # Add metadata
        interrupt_tool.__tool_type__ = ToolType.STRUCTURED_TOOL
        interrupt_tool.__tool_category__ = ToolCategory.COORDINATION
        interrupt_tool.__tool_capabilities__ = {
            ToolCapability.INTERRUPTIBLE,
            ToolCapability.STRUCTURED_OUTPUT,
            ToolCapability.VALIDATED_OUTPUT,
        }
        object.__setattr__(interrupt_tool, "is_interruptible", True)
        interrupt_tool.__interrupt_message__ = interrupt_message

        return interrupt_tool

    # Tool query methods

    def get_tools_by_capability(self, *capabilities: ToolCapability) -> list[str]:
        """Get tool names with specified capabilities."""
        matching = []
        for name, props in self._tool_properties.items():
            if all(cap in props.capabilities for cap in capabilities):
                matching.append(name)
        return matching

    def get_tools_by_category(self, category: ToolCategory) -> list[str]:
        """Get tool names in category."""
        return [
            name
            for name, props in self._tool_properties.items()
            if props.category == category
        ]

    def get_interruptible_tools(self) -> list[str]:
        """Get all interruptible tool names."""
        return [
            name
            for name, props in self._tool_properties.items()
            if props.is_interruptible
        ]

    def get_state_tools(self) -> list[str]:
        """Get all state-aware tool names."""
        return [
            name for name, props in self._tool_properties.items() if props.is_state_tool
        ]

    def get_tools_reading_state(self) -> list[str]:
        """Get tools that read from state."""
        return [
            name
            for name, props in self._tool_properties.items()
            if props.from_state_tool
        ]

    def get_tools_writing_state(self) -> list[str]:
        """Get tools that write to state."""
        return [
            name for name, props in self._tool_properties.items() if props.to_state_tool
        ]

    def get_tool_properties(self, tool_name: str) -> ToolProperties | None:
        """Get properties for specific tool."""
        return self._tool_properties.get(tool_name)

    # Helper methods

    def _filter_by_capabilities(
        self, tools: list[ToolLike], capabilities: list[ToolCapability]
    ) -> list[ToolLike]:
        """Filter tools by required capabilities."""
        filtered = []

        for tool in tools:
            tool_name = self._analyzer._get_tool_name(tool)
            props = self._tool_properties.get(tool_name)

            if props and all(cap in props.capabilities for cap in capabilities):
                filtered.append(tool)

        return filtered

    def _get_all_tools(self) -> list[ToolLike]:
        """Get all tools from various sources."""
        all_tools = []

        # Add directly specified tools
        if self.tools:
            for tool in self.tools:
                # Process BaseModel tools
                if isinstance(tool, BaseModel) and not isinstance(
                    tool, (BaseTool, Tool, StructuredTool)
                ):
                    try:
                        structured_tool = self._convert_model_to_tool(tool)
                        if structured_tool:
                            all_tools.append(structured_tool)
                        else:
                            logger.warning(
                                f"Could not convert model to tool: {type(tool).__name__}"
                            )
                    except Exception as e:
                        logger.warning(f"Error converting model to tool: {e}")
                else:
                    all_tools.append(tool)

        # Add tools from toolkits
        if self.toolkit:
            if isinstance(self.toolkit, list):
                for tk in self.toolkit:
                    if hasattr(tk, "get_tools"):
                        all_tools.extend(tk.get_tools())
                    elif isinstance(tk, BaseToolkit):
                        all_tools.extend(tk.tools)
            elif hasattr(self.toolkit, "get_tools"):
                all_tools.extend(self.toolkit.get_tools())
            elif isinstance(self.toolkit, BaseToolkit):
                all_tools.extend(self.toolkit.tools)

        return all_tools

    def _convert_model_to_tool(self, model: BaseModel) -> StructuredTool | None:
        """Convert Pydantic model to StructuredTool."""
        if not callable(model):
            return None

        call_method = model.__call__
        name = getattr(model, "name", model.__class__.__name__.lower())
        description = call_method.__doc__ or f"Tool for {name}"

        # Ensure function has docstring
        def model_tool(*args, **kwargs):
            """Model Tool."""
            return call_method(*args, **kwargs)

        model_tool.__name__ = name
        model_tool.__doc__ = description

        # Use tool decorator
        model_tool = tool(name)(model_tool)

        return model_tool

    # Type export methods

    @classmethod
    def get_tool_type(cls) -> type:
        """Get the universal ToolLike type for other components."""
        from .types import ToolLike

        return ToolLike

    @classmethod
    def get_analyzer(cls) -> ToolAnalyzer:
        """Get a tool analyzer instance."""
        return ToolAnalyzer()

    @classmethod
    def get_capability_enum(cls) -> type[ToolCapability]:
        """Get ToolCapability enum for other components."""
        from .types import ToolCapability

        return ToolCapability

    @classmethod
    def get_category_enum(cls) -> type[ToolCategory]:
        """Get ToolCategory enum for other components."""
        from .types import ToolCategory

        return ToolCategory

    # Engine conversion methods

    @classmethod
    def from_aug_llm_config(
        cls,
        config: AugLLMConfig,
        *,
        extract_tools: bool = True,
        include_structured_output: bool = True,
        name: str | None = None,
    ) -> ToolEngine:
        """Create ToolEngine from AugLLMConfig.

        Args:
            config: AugLLMConfig instance to convert
            extract_tools: Whether to extract tools from config
            include_structured_output: Whether to convert structured_output_model to tool
            name: Optional name for the engine

        Returns:
            ToolEngine with tools extracted from AugLLMConfig
        """
        tools = []

        if extract_tools and config.tools:
            tools.extend(config.tools)

        if include_structured_output and config.structured_output_model:
            # Convert structured output model to tool
            structured_tool = cls.create_structured_output_tool(
                func=lambda **kwargs: config.structured_output_model(**kwargs),
                name=f"{config.structured_output_model.__name__.lower()}_tool",
                description=f"Tool for {config.structured_output_model.__name__}",
                output_model=config.structured_output_model,
            )
            tools.append(structured_tool)

        return cls(
            tools=tools,
            name=name or f"tool_engine_from_{config.__class__.__name__.lower()}",
        )

    @classmethod
    def from_retriever_config(
        cls,
        config: BaseRetrieverConfig,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
        name: str | None = None,
    ) -> ToolEngine:
        """Create ToolEngine from BaseRetrieverConfig.

        Args:
            config: BaseRetrieverConfig instance to convert
            tool_name: Name for the retriever tool
            tool_description: Description for the retriever tool
            name: Optional name for the engine

        Returns:
            ToolEngine with retriever converted to tool
        """
        retriever_tool = cls.create_retriever_tool_from_config(
            retriever_config=config,
            name=tool_name or f"{config.__class__.__name__.lower()}_retriever",
            description=tool_description
            or f"Retriever tool from {config.__class__.__name__}",
        )

        return cls(
            tools=[retriever_tool],
            name=name or f"tool_engine_from_{config.__class__.__name__.lower()}",
            description=f"ToolEngine with retriever from {config.__class__.__name__}",
        )

    @classmethod
    def from_vectorstore_config(
        cls,
        config: VectorStoreConfig,
        *,
        search_tool_name: str | None = None,
        similarity_search_tool: bool = True,
        max_marginal_relevance_tool: bool = False,
        name: str | None = None,
    ) -> ToolEngine:
        """Create ToolEngine from VectorStoreConfig.

        Args:
            config: VectorStoreConfig instance to convert
            search_tool_name: Base name for search tools
            similarity_search_tool: Whether to create similarity search tool
            max_marginal_relevance_tool: Whether to create MMR search tool
            name: Optional name for the engine

        Returns:
            ToolEngine with vectorstore search tools
        """
        from haive.core.engine.vectorstore import VectorStoreConfig

        if not isinstance(config, VectorStoreConfig):
            raise TypeError(f"Expected VectorStoreConfig, got {type(config)}")

        tools = []
        base_name = search_tool_name or f"{config.__class__.__name__.lower()}_search"

        if similarity_search_tool:
            # Create similarity search tool
            similarity_tool = cls.create_retriever_tool_from_config(
                retriever_config=config,
                name=f"{base_name}_similarity",
                description=f"Similarity search using {config.__class__.__name__}",
            )
            tools.append(similarity_tool)

        if max_marginal_relevance_tool:
            # Create MMR search tool (if supported)
            mmr_tool = cls.create_retriever_tool_from_config(
                retriever_config=config,
                name=f"{base_name}_mmr",
                description=f"Max marginal relevance search using {config.__class__.__name__}",
            )
            tools.append(mmr_tool)

        return cls(
            tools=tools,
            name=name or f"tool_engine_from_{config.__class__.__name__.lower()}",
            description=f"ToolEngine with vectorstore tools from {config.__class__.__name__}",
        )

    @classmethod
    def from_document_engine(
        cls,
        engine: DocumentEngine,
        *,
        create_loader_tools: bool = True,
        create_splitter_tools: bool = True,
        create_processor_tools: bool = False,
        name: str | None = None,
    ) -> ToolEngine:
        """Create ToolEngine from DocumentEngine.

        Args:
            engine: DocumentEngine instance to convert
            create_loader_tools: Whether to create document loading tools
            create_splitter_tools: Whether to create document splitting tools
            create_processor_tools: Whether to create document processing tools
            name: Optional name for the engine

        Returns:
            ToolEngine with document processing tools
        """
        tools = []

        # Note: This is a placeholder - actual implementation would depend on
        # DocumentEngine's public API for extracting loaders/splitters
        if create_loader_tools:
            # Convert document loaders to tools
            loader_tool = cls.create_state_tool(
                func=lambda path, **kwargs: engine.load_document(path, **kwargs),
                name="document_loader",
                description="Load documents using DocumentEngine",
            )
            tools.append(loader_tool)

        if create_splitter_tools:
            # Convert splitters to tools
            splitter_tool = cls.create_state_tool(
                func=lambda content, **kwargs: engine.split_document(content, **kwargs),
                name="document_splitter",
                description="Split documents using DocumentEngine",
            )
            tools.append(splitter_tool)

        if create_processor_tools:
            # Convert processors to tools
            processor_tool = cls.create_state_tool(
                func=lambda docs, **kwargs: engine.process_documents(docs, **kwargs),
                name="document_processor",
                description="Process documents using DocumentEngine",
            )
            tools.append(processor_tool)

        return cls(
            tools=tools,
            name=name or f"tool_engine_from_{engine.__class__.__name__.lower()}",
            description=f"ToolEngine with document tools from {engine.__class__.__name__}",
        )

    @classmethod
    def from_multiple_engines(
        cls,
        engines: dict[str, InvokableEngine],
        *,
        engine_conversion_config: dict[str, dict] | None = None,
        name: str | None = None,
    ) -> ToolEngine:
        """Create ToolEngine from multiple engines.

        Args:
            engines: Dictionary of engine_name -> engine_instance
            engine_conversion_config: Configuration for each engine's conversion
            name: Optional name for the combined engine

        Returns:
            ToolEngine with tools from all engines
        """
        all_tools = []
        config = engine_conversion_config or {}

        for engine_name, engine in engines.items():
            engine_config = config.get(engine_name, {})

            # Route to appropriate conversion method
            if hasattr(engine, "__class__"):
                class_name = engine.__class__.__name__

                if "AugLLMConfig" in class_name:
                    converted = cls.from_aug_llm_config(engine, **engine_config)
                elif "RetrieverConfig" in class_name:
                    converted = cls.from_retriever_config(engine, **engine_config)
                elif "VectorStoreConfig" in class_name:
                    converted = cls.from_vectorstore_config(engine, **engine_config)
                elif "DocumentEngine" in class_name:
                    converted = cls.from_document_engine(engine, **engine_config)
                else:
                    # Generic conversion - try to extract tools if available
                    tools = getattr(engine, "tools", [])
                    if tools:
                        converted = cls(tools=tools, name=f"converted_{engine_name}")
                    else:
                        continue

                all_tools.extend(converted.tools or [])

        return cls(
            tools=all_tools,
            name=name or "multi_engine_tool_engine",
            description=f"ToolEngine combining {len(engines)} engines",
        )

    # Field validators

    @field_validator("engine_type")
    @classmethod
    def validate_engine_type(cls, v) -> Any:
        """Validate engine type is TOOL."""
        if v != EngineType.TOOL:
            raise ValueError("engine_type must be TOOL")
        return v

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v) -> Any:
        """Validate tools are of the correct type."""
        if v is None:
            return v

        # Just return the tools - analyzer will handle validation
        return v

    @field_validator("toolkit")
    @classmethod
    def validate_toolkit(cls, v) -> Any:
        """Validate toolkit is of the correct type."""
        if v is None:
            return v

        if isinstance(v, list):
            valid_toolkits = []
            for tk in v:
                if isinstance(tk, BaseToolkit) or hasattr(tk, "get_tools"):
                    valid_toolkits.append(tk)
                else:
                    logger.warning(
                        f"Ignoring invalid toolkit type: {type(tk).__name__}"
                    )
            return valid_toolkits

        if isinstance(v, BaseToolkit) or hasattr(v, "get_tools"):
            return v

        logger.warning(f"Ignoring invalid toolkit type: {type(v).__name__}")
        return None

    @field_validator("tool_choice")
    @classmethod
    def validate_tool_choice(cls, v) -> Any:
        """Validate tool_choice has a valid value."""
        valid_choices = ["auto", "required", "none"]
        if v not in valid_choices:
            raise ValueError(f"tool_choice must be one of {valid_choices}")
        return v
