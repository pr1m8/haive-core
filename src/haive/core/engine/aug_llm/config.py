"""AugLLM configuration system for enhanced LLM chains.

from typing import Any, Dict
This module provides a comprehensive configuration system for creating and
managing enhanced LLM chains within the Haive framework. The AugLLMConfig class
serves as a central configuration point that integrates prompts, tools, output
parsers, and structured output models with extensive validation and debugging
capabilities.

Key features:
- Flexible prompt template creation and management with support for few-shot learning
- Comprehensive tool integration with automatic discovery and configuration
- Structured output handling via two approaches (v1: parser-based, v2: tool-based)
- Rich debugging and validation to ensure proper configuration
- Pre/post processing hooks for customizing input and output
- Support for both synchronous and asynchronous execution

The configuration system is designed to be highly customizable while providing
sensible defaults and automatic detection of configuration requirements.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self, Union, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import (
    BaseOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from haive.core.common.mixins.structured_output_mixin import StructuredOutputMixin
    from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin
    from haive.core.models.llm.base import AzureLLMConfig, LLMConfig
else:

    class StructuredOutputMixin:
        pass

    class ToolRouteMixin:
        pass

    try:
        from haive.core.models.llm.base import AzureLLMConfig, LLMConfig
    except ImportError:
        AzureLLMConfig = None
        LLMConfig = None
from haive.core.engine.base import EngineType, InvokableEngine

logger = logging.getLogger(__name__)
console = Console()
logger.setLevel(logging.WARNING)
DEBUG_OUTPUT = os.getenv("HAIVE_DEBUG_CONFIG", "FALSE").lower() in ("true", "1", "yes")


def debug_print(*args, **kwargs) -> None:
    """Print debug output only if DEBUG_OUTPUT is enabled."""
    if DEBUG_OUTPUT:
        try:
            from rich import print as rprint

            rprint(*args, **kwargs)
        except ImportError:
            pass
    elif args:
        logger.debug(" ".join((str(arg) for arg in args)))


ParserType = Literal["pydantic", "pydantic_tools", "str", "json", "custom"]
StructuredOutputVersion = Literal["v1", "v2"]
ToolChoiceMode = Literal["auto", "required", "optional", "none"]


def _get_augllm_base_classes():
    """Dynamically load base classes only when needed."""
    from haive.core.common.mixins.structured_output_mixin import StructuredOutputMixin
    from haive.core.common.mixins.tool_route_mixin import ToolRouteMixin

    return (
        ToolRouteMixin,
        StructuredOutputMixin,
        InvokableEngine[
            Union[str, dict[str, Any], list[BaseMessage]],
            Union[BaseMessage, dict[str, Any]],
        ],
    )


class AugLLMConfig(*_get_augllm_base_classes()):
    """Configuration for creating enhanced LLM chains with flexible message handling.

    AugLLMConfig provides a structured way to configure and create LLM chains
    with prompts, tools, output parsers, and structured output models with
    comprehensive validation and automatic updates. It serves as the central
    configuration class for language model interactions in the Haive framework.

    This class integrates several key functionalities:
    1. Prompt template management with support for few-shot learning
    2. Tool integration and discovery with automatic routing
    3. Structured output handling (both parser-based and tool-based approaches)
    4. Message handling for chat-based LLMs
    5. Pre/post processing hooks for customization

    The configuration system is designed to be highly flexible while enforcing
    consistent patterns and proper validation, making it easier to create reliable
    language model interactions.

    Attributes:
        engine_type (EngineType): The type of engine (always LLM).
        llm_config (LLMConfig): Configuration for the LLM provider.
        prompt_template (Optional[BasePromptTemplate]): Template for structuring prompts.
        system_message (Optional[str]): System message for chat models.
        tools (Sequence[Union[Type[BaseTool], Type[BaseModel], Callable, StructuredTool, BaseModel]]):
            Tools that can be bound to the LLM.
        structured_output_model (Optional[Type[BaseModel]]): Pydantic model for structured outputs.
        structured_output_version (Optional[StructuredOutputVersion]):
            Version of structured output handling (v1: parser-based, v2: tool-based).
        temperature (Optional[float]): Temperature parameter for the LLM.
        max_tokens (Optional[int]): Maximum number of tokens to generate.
        preprocess (Optional[Callable]): Function to preprocess input before sending to LLM.
        postprocess (Optional[Callable]): Function to postprocess output from LLM.

    Examples:
        >>> from haive.core.engine.aug_llm.config import AugLLMConfig
        >>> from haive.core.models.llm.base import AzureLLMConfig
        >>> from pydantic import BaseModel, Field
        >>>
        >>> # Define a structured output model
        >>> class MovieReview(BaseModel):
        ...     title: str = Field(description="Title of the movie")
        ...     rating: int = Field(description="Rating from 1-10")
        ...     review: str = Field(description="Detailed review of the movie")
        >>>
        >>> # Create a basic configuration
        >>> config = AugLLMConfig(
        ...     name="movie_reviewer",
        ...     llm_config=AzureLLMConfig(model="gpt-4"),
        ...     system_message="You are a professional movie critic.",
        ...     structured_output_model=MovieReview,
        ...     temperature=0.7
        ... )
        >>>
        >>> # Create a runnable from the configuration
        >>> reviewer = config.create_runnable()
        >>>
        >>> # Use the runnable
        >>> result = reviewer.invoke("Review the movie 'Inception'")
    """

    engine_type: EngineType = Field(
        default=EngineType.LLM, description="The type of engine"
    )
    llm_config: LLMConfig = Field(
        default_factory=lambda: (
            AzureLLMConfig(model="gpt-4o") if AzureLLMConfig else None
        ),
        description="LLM provider configuration",
    )
    prompt_template: BasePromptTemplate | None = Field(
        default_factory=lambda: ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("placeholder", "{messages}")]
        ),
        description="Prompt template for the LLM",
    )
    system_message: str | None = Field(
        default=None, description="System message for chat models"
    )
    messages_placeholder_name: str = Field(
        default="messages",
        description="Name of the messages placeholder in chat templates",
    )
    add_messages_placeholder: bool = Field(
        default=True, description="Whether to automatically add MessagesPlaceholder"
    )
    force_messages_optional: bool = Field(
        default=True, description="Force messages placeholder to be optional"
    )
    examples: list[dict[str, Any]] | None = Field(
        default=None, description="Examples for few-shot prompting"
    )
    example_prompt: PromptTemplate | None = Field(
        default=None, description="Template for formatting few-shot examples"
    )
    prefix: str | None = Field(
        default=None, description="Text before examples in few-shot prompting"
    )
    suffix: str | None = Field(
        default=None, description="Text after examples in few-shot prompting"
    )
    example_separator: str = Field(
        default="\n\n", description="Separator between examples in few-shot prompting"
    )
    input_variables: list[str] | None = Field(
        default=None, description="Input variables for the prompt template"
    )
    schemas: Sequence[type[BaseModel] | Callable | BaseModel | Any] = Field(
        default_factory=list, description="Schemas for tools (no LangChain validation)"
    )
    pydantic_tools: list[type[BaseModel]] = Field(
        default_factory=list, description="Pydantic models for tool schemas"
    )
    use_tool_for_format_instructions: bool = Field(
        default=False,
        description="Use a single tool Pydantic model for format instructions",
    )
    tool_is_base_model: bool = Field(
        default=False,
        description="Whether a tool is a BaseModel type (detected automatically)",
    )
    force_tool_use: bool = Field(
        default=False, description="Whether to force the LLM to use a tool (any tool)"
    )
    force_tool_choice: bool | str | list[str] | None = Field(
        default=None, description="Force specific tool(s) to be used"
    )
    tool_choice_mode: ToolChoiceMode = Field(
        default="auto", description="Tool choice mode to use for binding tools"
    )
    structured_output_model: type[BaseModel] | None = Field(
        default=None, description="Pydantic model for structured output"
    )
    structured_output_version: StructuredOutputVersion | None = Field(
        default=None,
        description="Version of structured output handling: v1 (traditional), v2 (tool-based), None (disabled)",
    )
    if TYPE_CHECKING:
        output_parser: BaseOutputParser | None = Field(
            default=None, description="Parser for LLM output", exclude=True
        )
    else:
        output_parser: Any | None = Field(
            default=None, description="Parser for LLM output", exclude=True
        )
    parse_raw_output: bool = Field(
        default=False,
        description="Force parsing raw output even with structured output model",
    )
    include_format_instructions: bool = Field(
        default=True, description="Whether to include format instructions in the prompt"
    )
    parser_type: ParserType | None = Field(
        default=None,
        description="Parser type: 'pydantic', 'pydantic_tools', 'str', 'json', or 'custom'",
    )
    output_field_name: str | None = Field(
        default=None, description="Custom name for the primary output field in schema"
    )
    output_key: str | None = Field(
        default=None, description="Custom key for output when needed"
    )
    tool_kwargs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Parameters for tool instantiation"
    )
    bind_tools_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Parameters for binding tools to the LLM"
    )
    bind_tools_config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration for bind_tools"
    )
    preprocess: Callable[[Any], Any] | None = Field(
        default=None,
        description="Function to preprocess input before sending to LLM",
        exclude=True,
    )
    postprocess: Callable[[Any], Any] | None = Field(
        default=None,
        description="Function to postprocess output from LLM",
        exclude=True,
    )
    temperature: float | None = Field(
        default=None, description="Temperature parameter for the LLM"
    )
    max_tokens: int | None = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    runtime_options: dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime options for the LLM"
    )
    custom_runnables: list[Runnable] | None = Field(
        default=None, description="Custom runnables to add to the chain", exclude=True
    )
    partial_variables: dict[str, Any] = Field(
        default_factory=dict, description="Partial variables for the prompt template"
    )
    optional_variables: list[str] = Field(
        default_factory=list, description="Optional variables for the prompt template"
    )
    uses_messages_field: bool | None = Field(
        default=None,
        description="Explicitly specify if this engine uses a messages field. If None, auto-detected.",
    )
    _computed_input_fields: dict[str, tuple[type, Any]] = PrivateAttr(
        default_factory=dict
    )
    _computed_output_fields: dict[str, tuple[type, Any]] = PrivateAttr(
        default_factory=dict
    )
    _format_instructions_text: str | None = PrivateAttr(default=None)
    _is_processing_validation: bool = PrivateAttr(default=False)
    _tool_name_mapping: dict[str, str] = PrivateAttr(default_factory=dict)
    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    def model_post_init(self, __context) -> None:
        """Proper Pydantic post-initialization."""
        super().model_post_init(__context)
        self._initialize_tool_mixin()
        self._debug_initialization_summary()

    def _initialize_tool_mixin(self):
        """Initialize ToolListMixin functionality manually."""
        if not hasattr(self, "tool_routes"):
            self.tool_routes = {}
        self._sync_tool_routes()

    def _sync_tool_routes(self):
        """Synchronize tool_routes with current tools using enhanced detection."""
        if not self.tools:
            self.clear_tool_routes()
            return

        # Use enhanced tool analysis instead of basic sync
        for i, tool in enumerate(self.tools):
            tool_name = self._get_tool_name(tool, i)
            # Use _analyze_tool for smarter route detection
            route, metadata = self._analyze_tool(tool)
            metadata = metadata or {}
            metadata.update(
                {"source": "aug_llm_config", "index": i, "enhanced_detection": True}
            )
            self.set_tool_route(tool_name, route, metadata)
        if self.structured_output_model:
            # Fix routing for structured output model - use sanitized name
            from haive.core.utils.naming import sanitize_tool_name

            original_name = self.structured_output_model.__name__
            sanitized_name = sanitize_tool_name(original_name)

            # Check both original and sanitized names in tool_routes
            if original_name in self.tool_routes:
                metadata = self.get_tool_metadata(original_name) or {}
                metadata["is_structured_output"] = True
                metadata["purpose"] = "structured_output"
                metadata["version"] = self.structured_output_version
                metadata["tool_type"] = "structured_output_model"

                # Route to parse_output with sanitized name
                self.set_tool_route(sanitized_name, "parse_output", metadata)

                # Remove the original name if it's different
                if (
                    original_name != sanitized_name
                    and original_name in self.tool_routes
                ):
                    del self.tool_routes[original_name]
            elif sanitized_name in self.tool_routes:
                metadata = self.get_tool_metadata(sanitized_name) or {}
                metadata["is_structured_output"] = True
                metadata["purpose"] = "structured_output"
                metadata["version"] = self.structured_output_version
                metadata["tool_type"] = "structured_output_model"
                # Route to parse_output
                self.set_tool_route(sanitized_name, "parse_output", metadata)

    def _debug_initialization_summary(self):
        """Show rich initialization summary."""
        if not DEBUG_OUTPUT:
            return
        tree = Tree("🚀 [bold blue]AugLLMConfig Initialization[/bold blue]")
        basic = tree.add("📋 [cyan]Basic Configuration[/cyan]")
        basic.add(f"Name: [yellow]{self.name}[/yellow]")
        basic.add(f"ID: [yellow]{self.id}[/yellow]")
        basic.add(f"Engine Type: [yellow]{self.engine_type.value}[/yellow]")
        llm_info = tree.add("🤖 [cyan]LLM Configuration[/cyan]")
        llm_info.add(f"Provider: [yellow]{type(self.llm_config).__name__}[/yellow]")
        llm_info.add(
            f"Model: [yellow]{getattr(self.llm_config, 'model', 'Unknown')}[/yellow]"
        )
        tools_info = tree.add("🔧 [cyan]Tools Configuration[/cyan]")
        tools_info.add(f"Total Tools: [yellow]{len(self.tools)}[/yellow]")
        tools_info.add(f"Pydantic Tools: [yellow]{len(self.pydantic_tools)}[/yellow]")
        tools_info.add(f"Tool Routes: [yellow]{len(self.tool_routes)}[/yellow]")
        output_info = tree.add("📤 [cyan]Output Configuration[/cyan]")
        output_info.add(
            f"Structured Output Model: [yellow]{(self.structured_output_model.__name__ if self.structured_output_model else 'None')}[/yellow]"
        )
        output_info.add(
            f"Structured Output Version: [yellow]{self.structured_output_version or 'None'}[/yellow]"
        )
        output_info.add(f"Parser Type: [yellow]{self.parser_type or 'None'}[/yellow]")
        console.print(
            Panel(tree, title="Initialization Complete", border_style="green")
        )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v) -> Any:
        """Validate and auto-name tools."""
        if not v:
            return v
        validated_tools = []
        for _i, tool in enumerate(v):
            if hasattr(tool, "name") or isinstance(tool, type) or callable(tool):
                validated_tools.append(tool)
            else:
                validated_tools.append(tool)
        return validated_tools

    @field_validator("schemas")
    @classmethod
    def validate_schemas(cls, v) -> Any:
        """Validate and auto-name schemas."""
        if not v:
            return v
        return v

    @field_validator("structured_output_model")
    @classmethod
    def validate_structured_output_model(cls, v) -> Any:
        """Validate structured output model and default to tools-based validation."""
        if not v:
            return v
        if not issubclass(v, BaseModel):
            raise ValueError("structured_output_model must be a BaseModel subclass")
        return v

    @field_validator("prompt_template", mode="before")
    @classmethod
    def validate_prompt_template(cls, v) -> Any:
        """Validate and reconstruct prompt template from dict data."""
        if not v:
            return v
        if isinstance(v, BasePromptTemplate):
            return v
        if isinstance(v, dict):
            try:
                from langchain_core.load.load import load

                if "lc" in v and "type" in v and (v.get("type") == "constructor"):
                    logger.debug(
                        "Attempting to reconstruct prompt template using LangChain load"
                    )
                    reconstructed = load(v)
                    if isinstance(reconstructed, BasePromptTemplate):
                        logger.debug(
                            f"Successfully reconstructed: {type(reconstructed)}"
                        )
                        return reconstructed
            except Exception as e:
                logger.debug(
                    f"LangChain load failed: {e}, falling back to default template"
                )
            logger.debug("Creating default ChatPromptTemplate from dict")
            if "messages" in v:
                try:
                    messages = []
                    for msg_data in v.get("messages", []):
                        if isinstance(msg_data, dict):
                            msg_type = msg_data.get("type", "human")
                            if msg_type == "system":
                                messages.append(
                                    (
                                        "system",
                                        msg_data.get("prompt", {}).get(
                                            "template", "You are a helpful assistant."
                                        ),
                                    )
                                )
                            elif msg_type == "human":
                                messages.append(
                                    (
                                        "human",
                                        msg_data.get("prompt", {}).get(
                                            "template", "{input}"
                                        ),
                                    )
                                )
                            elif msg_type == "placeholder":
                                var_name = msg_data.get("variable_name", "messages")
                                messages.append(("placeholder", f"{{{var_name}}}"))
                    if messages:
                        return ChatPromptTemplate.from_messages(messages)
                except Exception as e:
                    logger.debug(f"Failed to reconstruct from messages: {e}")
            return ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful assistant."),
                    ("placeholder", "{messages}"),
                ]
            )
        logger.debug(f"Unexpected type {type(v)}, creating default ChatPromptTemplate")
        return ChatPromptTemplate.from_messages(
            [("system", "You are a helpful assistant."), ("placeholder", "{messages}")]
        )

    @model_validator(mode="before")
    @classmethod
    def set_default_structured_output_version(cls, data: dict[str, Any]):
        """Set default structured output version to v2 (tools) when model is provided but version is not."""
        if isinstance(data, dict):
            structured_output_model = data.get("structured_output_model")
            structured_output_version = data.get("structured_output_version")
            if structured_output_model and (not structured_output_version):
                data["structured_output_version"] = "v2"
        return data

    @model_validator(mode="before")
    @classmethod
    def ensure_structured_output_as_tool(cls, data: dict[str, Any]):
        """Ensure structured output model is properly configured for both v1 and v2."""
        if isinstance(data, dict):
            structured_output_model = data.get("structured_output_model")
            structured_output_version = data.get("structured_output_version")
            if not structured_output_model:
                return data
            tools = data.get("tools", [])
            if not isinstance(tools, list):
                tools = list(tools) if tools else []
            if structured_output_model not in tools:
                tools.append(structured_output_model)
                data["tools"] = tools
            if structured_output_version == "v2":
                data["force_tool_use"] = True
                data["tool_choice_mode"] = "required"
                if hasattr(structured_output_model, "__name__"):
                    # Use sanitized tool name for OpenAI compliance
                    from haive.core.utils.naming import sanitize_tool_name

                    data["force_tool_choice"] = sanitize_tool_name(
                        structured_output_model.__name__
                    )
        return data

    @model_validator(mode="before")
    @classmethod
    def default_schemas_to_tools(cls, data: dict[str, Any]):
        """Default schemas to tools if schemas isn't provided but tools has values."""
        if isinstance(data, dict):
            tools = data.get("tools", [])
            schemas = data.get("schemas", [])
            if tools and (not schemas):
                data["schemas"] = tools
        return data

    @model_validator(mode="after")
    def comprehensive_validation_and_setup(self) -> Self:
        """Comprehensive validation and setup after initialization."""
        if self._is_processing_validation:
            return self
        self._is_processing_validation = True
        try:
            debug_print(
                "🔍 [bold blue]Starting comprehensive validation and setup[/bold blue]"
            )
            self._process_and_validate_tools()
            self._create_prompt_template_if_needed()
            self._ensure_messages_placeholder_handling()
            self._apply_partial_variables()
            self._apply_optional_variables()
            self._setup_format_instructions()
            self._setup_output_handling()
            self._configure_tool_choice()
            if self.uses_messages_field is None:
                self.uses_messages_field = self._detect_uses_messages_field()
            self._compute_schema_fields()
            self._final_validation_check()
            self._debug_final_configuration()
            debug_print(
                "✅ [bold green]Comprehensive validation and setup complete[/bold green]"
            )
        except Exception as e:
            debug_print(f"❌ [bold red]Error during validation: {e}[/bold red]")
            raise
        finally:
            self._is_processing_validation = False
        return self

    def _analyze_tool(self, tool: Any) -> tuple[str, dict[str, Any] | None]:
        """Analyze tool with AugLLM-specific context awareness.

        Extends ToolRouteMixin's analysis with structured output detection.
        Routes structured output models to 'parse_output' for ValidationNodeConfigV2.
        """
        # Check if this is the configured structured output model
        if self.structured_output_model and tool == self.structured_output_model:
            return "parse_output", {
                "purpose": "structured_output",
                "version": self.structured_output_version,
                "force_choice": self.structured_output_version == "v2",
                "class_name": tool.__name__ if hasattr(tool, "__name__") else str(tool),
                "tool_type": "structured_output_model",
                "is_structured_output": True,
            }

        # Check if tool has STRUCTURED_OUTPUT capability
        from haive.core.engine.tool.types import ToolCapability

        capabilities = getattr(tool, "__tool_capabilities__", set())
        if ToolCapability.STRUCTURED_OUTPUT in capabilities:
            tool_name = getattr(tool, "name", getattr(tool, "__name__", "unknown"))
            return "parse_output", {
                "tool_name": tool_name,
                "tool_type": "structured_output_tool",
                "capabilities": list(capabilities),
                "has_structured_output": True,
                "purpose": "structured_output",
            }

        return super()._analyze_tool(tool)

    def _process_and_validate_tools(self):
        """Process tools using unified ToolRouteMixin functionality."""
        debug_print("🔧 [blue]Processing and validating tools...[/blue]")
        if not self.tools:
            self.clear_tools()
            self.pydantic_tools = []
            self.tool_is_base_model = False
            self._tool_name_mapping = {}
            debug_print(
                "📝 [yellow]No tools provided - cleared tool-related fields[/yellow]"
            )
            return
        basemodel_tools = []
        tool_names = []
        tool_name_mapping = {}
        for i, tool in enumerate(self.tools):
            self.add_tool(tool)
            tool_name = self._get_tool_name(tool, i)
            self.tool_routes.get(tool_name, "unknown")
            actual_tool_name = tool_name
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                basemodel_tools.append(tool)
                if tool not in self.pydantic_tools:
                    self.pydantic_tools.append(tool)
                    debug_print(
                        f"➕ [green]Added BaseModel {tool.__name__} to pydantic_tools[/green]"
                    )
            if tool_name:
                tool_names.append(tool_name)
                tool_name_mapping[tool_name] = actual_tool_name
        self._tool_name_mapping = tool_name_mapping
        current_basemodel_tools = set(basemodel_tools)
        self.pydantic_tools = [
            tool for tool in self.pydantic_tools if tool in current_basemodel_tools
        ]
        self.tool_is_base_model = len(basemodel_tools) > 0
        if not hasattr(self, "metadata"):
            self.metadata = {}
        self.metadata["tool_names"] = tool_names
        self.metadata["has_basemodel_tools"] = bool(basemodel_tools)
        self.metadata["basemodel_tool_count"] = len(basemodel_tools)
        self.metadata["tool_name_mapping"] = tool_name_mapping
        debug_print(
            f"📊 [cyan]Tool processing complete: {len(tool_names)} tools, {len(basemodel_tools)} BaseModel tools[/cyan]"
        )

    def _setup_format_instructions(self):
        """Set up format instructions with proper validation - for both v1 and v2."""
        debug_print("📝 [blue]Setting up format instructions...[/blue]")
        if "format_instructions" in self.partial_variables:
            del self.partial_variables["format_instructions"]
            self._format_instructions_text = None
        should_setup = self._should_setup_format_instructions()
        if not should_setup:
            debug_print(
                "🚫 [yellow]Format instructions not needed - conditions not met[/yellow]"
            )
            return
        try:
            debug_print(
                f"📋 [green]Setting up format instructions for: {self.structured_output_model.__name__}[/green]"
            )
            parser = PydanticOutputParser(pydantic_object=self.structured_output_model)
            instructions = parser.get_format_instructions()
            self.partial_variables["format_instructions"] = instructions
            self._format_instructions_text = instructions
            debug_print(
                "✅ [green]Format instructions added using PydanticOutputParser[/green]"
            )
        except Exception as e:
            debug_print(f"❌ [red]Error setting up format instructions: {e}[/red]")

    def _should_setup_format_instructions(self) -> bool:
        """Determine if format instructions should be set up."""
        if not self.include_format_instructions:
            debug_print("❌ [yellow]include_format_instructions is False[/yellow]")
            return False
        if "format_instructions" in self.partial_variables:
            debug_print(
                "❌ [yellow]format_instructions already exists in partial_variables[/yellow]"
            )
            return False
        if not self.structured_output_model:
            debug_print("❌ [yellow]No structured_output_model set[/yellow]")
            return False
        debug_print("✅ [green]Conditions met for format instructions[/green]")
        return True

    def _setup_output_handling(self):
        """Set up output handling based on configuration with proper validation."""
        debug_print("📤 [blue]Setting up output handling...[/blue]")
        if self.parse_raw_output:
            debug_print("📄 [yellow]Using StrOutputParser for raw output[/yellow]")
            self.output_parser = StrOutputParser()
            self.parser_type = "str"
            return
        if self.output_parser:
            debug_print("🎯 [cyan]Using explicitly provided output_parser[/cyan]")
            if not self.parser_type:
                if isinstance(self.output_parser, StrOutputParser):
                    self.parser_type = "str"
                elif isinstance(self.output_parser, PydanticOutputParser):
                    self.parser_type = "pydantic"
                elif isinstance(self.output_parser, PydanticToolsParser):
                    self.parser_type = "pydantic_tools"
                else:
                    self.parser_type = "custom"
            return
        if self.structured_output_model and self.structured_output_version == "v2":
            debug_print(
                "[cyan]V2 structured output: NO PARSER (tool-based approach)[/cyan]"
            )
            self.output_parser = None
            self.parser_type = None
            debug_print(
                "[green]V2 mode: output_parser and parser_type set to None[/green]"
            )
            return
        if self.structured_output_model and self.structured_output_version == "v1":
            debug_print("[cyan]V1 structured output: setting up parser[/cyan]")
            self.parser_type = "pydantic"
            self.output_parser = PydanticOutputParser(
                pydantic_object=self.structured_output_model
            )
            return
        if self.pydantic_tools and (not self.structured_output_model):
            debug_print(
                "🔧 [cyan]Pydantic tools detected but no structured output model[/cyan]"
            )
            if self.parser_type == "pydantic_tools":
                self.output_parser = PydanticToolsParser(tools=self.pydantic_tools)
                debug_print(
                    "[green]Created PydanticToolsParser for explicit pydantic tools[/green]"
                )
            else:
                debug_print(
                    "🎯 [cyan]No automatic parser set for pydantic tools - user must specify parser_type[/cyan]"
                )
        else:
            debug_print("📝 [yellow]No specific output parser configuration[/yellow]")

    def _setup_v2_structured_output(self):
        """Setup v2 (tool-based) approach - force tool usage with format instructions, NO parsing."""
        debug_print(
            f"🔧 [cyan]Setting up v2 approach (tool + format instructions, NO PARSER) with {self.structured_output_model.__name__}[/cyan]"
        )
        if self.structured_output_model not in self.tools:
            self.tools = list(self.tools) if self.tools else []
            self.tools.append(self.structured_output_model)
            debug_print(
                f"➕ [green]Added {self.structured_output_model.__name__} to tools[/green]"
            )
        if self.structured_output_model not in self.pydantic_tools:
            self.pydantic_tools.append(self.structured_output_model)
        self.parser_type = None
        self.output_parser = None
        debug_print(
            "🎯 [green]V2 mode: NO PARSER - tool forcing with format instructions only[/green]"
        )
        self.force_tool_use = True
        self.tool_choice_mode = "required"
        model_class_name = self.structured_output_model.__name__
        actual_tool_name = model_class_name
        self._tool_name_mapping[model_class_name] = actual_tool_name
        self.force_tool_choice = actual_tool_name
        debug_print(
            f"🎯 [green]Set force_tool_choice to '{actual_tool_name}' (exact class name)[/green]"
        )
        if self.include_format_instructions:
            try:
                parser = PydanticOutputParser(
                    pydantic_object=self.structured_output_model
                )
                instructions = parser.get_format_instructions()
                self.partial_variables["format_instructions"] = instructions
                self._format_instructions_text = instructions
                debug_print(
                    "✅ [green]Added format instructions from PydanticOutputParser for structured output model[/green]"
                )
            except Exception as e:
                debug_print(f"❌ [red]Error setting up format instructions: {e}[/red]")
        self._update_bind_tools_kwargs_for_v2()

    def _setup_v1_structured_output(self):
        """Setup v1 (traditional) structured output."""
        debug_print(
            f"📋 [cyan]Setting up v1 structured output with {self.structured_output_model.__name__}[/cyan]"
        )
        self.parser_type = "pydantic"
        self.output_parser = PydanticOutputParser(
            pydantic_object=self.structured_output_model
        )

    def _configure_tool_choice(self):
        """Configure tool choice based on available tools and settings."""
        if not self.tools:
            self.force_tool_use = False
            self.force_tool_choice = None
            self.tool_choice_mode = "auto"
            debug_print("🚫 [yellow]No tools - disabling tool choice[/yellow]")
            return
        debug_print("⚙️ [blue]Configuring tool choice...[/blue]")
        tool_names = self.metadata.get("tool_names", [])
        if isinstance(self.force_tool_choice, bool):
            if self.force_tool_choice:
                self.tool_choice_mode = "required"
                self.force_tool_use = True
                self.force_tool_choice = None
            else:
                self.tool_choice_mode = "optional"
                self.force_tool_use = False
                self.force_tool_choice = None
            debug_print(
                f"🔄 [yellow]Converted boolean force_tool_choice to mode: {self.tool_choice_mode}[/yellow]"
            )
        elif isinstance(self.force_tool_choice, str):
            self.force_tool_use = True
            self.tool_choice_mode = "required"
            actual_tool_names = list(self._tool_name_mapping.values())
            if (
                self.force_tool_choice not in actual_tool_names
                and self.force_tool_choice not in tool_names
            ):
                debug_print(
                    f"⚠️ [yellow]Warning: force_tool_choice '{self.force_tool_choice}' not in available tools: {actual_tool_names}[/yellow]"
                )
        elif (
            isinstance(self.force_tool_choice, list | tuple) and self.force_tool_choice
        ):
            self.force_tool_use = True
            self.tool_choice_mode = "required"
            self.force_tool_choice = self.force_tool_choice[0]
            debug_print(
                f"📝 [yellow]Multiple forced tools not supported - using first: {self.force_tool_choice}[/yellow]"
            )
        elif self.force_tool_use and (not self.force_tool_choice):
            self.tool_choice_mode = "required"
            if len(tool_names) == 1:
                display_name = tool_names[0]
                actual_name = self._tool_name_mapping.get(display_name, display_name)
                self.force_tool_choice = actual_name
                debug_print(
                    f"🎯 [green]Auto-selected single tool: {self.force_tool_choice}[/green]"
                )
        if self.structured_output_version != "v2":
            self._update_bind_tools_kwargs()

    def _update_bind_tools_kwargs(self):
        """Update bind_tools_kwargs based on current tool choice configuration."""
        if self.tool_choice_mode == "required":
            if self.force_tool_choice:
                self.bind_tools_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": self.force_tool_choice},
                }
            else:
                self.bind_tools_kwargs["tool_choice"] = "required"
        elif self.tool_choice_mode == "auto":
            self.bind_tools_kwargs["tool_choice"] = "auto"
        elif self.tool_choice_mode == "none":
            self.bind_tools_kwargs["tool_choice"] = "none"
        else:
            self.bind_tools_kwargs["tool_choice"] = "auto"

    def _update_bind_tools_kwargs_for_v2(self):
        """Update bind_tools_kwargs specifically for v2 structured output."""
        if self.force_tool_choice:
            self.bind_tools_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": self.force_tool_choice},
            }
            debug_print(
                f"🔧 [green]Set bind_tools_kwargs for v2: forcing tool '{self.force_tool_choice}'[/green]"
            )
        else:
            self.bind_tools_kwargs["tool_choice"] = "required"
            debug_print(
                "🔧 [yellow]Set bind_tools_kwargs for v2: 'required' (no specific tool)[/yellow]"
            )

    def _create_prompt_template_if_needed(self):
        """Create appropriate prompt template based on available components."""
        if self.prompt_template is not None:
            return
        debug_print("📝 [blue]Creating prompt template...[/blue]")
        if self.examples and self.example_prompt and self.prefix and self.suffix:
            debug_print("📚 [green]Creating FewShotPromptTemplate[/green]")
            self._create_few_shot_template()
        elif self.examples and isinstance(self.example_prompt, ChatPromptTemplate):
            debug_print("💬 [green]Creating FewShotChatMessagePromptTemplate[/green]")
            self._create_few_shot_chat_template()
        elif self.system_message:
            debug_print(
                "🤖 [green]Creating ChatPromptTemplate from system message[/green]"
            )
            self._create_chat_template_from_system()
        elif self.add_messages_placeholder:
            debug_print("📋 [green]Creating default ChatPromptTemplate[/green]")
            self._create_default_chat_template()

    def _ensure_messages_placeholder_handling(self):
        """Ensure messages placeholder is properly handled based on configuration."""
        if not self.prompt_template:
            return
        if isinstance(self.prompt_template, ChatPromptTemplate):
            debug_print(
                "💬 [blue]Handling messages placeholder for ChatPromptTemplate[/blue]"
            )
            self._handle_chat_template_messages_placeholder()
        elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
            debug_print(
                "📚 [blue]FewShotChatMessagePromptTemplate detected - messages enabled[/blue]"
            )
            self.uses_messages_field = True
        elif isinstance(self.prompt_template, FewShotPromptTemplate):
            debug_print(
                "📝 [blue]FewShotPromptTemplate detected - messages not applicable[/blue]"
            )
            self.uses_messages_field = False
        else:
            debug_print("🔍 [blue]Checking template for message variables[/blue]")
            self._check_template_for_messages_variables()

    def _handle_chat_template_messages_placeholder(self):
        """Handle messages placeholder in ChatPromptTemplate."""
        messages = list(self.prompt_template.messages)
        has_messages_placeholder = False
        messages_placeholder_index = -1
        for i, msg in enumerate(messages):
            if (
                isinstance(msg, MessagesPlaceholder)
                and getattr(msg, "variable_name", "") == self.messages_placeholder_name
            ):
                has_messages_placeholder = True
                messages_placeholder_index = i
                break
        if not has_messages_placeholder and self.add_messages_placeholder:
            should_be_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )
            new_placeholder = MessagesPlaceholder(
                variable_name=self.messages_placeholder_name,
                optional=should_be_optional,
            )
            messages.append(new_placeholder)
            self._update_chat_template_messages(messages)
            debug_print(
                f"➕ [green]Added messages placeholder (optional={should_be_optional})[/green]"
            )
        elif has_messages_placeholder:
            placeholder = messages[messages_placeholder_index]
            should_be_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )
            if (
                hasattr(placeholder, "optional")
                and placeholder.optional != should_be_optional
            ):
                messages[messages_placeholder_index] = MessagesPlaceholder(
                    variable_name=self.messages_placeholder_name,
                    optional=should_be_optional,
                )
                self._update_chat_template_messages(messages)
                debug_print(
                    f"🔄 [cyan]Updated messages placeholder optional={should_be_optional}[/cyan]"
                )

    def _update_chat_template_messages(self, messages: list[Any]):
        """Update ChatPromptTemplate with new messages list."""
        partial_vars = getattr(self.prompt_template, "partial_variables", {}) or {}
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        if partial_vars:
            self.prompt_template = self.prompt_template.partial(**partial_vars)
        self.uses_messages_field = True

    def _create_default_chat_template(self):
        """Create a default ChatPromptTemplate with optional messages placeholder."""
        should_be_optional = (
            self.force_messages_optional
            or self.messages_placeholder_name in self.optional_variables
        )
        messages = [
            MessagesPlaceholder(
                variable_name=self.messages_placeholder_name,
                optional=should_be_optional,
            )
        ]
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.uses_messages_field = True

    def _create_chat_template_from_system(self):
        """Create a ChatPromptTemplate from system_message."""
        messages = [SystemMessage(content=self.system_message)]
        if self.add_messages_placeholder:
            should_be_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )
            messages.append(
                MessagesPlaceholder(
                    variable_name=self.messages_placeholder_name,
                    optional=should_be_optional,
                )
            )
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.uses_messages_field = True

    def _create_few_shot_template(self):
        """Create a FewShotPromptTemplate from examples and example_prompt."""
        self.prompt_template = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
            prefix=self.prefix,
            suffix=self.suffix,
            input_variables=self.input_variables or [],
            example_separator=self.example_separator,
            partial_variables=self.partial_variables,
            optional_variables=self.optional_variables,
        )
        self.uses_messages_field = False

    def _create_few_shot_chat_template(self):
        """Create a FewShotChatMessagePromptTemplate using example_prompt."""
        prefix_messages = []
        if self.system_message:
            prefix_messages = [SystemMessage(content=self.system_message)]
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=self.examples, example_prompt=self.example_prompt
        )
        messages = [*prefix_messages, few_shot_prompt]
        if self.add_messages_placeholder:
            should_be_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )
            messages.append(
                MessagesPlaceholder(
                    variable_name=self.messages_placeholder_name,
                    optional=should_be_optional,
                )
            )
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.uses_messages_field = True

    def _check_template_for_messages_variables(self):
        """Check if the template uses messages variables."""
        if hasattr(self.prompt_template, "input_variables"):
            input_vars = getattr(self.prompt_template, "input_variables", [])
            if self.messages_placeholder_name in input_vars:
                self.uses_messages_field = True
                return
        self.uses_messages_field = False

    def _apply_partial_variables(self):
        """Apply partial variables to the prompt template."""
        if not self.prompt_template or not self.partial_variables:
            return
        try:
            if hasattr(self.prompt_template, "partial"):
                self.prompt_template = self.prompt_template.partial(
                    **self.partial_variables
                )
                debug_print(
                    f"✅ [green]Applied {len(self.partial_variables)} partial variables[/green]"
                )
        except Exception as e:
            debug_print(f"❌ [red]Error applying partial variables: {e}[/red]")

    def _apply_optional_variables(self):
        """Apply optional variables to the prompt template."""
        if not self.optional_variables or not self.prompt_template:
            return
        if isinstance(self.prompt_template, ChatPromptTemplate):
            if self.messages_placeholder_name in self.optional_variables:
                self._handle_chat_template_messages_placeholder()
        elif hasattr(self.prompt_template, "optional_variables"):
            if not hasattr(self.prompt_template.optional_variables, "extend"):
                self.prompt_template.optional_variables = list(
                    self.prompt_template.optional_variables
                )
            for var in self.optional_variables:
                if var not in self.prompt_template.optional_variables:
                    self.prompt_template.optional_variables.append(var)

    def _detect_uses_messages_field(self) -> bool:
        """Detect if this LLM configuration uses a messages field."""
        if not self.add_messages_placeholder:
            if isinstance(self.prompt_template, ChatPromptTemplate):
                for msg in self.prompt_template.messages:
                    if (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == self.messages_placeholder_name
                    ):
                        return True
            return False
        if self.tools or self.system_message:
            return True
        if self.prompt_template:
            if isinstance(
                self.prompt_template,
                ChatPromptTemplate | FewShotChatMessagePromptTemplate,
            ):
                return True
            if isinstance(self.prompt_template, FewShotPromptTemplate):
                return False
            if hasattr(self.prompt_template, "input_variables"):
                return (
                    self.messages_placeholder_name
                    in self.prompt_template.input_variables
                )
        return True

    def _compute_schema_fields(self):
        """Compute input and output schema fields."""
        debug_print("📊 [blue]Computing schema fields...[/blue]")
        self._computed_input_fields = self._compute_input_fields()
        self._computed_output_fields = self._compute_output_fields()
        debug_print(
            f"📥 [cyan]Input fields: {list(self._computed_input_fields.keys())}[/cyan]"
        )
        debug_print(
            f"📤 [cyan]Output fields: {list(self._computed_output_fields.keys())}[/cyan]"
        )

    def _compute_input_fields(self) -> dict[str, tuple[type, Any]]:
        """Compute input fields based on prompt template and configuration."""
        from typing import Any as AnyType
        from typing import Optional as OptionalType

        fields = {}
        required_vars = self._get_input_variables()
        partial_vars = {}
        if self.prompt_template and hasattr(self.prompt_template, "partial_variables"):
            partial_vars = getattr(self.prompt_template, "partial_variables", {})
        if self.uses_messages_field:
            is_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )
            if is_optional:
                fields[self.messages_placeholder_name] = (
                    OptionalType[list[BaseMessage]],
                    Field(default_factory=list),
                )
            else:
                fields[self.messages_placeholder_name] = (
                    list[BaseMessage],
                    Field(default_factory=list),
                )
        for var in required_vars:
            if (
                var != self.messages_placeholder_name
                and var not in fields
                and (var not in partial_vars)
            ):
                fields[var] = (AnyType, None)
        for var in self.optional_variables:
            if var != self.messages_placeholder_name and var not in fields:
                fields[var] = (OptionalType[AnyType], None)
        for var, default_value in partial_vars.items():
            if (
                var != self.messages_placeholder_name
                and var not in fields
                and (var != "format_instructions")
            ):
                fields[var] = (OptionalType[AnyType], Field(default=default_value))
        return fields

    def _compute_output_fields(self) -> dict[str, tuple[type, Any]]:
        """Compute output fields based on configuration."""
        from typing import Any as AnyType

        fields = {}
        if self.structured_output_version == "v2":
            fields[self.messages_placeholder_name] = (
                list[BaseMessage],
                Field(default_factory=list),
            )
            return fields
        if self.output_parser:
            if self.parser_type == "str" or isinstance(
                self.output_parser, StrOutputParser
            ):
                field_name = self.output_field_name or self.output_key or "content"
                fields[field_name] = (str, None)
            elif self.parser_type == "json":
                field_name = self.output_field_name or self.output_key or "content"
                fields[field_name] = (dict[str, AnyType], None)
            elif self.parser_type == "pydantic" and hasattr(
                self.output_parser, "pydantic_object"
            ):
                pydantic_model = self.output_parser.pydantic_object
                if hasattr(pydantic_model, "model_fields"):
                    for field_name, field_info in pydantic_model.model_fields.items():
                        fields[field_name] = (field_info.annotation, field_info.default)
                else:
                    model_name = self.output_field_name or self.output_key or "result"
                    fields[model_name] = (pydantic_model, None)
            elif self.parser_type == "pydantic_tools":
                field_name = self.output_field_name or self.output_key or "tool_result"
                fields[field_name] = (dict[str, AnyType], None)
            else:
                field_name = self.output_field_name or self.output_key or "content"
                fields[field_name] = (AnyType, None)
            if self.uses_messages_field:
                fields[self.messages_placeholder_name] = (
                    list[BaseMessage],
                    Field(default_factory=list),
                )
            return fields
        fields[self.messages_placeholder_name] = (
            list[BaseMessage],
            Field(default_factory=list),
        )
        return fields

    def _get_input_variables(self) -> set[str]:
        """Get all input variables required by the prompt template."""
        all_vars = set()
        if not self.prompt_template:
            return (
                {self.messages_placeholder_name} if self.uses_messages_field else set()
            )
        if hasattr(self.prompt_template, "input_variables"):
            vars_list = getattr(self.prompt_template, "input_variables", [])
            all_vars.update(vars_list)
        if isinstance(self.prompt_template, ChatPromptTemplate):
            for msg in self.prompt_template.messages:
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "input_variables"):
                    all_vars.update(msg.prompt.input_variables)
                if hasattr(msg, "variable_name"):
                    var_name = msg.variable_name
                    if (
                        not getattr(msg, "optional", False)
                        or not self.force_messages_optional
                    ):
                        all_vars.add(var_name)
        partial_vars = set(self.partial_variables.keys())
        if hasattr(self.prompt_template, "partial_variables"):
            partial_vars.update(
                getattr(self.prompt_template, "partial_variables", {}).keys()
            )
        all_vars = all_vars - partial_vars - set(self.optional_variables)
        return all_vars

    def _format_model_schema(
        self, model_name: str, schema: dict[str, Any], as_section: bool = False
    ) -> str:
        """Format a model schema as instructions."""
        schema_json = json.dumps(schema, indent=2)
        header = f"## {model_name}\n" if as_section else ""
        return f"{header}You must format your response as JSON that matches this schema:\n\n```json\n{schema_json}\n```\n\nThe output should be valid JSON that conforms to the {model_name} schema."

    def _final_validation_check(self):
        """Perform final validation checks."""
        debug_print("🔍 [blue]Performing final validation checks...[/blue]")
        if self.structured_output_model and self.structured_output_version:
            if self.structured_output_model not in self.pydantic_tools:
                debug_print(
                    "⚠️ [yellow]Structured output model not in pydantic_tools - adding[/yellow]"
                )
                self.pydantic_tools.append(self.structured_output_model)
        if self.force_tool_choice and (not self.tools):
            debug_print(
                "⚠️ [yellow]force_tool_choice set but no tools available[/yellow]"
            )
            self.force_tool_choice = None
            self.force_tool_use = False
        if self.structured_output_version == "v2":
            if self.output_parser is not None:
                debug_print(
                    "🔧 [yellow]V2 detected: clearing output_parser (should be None)[/yellow]"
                )
                self.output_parser = None
            if self.parser_type is not None:
                debug_print(
                    "🔧 [yellow]V2 detected: clearing parser_type (should be None)[/yellow]"
                )
                self.parser_type = None
            if self.structured_output_model:
                if self.structured_output_model not in self.tools:
                    self.tools = list(self.tools) if self.tools else []
                    self.tools.append(self.structured_output_model)
                    debug_print(
                        "🔧 [green]Added structured_output_model to tools for v2[/green]"
                    )
                # Use sanitized tool name for OpenAI compliance
                from haive.core.utils.naming import sanitize_tool_name

                expected_tool_name = sanitize_tool_name(
                    self.structured_output_model.__name__
                )
                if self.force_tool_choice != expected_tool_name:
                    self.force_tool_choice = expected_tool_name
                    debug_print(
                        f"🔧 [green]Set force_tool_choice to '{expected_tool_name}' for v2[/green]"
                    )
                if self.tool_choice_mode != "required":
                    self.tool_choice_mode = "required"
                    debug_print(
                        "🔧 [green]Set tool_choice_mode to 'required' for v2[/green]"
                    )
                if not self.force_tool_use:
                    self.force_tool_use = True
                    debug_print("🔧 [green]Set force_tool_use to True for v2[/green]")
        elif self.structured_output_version == "v1" and self.structured_output_model:
            if self.output_parser is None and self.parser_type != "pydantic":
                debug_print(
                    "🔧 [yellow]V1 detected: setting up PydanticOutputParser[/yellow]"
                )
                self.output_parser = PydanticOutputParser(
                    pydantic_object=self.structured_output_model
                )
                self.parser_type = "pydantic"
        debug_print("✅ [green]Final validation checks complete[/green]")

    def _debug_final_configuration(self):
        """Debug final configuration state."""
        if not DEBUG_OUTPUT:
            return
        tree = Tree("🎯 [bold blue]Final Configuration State[/bold blue]")
        tools_section = tree.add("🔧 [cyan]Tools Configuration[/cyan]")
        tools_section.add(f"Total Tools: [yellow]{len(self.tools)}[/yellow]")
        tools_section.add(
            f"Pydantic Tools: [yellow]{len(self.pydantic_tools)}[/yellow]"
        )
        tools_section.add(f"Tool Routes: [yellow]{len(self.tool_routes)}[/yellow]")
        tools_section.add(
            f"Tool Is BaseModel: [yellow]{self.tool_is_base_model}[/yellow]"
        )
        tools_section.add(f"Force Tool Use: [yellow]{self.force_tool_use}[/yellow]")
        tools_section.add(f"Tool Choice Mode: [yellow]{self.tool_choice_mode}[/yellow]")
        tools_section.add(
            f"Force Tool Choice: [yellow]{self.force_tool_choice}[/yellow]"
        )
        output_section = tree.add("📤 [cyan]Output Configuration[/cyan]")
        output_section.add(
            f"Structured Output Model: [yellow]{(self.structured_output_model.__name__ if self.structured_output_model else 'None')}[/yellow]"
        )
        output_section.add(
            f"Structured Output Version: [yellow]{self.structured_output_version or 'None'}[/yellow]"
        )
        output_section.add(
            f"Parser Type: [yellow]{self.parser_type or 'None'}[/yellow]"
        )
        output_section.add(
            f"Format Instructions: [yellow]{('Set' if self._format_instructions_text else 'None')}[/yellow]"
        )
        schema_section = tree.add("📊 [cyan]Schema Configuration[/cyan]")
        schema_section.add(
            f"Uses Messages Field: [yellow]{self.uses_messages_field}[/yellow]"
        )
        schema_section.add(
            f"Input Fields: [yellow]{len(self._computed_input_fields)}[/yellow]"
        )
        schema_section.add(
            f"Output Fields: [yellow]{len(self._computed_output_fields)}[/yellow]"
        )
        if self._tool_name_mapping:
            mapping_section = tree.add("🏷️ [cyan]Tool Name Mapping[/cyan]")
            for display_name, actual_name in self._tool_name_mapping.items():
                mapping_section.add(f"{display_name} → [yellow]{actual_name}[/yellow]")
        console.print(Panel(tree, title="Configuration Complete", border_style="green"))

    def _debug_log(self, title: str, content: dict[str, Any]):
        """Pretty print debug information."""
        if not DEBUG_OUTPUT:
            return
        table = Table(title=title, title_justify="left", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")
        for key, value in content.items():
            if value is not None:
                formatted_value = str(value)
                if len(formatted_value) > 100:
                    formatted_value = formatted_value[:97] + "..."
                table.add_row(key, formatted_value)
        console.print(Panel(table, expand=False))

    def _get_prompt_template_info(self) -> str:
        """Get detailed information about the prompt template."""
        if not self.prompt_template:
            return "None"
        info = f"{type(self.prompt_template).__name__}"
        if isinstance(self.prompt_template, ChatPromptTemplate):
            msg_count = len(self.prompt_template.messages)
            has_placeholder = any(
                (
                    isinstance(msg, MessagesPlaceholder)
                    and getattr(msg, "variable_name", "")
                    == self.messages_placeholder_name
                    for msg in self.prompt_template.messages
                )
            )
            info += f" ({msg_count} messages, placeholder={has_placeholder})"
        elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
            example_count = len(getattr(self.prompt_template, "examples", []))
            info += f" ({example_count} examples)"
        return info

    def get_input_fields(self) -> dict[str, tuple[type, Any]]:
        """Get schema fields for input."""
        return self._computed_input_fields

    def get_output_fields(self) -> dict[str, tuple[type, Any]]:
        """Get schema fields for output."""
        return self._computed_output_fields

    def create_runnable(
        self, runnable_config: RunnableConfig | None = None
    ) -> Runnable:
        """Create a runnable LLM chain based on this configuration."""
        from haive.core.engine.aug_llm.factory import AugLLMFactory

        config_params = self.apply_runnable_config(runnable_config)
        factory = AugLLMFactory(self, config_params)
        return factory.create_runnable()

    def apply_runnable_config(
        self, runnable_config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        """Extract parameters from runnable_config relevant to this engine."""
        params = super().apply_runnable_config(runnable_config)
        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]
            aug_llm_params = [
                "tools",
                "force_tool_choice",
                "force_tool_use",
                "tool_choice_mode",
                "temperature",
                "max_tokens",
                "system_message",
                "partial_variables",
                "parse_raw_output",
                "messages_placeholder_name",
                "optional_variables",
                "include_format_instructions",
                "parser_type",
                "pydantic_tools",
                "add_messages_placeholder",
                "force_messages_optional",
                "use_tool_for_format_instructions",
                "structured_output_version",
                "output_field_name",
                "output_key",
            ]
            for param in aug_llm_params:
                if param in configurable:
                    params[param] = configurable[param]
        return params

    def _process_input(
        self, input_data: str | dict[str, Any] | list[BaseMessage]
    ) -> dict[str, Any]:
        """Process input into a format usable by the runnable."""
        debug_print("[blue]Processing input data[/blue]")
        required_vars = self._get_input_variables()
        if isinstance(input_data, dict):
            debug_print("[green]Input is already a dictionary[/green]")
            return input_data
        if isinstance(input_data, str):
            result = {}
            if (
                self.uses_messages_field
                and self.messages_placeholder_name not in result
            ):
                result[self.messages_placeholder_name] = [
                    HumanMessage(content=input_data)
                ]
                debug_print("[green]Added string to messages field[/green]")
            for var in required_vars:
                if var != self.messages_placeholder_name:
                    result[var] = input_data
                    debug_print(f"[cyan]Added string to field: {var}[/cyan]")
            return result
        if isinstance(input_data, list) and all(
            (isinstance(item, BaseMessage) for item in input_data)
        ):
            result = {self.messages_placeholder_name: input_data}
            debug_print("[green]Added message list to messages field[/green]")
            return result
        debug_print("[yellow]Converting unknown input to human message[/yellow]")
        return {self.messages_placeholder_name: [HumanMessage(content=str(input_data))]}

    def get_format_instructions(
        self, model: type[BaseModel] | None = None, as_tools: bool = False
    ) -> str:
        """Get format instructions for a model without changing the config."""
        target_model = model
        if target_model is None:
            if self.structured_output_model:
                target_model = self.structured_output_model
            elif self.pydantic_tools:
                if as_tools:
                    tool_instructions = []
                    for tool_model in self.pydantic_tools:
                        schema = tool_model.model_json_schema()
                        model_name = tool_model.__name__
                        tool_instructions.append(
                            self._format_model_schema(
                                model_name, schema, as_section=True
                            )
                        )
                    if len(tool_instructions) == 1:
                        return tool_instructions[0]
                    return (
                        "You must respond using one of the following formats:\n\n"
                        + "\n\n".join(tool_instructions)
                    )
                target_model = self.pydantic_tools[0]
            elif (
                len(self.tools) == 1
                and isinstance(self.tools[0], type)
                and issubclass(self.tools[0], BaseModel)
            ):
                target_model = self.tools[0]
        if not target_model:
            debug_print("[yellow]No model available for format instructions[/yellow]")
            return ""
        try:
            schema = target_model.model_json_schema()
            model_name = target_model.__name__
            return self._format_model_schema(model_name, schema)
        except Exception as e:
            debug_print(f"[yellow]Error generating format instructions: {e}[/yellow]")
            return ""

    def add_format_instructions(
        self,
        model: type[BaseModel] | None = None,
        as_tools: bool = False,
        var_name: str = "format_instructions",
    ) -> AugLLMConfig:
        """Add format instructions to partial_variables without changing structured output configuration."""
        instructions = self.get_format_instructions(model, as_tools)
        if instructions:
            self.partial_variables[var_name] = instructions
            self._apply_partial_variables()
            debug_print(f"[green]Added format instructions to {var_name}[/green]")
        return self

    def add_system_message(self, content: str) -> AugLLMConfig:
        """Add or update system message in the prompt template."""
        debug_print("[blue]Adding/updating system message[/blue]")
        self.system_message = content
        if isinstance(self.prompt_template, ChatPromptTemplate):
            new_messages = []
            has_system = False
            for msg in self.prompt_template.messages:
                if hasattr(msg, "role") and msg.role == "system":
                    new_messages.append(SystemMessage(content=content))
                    has_system = True
                    debug_print("[yellow]Updated existing system message[/yellow]")
                else:
                    new_messages.append(msg)
            if not has_system:
                new_messages.insert(0, SystemMessage(content=content))
                debug_print("[green]Added new system message[/green]")
            self._update_chat_template_messages(new_messages)
        else:
            debug_print("[green]Creating new chat template with system message[/green]")
            self._create_chat_template_from_system()
        self.uses_messages_field = True
        return self

    def add_human_message(self, content: str) -> AugLLMConfig:
        """Add a human message to the prompt template."""
        debug_print("[blue]Adding human message[/blue]")
        if isinstance(self.prompt_template, ChatPromptTemplate):
            new_messages = list(self.prompt_template.messages)
            new_messages.append(HumanMessage(content=content))
            debug_print("[green]Added human message to existing template[/green]")
            self._update_chat_template_messages(new_messages)
        else:
            messages = []
            if self.system_message:
                messages.append(SystemMessage(content=self.system_message))
            messages.append(HumanMessage(content=content))
            if self.add_messages_placeholder:
                should_be_optional = (
                    self.force_messages_optional
                    or self.messages_placeholder_name in self.optional_variables
                )
                messages.append(
                    MessagesPlaceholder(
                        variable_name=self.messages_placeholder_name,
                        optional=should_be_optional,
                    )
                )
            self.prompt_template = ChatPromptTemplate.from_messages(messages)
            debug_print("[green]Created new chat template with human message[/green]")
        self.uses_messages_field = True
        return self

    def replace_message(self, index: int, message: str | BaseMessage) -> AugLLMConfig:
        """Replace a message in the prompt template."""
        if not isinstance(self.prompt_template, ChatPromptTemplate):
            raise ValueError("Can only replace messages in a ChatPromptTemplate")
        debug_print(f"[blue]Replacing message at index {index}[/blue]")
        if isinstance(message, str):
            if index < len(self.prompt_template.messages):
                old_msg = self.prompt_template.messages[index]
                if hasattr(old_msg, "role"):
                    role = old_msg.role
                    if role == "system":
                        message = SystemMessage(content=message)
                    elif role == "human":
                        message = HumanMessage(content=message)
                    elif role == "ai":
                        message = AIMessage(content=message)
                    else:
                        message = HumanMessage(content=message)
                else:
                    message = HumanMessage(content=message)
            else:
                message = HumanMessage(content=message)
        if index < len(self.prompt_template.messages):
            new_messages = list(self.prompt_template.messages)
            new_messages[index] = message
            self._update_chat_template_messages(new_messages)
            debug_print(f"[green]Replaced message at index {index}[/green]")
        return self

    def remove_message(self, index: int) -> AugLLMConfig:
        """Remove a message from the prompt template."""
        if not isinstance(self.prompt_template, ChatPromptTemplate):
            raise ValueError("Can only remove messages from a ChatPromptTemplate")
        debug_print(f"[blue]Removing message at index {index}[/blue]")
        if index < len(self.prompt_template.messages):
            new_messages = list(self.prompt_template.messages)
            removed = new_messages.pop(index)
            self._update_chat_template_messages(new_messages)
            if (
                isinstance(removed, MessagesPlaceholder)
                and removed.variable_name == self.messages_placeholder_name
            ):
                if self.add_messages_placeholder:
                    self._ensure_messages_placeholder_handling()
                else:
                    self.uses_messages_field = self._detect_uses_messages_field()
            debug_print(f"[green]Removed message at index {index}[/green]")
        return self

    def add_optional_variable(self, var_name: str) -> AugLLMConfig:
        """Add an optional variable to the prompt template."""
        if var_name not in self.optional_variables:
            self.optional_variables.append(var_name)
            debug_print(f"[blue]Added optional variable: {var_name}[/blue]")
            self._apply_optional_variables()
        return self

    def with_structured_output(
        self,
        model: type[BaseModel],
        include_instructions: bool = True,
        version: str = "v2",
    ) -> AugLLMConfig:
        """Configure with Pydantic structured output."""
        debug_print(
            f"[blue]Configuring with structured output (version {version})[/blue]"
        )
        self.structured_output_model = model
        self.include_format_instructions = include_instructions
        if version not in ["v1", "v2"]:
            debug_print(
                f"[yellow]Invalid version '{version}' - defaulting to 'v2'[/yellow]"
            )
            version = "v2"
        self.structured_output_version = cast(StructuredOutputVersion, version)
        if not self._is_processing_validation:
            self.comprehensive_validation_and_setup()
        return self

    def with_pydantic_tools(
        self,
        tool_models: list[type[BaseModel]],
        include_instructions: bool = True,
        force_use: bool = False,
    ) -> AugLLMConfig:
        """Configure with Pydantic tools output parsing."""
        debug_print("[blue]Configuring with pydantic tools[/blue]")
        self.pydantic_tools = tool_models
        self.tools = list(self.tools) if self.tools else []
        for model in tool_models:
            if model not in self.tools:
                self.tools.append(model)
        self.include_format_instructions = include_instructions
        if force_use:
            self.force_tool_use = True
            self.tool_choice_mode = "required"
            if len(tool_models) == 1:
                model_name = tool_models[0].__name__
                self.force_tool_choice = model_name
                debug_print(f"[green]Forcing use of tool: {model_name}[/green]")
        if not self._is_processing_validation:
            self.comprehensive_validation_and_setup()
        return self

    def with_format_instructions(
        self,
        model: type[BaseModel],
        as_tool: bool = False,
        var_name: str = "format_instructions",
    ) -> AugLLMConfig:
        """Add format instructions without setting up structured output or parser."""
        instructions = self.get_format_instructions(model, as_tool)
        if instructions:
            self.partial_variables[var_name] = instructions
            self._apply_partial_variables()
            debug_print(f"[green]Added format instructions to {var_name}[/green]")
        return self

    def with_tools(
        self,
        tools: list[
            type[BaseTool]
            | type[BaseModel]
            | Callable
            | StructuredTool
            | BaseTool
            | str
        ],
        force_use: bool = False,
        specific_tool: str | None = None,
    ) -> AugLLMConfig:
        """Configure with specified tools."""
        debug_print("[blue]Configuring with tools[/blue]")
        self.tools = tools
        self.force_tool_use = force_use
        if specific_tool:
            self.force_tool_choice = specific_tool
            self.tool_choice_mode = "required"
            debug_print(f"[green]Forcing use of specific tool: {specific_tool}[/green]")
        elif force_use:
            self.tool_choice_mode = "required"
            debug_print("[green]Forcing use of any tool[/green]")
        if not self._is_processing_validation:
            self.comprehensive_validation_and_setup()
        return self

    def add_prompt_template(self, prompt_template: BasePromptTemplate) -> AugLLMConfig:
        """Add a prompt template to the configuration."""
        debug_print(
            f"[blue]Adding prompt template: {type(prompt_template).__name__}[/blue]"
        )
        self.prompt_template = prompt_template
        if not self._is_processing_validation:
            self.comprehensive_validation_and_setup()
        debug_print(
            f"[green]Added prompt template: {type(prompt_template).__name__}[/green]"
        )
        return self

    def add_tool(
        self, tool: Any, name: str | None = None, route: str | None = None
    ) -> AugLLMConfig:
        """Add a single tool with optional name and route."""
        tool_was_added = False
        if tool not in self.tools:
            self.tools = [*list(self.tools), tool]
            tool_was_added = True
            if name or route:
                auto_name = name or (
                    getattr(tool, "name", None)
                    or getattr(tool, "__name__", f"tool_{len(self.tools)}")
                )
                auto_route = route or "manual"
                self.tool_routes[auto_name] = auto_route
            debug_print(f"➕ [green]Added tool: {name or type(tool).__name__}[/green]")

        # Always sync tool routes to ensure routing is updated even for existing tools
        self._sync_tool_routes()
        return self

    def remove_tool(self, tool: Any) -> AugLLMConfig:
        """Remove a tool and update all related configurations."""
        if tool in self.tools:
            self.tools = [t for t in self.tools if t != tool]
            if tool in self.pydantic_tools:
                self.pydantic_tools = [t for t in self.pydantic_tools if t != tool]
            tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if tool_name and tool_name in self.tool_routes:
                del self.tool_routes[tool_name]
            debug_print(
                f"➖ [yellow]Removed tool: {tool_name or type(tool).__name__}[/yellow]"
            )
            self._sync_tool_routes()
            if not self._is_processing_validation:
                self._compute_schema_fields()
        return self

    def instantiate_llm(self) -> Any:
        """Instantiate the LLM based on the configuration."""
        return self.llm_config.instantiate()

    def _create_tool_implementation(self, name: str, description: str, **kwargs) -> Any:
        """Create tool implementation specialized for AugLLMConfig.

        Can create:
        - LLM function tools that invoke the configured LLM
        - Retriever tools if configured as retriever
        - Pydantic tools from structured output models
        """
        route = self.get_tool_route(name)
        if route == "retriever" and hasattr(self, "instantiate"):
            return self._create_retriever_tool(name, description, **kwargs)
        if self.structured_output_model and route == "pydantic_model":
            return self._create_structured_output_tool(name, description, **kwargs)
        return self._create_llm_function_tool(name, description, **kwargs)

    def _create_llm_function_tool(self, name: str, description: str, **kwargs) -> Any:
        """Create a function tool that invokes this LLM configuration."""
        try:
            from langchain_core.tools import StructuredTool
            from pydantic import BaseModel, Field

            input_fields = self.get_input_fields()

            class LLMInput(BaseModel):
                pass

            for field_name, (_field_type, field_default) in input_fields.items():
                if field_default is not None:
                    setattr(
                        LLMInput,
                        field_name,
                        Field(
                            default=field_default, description=f"Input for {field_name}"
                        ),
                    )
                else:
                    setattr(
                        LLMInput,
                        field_name,
                        Field(description=f"Input for {field_name}"),
                    )

            def llm_function(**inputs) -> Any:
                """Invoke the configured LLM with inputs."""
                runnable = self.create_runnable()
                return runnable.invoke(inputs)

            llm_function.__name__ = name
            llm_function.__doc__ = description
            return StructuredTool.from_function(
                func=llm_function,
                name=name,
                description=description,
                args_schema=LLMInput,
                **kwargs,
            )
        except ImportError:
            raise ImportError("langchain_core.tools is required for LLM function tools")

    def _create_structured_output_tool(
        self, name: str, description: str, **kwargs
    ) -> Any:
        """Create a tool from the structured output model."""
        if not self.structured_output_model:
            raise ValueError("No structured output model configured")
        tool_class = self.structured_output_model
        metadata = {
            "llm_config": self.name or "anonymous",
            "version": self.structured_output_version,
            "tool_type": "structured_output",
        }
        self.set_tool_route(name, "structured_output", metadata)
        return tool_class

    def add_tool_with_route(
        self,
        tool: Any,
        route: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AugLLMConfig:
        """Add a tool with explicit route and metadata."""
        if tool not in self.tools:
            self.tools = [*list(self.tools), tool]
        tool_name = name or (
            getattr(tool, "name", None)
            or getattr(tool, "__name__", f"tool_{len(self.tools)}")
        )
        self.set_tool_route(tool_name, route, metadata)
        debug_print(f"➕ [green]Added tool with route: {tool_name} -> {route}[/green]")
        self._sync_tool_routes()
        return self

    def create_tool_from_config(
        self, config: Any, name: str | None = None, route: str | None = None, **kwargs
    ) -> Any:
        """Create a tool from another config object.

        Args:
            config: Configuration object that has a to_tool method
            name: Tool name
            route: Tool route to set
            **kwargs: Additional kwargs for tool creation

        Returns:
            Created tool
        """
        if not hasattr(config, "to_tool"):
            raise ValueError(
                f"Config {type(config).__name__} does not support to_tool conversion"
            )
        tool = config.to_tool(name=name, **kwargs)
        if route:
            self.add_tool_with_route(tool, route, name)
        else:
            self.add_tool(tool, name)
        return tool

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig, **kwargs):
        """Create from an existing LLMConfig."""
        return cls(llm_config=llm_config, **kwargs)

    @classmethod
    def from_prompt(
        cls, prompt: BasePromptTemplate, llm_config: LLMConfig | None = None, **kwargs
    ):
        """Create from a prompt template."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print(f"[blue]Creating AugLLMConfig from {type(prompt).__name__}[/blue]")
        partial_variables = kwargs.pop("partial_variables", {})
        optional_variables = []
        if hasattr(prompt, "optional_variables") and getattr(
            prompt, "optional_variables", None
        ):
            optional_variables = list(getattr(prompt, "optional_variables", []))
        if "optional_variables" in kwargs:
            optional_variables = kwargs.pop("optional_variables")
        uses_messages = kwargs.pop("uses_messages_field", None)
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        if uses_messages is None:
            if isinstance(prompt, ChatPromptTemplate):
                uses_messages = any(
                    (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == messages_placeholder_name
                        or (hasattr(msg, "role") and msg.role == "system")
                        for msg in prompt.messages
                    )
                )
            elif isinstance(prompt, FewShotChatMessagePromptTemplate):
                uses_messages = True
            else:
                uses_messages = False
        config = cls(
            prompt_template=prompt,
            llm_config=llm_config,
            partial_variables=partial_variables,
            uses_messages_field=uses_messages,
            messages_placeholder_name=messages_placeholder_name,
            optional_variables=optional_variables,
            **kwargs,
        )
        debug_print("[green]Successfully created AugLLMConfig from prompt[/green]")
        return config

    @classmethod
    def from_system_prompt(
        cls, system_prompt: str, llm_config: LLMConfig | None = None, **kwargs
    ):
        """Create from a system prompt string."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print("[blue]Creating AugLLMConfig from system prompt string[/blue]")
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        is_optional = messages_placeholder_name in optional_variables
        messages = [SystemMessage(content=system_prompt)]
        if add_messages_placeholder:
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        prompt = ChatPromptTemplate.from_messages(messages)
        return cls(
            prompt_template=prompt,
            system_message=system_prompt,
            llm_config=llm_config,
            uses_messages_field=True,
            messages_placeholder_name=messages_placeholder_name,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            **kwargs,
        )

    @classmethod
    def from_few_shot(
        cls,
        examples: list[dict[str, Any]],
        example_prompt: PromptTemplate,
        prefix: str,
        suffix: str,
        input_variables: list[str],
        llm_config: LLMConfig | None = None,
        **kwargs,
    ):
        """Create with few-shot examples."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print("[blue]Creating AugLLMConfig with few-shot examples[/blue]")
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")
        optional_variables = kwargs.pop("optional_variables", [])
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            partial_variables=partial_variables,
            optional_variables=optional_variables,
        )
        return cls(
            prompt_template=few_shot_prompt,
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            llm_config=llm_config,
            uses_messages_field=False,
            partial_variables=partial_variables,
            optional_variables=optional_variables,
            **kwargs,
        )

    @classmethod
    def from_few_shot_chat(
        cls,
        examples: list[dict[str, Any]],
        example_prompt: ChatPromptTemplate,
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        **kwargs,
    ):
        """Create with few-shot examples for chat templates."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print("[blue]Creating AugLLMConfig with few-shot chat examples[/blue]")
        partial_variables = kwargs.pop("partial_variables", {})
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples, example_prompt=example_prompt
        )
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(few_shot_prompt)
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        prompt = ChatPromptTemplate.from_messages(messages)
        return cls(
            prompt_template=prompt,
            examples=examples,
            example_prompt=example_prompt,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,
            messages_placeholder_name=messages_placeholder_name,
            partial_variables=partial_variables,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            **kwargs,
        )

    @classmethod
    def from_system_and_few_shot(
        cls,
        system_message: str,
        examples: list[dict[str, Any]],
        example_prompt: PromptTemplate,
        prefix: str,
        suffix: str,
        input_variables: list[str],
        llm_config: LLMConfig | None = None,
        **kwargs,
    ):
        """Create with system message and few-shot examples."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print(
            "[blue]Creating AugLLMConfig with system message and few-shot examples[/blue]"
        )
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")
        optional_variables = kwargs.pop("optional_variables", [])
        enhanced_prefix = f"{system_message}\n\n{prefix}"
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=enhanced_prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            partial_variables=partial_variables,
            optional_variables=optional_variables,
        )
        return cls(
            prompt_template=few_shot_prompt,
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            system_message=system_message,
            input_variables=input_variables,
            example_separator=example_separator,
            llm_config=llm_config,
            uses_messages_field=False,
            partial_variables=partial_variables,
            optional_variables=optional_variables,
            **kwargs,
        )

    @classmethod
    def from_tools(
        cls,
        tools: list[BaseTool | type[BaseTool] | str | type[BaseModel]],
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        use_tool_for_format_instructions: bool | None = None,
        force_tool_use: bool = False,
        **kwargs,
    ):
        """Create with specified tools."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print("[blue]Creating AugLLMConfig with tools[/blue]")
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        force_tool_choice = kwargs.pop("force_tool_choice", None)
        tool_choice_mode = kwargs.pop("tool_choice_mode", "auto")
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        prompt_template = (
            ChatPromptTemplate.from_messages(messages) if messages else None
        )
        instance = cls(
            tools=tools,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,
            messages_placeholder_name=messages_placeholder_name,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            use_tool_for_format_instructions=use_tool_for_format_instructions,
            force_tool_use=force_tool_use,
            force_tool_choice=force_tool_choice,
            tool_choice_mode=tool_choice_mode,
            **kwargs,
        )
        return instance

    @classmethod
    def from_pydantic_tools(
        cls,
        tool_models: list[type[BaseModel]],
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        include_instructions: bool = True,
        force_tool_use: bool = False,
        **kwargs,
    ):
        """Create with Pydantic tool models."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print("[blue]Creating AugLLMConfig with pydantic tools[/blue]")
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        force_tool_choice = kwargs.pop("force_tool_choice", None)
        tool_choice_mode = kwargs.pop("tool_choice_mode", "auto")
        if force_tool_use:
            tool_choice_mode = "required"
            if len(tool_models) == 1:
                # Use sanitized tool name for OpenAI compliance
                from haive.core.utils.naming import sanitize_tool_name

                force_tool_choice = sanitize_tool_name(tool_models[0].__name__)
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        prompt_template = (
            ChatPromptTemplate.from_messages(messages) if messages else None
        )
        partial_variables = kwargs.pop("partial_variables", {})
        instance = cls(
            pydantic_tools=tool_models,
            tools=tool_models,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,
            messages_placeholder_name=messages_placeholder_name,
            partial_variables=partial_variables,
            include_format_instructions=include_instructions,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            force_tool_use=force_tool_use,
            force_tool_choice=force_tool_choice,
            tool_choice_mode=tool_choice_mode,
            **kwargs,
        )
        if include_instructions:
            instance.add_format_instructions(None, True)
        return instance

    @classmethod
    def from_format_instructions(
        cls,
        model: type[BaseModel],
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        as_tool: bool = False,
        var_name: str = "format_instructions",
        **kwargs,
    ):
        """Create config with format instructions but without structured output."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print(
            f"[blue]Creating AugLLMConfig with format instructions from {model.__name__}[/blue]"
        )
        config = cls.from_system_prompt(
            system_message=system_message or "", llm_config=llm_config, **kwargs
        )
        config.add_format_instructions(model, as_tool, var_name)
        return config

    @classmethod
    def from_structured_output_v1(
        cls,
        model: type[BaseModel],
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        include_instructions: bool = True,
        **kwargs,
    ):
        """Create with v1 structured output using traditional parsing."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print(
            f"[blue]Creating AugLLMConfig with v1 structured output using {model.__name__}[/blue]"
        )
        kwargs["structured_output_version"] = "v1"
        kwargs["structured_output_model"] = model
        kwargs["include_format_instructions"] = include_instructions
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        force_messages_optional = kwargs.pop("force_messages_optional", True)
        if add_messages_placeholder:
            optional_variables = kwargs.get("optional_variables", [])
            is_optional = (
                force_messages_optional
                or messages_placeholder_name in optional_variables
            )
            if is_optional and messages_placeholder_name not in optional_variables:
                optional_variables.append(messages_placeholder_name)
                kwargs["optional_variables"] = optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        if messages:
            prompt_template = ChatPromptTemplate.from_messages(messages)
            kwargs["prompt_template"] = prompt_template
        instance = cls(
            llm_config=llm_config,
            system_message=system_message,
            messages_placeholder_name=messages_placeholder_name,
            add_messages_placeholder=add_messages_placeholder,
            force_messages_optional=force_messages_optional,
            **kwargs,
        )
        return instance

    @classmethod
    def from_structured_output_v2(
        cls,
        model: type[BaseModel],
        system_message: str | None = None,
        llm_config: LLMConfig | None = None,
        include_instructions: bool = False,
        output_field_name: str | None = None,
        **kwargs,
    ):
        """Create with v2 structured output using the tool-based approach."""
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")
        debug_print(
            f"[blue]Creating AugLLMConfig with v2 structured output using {model.__name__}[/blue]"
        )
        kwargs["structured_output_version"] = "v2"
        kwargs["structured_output_model"] = model
        kwargs["force_tool_use"] = True
        kwargs["tool_choice_mode"] = "required"
        kwargs["include_format_instructions"] = include_instructions
        if "force_tool_choice" not in kwargs:
            # Use sanitized tool name for OpenAI compliance
            from haive.core.utils.naming import sanitize_tool_name

            kwargs["force_tool_choice"] = sanitize_tool_name(model.__name__)
        if output_field_name:
            kwargs["output_field_name"] = output_field_name
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        force_messages_optional = kwargs.pop("force_messages_optional", True)
        if add_messages_placeholder:
            optional_variables = kwargs.get("optional_variables", [])
            is_optional = (
                force_messages_optional
                or messages_placeholder_name in optional_variables
            )
            if is_optional and messages_placeholder_name not in optional_variables:
                optional_variables.append(messages_placeholder_name)
                kwargs["optional_variables"] = optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )
        if messages:
            prompt_template = ChatPromptTemplate.from_messages(messages)
            kwargs["prompt_template"] = prompt_template
        instance = cls(
            llm_config=llm_config,
            system_message=system_message,
            messages_placeholder_name=messages_placeholder_name,
            add_messages_placeholder=add_messages_placeholder,
            force_messages_optional=force_messages_optional,
            **kwargs,
        )
        return instance

    def debug_tool_configuration(self) -> AugLLMConfig:
        """Print detailed debug information about tool configuration."""
        console.print("\n" + "=" * 80)
        console.print("[bold blue]🔧 TOOL CONFIGURATION DEBUG[/bold blue]")
        console.print("=" * 80)
        basic_tree = Tree("📋 [cyan]Basic Tool Information[/cyan]")
        basic_tree.add(f"Total Tools: [yellow]{len(self.tools)}[/yellow]")
        basic_tree.add(f"Pydantic Tools: [yellow]{len(self.pydantic_tools)}[/yellow]")
        basic_tree.add(f"Tool Is BaseModel: [yellow]{self.tool_is_base_model}[/yellow]")
        basic_tree.add(f"Force Tool Use: [yellow]{self.force_tool_use}[/yellow]")
        basic_tree.add(f"Tool Choice Mode: [yellow]{self.tool_choice_mode}[/yellow]")
        basic_tree.add(f"Force Tool Choice: [yellow]{self.force_tool_choice}[/yellow]")
        console.print(basic_tree)
        if self.tools:
            tools_tree = Tree("🔧 [cyan]Tool Details[/cyan]")
            for i, tool in enumerate(self.tools):
                tool_info = f"[{i}] {type(tool).__name__}"
                if hasattr(tool, "name"):
                    tool_info += f" (name: {tool.name})"
                if hasattr(tool, "__name__"):
                    tool_info += f" (__name__: {tool.__name__})"
                tools_tree.add(tool_info)
            console.print(tools_tree)
        if self.tool_routes:
            routes_tree = Tree("🛤️ [cyan]Tool Routes[/cyan]")
            for name, route in self.tool_routes.items():
                routes_tree.add(f"{name} → [yellow]{route}[/yellow]")
            console.print(routes_tree)
        if self._tool_name_mapping:
            mapping_tree = Tree("🏷️ [cyan]Tool Name Mapping[/cyan]")
            for display_name, actual_name in self._tool_name_mapping.items():
                mapping_tree.add(f"{display_name} → [yellow]{actual_name}[/yellow]")
            console.print(mapping_tree)
        if self.bind_tools_kwargs:
            bind_tree = Tree("⚙️ [cyan]Bind Tools Kwargs[/cyan]")
            for key, value in self.bind_tools_kwargs.items():
                bind_tree.add(f"{key}: [yellow]{value}[/yellow]")
            console.print(bind_tree)
        if self.structured_output_model:
            struct_tree = Tree("📤 [cyan]Structured Output[/cyan]")
            struct_tree.add(
                f"Model: [yellow]{self.structured_output_model.__name__}[/yellow]"
            )
            struct_tree.add(
                f"Version: [yellow]{self.structured_output_version}[/yellow]"
            )
            struct_tree.add(f"Parser Type: [yellow]{self.parser_type}[/yellow]")
            console.print(struct_tree)
        console.print("=" * 80 + "\n")
        return self

    def add_prompt_template(self, name: str, template: BasePromptTemplate) -> None:
        """Add a named prompt template for easy switching.

        Args:
            name: Unique name for the template
            template: The prompt template to store
        """
        if not hasattr(self, "_prompt_templates"):
            self._prompt_templates = {}
        self._prompt_templates[name] = template
        debug_print(f"Added prompt template '{name}': {type(template).__name__}")

    def use_prompt_template(self, name: str) -> AugLLMConfig:
        """Switch to using a specific named template.

        Args:
            name: Name of the template to activate

        Returns:
            Self for method chaining

        Raises:
            ValueError: If template name not found
        """
        if not hasattr(self, "_prompt_templates"):
            self._prompt_templates = {}
        if name not in self._prompt_templates:
            available = list(self._prompt_templates.keys())
            raise ValueError(f"Template '{name}' not found. Available: {available}")
        self.prompt_template = self._prompt_templates[name]
        if not hasattr(self, "_active_template"):
            self._active_template = None
        self._active_template = name
        debug_print(f"Activated prompt template '{name}'")
        return self

    def remove_prompt_template(self, name: str | None = None) -> AugLLMConfig:
        """Remove a template or disable the active one.

        Args:
            name: Template name to remove. If None, disables active template.

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "_prompt_templates"):
            self._prompt_templates = {}
        if not hasattr(self, "_active_template"):
            self._active_template = None
        if not hasattr(self, "_original_template"):
            self._original_template = self.prompt_template
        if name is None:
            self.prompt_template = self._original_template
            self._active_template = None
            debug_print("Disabled active prompt template")
        elif name in self._prompt_templates:
            del self._prompt_templates[name]
            if self._active_template == name:
                self._active_template = None
                self.prompt_template = self._original_template
            debug_print(f"Removed prompt template '{name}'")
        else:
            debug_print(f"Template '{name}' not found for removal")
        return self

    def list_prompt_templates(self) -> list[str]:
        """List available template names."""
        if not hasattr(self, "_prompt_templates"):
            self._prompt_templates = {}
        return list(self._prompt_templates.keys())

    def get_active_template(self) -> str | None:
        """Get the name of the currently active template."""
        if not hasattr(self, "_active_template"):
            self._active_template = None
        return self._active_template

    def remove_tool(self, tool: Any) -> AugLLMConfig:
        """Remove a tool from the configuration.

        Args:
            tool: Tool instance to remove

        Returns:
            Self for method chaining
        """
        if self.tools:
            current_tools = list(self.tools)
            if tool in current_tools:
                current_tools.remove(tool)
                self.tools = current_tools
                debug_print(
                    f"Removed tool: {getattr(tool, 'name', type(tool).__name__)}"
                )
            else:
                debug_print(
                    f"Tool {getattr(tool, 'name', type(tool).__name__)} not found for removal"
                )
        return self

    def clear_tools(self) -> AugLLMConfig:
        """Clear all tools.

        Returns:
            Self for method chaining
        """
        self.tools.clear()
        debug_print("Cleared all tools")
        return self

    def instantiate_llm(self) -> Any:
        """Instantiate the LLM based on the configuration."""
        return self.llm_config.instantiate()
