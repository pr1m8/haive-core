"""
AugLLM configuration system for enhanced LLM chains.

Provides a structured way to configure and create LLM chains with prompts,
tools, output parsers, and structured output models with rich debugging.
"""

import inspect
import json
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

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
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, BaseToolkit, StructuredTool, Tool
from pydantic import BaseModel, Field, model_validator
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from haive.core.engine.base import EngineType, InvokableEngine
from haive.core.models.llm.base import AzureLLMConfig, LLMConfig

logger = logging.getLogger(__name__)
console = Console()

# Literal types for better type safety
ParserType = Literal["pydantic", "pydantic_tools", "str", "json", "custom"]
StructuredOutputVersion = Literal["v1", "v2"]
ToolChoiceMode = Literal["auto", "required", "optional", "none"]


class AugLLMConfig(
    InvokableEngine[
        Union[str, Dict[str, Any], List[BaseMessage]],
        Union[BaseMessage, Dict[str, Any]],
    ]
):
    """
    Configuration for creating enhanced LLM chains with flexible message handling.

    AugLLMConfig provides a structured way to configure and create LLM chains
    with prompts, tools, output parsers, and structured output models.
    """

    engine_type: EngineType = Field(
        default=EngineType.LLM, description="The type of engine"
    )

    # Core LLM configuration
    llm_config: LLMConfig = Field(
        default=AzureLLMConfig(model="gpt-4o"), description="LLM provider configuration"
    )

    # Prompt components
    prompt_template: Optional[BasePromptTemplate] = Field(
        default=None, description="Prompt template for the LLM"
    )
    system_message: Optional[str] = Field(
        default=None, description="System message for chat models"
    )

    # Message placeholder configuration
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

    # Few-shot components
    examples: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Examples for few-shot prompting"
    )
    example_prompt: Optional[PromptTemplate] = Field(
        default=None, description="Template for formatting few-shot examples"
    )
    prefix: Optional[str] = Field(
        default=None, description="Text before examples in few-shot prompting"
    )
    suffix: Optional[str] = Field(
        default=None, description="Text after examples in few-shot prompting"
    )
    example_separator: str = Field(
        default="\n\n", description="Separator between examples in few-shot prompting"
    )
    input_variables: Optional[List[str]] = Field(
        default=None, description="Input variables for the prompt template"
    )

    # Tools
    tools: Sequence[
        Union[Type[BaseTool], Type[BaseModel], Callable, StructuredTool, BaseModel]
    ] = Field(default_factory=list, description="The tools to use for the node")
    pydantic_tools: List[Type[BaseModel]] = Field(
        default_factory=list, description="Pydantic models for tool schemas"
    )

    # Flags for format instructions from tools
    use_tool_for_format_instructions: bool = Field(
        default=False,
        description="Use a single tool Pydantic model for format instructions",
    )
    tool_is_base_model: bool = Field(
        default=False,
        description="Whether a tool is a BaseModel type (detected automatically)",
    )

    # Tool selection options
    force_tool_use: bool = Field(
        default=False, description="Whether to force the LLM to use a tool (any tool)"
    )
    force_tool_choice: Optional[Union[bool, str, List[str]]] = Field(
        default=None, description="Force specific tool(s) to be used"
    )
    tool_choice_mode: ToolChoiceMode = Field(
        default="auto", description="Tool choice mode to use for binding tools"
    )

    # Output handling
    structured_output_model: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured output"
    )
    structured_output_version: StructuredOutputVersion = Field(
        default="v1",
        description="Version of structured output handling: v1 (traditional), v2 (tool-based)",
    )
    output_parser: Optional[BaseOutputParser] = Field(
        default=None, description="Parser for LLM output"
    )
    parse_raw_output: bool = Field(
        default=False,
        description="Force parsing raw output even with structured output model",
    )
    include_format_instructions: bool = Field(
        default=True, description="Whether to include format instructions in the prompt"
    )
    parser_type: ParserType = Field(
        default="pydantic",
        description="Parser type: 'pydantic', 'pydantic_tools', 'str', 'json', or 'custom'",
    )

    # Output field naming
    output_field_name: Optional[str] = Field(
        default=None, description="Custom name for the primary output field in schema"
    )

    # Tool configuration
    tool_kwargs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Parameters for tool instantiation"
    )
    bind_tools_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters for binding tools to the LLM"
    )
    bind_tools_config: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration for bind_tools"
    )

    # Pre/post processing
    preprocess: Optional[Callable[[Any], Any]] = Field(
        default=None,
        description="Function to preprocess input before sending to LLM",
        exclude=True,  # Exclude from serialization
    )
    postprocess: Optional[Callable[[Any], Any]] = Field(
        default=None,
        description="Function to postprocess output from LLM",
        exclude=True,  # Exclude from serialization
    )

    # Runtime options
    temperature: Optional[float] = Field(
        default=None, description="Temperature parameter for the LLM"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    runtime_options: Dict[str, Any] = Field(
        default_factory=dict, description="Additional runtime options for the LLM"
    )

    # Custom runnables to chain
    custom_runnables: Optional[List[Runnable]] = Field(
        default=None,
        description="Custom runnables to add to the chain",
        exclude=True,  # Exclude from serialization
    )

    # Partial variables for templates
    partial_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Partial variables for the prompt template"
    )

    # Optional variables for templates
    optional_variables: List[str] = Field(
        default_factory=list, description="Optional variables for the prompt template"
    )

    # Message field detection
    uses_messages_field: Optional[bool] = Field(
        default=None,
        description="Explicitly specify if this engine uses a messages field. If None, auto-detected.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **kwargs):
        """Initialize with debug logging."""
        super().__init__(**kwargs)

    def _debug_log(self, title: str, content: Dict[str, Any]):
        """Pretty print debug information."""
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

    @model_validator(mode="after")
    def validate_and_setup(self):
        """Validate configuration and set up components after initialization."""
        # Debug logging
        self._debug_log(
            "AugLLMConfig Initialization",
            {
                "name": self.name,
                "system_message": self.system_message is not None,
                "prompt_template": (
                    type(self.prompt_template).__name__
                    if self.prompt_template
                    else None
                ),
                "force_messages_optional": self.force_messages_optional,
                "add_messages_placeholder": self.add_messages_placeholder,
                "messages_placeholder_name": self.messages_placeholder_name,
                "tools": len(self.tools),
                "pydantic_tools": len(self.pydantic_tools),
                "use_tool_for_format_instructions": self.use_tool_for_format_instructions,
                "force_tool_use": self.force_tool_use,
                "force_tool_choice": self.force_tool_choice,
                "tool_choice_mode": self.tool_choice_mode,
                "structured_output_version": self.structured_output_version,
            },
        )

        # Check for BaseModel in tools and validate tool configuration
        self._process_tools()

        # Create prompt template components if needed
        self._create_prompt_template_if_needed()

        # Ensure messages placeholder is properly handled
        self._ensure_messages_placeholder_handling()

        # Apply partial variables to the prompt template if needed
        self._apply_partial_variables()

        # Apply optional variables to the prompt template
        self._apply_optional_variables()

        # Generate format instructions if needed
        self._setup_format_instructions()

        # Set up output parser if structured_output_model is provided but no output_parser
        self._setup_output_handling()

        # Configure tool choice options
        self._configure_tool_choice()

        # Auto-detect uses_messages_field if not explicitly set
        if self.uses_messages_field is None:
            self.uses_messages_field = self._detect_uses_messages_field()

        # Debug final state
        self._debug_log(
            "Final Configuration",
            {
                "uses_messages_field": self.uses_messages_field,
                "messages_optional": self.force_messages_optional,
                "optional_variables": self.optional_variables,
                "prompt_template": self._get_prompt_template_info(),
                "has_format_instructions": "format_instructions"
                in self.partial_variables,
                "tool_is_base_model": self.tool_is_base_model,
                "use_tool_for_format_instructions": self.use_tool_for_format_instructions,
                "tool_choice_mode": self.tool_choice_mode,
                "force_tool_use": self.force_tool_use,
                "force_tool_choice": self.force_tool_choice,
                "structured_output_version": self.structured_output_version,
                "output_field_name": self.output_field_name,
            },
        )

        return self

    def _process_tools(self):
        """Process tools, detect BaseModel tools, and validate tool configuration."""
        # Skip if no tools
        if not self.tools:
            return

        # 1. Check for BaseModel types in tools
        basemodel_tools = []
        tool_names = []

        # Process each tool to detect types and names
        for tool in self.tools:
            # Case 1: Tool is a BaseModel type
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                basemodel_tools.append(tool)
                tool_names.append(getattr(tool, "__name__", "UnnamedModel"))

                # Add to pydantic_tools if not already there
                if tool not in self.pydantic_tools:
                    self.pydantic_tools.append(tool)
                    rprint(
                        f"[green]Added BaseModel {tool.__name__} to pydantic_tools[/green]"
                    )

            # Case 2: Tool is a BaseTool instance or type
            elif isinstance(tool, BaseTool) or (
                isinstance(tool, type) and issubclass(tool, BaseTool)
            ):
                # Get tool name
                if isinstance(tool, type):
                    # For tool classes, use class name or first declared class attribute
                    tool_names.append(getattr(tool, "name", tool.__name__))
                else:
                    # For tool instances, use name property
                    tool_names.append(getattr(tool, "name", "unnamed_tool"))

            # Case 3: Tool is a string (reference to a tool)
            elif isinstance(tool, str):
                tool_names.append(tool)

            # Case 4: Callable function
            elif callable(tool) and not isinstance(tool, type):
                # Add function name as tool name
                func_name = getattr(tool, "__name__", "unnamed_function")
                tool_names.append(func_name)

            # Case 5: Other tool types (log warning)
            else:
                tool_type = type(tool).__name__
                rprint(f"[yellow]Unrecognized tool type: {tool_type}[/yellow]")

        # Set flag if a single BaseModel tool is found (for format instructions)
        if len(basemodel_tools) == 1:
            self.tool_is_base_model = True

            # Auto-enable format instructions unless explicitly disabled
            if self.use_tool_for_format_instructions is None:
                self.use_tool_for_format_instructions = True
                rprint(
                    f"[green]Auto-enabled format instructions from {basemodel_tools[0].__name__}[/green]"
                )

        # If multiple BaseModel tools, log a warning
        elif len(basemodel_tools) > 1:
            tool_names_str = ", ".join([t.__name__ for t in basemodel_tools])
            rprint(f"[yellow]Multiple BaseModel tools found: {tool_names_str}[/yellow]")

        # Store discovered tool names for later validation
        self.metadata["tool_names"] = tool_names
        self.metadata["has_basemodel_tools"] = bool(basemodel_tools)

    def _configure_tool_choice(self):
        """Configure tool choice based on available tools and settings."""
        # Skip if no tools
        if not self.tools:
            self.force_tool_use = False
            self.force_tool_choice = None
            self.tool_choice_mode = "none"
            return

        # Get discovered tool names
        tool_names = self.metadata.get("tool_names", [])

        # Handle Boolean force_tool_choice - convert to 'auto' mode
        if isinstance(self.force_tool_choice, bool):
            if self.force_tool_choice:
                self.tool_choice_mode = "required"
                self.force_tool_use = True
                # Convert to 'auto' (any tool)
                self.force_tool_choice = None
            else:
                self.tool_choice_mode = "optional"
                self.force_tool_use = False
                self.force_tool_choice = None
            rprint(
                f"[yellow]Converted boolean force_tool_choice to mode: {self.tool_choice_mode}[/yellow]"
            )

        # Handle string force_tool_choice - specific tool name
        elif isinstance(self.force_tool_choice, str):
            self.force_tool_use = True
            self.tool_choice_mode = "required"

            # Validate tool name exists when possible
            if tool_names and self.force_tool_choice not in tool_names:
                rprint(
                    f"[yellow]Warning: Specified force_tool_choice '{self.force_tool_choice}' not found in detected tool names: {tool_names}[/yellow]"
                )

        # Handle list force_tool_choice - not directly supported, use first item
        elif (
            isinstance(self.force_tool_choice, (list, tuple)) and self.force_tool_choice
        ):
            self.force_tool_use = True
            self.tool_choice_mode = "required"
            self.force_tool_choice = self.force_tool_choice[0]
            rprint(
                f"[yellow]Multiple forced tools not supported - using first item: {self.force_tool_choice}[/yellow]"
            )

        # Handle force_tool_use without specific choice
        elif self.force_tool_use and not self.force_tool_choice:
            self.tool_choice_mode = "required"

            # For single tool scenarios, auto-select that tool
            if len(tool_names) == 1:
                self.force_tool_choice = tool_names[0]
                rprint(
                    f"[green]Auto-configured force_tool_choice to use: {self.force_tool_choice}[/green]"
                )

        # Set bind_tools_kwargs based on mode
        if self.tool_choice_mode == "required":
            self.bind_tools_kwargs["tool_choice"] = "required"
            if self.force_tool_choice:
                self.bind_tools_kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": self.force_tool_choice},
                }
        elif self.tool_choice_mode == "auto":
            self.bind_tools_kwargs["tool_choice"] = "auto"
        elif self.tool_choice_mode == "none":
            self.bind_tools_kwargs["tool_choice"] = "none"

    def _setup_output_handling(self):
        """Set up output handling based on configuration."""
        # Case 1: Raw output parsing requested
        if self.parse_raw_output:
            rprint("[yellow]Using StrOutputParser for raw output[/yellow]")
            self.output_parser = StrOutputParser()
            self.parser_type = "str"
            return

        # Case 2: Structured output model exists but no parser yet
        if self.structured_output_model and not self.output_parser:
            # Check if we're using v2 structured output
            if self.structured_output_version == "v2":
                # For v2, configure for tool-based approach
                rprint(
                    f"[cyan]Using structured output v2 with model: {self.structured_output_model.__name__}[/cyan]"
                )

                # Ensure the model is in tools list if not already
                if self.structured_output_model not in self.tools:
                    self.tools = list(self.tools) if self.tools else []
                    self.tools.append(self.structured_output_model)
                    rprint(
                        f"[green]Added {self.structured_output_model.__name__} to tools[/green]"
                    )

                # Ensure model is in pydantic_tools
                if self.structured_output_model not in self.pydantic_tools:
                    self.pydantic_tools.append(self.structured_output_model)

                # Set parser type
                self.parser_type = "pydantic_tools"

                # Force tool usage for this model
                self.force_tool_use = True
                self.tool_choice_mode = "required"

                # Only set force_tool_choice if we have a single model and no specific choice yet
                if not self.force_tool_choice and len(self.pydantic_tools) == 1:
                    # Use model name or a lowercase variant to match tool naming conventions
                    model_name = self.structured_output_model.__name__.lower()
                    self.force_tool_choice = model_name
                    rprint(
                        f"[green]Auto-set force_tool_choice to '{model_name}'[/green]"
                    )

                # Create output parser for v2
                self.output_parser = PydanticToolsParser(
                    tools=[self.structured_output_model]
                )

            else:
                # For v1, use traditional Pydantic output parser
                rprint(
                    f"[cyan]Using structured output v1 with model: {self.structured_output_model.__name__}[/cyan]"
                )
                self.parser_type = "pydantic"
                self.output_parser = PydanticOutputParser(
                    pydantic_object=self.structured_output_model
                )

        # Case 3: Pydantic tools exist but no parser yet
        elif self.pydantic_tools and not self.output_parser:
            rprint("[cyan]Setting up pydantic tools parser[/cyan]")
            self.parser_type = "pydantic_tools"
            self.output_parser = PydanticToolsParser(tools=self.pydantic_tools)

    def _setup_format_instructions(self):
        """Set up format instructions based on configuration without affecting structured output."""
        if (
            not self.include_format_instructions
            or "format_instructions" in self.partial_variables
        ):
            return

        rprint("[blue]Setting up format instructions[/blue]")

        # Case 1: Structured output model exists
        if self.structured_output_model and not self.parse_raw_output:
            try:
                # For v2, generate simple schema-based instructions
                if self.structured_output_version == "v2":
                    # Generate schema-based instructions for the model
                    schema = self.structured_output_model.schema()
                    model_name = self.structured_output_model.__name__
                    instructions = self._format_model_schema(model_name, schema)
                    self.partial_variables["format_instructions"] = instructions
                    rprint(
                        f"[green]Added schema-based instructions for {model_name} (v2)[/green]"
                    )
                else:
                    # For v1, try PydanticOutputParser
                    temp_parser = PydanticOutputParser(
                        pydantic_object=self.structured_output_model
                    )
                    if hasattr(temp_parser, "get_format_instructions"):
                        self.partial_variables["format_instructions"] = (
                            temp_parser.get_format_instructions()
                        )
                        rprint(
                            "[green]Added format instructions from structured_output_model (v1)[/green]"
                        )
                    else:
                        # Fallback to schema-based instructions
                        schema = self.structured_output_model.schema()
                        model_name = self.structured_output_model.__name__
                        instructions = self._format_model_schema(model_name, schema)
                        self.partial_variables["format_instructions"] = instructions
                        rprint(
                            f"[green]Added schema-based instructions for {model_name} (v1 fallback)[/green]"
                        )
            except Exception as e:
                rprint(f"[yellow]Error setting up format instructions: {e}[/yellow]")

        # Case 2: Pydantic tools exist
        elif self.pydantic_tools and not self.parse_raw_output:
            try:
                # Generate schema-based instructions for all tools
                tool_instructions = []
                for tool_model in self.pydantic_tools:
                    schema = tool_model.schema()
                    model_name = tool_model.__name__
                    tool_instructions.append(
                        self._format_model_schema(model_name, schema, as_section=True)
                    )

                # Combine instructions
                if len(tool_instructions) == 1:
                    self.partial_variables["format_instructions"] = tool_instructions[0]
                else:
                    combined = (
                        "You must respond using one of the following formats:\n\n"
                        + "\n\n".join(tool_instructions)
                    )
                    self.partial_variables["format_instructions"] = combined

                rprint(
                    f"[green]Added schema-based instructions for {len(self.pydantic_tools)} tools[/green]"
                )
            except Exception as e:
                rprint(
                    f"[yellow]Error setting up format instructions for tools: {e}[/yellow]"
                )

        # Apply instructions to template if it exists
        if "format_instructions" in self.partial_variables and self.prompt_template:
            self._apply_partial_variables()

    def _format_model_schema(
        self, model_name: str, schema: Dict[str, Any], as_section: bool = False
    ) -> str:
        """Format a model schema as instructions.

        Args:
            model_name: Name of the model
            schema: JSON schema dictionary
            as_section: Whether to format as a section (for multiple tools)

        Returns:
            Formatted instructions
        """
        # Format JSON with indentation
        schema_json = json.dumps(schema, indent=2)

        # Create header if needed
        header = f"## {model_name}\n" if as_section else ""

        # Format instructions
        return f"""{header}You must format your response as JSON that matches this schema:

```json
{schema_json}
```

The output should be valid JSON that conforms to the {model_name} schema.
"""

    def get_format_instructions(
        self, model: Optional[Type[BaseModel]] = None, as_tools: bool = False
    ) -> str:
        """
        Get format instructions for a model without changing the config.

        Args:
            model: Optional Pydantic model. If None, uses structured_output_model or first pydantic_tool
            as_tools: Whether to format as tools (using PydanticToolsParser)

        Returns:
            Format instructions string
        """
        # Figure out which model to use
        target_model = model
        if target_model is None:
            if self.structured_output_model:
                target_model = self.structured_output_model
            elif self.pydantic_tools:
                if as_tools:
                    # Use all tools
                    tool_instructions = []
                    for tool_model in self.pydantic_tools:
                        schema = tool_model.schema()
                        model_name = tool_model.__name__
                        tool_instructions.append(
                            self._format_model_schema(
                                model_name, schema, as_section=True
                            )
                        )

                    # Combine instructions
                    if len(tool_instructions) == 1:
                        return tool_instructions[0]
                    else:
                        return (
                            "You must respond using one of the following formats:\n\n"
                            + "\n\n".join(tool_instructions)
                        )
                else:
                    # Use first tool
                    target_model = self.pydantic_tools[0]
            elif (
                len(self.tools) == 1
                and isinstance(self.tools[0], type)
                and issubclass(self.tools[0], BaseModel)
            ):
                target_model = self.tools[0]

        if not target_model:
            rprint("[yellow]No model available for format instructions[/yellow]")
            return ""

        # Generate schema-based instructions for the model
        try:
            schema = target_model.schema()
            model_name = target_model.__name__
            return self._format_model_schema(model_name, schema)
        except Exception as e:
            rprint(f"[yellow]Error generating format instructions: {e}[/yellow]")
            return ""

    def add_format_instructions(
        self,
        model: Optional[Type[BaseModel]] = None,
        as_tools: bool = False,
        var_name: str = "format_instructions",
    ) -> "AugLLMConfig":
        """
        Add format instructions to partial_variables without changing structured output configuration.

        Args:
            model: Optional Pydantic model. If None, uses existing configuration
            as_tools: Whether to format as tools
            var_name: Name of the partial variable to add

        Returns:
            Self for chaining
        """
        instructions = self.get_format_instructions(model, as_tools)
        if instructions:
            self.partial_variables[var_name] = instructions
            self._apply_partial_variables()
            rprint(f"[green]Added format instructions to {var_name}[/green]")
        return self

    def _create_prompt_template_if_needed(self):
        """Create appropriate prompt template based on available components."""
        if self.prompt_template is not None:
            return

        # Create FewShotPromptTemplate if components are available
        if self.examples and self.example_prompt and self.prefix and self.suffix:
            rprint("[green]Creating FewShotPromptTemplate[/green]")
            self._create_few_shot_template()

        # Handle FewShotChatMessagePromptTemplate scenario
        elif self.examples and isinstance(self.example_prompt, ChatPromptTemplate):
            rprint("[green]Creating FewShotChatMessagePromptTemplate[/green]")
            self._create_few_shot_chat_template()

        # Create ChatPromptTemplate from system message
        elif self.system_message:
            rprint("[green]Creating ChatPromptTemplate from system message[/green]")
            self._create_chat_template_from_system()

        # Create default ChatPromptTemplate
        elif self.add_messages_placeholder and not self.prompt_template:
            rprint("[green]Creating default ChatPromptTemplate[/green]")
            self._create_default_chat_template()

    def _ensure_messages_placeholder_handling(self):
        """Ensure messages placeholder is properly handled based on configuration."""
        if not self.prompt_template:
            return

        if isinstance(self.prompt_template, ChatPromptTemplate):
            rprint("[blue]Handling messages placeholder for ChatPromptTemplate[/blue]")
            self._handle_chat_template_messages_placeholder()

        elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
            rprint(
                "[blue]Handling messages placeholder for FewShotChatMessagePromptTemplate[/blue]"
            )
            # This template type handles messages differently
            self.uses_messages_field = True

        elif isinstance(self.prompt_template, FewShotPromptTemplate):
            rprint(
                "[blue]FewShotPromptTemplate detected - messages not applicable[/blue]"
            )
            self.uses_messages_field = False

        else:
            rprint("[blue]Checking for messages variables in template[/blue]")
            self._check_template_for_messages_variables()

    def _handle_chat_template_messages_placeholder(self):
        """Handle messages placeholder in ChatPromptTemplate."""
        messages = list(self.prompt_template.messages)
        has_messages_placeholder = False
        messages_placeholder_index = -1

        # Check existing placeholders
        for i, msg in enumerate(messages):
            if (
                isinstance(msg, MessagesPlaceholder)
                and getattr(msg, "variable_name", "") == self.messages_placeholder_name
            ):
                has_messages_placeholder = True
                messages_placeholder_index = i
                rprint(
                    f"[yellow]Found existing messages placeholder at index {i}[/yellow]"
                )
                break

        # Add placeholder if needed
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
            rprint(
                f"[green]Added messages placeholder (optional={should_be_optional})[/green]"
            )

            # Create new template
            self._update_chat_template_messages(messages)

        # Update existing placeholder if needed
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
                rprint(
                    f"[cyan]Updating messages placeholder optional status: {should_be_optional}[/cyan]"
                )
                messages[messages_placeholder_index] = MessagesPlaceholder(
                    variable_name=self.messages_placeholder_name,
                    optional=should_be_optional,
                )
                self._update_chat_template_messages(messages)

    def _update_chat_template_messages(self, messages: List[Any]):
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

        # Only add MessagesPlaceholder if add_messages_placeholder is True
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
        # Few-shot prompts typically don't use messages
        self.uses_messages_field = False

    def _create_few_shot_chat_template(self):
        """Create a FewShotChatMessagePromptTemplate using example_prompt."""
        # Get system message if available
        prefix_messages = []
        if self.system_message:
            prefix_messages = [SystemMessage(content=self.system_message)]

        # Build few shot template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=self.examples,
            example_prompt=self.example_prompt,
        )

        # Build full message sequence
        messages = prefix_messages + [few_shot_prompt]

        # Add messages placeholder if requested
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

        # Create complete prompt
        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.uses_messages_field = True
        rprint("[green]Created FewShotChatMessagePromptTemplate[/green]")

    def _check_template_for_messages_variables(self):
        """Check if the template uses messages variables."""
        # Check input variables
        if hasattr(self.prompt_template, "input_variables"):
            input_vars = getattr(self.prompt_template, "input_variables", [])
            if self.messages_placeholder_name in input_vars:
                self.uses_messages_field = True
                rprint(
                    f"[yellow]Found {self.messages_placeholder_name} in input variables[/yellow]"
                )
                return

        # Default to false for non-chat templates
        self.uses_messages_field = False
        rprint("[yellow]No message variables found in template[/yellow]")

    def _apply_partial_variables(self):
        """Apply partial variables to the prompt template."""
        if not self.prompt_template or not self.partial_variables:
            return

        rprint("[cyan]Applying partial variables to template[/cyan]")

        try:
            if hasattr(self.prompt_template, "partial"):
                self.prompt_template = self.prompt_template.partial(
                    **self.partial_variables
                )
                rprint("[green]Successfully applied partial variables[/green]")

                # Debug log
                self._debug_log(
                    "Applied Partial Variables",
                    {
                        "variables": self.partial_variables,
                        "template_type": type(self.prompt_template).__name__,
                    },
                )
            else:
                rprint("[yellow]Template does not support partial variables[/yellow]")
        except Exception as e:
            rprint(f"[red]Error applying partial variables: {e}[/red]")

    def _apply_optional_variables(self):
        """Apply optional variables to the prompt template."""
        if not self.optional_variables or not self.prompt_template:
            return

        rprint("[cyan]Applying optional variables to template[/cyan]")

        # For ChatPromptTemplate, handle message placeholder optionality
        if isinstance(self.prompt_template, ChatPromptTemplate):
            if self.messages_placeholder_name in self.optional_variables:
                self._handle_chat_template_messages_placeholder()

        # For other template types
        else:
            if hasattr(self.prompt_template, "optional_variables"):
                if not hasattr(self.prompt_template.optional_variables, "extend"):
                    self.prompt_template.optional_variables = list(
                        self.prompt_template.optional_variables
                    )

                for var in self.optional_variables:
                    if var not in self.prompt_template.optional_variables:
                        self.prompt_template.optional_variables.append(var)

                rprint(
                    f"[green]Applied optional variables: {self.optional_variables}[/green]"
                )

    def _detect_uses_messages_field(self) -> bool:
        """Detect if this LLM configuration uses a messages field."""
        rprint("[blue]Auto-detecting messages field usage[/blue]")

        # Check for explicit configuration
        if not self.add_messages_placeholder:
            rprint(
                "[yellow]add_messages_placeholder is False - checking template[/yellow]"
            )
            if isinstance(self.prompt_template, ChatPromptTemplate):
                for msg in self.prompt_template.messages:
                    if (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == self.messages_placeholder_name
                    ):
                        rprint("[green]Found messages placeholder in template[/green]")
                        return True
            return False

        # Default to True for tools and system_message
        if self.tools or self.system_message:
            rprint("[green]Tools or system message present - using messages[/green]")
            return True

        # Check prompt template type
        if self.prompt_template:
            if isinstance(self.prompt_template, ChatPromptTemplate):
                rprint("[green]ChatPromptTemplate - using messages[/green]")
                return True
            elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
                rprint(
                    "[green]FewShotChatMessagePromptTemplate - using messages[/green]"
                )
                return True
            elif isinstance(self.prompt_template, FewShotPromptTemplate):
                rprint("[yellow]FewShotPromptTemplate - not using messages[/yellow]")
                return False
            else:
                # Check for messages in input variables
                if hasattr(self.prompt_template, "input_variables"):
                    if (
                        self.messages_placeholder_name
                        in self.prompt_template.input_variables
                    ):
                        rprint(
                            f"[green]Found {self.messages_placeholder_name} in template variables[/green]"
                        )
                        return True

        # Default to True for safety
        rprint("[green]Default to True for safety[/green]")
        return True

    def _get_prompt_template_info(self) -> str:
        """Get detailed information about the prompt template."""
        if not self.prompt_template:
            return "None"

        info = f"{type(self.prompt_template).__name__}"

        if isinstance(self.prompt_template, ChatPromptTemplate):
            msg_count = len(self.prompt_template.messages)
            has_placeholder = any(
                isinstance(msg, MessagesPlaceholder)
                and getattr(msg, "variable_name", "") == self.messages_placeholder_name
                for msg in self.prompt_template.messages
            )
            info += f" ({msg_count} messages, placeholder={has_placeholder})"
        elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
            example_count = len(getattr(self.prompt_template, "examples", []))
            info += f" ({example_count} examples)"

        return info

    def _get_input_variables(self) -> Set[str]:
        """Get all input variables required by the prompt template, excluding partials and optionals."""
        all_vars = set()

        # No template = just messages if used
        if not self.prompt_template:
            return (
                {self.messages_placeholder_name} if self.uses_messages_field else set()
            )

        # Direct input_variables attribute
        if hasattr(self.prompt_template, "input_variables"):
            vars_list = getattr(self.prompt_template, "input_variables", [])
            all_vars.update(vars_list)
            rprint(f"[cyan]Template input variables: {vars_list}[/cyan]")

        # Chat templates message variables
        if isinstance(self.prompt_template, ChatPromptTemplate):
            for i, msg in enumerate(self.prompt_template.messages):
                # Check message prompt templates
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "input_variables"):
                    msg_vars = msg.prompt.input_variables
                    all_vars.update(msg_vars)
                    rprint(f"[cyan]Message {i} variables: {msg_vars}[/cyan]")

                # Check variable_name for placeholders
                if hasattr(msg, "variable_name"):
                    var_name = getattr(msg, "variable_name")
                    # Only add if not optional or if forced
                    if (
                        not getattr(msg, "optional", False)
                        or not self.force_messages_optional
                    ):
                        all_vars.add(var_name)
                        rprint(
                            f"[cyan]Placeholder variable: {var_name} (optional={getattr(msg, 'optional', False)})[/cyan]"
                        )

        # Few shot chat template variables
        elif isinstance(self.prompt_template, FewShotChatMessagePromptTemplate):
            if hasattr(self.prompt_template, "example_prompt") and hasattr(
                self.prompt_template.example_prompt, "input_variables"
            ):
                example_vars = self.prompt_template.example_prompt.input_variables
                all_vars.update(example_vars)
                rprint(f"[cyan]Few shot example variables: {example_vars}[/cyan]")

        # Remove partial variables
        partial_vars = set(self.partial_variables.keys())
        if hasattr(self.prompt_template, "partial_variables"):
            template_partials = getattr(self.prompt_template, "partial_variables", {})
            partial_vars.update(template_partials.keys())

        # Remove optional variables
        optional_vars = set(self.optional_variables)
        if hasattr(self.prompt_template, "optional_variables"):
            template_optionals = getattr(self.prompt_template, "optional_variables", [])
            optional_vars.update(template_optionals)

        # Remove partials and optionals
        result = all_vars - partial_vars - optional_vars

        rprint(f"[green]Final required variables: {result}[/green]")

        # If empty, default to messages for safety based on uses_messages_field
        if (
            not result
            and self.uses_messages_field
            and self.messages_placeholder_name not in optional_vars
        ):
            rprint(
                f"[yellow]No variables found - defaulting to {self.messages_placeholder_name}[/yellow]"
            )
            return {self.messages_placeholder_name}

        return result

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get schema fields based on prompt template and configuration.

        Implements abstract method from Engine base class.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Any as AnyType
        from typing import List as ListType
        from typing import Optional as OptionalType

        fields = {}

        # Get required input variables
        required_vars = self._get_input_variables()

        # Get type information from prompt template if available
        input_types = {}
        if (
            hasattr(self.prompt_template, "input_types")
            and self.prompt_template.input_types
        ):
            input_types = self.prompt_template.input_types
            rprint(f"[cyan]Found input types in prompt template: {input_types}[/cyan]")

        # Handle messages field specially
        if self.uses_messages_field:
            is_optional = (
                self.force_messages_optional
                or self.messages_placeholder_name in self.optional_variables
            )

            if is_optional:
                fields[self.messages_placeholder_name] = (
                    OptionalType[ListType[BaseMessage]],
                    Field(default_factory=list),
                )
                rprint(f"[green]Messages field added as optional[/green]")
            else:
                fields[self.messages_placeholder_name] = (
                    ListType[BaseMessage],
                    Field(default_factory=list),
                )
                rprint(f"[green]Messages field added as required[/green]")

        # Process all other required variables
        for var in required_vars:
            if var != self.messages_placeholder_name and var not in fields:
                # Look for type in input_types
                if var in input_types:
                    var_type = input_types[var]
                    rprint(
                        f"[green]Using type from prompt template for {var}: {var_type}[/green]"
                    )
                else:
                    var_type = AnyType
                    rprint(f"[yellow]Using Any type for {var}[/yellow]")

                # Add field
                fields[var] = (var_type, Field(...))

        # Process optional variables
        for var in self.optional_variables:
            if var != self.messages_placeholder_name and var not in fields:
                # Get type directly from input_types or default to Any
                var_type = input_types.get(var, AnyType)
                fields[var] = (OptionalType[var_type], None)
                rprint(
                    f"[cyan]Added optional field: {var} with type Optional[{var_type}][/cyan]"
                )

        # Debug final fields
        self._debug_log(
            "Input Fields",
            {name: f"{type_info[0]}" for name, type_info in fields.items()},
        )

        return fields

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get output fields based on structured_output_model and output_parser.

        Implements abstract method from Engine base class.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Any as AnyType
        from typing import Dict
        from typing import List as ListType
        from typing import Optional as OptionalType

        fields = {}

        # Use structured output v2 (tool-based approach)
        if self.structured_output_version == "v2" and (
            self.structured_output_model or self.pydantic_tools
        ):
            # For v2, we return a tool result
            tool_name = self.output_field_name or "tool_result"

            # Determine the model to use
            if self.structured_output_model:
                fields[tool_name] = (self.structured_output_model, None)
            elif self.pydantic_tools:
                # Use a union of all tool models or the first one
                if len(self.pydantic_tools) == 1:
                    fields[tool_name] = (self.pydantic_tools[0], None)
                else:
                    # Multiple tools - use the custom name with Dict for now
                    fields[tool_name] = (Dict[str, AnyType], None)

            # Always include a messages field for v2 since we're using tool calling
            fields[self.messages_placeholder_name] = (
                ListType[BaseMessage],
                Field(default_factory=list),
            )

            rprint(f"[green]Using v2 output schema with field: {tool_name}[/green]")
            return fields

        # Use structured_output_model if available and not parsing raw (v1 approach)
        if self.structured_output_model and not self.parse_raw_output:
            # Extract fields from the model
            if hasattr(self.structured_output_model, "model_fields"):
                # Pydantic v2
                for (
                    field_name,
                    field_info,
                ) in self.structured_output_model.model_fields.items():
                    fields[field_name] = (field_info.annotation, field_info.default)
            else:
                # Fallback to using the model as a single field
                model_name = self.output_field_name or (
                    getattr(self.structured_output_model, "__name__", "").lower()
                    or "result"
                )
                fields[model_name] = (self.structured_output_model, None)

        # Handle Pydantic tools if specified
        elif self.pydantic_tools and self.parser_type == "pydantic_tools":
            # Get fields from all tool models or use a single field
            if self.output_field_name:
                # Use custom field name for all tools
                fields[self.output_field_name] = (Dict[str, AnyType], None)
            else:
                # Create a field for each tool model
                for tool_model in self.pydantic_tools:
                    model_name = getattr(tool_model, "__name__", "").lower()
                    fields[model_name] = (tool_model, None)

        # Handle output parser types if structured_output_model not used or parsing raw
        elif self.output_parser or self.parse_raw_output:
            # Get default field name
            field_name = self.output_field_name or "content"
            parser_name = (
                type(self.output_parser).__name__ if self.output_parser else ""
            )

            # String-based parsers
            if (
                parser_name in ["StrOutputParser", "StringOutputParser"]
                or self.parse_raw_output
                or self.parser_type == "str"
            ):
                fields[field_name] = (str, None)

            # JSON-based parsers
            elif (
                parser_name in ["JsonOutputParser", "JSONLinesOutputParser"]
                or self.parser_type == "json"
            ):
                fields[field_name] = (Dict[str, AnyType], None)

            # PydanticOutputParser
            elif (
                parser_name == "PydanticOutputParser" or self.parser_type == "pydantic"
            ) and hasattr(self.output_parser, "pydantic_object"):
                # Extract from PydanticOutputParser
                pydantic_model = self.output_parser.pydantic_object
                if hasattr(pydantic_model, "model_fields"):  # Pydantic v2
                    for field_name, field_info in pydantic_model.model_fields.items():
                        fields[field_name] = (field_info.annotation, field_info.default)

            # PydanticToolsParser
            elif (
                parser_name == "PydanticToolsParser"
                or self.parser_type == "pydantic_tools"
            ) and hasattr(self.output_parser, "tools"):
                # Use custom field name if provided
                if self.output_field_name:
                    fields[self.output_field_name] = (Dict[str, AnyType], None)
                else:
                    # For each tool model, add its fields
                    for tool_model in self.output_parser.tools:
                        model_name = getattr(tool_model, "__name__", "").lower()
                        fields[model_name] = (tool_model, None)

            # List-based parsers
            elif parser_name in ["ListOutputParser", "CSVOutputParser"]:
                list_field = self.output_field_name or "items"
                fields[list_field] = (ListType[AnyType], None)

            # Default parser output
            else:
                fields[field_name] = (AnyType, None)

        # Default output fields
        if not fields:
            # Use a simple content field
            content_field = self.output_field_name or "content"
            fields[content_field] = (OptionalType[str], None)

            # Include messages field if using messages
            if self.uses_messages_field:
                fields[self.messages_placeholder_name] = (
                    ListType[BaseMessage],
                    Field(default_factory=list),
                )

        # Debug output fields
        self._debug_log(
            "Output Fields",
            {name: f"{type_info[0]}" for name, type_info in fields.items()},
        )

        return fields

    def create_runnable(
        self, runnable_config: Optional[RunnableConfig] = None
    ) -> Runnable:
        """
        Create a runnable LLM chain based on this configuration.

        Args:
            runnable_config: Optional runtime configuration

        Returns:
            A runnable LLM chain
        """
        from haive.core.engine.aug_llm.factory import AugLLMFactory

        # Extract config parameters from runnable_config
        config_params = self.apply_runnable_config(runnable_config)

        # Create factory with config params
        factory = AugLLMFactory(self, config_params)

        # Build the runnable chain
        return factory.create_runnable()

    def apply_runnable_config(
        self, runnable_config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """
        Extract parameters from runnable_config relevant to this engine.

        Args:
            runnable_config: Runtime configuration

        Returns:
            Dictionary of relevant parameters
        """
        # Start with common parameters from base class
        params = super().apply_runnable_config(runnable_config)

        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]

            # AugLLM specific parameters to extract
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
            ]

            for param in aug_llm_params:
                if param in configurable:
                    params[param] = configurable[param]

        # Debug extracted params
        self._debug_log("Runtime Config Parameters", params)

        return params

    def _process_input(
        self, input_data: Union[str, Dict[str, Any], List[BaseMessage]]
    ) -> Dict[str, Any]:
        """Process input into a format usable by the runnable."""
        rprint("[blue]Processing input data[/blue]")

        # Find input variables required by the prompt template
        required_vars = self._get_input_variables()

        # Handle dictionary input
        if isinstance(input_data, dict):
            rprint("[green]Input is already a dictionary[/green]")
            # Simply return the input dict - all needed fields should be there
            return input_data

        # Handle string input
        if isinstance(input_data, str):
            result = {}

            # If we need messages
            if (
                self.uses_messages_field
                and self.messages_placeholder_name not in result
            ):
                result[self.messages_placeholder_name] = [
                    HumanMessage(content=input_data)
                ]
                rprint(f"[green]Added string to messages field[/green]")

            # For other variables, use the string directly
            for var in required_vars:
                if var != self.messages_placeholder_name:
                    result[var] = input_data
                    rprint(f"[cyan]Added string to field: {var}[/cyan]")

            return result

        # Handle list of messages
        if isinstance(input_data, list) and all(
            isinstance(item, BaseMessage) for item in input_data
        ):
            result = {self.messages_placeholder_name: input_data}
            rprint(f"[green]Added message list to messages field[/green]")
            return result

        # Default case - convert to human message
        rprint("[yellow]Converting unknown input to human message[/yellow]")
        return {self.messages_placeholder_name: [HumanMessage(content=str(input_data))]}

    def add_system_message(self, content: str) -> "AugLLMConfig":
        """Add or update system message in the prompt template.

        Args:
            content: System message content

        Returns:
            Self for chaining
        """
        rprint(f"[blue]Adding/updating system message[/blue]")

        # Update system_message property
        self.system_message = content

        # Update prompt template if it's a chat template
        if isinstance(self.prompt_template, ChatPromptTemplate):
            new_messages = []
            has_system = False

            # Check existing messages
            for msg in self.prompt_template.messages:
                if hasattr(msg, "role") and msg.role == "system":
                    # Replace existing system message
                    new_messages.append(SystemMessage(content=content))
                    has_system = True
                    rprint("[yellow]Updated existing system message[/yellow]")
                else:
                    new_messages.append(msg)

            # Add system message if none exists
            if not has_system:
                new_messages.insert(0, SystemMessage(content=content))
                rprint("[green]Added new system message[/green]")

            # Create new template with updated messages
            self._update_chat_template_messages(new_messages)
        else:
            # Create chat template if none exists
            rprint("[green]Creating new chat template with system message[/green]")
            self._create_chat_template_from_system()

        self.uses_messages_field = True

        return self

    def add_human_message(self, content: str) -> "AugLLMConfig":
        """Add a human message to the prompt template.

        Args:
            content: Human message content

        Returns:
            Self for chaining
        """
        rprint(f"[blue]Adding human message[/blue]")

        if isinstance(self.prompt_template, ChatPromptTemplate):
            # Add to existing chat template
            new_messages = list(self.prompt_template.messages)
            new_messages.append(HumanMessage(content=content))
            rprint("[green]Added human message to existing template[/green]")

            # Create new template
            self._update_chat_template_messages(new_messages)
        else:
            # Create new chat template
            messages = []
            if self.system_message:
                messages.append(SystemMessage(content=self.system_message))
            messages.append(HumanMessage(content=content))

            # Only add MessagesPlaceholder if auto-add is enabled
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
            rprint("[green]Created new chat template with human message[/green]")

        self.uses_messages_field = True

        return self

    def replace_message(
        self, index: int, message: Union[str, BaseMessage]
    ) -> "AugLLMConfig":
        """Replace a message in the prompt template.

        Args:
            index: Index of message to replace
            message: New message content or BaseMessage object

        Returns:
            Self for chaining
        """
        if not isinstance(self.prompt_template, ChatPromptTemplate):
            raise ValueError("Can only replace messages in a ChatPromptTemplate")

        rprint(f"[blue]Replacing message at index {index}[/blue]")

        # Convert string to message if needed
        if isinstance(message, str):
            # Determine message role based on the message being replaced
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

        # Replace the message
        if index < len(self.prompt_template.messages):
            new_messages = list(self.prompt_template.messages)
            new_messages[index] = message

            self._update_chat_template_messages(new_messages)
            rprint(f"[green]Replaced message at index {index}[/green]")

        return self

    def remove_message(self, index: int) -> "AugLLMConfig":
        """Remove a message from the prompt template.

        Args:
            index: Index of message to remove

        Returns:
            Self for chaining
        """
        if not isinstance(self.prompt_template, ChatPromptTemplate):
            raise ValueError("Can only remove messages from a ChatPromptTemplate")

        rprint(f"[blue]Removing message at index {index}[/blue]")

        if index < len(self.prompt_template.messages):
            new_messages = list(self.prompt_template.messages)
            removed = new_messages.pop(index)

            # Create new template
            self._update_chat_template_messages(new_messages)

            # Update uses_messages_field if we removed the MessagesPlaceholder
            if (
                isinstance(removed, MessagesPlaceholder)
                and removed.variable_name == self.messages_placeholder_name
            ):
                # Re-add the placeholder if add_messages_placeholder is True
                if self.add_messages_placeholder:
                    self._ensure_messages_placeholder_handling()
                else:
                    # Check if there's still a messages placeholder
                    self.uses_messages_field = self._detect_uses_messages_field()

            rprint(f"[green]Removed message at index {index}[/green]")

        return self

    def add_optional_variable(self, var_name: str) -> "AugLLMConfig":
        """Add an optional variable to the prompt template.

        Args:
            var_name: Name of the optional variable

        Returns:
            Self for chaining
        """
        if var_name not in self.optional_variables:
            self.optional_variables.append(var_name)
            rprint(f"[blue]Added optional variable: {var_name}[/blue]")

            # Apply optional variables to prompt template
            self._apply_optional_variables()

        return self

    def with_structured_output(
        self,
        model: Type[BaseModel],
        include_instructions: bool = True,
        version: str = "v1",
    ) -> "AugLLMConfig":
        """Configure with Pydantic structured output.

        Args:
            model: Pydantic model for structured output
            include_instructions: Whether to include format instructions in prompt
            version: Version of structured output handling - "v1" (traditional) or "v2" (tool-based)

        Returns:
            Self for chaining
        """
        rprint(f"[blue]Configuring with structured output (version {version})[/blue]")

        # Set the structured output model
        self.structured_output_model = model
        self.include_format_instructions = include_instructions

        # Validate version
        if version not in ["v1", "v2"]:
            rprint(f"[yellow]Invalid version '{version}' - defaulting to 'v1'[/yellow]")
            version = "v1"

        self.structured_output_version = cast(StructuredOutputVersion, version)

        # Configure based on version
        if version == "v2":
            # For v2, use pydantic_tools approach
            self.parser_type = "pydantic_tools"

            # Add model to pydantic_tools if not already there
            if model not in self.pydantic_tools:
                self.pydantic_tools.append(model)

            # Add to tools if not already there
            if model not in self.tools:
                self.tools = list(self.tools) if self.tools else []
                self.tools.append(model)

            # Set up tool-based parsing
            self.output_parser = PydanticToolsParser(tools=[model])

            # Configure tool choice to use this model
            self.force_tool_use = True
            self.tool_choice_mode = "required"

            # Auto-configure force_tool_choice if not set
            if not self.force_tool_choice:
                model_name = model.__name__.lower()
                self.force_tool_choice = model_name
                rprint(f"[green]Auto-set force_tool_choice to '{model_name}'[/green]")

            rprint("[green]Set up tool-based structured output (v2)[/green]")
        else:
            # For v1, use traditional approach
            self.parser_type = "pydantic"
            self.output_parser = PydanticOutputParser(pydantic_object=model)
            rprint("[green]Set up traditional structured output (v1)[/green]")

        # Add format instructions if requested
        if include_instructions:
            instructions = self.get_format_instructions(model)
            if instructions:
                self.partial_variables["format_instructions"] = instructions
                rprint(f"[green]Added format instructions for {version} output[/green]")

                # Apply to template if it exists
                self._apply_partial_variables()

        return self

    def with_pydantic_tools(
        self,
        tool_models: List[Type[BaseModel]],
        include_instructions: bool = True,
        force_use: bool = False,
    ) -> "AugLLMConfig":
        """Configure with Pydantic tools output parsing.

        Args:
            tool_models: List of Pydantic models for tool schemas
            include_instructions: Whether to include format instructions in prompt
            force_use: Whether to force the LLM to use these tools

        Returns:
            Self for chaining
        """
        rprint("[blue]Configuring with pydantic tools[/blue]")

        # Set the pydantic tools
        self.pydantic_tools = tool_models

        # Add to tools list if not already there
        self.tools = list(self.tools) if self.tools else []
        for model in tool_models:
            if model not in self.tools:
                self.tools.append(model)

        self.parser_type = "pydantic_tools"
        self.include_format_instructions = include_instructions

        # Configure tool use if requested
        if force_use:
            self.force_tool_use = True
            self.tool_choice_mode = "required"

            # If single tool, set it as the forced choice
            if len(tool_models) == 1:
                model_name = tool_models[0].__name__.lower()
                self.force_tool_choice = model_name
                rprint(f"[green]Forcing use of tool: {model_name}[/green]")

        # Setup parser
        self.output_parser = PydanticToolsParser(tools=tool_models)

        # Add format instructions if needed
        if include_instructions:
            instructions = self.get_format_instructions(None, True)
            if instructions:
                self.partial_variables["format_instructions"] = instructions
                rprint("[green]Added format instructions for pydantic tools[/green]")

                # Apply to template if it exists
                self._apply_partial_variables()

        return self

    def with_format_instructions(
        self,
        model: Type[BaseModel],
        as_tool: bool = False,
        var_name: str = "format_instructions",
    ) -> "AugLLMConfig":
        """
        Add format instructions without setting up structured output or parser.

        Args:
            model: Pydantic model to use for format instructions
            as_tool: Whether to format using PydanticToolsParser
            var_name: Variable name to add instructions to

        Returns:
            Self for chaining
        """
        # Get instructions
        instructions = self.get_format_instructions(model, as_tool)

        # Add to partial variables if instructions were generated
        if instructions:
            self.partial_variables[var_name] = instructions
            self._apply_partial_variables()
            rprint(f"[green]Added format instructions to {var_name}[/green]")

        return self

    def with_tools(
        self,
        tools: List[
            Union[
                Type[BaseTool], Type[BaseModel], Callable, StructuredTool, BaseTool, str
            ]
        ],
        force_use: bool = False,
        specific_tool: Optional[str] = None,
    ) -> "AugLLMConfig":
        """
        Configure with specified tools.

        Args:
            tools: List of tools to make available
            force_use: Whether to force the LLM to use a tool (any tool)
            specific_tool: Force a specific tool to be used (by name)

        Returns:
            Self for chaining
        """
        rprint("[blue]Configuring with tools[/blue]")

        # Set tools
        self.tools = tools

        # Process tools to detect types and validate
        self._process_tools()

        # Configure tool choice
        self.force_tool_use = force_use

        if specific_tool:
            self.force_tool_choice = specific_tool
            self.tool_choice_mode = "required"
            rprint(f"[green]Forcing use of specific tool: {specific_tool}[/green]")
        elif force_use:
            self.tool_choice_mode = "required"
            rprint("[green]Forcing use of any tool[/green]")

        # Configure tool choice settings
        self._configure_tool_choice()

        return self

    def add_prompt_template(
        self, prompt_template: BasePromptTemplate
    ) -> "AugLLMConfig":
        """
        Add a prompt template to the configuration.

        Args:
            prompt_template: Prompt template to use

        Returns:
            Self for chaining
        """
        rprint(f"[blue]Adding prompt template: {type(prompt_template).__name__}[/blue]")

        # Set prompt template
        self.prompt_template = prompt_template

        # Handle messages placeholder if it's a chat template
        if isinstance(prompt_template, ChatPromptTemplate):
            self._handle_chat_template_messages_placeholder()

        # Apply partial variables if any
        if self.partial_variables:
            self._apply_partial_variables()

        # Apply optional variables if any
        if self.optional_variables:
            self._apply_optional_variables()

        # Update uses_messages_field detection
        self.uses_messages_field = self._detect_uses_messages_field()

        rprint(
            f"[green]Added prompt template: {type(prompt_template).__name__}[/green]"
        )
        return self

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig, **kwargs):
        """Create from an existing LLMConfig.

        Args:
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        return cls(llm_config=llm_config, **kwargs)

    @classmethod
    def from_prompt(
        cls,
        prompt: BasePromptTemplate,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create from a prompt template.

        Args:
            prompt: Prompt template to use
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint(f"[blue]Creating AugLLMConfig from {type(prompt).__name__}[/blue]")

        # Handle partial variables if provided in kwargs
        partial_variables = kwargs.pop("partial_variables", {})

        # Extract optional variables if present in the prompt
        optional_variables = []
        if hasattr(prompt, "optional_variables") and getattr(
            prompt, "optional_variables", None
        ):
            optional_variables = list(getattr(prompt, "optional_variables", []))

        # Override with explicit kwargs if provided
        if "optional_variables" in kwargs:
            optional_variables = kwargs.pop("optional_variables")

        # Detect if this is a messages-based prompt
        uses_messages = kwargs.pop("uses_messages_field", None)
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")

        if uses_messages is None:
            # Auto-detect based on prompt type
            if isinstance(prompt, ChatPromptTemplate):
                # Check if any message is a MessagesPlaceholder
                uses_messages = any(
                    (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == messages_placeholder_name
                    )
                    or hasattr(msg, "role")
                    and msg.role == "system"
                    for msg in prompt.messages
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

        rprint("[green]Successfully created AugLLMConfig from prompt[/green]")
        return config

    @classmethod
    def from_system_prompt(
        cls, system_prompt: str, llm_config: Optional[LLMConfig] = None, **kwargs
    ):
        """Create from a system prompt string.

        Args:
            system_prompt: System prompt string
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint("[blue]Creating AugLLMConfig from system prompt string[/blue]")

        # Get messages placeholder name and optional flag
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

        # Check if messages is optional
        is_optional = messages_placeholder_name in optional_variables

        # Create messages list with system message
        messages = [SystemMessage(content=system_prompt)]

        # Only add MessagesPlaceholder if auto-add is enabled
        if add_messages_placeholder:
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )

        # Create template
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
        examples: List[Dict[str, Any]],
        example_prompt: PromptTemplate,
        prefix: str,
        suffix: str,
        input_variables: List[str],
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create with few-shot examples.

        Args:
            examples: List of examples as dictionaries
            example_prompt: Template for formatting examples
            prefix: Text before examples
            suffix: Text after examples
            input_variables: Input variables for the prompt
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint("[blue]Creating AugLLMConfig with few-shot examples[/blue]")

        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")

        # Extract optional variables
        optional_variables = kwargs.pop("optional_variables", [])

        # Create few-shot prompt template
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
            uses_messages_field=False,  # FewShotPromptTemplate typically doesn't use messages
            partial_variables=partial_variables,
            optional_variables=optional_variables,
            **kwargs,
        )

    @classmethod
    def from_few_shot_chat(
        cls,
        examples: List[Dict[str, Any]],
        example_prompt: ChatPromptTemplate,
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create with few-shot examples for chat templates.

        Args:
            examples: List of examples as dictionaries
            example_prompt: Chat template for formatting examples
            system_message: Optional system message
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint("[blue]Creating AugLLMConfig with few-shot chat examples[/blue]")

        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")

        # Extract optional variables
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

        # Build few shot template
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
        )

        # Create messages list
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Add few-shot examples
        messages.append(few_shot_prompt)

        # Add messages placeholder if needed
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )

        # Create prompt template
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
        examples: List[Dict[str, Any]],
        example_prompt: PromptTemplate,
        prefix: str,
        suffix: str,
        input_variables: List[str],
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create with system message and few-shot examples.

        Args:
            system_message: System message to use
            examples: List of examples as dictionaries
            example_prompt: Template for formatting examples
            prefix: Text before examples
            suffix: Text after examples
            input_variables: Input variables for the prompt
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint(
            "[blue]Creating AugLLMConfig with system message and few-shot examples[/blue]"
        )

        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")

        # Extract optional variables
        optional_variables = kwargs.pop("optional_variables", [])

        # Create a prefix with system message
        enhanced_prefix = f"{system_message}\n\n{prefix}"

        # Create few-shot prompt template
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
            uses_messages_field=False,  # FewShotPromptTemplate typically doesn't use messages
            partial_variables=partial_variables,
            optional_variables=optional_variables,
            **kwargs,
        )

    @classmethod
    def from_tools(
        cls,
        tools: List[Union[BaseTool, Type[BaseTool], str, Type[BaseModel]]],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        use_tool_for_format_instructions: Optional[
            bool
        ] = None,  # Optional to allow auto-detection
        force_tool_use: bool = False,
        **kwargs,
    ):
        """Create with specified tools.

        Args:
            tools: List of tools to make available
            system_message: Optional system message
            llm_config: LLM configuration (optional)
            use_tool_for_format_instructions: Whether to use a BaseModel tool for format instructions
            force_tool_use: Whether to force the LLM to use a tool
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint("[blue]Creating AugLLMConfig with tools[/blue]")

        # Get messages placeholder configuration
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

        # Get tool choice configuration
        force_tool_choice = kwargs.pop("force_tool_choice", None)
        tool_choice_mode = kwargs.pop("tool_choice_mode", "auto")

        # Create messages list
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Add MessagesPlaceholder if auto-add is enabled
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )

        # Create template
        prompt_template = (
            ChatPromptTemplate.from_messages(messages) if messages else None
        )

        # Create instance
        instance = cls(
            tools=tools,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,  # Tool-using LLMs always use messages
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
        tool_models: List[Type[BaseModel]],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        include_instructions: bool = True,
        force_tool_use: bool = False,
        **kwargs,
    ):
        """Create with Pydantic tool models.

        Args:
            tool_models: List of Pydantic models for tool schemas
            system_message: Optional system message
            llm_config: LLM configuration (optional)
            include_instructions: Whether to include format instructions
            force_tool_use: Whether to force the LLM to use these tools
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint("[blue]Creating AugLLMConfig with pydantic tools[/blue]")

        # Get messages placeholder configuration
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

        # Tool choice configuration
        force_tool_choice = kwargs.pop("force_tool_choice", None)
        tool_choice_mode = kwargs.pop("tool_choice_mode", "auto")

        if force_tool_use:
            tool_choice_mode = "required"
            # If there's only one tool, set it as the forced choice
            if len(tool_models) == 1:
                force_tool_choice = tool_models[0].__name__.lower()

        # Create messages list
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Add MessagesPlaceholder if auto-add is enabled
        if add_messages_placeholder:
            is_optional = messages_placeholder_name in optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )

        # Create template
        prompt_template = (
            ChatPromptTemplate.from_messages(messages) if messages else None
        )

        # Prepare partial variables with format instructions if needed
        partial_variables = kwargs.pop("partial_variables", {})

        # Create instance first
        instance = cls(
            pydantic_tools=tool_models,
            tools=tool_models,  # Also add as tools for consistency
            parser_type="pydantic_tools",
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

        # Add format instructions if needed - using the instance's method to avoid error
        if include_instructions:
            instance.add_format_instructions(None, True)

        return instance

    @classmethod
    def from_format_instructions(
        cls,
        model: Type[BaseModel],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        as_tool: bool = False,
        var_name: str = "format_instructions",
        **kwargs,
    ):
        """Create config with format instructions but without structured output.

        Args:
            model: Pydantic model to use for format instructions
            system_message: Optional system message
            llm_config: LLM config
            as_tool: Whether to format as a tool
            var_name: Variable name for format instructions
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint(
            f"[blue]Creating AugLLMConfig with format instructions from {model.__name__}[/blue]"
        )

        # Create the config first
        config = cls.from_system_prompt(
            system_message=system_message or "", llm_config=llm_config, **kwargs
        )

        # Add format instructions using instance method
        config.add_format_instructions(model, as_tool, var_name)
        return config

    @classmethod
    def from_structured_output_v2(
        cls,
        model: Type[BaseModel],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        include_instructions: bool = True,
        output_field_name: Optional[str] = None,
        **kwargs,
    ):
        """Create with v2 structured output using the tool-based approach.

        Args:
            model: Pydantic model for structured output
            system_message: Optional system message
            llm_config: LLM configuration
            include_instructions: Whether to add format instructions
            output_field_name: Custom name for the output field
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        rprint(
            f"[blue]Creating AugLLMConfig with v2 structured output using {model.__name__}[/blue]"
        )

        # Ensure proper settings for v2
        kwargs["structured_output_version"] = "v2"
        kwargs["structured_output_model"] = model
        kwargs["force_tool_use"] = True
        kwargs["tool_choice_mode"] = "required"

        # Auto-set tool name if not specified
        if "force_tool_choice" not in kwargs:
            kwargs["force_tool_choice"] = model.__name__.lower()

        # Set output field name if specified
        if output_field_name:
            kwargs["output_field_name"] = output_field_name

        # Create messages list
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Get message parameters
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)
        force_messages_optional = kwargs.pop("force_messages_optional", True)

        # Add MessagesPlaceholder if auto-add is enabled
        if add_messages_placeholder:
            optional_variables = kwargs.get("optional_variables", [])
            is_optional = (
                force_messages_optional
                or messages_placeholder_name in optional_variables
            )

            # Ensure optional_variables list includes messages_placeholder_name if needed
            if is_optional and messages_placeholder_name not in optional_variables:
                optional_variables.append(messages_placeholder_name)
                kwargs["optional_variables"] = optional_variables

            # Add the placeholder
            messages.append(
                MessagesPlaceholder(
                    variable_name=messages_placeholder_name, optional=is_optional
                )
            )

        # Create prompt template
        if messages:
            prompt_template = ChatPromptTemplate.from_messages(messages)
            kwargs["prompt_template"] = prompt_template

        # Now create the config
        instance = cls(
            llm_config=llm_config,
            include_format_instructions=include_instructions,
            system_message=system_message,
            messages_placeholder_name=messages_placeholder_name,
            add_messages_placeholder=add_messages_placeholder,
            force_messages_optional=force_messages_optional,
            **kwargs,
        )

        # Add the model to tools explicitly
        if model not in instance.tools:
            instance.tools = list(instance.tools) if instance.tools else []
            instance.tools.append(model)

        # Add to pydantic_tools
        if model not in instance.pydantic_tools:
            instance.pydantic_tools.append(model)

        # Force setting up output handling
        instance._setup_output_handling()

        # Add format instructions if needed
        if include_instructions:
            instance.add_format_instructions(model)

        return instance

    def instantiate_llm(self) -> Any:
        """Instantiate the LLM based on the configuration."""
        return self.llm_config.instantiate()
