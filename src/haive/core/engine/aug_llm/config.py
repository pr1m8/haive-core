"""
AugLLM configuration system for enhanced LLM chains.

Provides a structured way to configure and create LLM chains with prompts,
tools, output parsers, and structured output models.
"""

import inspect
import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
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
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from haive.core.engine.base import EngineType, InvokableEngine
from haive.core.models.llm.base import AzureLLMConfig, LLMConfig

logger = logging.getLogger(__name__)


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

    # Optional variables for prompt templates
    optional_variables: List[str] = Field(
        default_factory=list, description="Optional variables for the prompt template"
    )

    # Message placeholder configuration
    messages_placeholder_name: str = Field(
        default="messages",
        description="Name of the messages placeholder in chat templates",
    )
    add_messages_placeholder: bool = Field(
        default=True,
        description="Whether to automatically add MessagesPlaceholder if not present",
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
    tools: List[Union[BaseTool, Type[BaseTool], str]] = Field(
        default_factory=list, description="List of tools to make available to the LLM"
    )
    pydantic_tools: List[Type[BaseModel]] = Field(
        default_factory=list, description="Pydantic models for tool schemas"
    )

    # Output handling
    structured_output_model: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured output"
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
    parser_type: str = Field(
        default="pydantic",
        description="Parser type: 'pydantic', 'pydantic_tools', or 'custom'",
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
    force_tool_choice: Optional[str] = Field(
        default=None, description="Force the LLM to use this specific tool"
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

    # Message field detection
    uses_messages_field: Optional[bool] = Field(
        default=None,
        description="Explicitly specify if this engine uses a messages field. If None, auto-detected.",
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_and_setup(self):
        """Validate configuration and set up components after initialization."""
        # Auto-detect uses_messages_field if not explicitly set
        if self.uses_messages_field is None:
            self.uses_messages_field = self._detect_uses_messages_field()

        # Handle messages placeholder in template
        self._ensure_messages_placeholder()

        # Apply partial variables to the prompt template if needed
        if self.prompt_template and self.partial_variables:
            self._apply_partial_variables()

        # Apply optional variables to the prompt template
        self._apply_optional_variables()

        # Create a FewShotPromptTemplate if examples and example_prompt are provided but no prompt_template
        if (
            self.examples
            and self.example_prompt
            and not self.prompt_template
            and self.prefix
            and self.suffix
        ):
            self._create_few_shot_template()

        # Create a simple chat template with system message if only system_message is provided
        elif self.system_message and not self.prompt_template:
            self._create_chat_template_from_system()

        # Set up output parser if structured_output_model is provided but no output_parser
        if self.structured_output_model and not self.output_parser:
            self._setup_output_parser()

        # Setup pydantic tools parser if needed
        elif self.pydantic_tools and not self.output_parser:
            self._setup_pydantic_tools_parser()

        return self

    def _apply_partial_variables(self):
        """Apply partial variables to the prompt template."""
        if hasattr(self.prompt_template, "partial_variables"):
            # Create a new partial_variables dict to avoid modifying the original
            existing_partials = (
                getattr(self.prompt_template, "partial_variables", {}) or {}
            )
            updated_partials = {**existing_partials, **self.partial_variables}
            self.prompt_template.partial_variables = updated_partials

    def _apply_optional_variables(self):
        """Apply optional variables to the prompt template."""
        if not self.optional_variables or not self.prompt_template:
            return

        # If messages_placeholder is not required, add it to optional variables
        if (
            not self.add_messages_placeholder
            and self.messages_placeholder_name not in self.optional_variables
        ):
            self.optional_variables.append(self.messages_placeholder_name)

        # Apply to PromptTemplate
        if isinstance(self.prompt_template, PromptTemplate):
            # Set optional_variables if supported
            if hasattr(self.prompt_template, "optional_variables"):
                self.prompt_template.optional_variables = self.optional_variables

        # Apply to FewShotPromptTemplate
        elif isinstance(self.prompt_template, FewShotPromptTemplate):
            # Set optional_variables if supported
            if hasattr(self.prompt_template, "optional_variables"):
                self.prompt_template.optional_variables = self.optional_variables

        # Apply to ChatPromptTemplate - needs special handling
        elif isinstance(self.prompt_template, ChatPromptTemplate):
            # For ChatPromptTemplate, we need to handle the optional messages placeholder
            if self.messages_placeholder_name in self.optional_variables:
                # Find any MessagesPlaceholder and mark it as optional
                for i, msg in enumerate(self.prompt_template.messages):
                    if (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == self.messages_placeholder_name
                    ):
                        # Set optional flag if available
                        if hasattr(msg, "optional"):
                            msg.optional = True

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

    def _create_chat_template_from_system(self):
        """Create a ChatPromptTemplate from system_message."""
        messages = [SystemMessage(content=self.system_message)]

        # Only add MessagesPlaceholder if add_messages_placeholder is True
        if self.add_messages_placeholder:
            is_optional = self.messages_placeholder_name in self.optional_variables
            messages.append(
                MessagesPlaceholder(
                    variable_name=self.messages_placeholder_name, optional=is_optional
                )
            )

        self.prompt_template = ChatPromptTemplate.from_messages(messages)
        self.uses_messages_field = True

    def _setup_output_parser(self):
        """Set up output parser for structured output."""
        if self.parse_raw_output:
            # Use a string parser to get raw output
            self.output_parser = StrOutputParser()
        elif self.parser_type == "pydantic" and self.include_format_instructions:
            # Create a PydanticOutputParser for the model
            self.output_parser = PydanticOutputParser(
                pydantic_object=self.structured_output_model
            )

            # Add format instructions to partial variables
            if self.include_format_instructions:
                format_instructions = self.output_parser.get_format_instructions()
                self.partial_variables["format_instructions"] = format_instructions

                # Apply to prompt template if it exists
                self._apply_partial_variables()

    def _setup_pydantic_tools_parser(self):
        """Set up parser for pydantic tools."""
        if self.parser_type == "pydantic_tools":
            # Create PydanticToolsParser
            self.output_parser = PydanticToolsParser(tools=self.pydantic_tools)

            # Add format instructions if needed
            if self.include_format_instructions and hasattr(
                self.output_parser, "get_format_instructions"
            ):
                format_instructions = self.output_parser.get_format_instructions()
                self.partial_variables["format_instructions"] = format_instructions

                # Apply to prompt template if it exists
                self._apply_partial_variables()

    def _ensure_messages_placeholder(self):
        """Ensure chat templates have a MessagesPlaceholder if needed."""
        if not self.prompt_template or not isinstance(
            self.prompt_template, ChatPromptTemplate
        ):
            return

        if not self.add_messages_placeholder:
            return

        # Check if a MessagesPlaceholder is already present
        has_messages_placeholder = False
        for msg in self.prompt_template.messages:
            if (
                isinstance(msg, MessagesPlaceholder)
                and getattr(msg, "variable_name", "") == self.messages_placeholder_name
            ):
                has_messages_placeholder = True
                # Check if it needs to be marked optional
                if (
                    self.messages_placeholder_name in self.optional_variables
                    and hasattr(msg, "optional")
                ):
                    msg.optional = True
                break

        # Add MessagesPlaceholder if not present and auto-add is enabled
        if not has_messages_placeholder and self.add_messages_placeholder:
            new_messages = list(self.prompt_template.messages)
            is_optional = self.messages_placeholder_name in self.optional_variables

            # Create with optional flag if needed
            new_placeholder = MessagesPlaceholder(
                variable_name=self.messages_placeholder_name, optional=is_optional
            )
            new_messages.append(new_placeholder)

            # Create new template with the messages
            partial_vars = getattr(self.prompt_template, "partial_variables", None)
            self.prompt_template = ChatPromptTemplate.from_messages(
                new_messages, partial_variables=partial_vars
            )
            self.uses_messages_field = True

    def _detect_uses_messages_field(self) -> bool:
        """Detect if this LLM configuration uses a messages field.

        Returns:
            True if messages field is detected, False otherwise
        """
        # If explicitly set to not add messages placeholder, don't assume we use messages
        # unless there's already a messages placeholder
        if self.add_messages_placeholder is False:
            # Only check for existing messages placeholder
            if self.prompt_template and isinstance(
                self.prompt_template, ChatPromptTemplate
            ):
                for msg in self.prompt_template.messages:
                    if (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "")
                        == self.messages_placeholder_name
                    ):
                        return True
            return self.messages_placeholder_name in self.optional_variables

        # Default to True for tools and system_message
        if self.tools or self.system_message:
            return True

        # Check if prompt_template contains MessagesPlaceholder
        if self.prompt_template:
            # Handle different template types differently
            if isinstance(self.prompt_template, ChatPromptTemplate):
                # Check all messages for a MessagesPlaceholder
                for msg in self.prompt_template.messages:
                    if isinstance(msg, MessagesPlaceholder):
                        return True
                    # Check for variable_name attribute that might be "messages"
                    if (
                        hasattr(msg, "variable_name")
                        and msg.variable_name == self.messages_placeholder_name
                    ):
                        return True
                    # Check nested message prompts
                    if hasattr(msg, "prompt") and hasattr(
                        msg.prompt, "input_variables"
                    ):
                        if self.messages_placeholder_name in msg.prompt.input_variables:
                            return True

                # Default to True for chat templates
                return True

            if isinstance(self.prompt_template, FewShotPromptTemplate):
                # FewShotPromptTemplate does not typically use messages
                return False

            # Check prompt template input variables
            if hasattr(self.prompt_template, "input_variables"):
                if (
                    self.messages_placeholder_name
                    in self.prompt_template.input_variables
                ):
                    return True

        # Default to True - safer to assume we use messages
        return True

    def _get_input_variables(self) -> Set[str]:
        """Get all input variables required by the prompt template, excluding partials and optionals.

        Returns:
            Set of required input variable names
        """
        if not self.prompt_template:
            return (
                {self.messages_placeholder_name} if self.uses_messages_field else set()
            )

        # Get all input variables from the template
        all_vars = set()

        # Direct input_variables attribute
        if hasattr(self.prompt_template, "input_variables"):
            all_vars.update(self.prompt_template.input_variables)

        # Chat templates message variables
        if isinstance(self.prompt_template, ChatPromptTemplate):
            for msg in self.prompt_template.messages:
                # Check message prompt templates
                if hasattr(msg, "prompt") and hasattr(msg.prompt, "input_variables"):
                    all_vars.update(msg.prompt.input_variables)
                # Check variable_name for placeholders
                if hasattr(msg, "variable_name"):
                    var_name = getattr(msg, "variable_name")
                    # Only add if not optional
                    if not getattr(msg, "optional", False):
                        all_vars.add(var_name)

        # Few-shot template variables
        if isinstance(self.prompt_template, FewShotPromptTemplate):
            if hasattr(self.prompt_template, "input_variables"):
                all_vars.update(self.prompt_template.input_variables)

        # Remove partial variables
        partial_vars = set()
        if hasattr(self.prompt_template, "partial_variables"):
            partial_vars.update(
                getattr(self.prompt_template, "partial_variables", {}).keys()
            )
        partial_vars.update(self.partial_variables.keys())

        # Remove optional variables
        optional_vars = set(self.optional_variables)
        if hasattr(self.prompt_template, "optional_variables"):
            optional_vars.update(
                getattr(self.prompt_template, "optional_variables", [])
            )

        # Return variables that aren't partials or optionals
        result = all_vars - partial_vars - optional_vars

        # If empty, default to messages for safety based on uses_messages_field
        if (
            not result
            and self.uses_messages_field
            and self.messages_placeholder_name not in optional_vars
        ):
            return {self.messages_placeholder_name}

        return result

    def get_input_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get schema fields based on prompt template and configuration.

        Implements abstract method from Engine base class.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import List as ListType
        from typing import Optional as OptionalType

        fields = {}

        # Get required input variables, excluding partials and optionals
        required_vars = self._get_input_variables()

        # Add messages field if needed (as required or optional)
        if self.uses_messages_field:
            if self.messages_placeholder_name in self.optional_variables:
                # Optional messages field
                fields[self.messages_placeholder_name] = (
                    OptionalType[ListType[BaseMessage]],
                    None,
                )
            else:
                # Required messages field
                fields[self.messages_placeholder_name] = (
                    ListType[BaseMessage],
                    Field(default_factory=list),
                )

        # Add fields for all required variables
        for var in required_vars:
            if var != self.messages_placeholder_name and var not in fields:
                # Try to get type information from prompt template
                var_type = str  # Default type
                if (
                    hasattr(self.prompt_template, "input_types")
                    and var in self.prompt_template.input_types
                ):
                    var_type = self.prompt_template.input_types[var]

                fields[var] = (var_type, Field(...))  # Required field

        # Add optional variables as optional fields
        for var in self.optional_variables:
            if var not in fields and var != self.messages_placeholder_name:
                # Try to get type information from prompt template
                var_type = str  # Default type
                if (
                    hasattr(self.prompt_template, "input_types")
                    and var in self.prompt_template.input_types
                ):
                    var_type = self.prompt_template.input_types[var]

                fields[var] = (OptionalType[var_type], None)

        return fields

    def get_output_fields(self) -> Dict[str, Tuple[Type, Any]]:
        """
        Get output fields based on structured_output_model and output_parser.

        Implements abstract method from Engine base class.

        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Dict
        from typing import List as ListType
        from typing import Optional as OptionalType

        fields = {}

        # Use structured_output_model if available and not parsing raw
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
                model_name = (
                    getattr(self.structured_output_model, "__name__", "").lower()
                    or "result"
                )
                fields[model_name] = (self.structured_output_model, None)

        # Handle Pydantic tools if specified
        elif self.pydantic_tools and self.parser_type == "pydantic_tools":
            # Get fields from all tool models
            for tool_model in self.pydantic_tools:
                if hasattr(tool_model, "model_fields"):
                    # Pydantic v2 - get a representative field to include in output
                    model_name = getattr(tool_model, "__name__", "").lower()
                    fields[model_name] = (tool_model, None)

        # Handle output parser types if structured_output_model not used or parsing raw
        elif self.output_parser or self.parse_raw_output:
            parser_name = (
                type(self.output_parser).__name__ if self.output_parser else ""
            )

            # String-based parsers
            if (
                parser_name in ["StrOutputParser", "StringOutputParser"]
                or self.parse_raw_output
            ):
                fields["content"] = (str, None)

            # JSON-based parsers
            elif parser_name in ["JsonOutputParser", "JSONLinesOutputParser"]:
                fields["result"] = (Dict[str, Any], None)

            # PydanticOutputParser
            elif parser_name == "PydanticOutputParser" and hasattr(
                self.output_parser, "pydantic_object"
            ):
                # Extract from PydanticOutputParser
                pydantic_model = self.output_parser.pydantic_object
                if hasattr(pydantic_model, "model_fields"):  # Pydantic v2
                    for field_name, field_info in pydantic_model.model_fields.items():
                        fields[field_name] = (field_info.annotation, field_info.default)

            # PydanticToolsParser
            elif parser_name == "PydanticToolsParser" and hasattr(
                self.output_parser, "tools"
            ):
                # For each tool model, add its fields
                for tool_model in self.output_parser.tools:
                    if hasattr(tool_model, "model_fields"):
                        model_name = getattr(tool_model, "__name__", "").lower()
                        fields[model_name] = (tool_model, None)

            # List-based parsers
            elif parser_name in ["ListOutputParser", "CSVOutputParser"]:
                fields["items"] = (ListType[Any], None)

            # Default parser output
            else:
                fields["output"] = (Any, None)

        # Default output fields
        if not fields:
            fields["content"] = (OptionalType[str], None)
            fields[self.messages_placeholder_name] = (
                ListType[BaseMessage],
                Field(default_factory=list),
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
            ]

            for param in aug_llm_params:
                if param in configurable:
                    params[param] = configurable[param]

        return params

    def _process_input(
        self, input_data: Union[str, Dict[str, Any], List[BaseMessage]]
    ) -> Dict[str, Any]:
        """Process input into a format usable by the runnable."""
        # Find input variables required by the prompt template
        required_vars = self._get_input_variables()

        # Handle dictionary input
        if isinstance(input_data, dict):
            # Simply return the input dict - all needed fields should be there
            return input_data

        # Handle string input
        if isinstance(input_data, str):
            result = {}

            # If we need messages
            if self.uses_messages_field:
                result[self.messages_placeholder_name] = [
                    HumanMessage(content=input_data)
                ]

            # For other variables, use the string directly
            for var in required_vars:
                if var != self.messages_placeholder_name:
                    result[var] = input_data

            return result

        # Handle list of messages
        if isinstance(input_data, list) and all(
            isinstance(item, BaseMessage) for item in input_data
        ):
            result = {self.messages_placeholder_name: input_data}
            return result

        # Default case - convert to human message
        return {self.messages_placeholder_name: [HumanMessage(content=str(input_data))]}

    def add_system_message(self, content: str) -> "AugLLMConfig":
        """Add or update system message in the prompt template.

        Args:
            content: System message content

        Returns:
            Self for chaining
        """
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
                else:
                    new_messages.append(msg)

            # Add system message if none exists
            if not has_system:
                new_messages.insert(0, SystemMessage(content=content))

            # Create new template with updated messages
            partial_vars = getattr(self.prompt_template, "partial_variables", None)
            self.prompt_template = ChatPromptTemplate.from_messages(
                new_messages, partial_variables=partial_vars
            )
        else:
            # Create chat template if none exists
            messages = [SystemMessage(content=content)]

            # Only add MessagesPlaceholder if auto-add is enabled
            if self.add_messages_placeholder:
                is_optional = self.messages_placeholder_name in self.optional_variables
                messages.append(
                    MessagesPlaceholder(
                        variable_name=self.messages_placeholder_name,
                        optional=is_optional,
                    )
                )

            self.prompt_template = ChatPromptTemplate.from_messages(messages)

        # Update uses_messages_field
        self.uses_messages_field = True

        return self

    def add_human_message(self, content: str) -> "AugLLMConfig":
        """Add a human message to the prompt template.

        Args:
            content: Human message content

        Returns:
            Self for chaining
        """
        if isinstance(self.prompt_template, ChatPromptTemplate):
            # Add to existing chat template
            new_messages = list(self.prompt_template.messages)
            new_messages.append(HumanMessage(content=content))

            # Create new template
            partial_vars = getattr(self.prompt_template, "partial_variables", None)
            self.prompt_template = ChatPromptTemplate.from_messages(
                new_messages, partial_variables=partial_vars
            )
        else:
            # Create new chat template
            messages = []
            if self.system_message:
                messages.append(SystemMessage(content=self.system_message))
            messages.append(HumanMessage(content=content))

            # Only add MessagesPlaceholder if auto-add is enabled
            if self.add_messages_placeholder:
                is_optional = self.messages_placeholder_name in self.optional_variables
                messages.append(
                    MessagesPlaceholder(
                        variable_name=self.messages_placeholder_name,
                        optional=is_optional,
                    )
                )

            self.prompt_template = ChatPromptTemplate.from_messages(messages)

        # Update uses_messages_field
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

            # Create new template
            partial_vars = getattr(self.prompt_template, "partial_variables", None)
            self.prompt_template = ChatPromptTemplate.from_messages(
                new_messages, partial_variables=partial_vars
            )

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

        if index < len(self.prompt_template.messages):
            new_messages = list(self.prompt_template.messages)
            removed = new_messages.pop(index)

            # Create new template
            partial_vars = getattr(self.prompt_template, "partial_variables", None)
            self.prompt_template = ChatPromptTemplate.from_messages(
                new_messages, partial_variables=partial_vars
            )

            # Update uses_messages_field if we removed the MessagesPlaceholder
            if (
                isinstance(removed, MessagesPlaceholder)
                and removed.variable_name == self.messages_placeholder_name
            ):
                # Re-add the placeholder if add_messages_placeholder is True
                if self.add_messages_placeholder:
                    self._ensure_messages_placeholder()
                else:
                    # Check if there's still a messages placeholder
                    self.uses_messages_field = self._detect_uses_messages_field()

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

            # Update for the messages placeholder if it matches
            if var_name == self.messages_placeholder_name and isinstance(
                self.prompt_template, ChatPromptTemplate
            ):
                # Find and update the MessagesPlaceholder
                for msg in self.prompt_template.messages:
                    if (
                        isinstance(msg, MessagesPlaceholder)
                        and getattr(msg, "variable_name", "") == var_name
                        and hasattr(msg, "optional")
                    ):
                        msg.optional = True
                        break

            # Apply optional variables to prompt template
            self._apply_optional_variables()

        return self

    def with_structured_output(
        self, model: Type[BaseModel], include_instructions: bool = True
    ) -> "AugLLMConfig":
        """Configure with Pydantic structured output.

        Args:
            model: Pydantic model for structured output
            include_instructions: Whether to include format instructions in prompt

        Returns:
            Self for chaining
        """
        # Set the structured output model
        self.structured_output_model = model
        self.parser_type = "pydantic"
        self.include_format_instructions = include_instructions

        # Create and configure parser if needed
        self._setup_output_parser()

        return self

    def with_pydantic_tools(
        self, tool_models: List[Type[BaseModel]], include_instructions: bool = True
    ) -> "AugLLMConfig":
        """Configure with Pydantic tools output parsing.

        Args:
            tool_models: List of Pydantic models for tool schemas
            include_instructions: Whether to include format instructions in prompt

        Returns:
            Self for chaining
        """
        # Set the pydantic tools
        self.pydantic_tools = tool_models
        self.parser_type = "pydantic_tools"
        self.include_format_instructions = include_instructions

        # Setup parser
        self._setup_pydantic_tools_parser()

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
    def from_few_shot_chat(
        cls,
        examples: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        human_template: Optional[str] = None,
        ai_template: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create with few-shot examples for a chat prompt.

        Args:
            examples: List of example dictionaries
            system_message: Optional system message
            human_template: Template for human messages
            ai_template: Template for AI responses
            llm_config: Optional LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")

        # Extract optional variables
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

        # Create messages for the prompt
        messages = []

        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # Process examples if templates are provided
        if examples and (human_template or ai_template):
            for example in examples:
                if human_template:
                    # Create a formatted human message
                    human_content = human_template
                    for key, value in example.items():
                        if isinstance(value, str):
                            placeholder = "{" + key + "}"
                            human_content = human_content.replace(placeholder, value)
                    messages.append(HumanMessage(content=human_content))

                if ai_template:
                    # Create a formatted AI message
                    ai_content = ai_template
                    for key, value in example.items():
                        if isinstance(value, str):
                            placeholder = "{" + key + "}"
                            ai_content = ai_content.replace(placeholder, value)
                    messages.append(AIMessage(content=ai_content))

        # Add messages placeholder for user input if auto-add is enabled
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
    def from_tools(
        cls,
        tools: List[Union[BaseTool, Type[BaseTool], str]],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        **kwargs,
    ):
        """Create with specified tools.

        Args:
            tools: List of tools to make available
            system_message: Optional system message
            llm_config: LLM configuration (optional)
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        # Get messages placeholder configuration
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

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

        return cls(
            tools=tools,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,  # Tool-using LLMs always use messages
            messages_placeholder_name=messages_placeholder_name,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            **kwargs,
        )

    @classmethod
    def from_pydantic_tools(
        cls,
        tool_models: List[Type[BaseModel]],
        system_message: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        include_instructions: bool = True,
        **kwargs,
    ):
        """Create with Pydantic tool models.

        Args:
            tool_models: List of Pydantic models for tool schemas
            system_message: Optional system message
            llm_config: LLM configuration (optional)
            include_instructions: Whether to include format instructions
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Use default LLM config if none provided
        if llm_config is None:
            llm_config = AzureLLMConfig(model="gpt-4o")

        # Get messages placeholder configuration
        messages_placeholder_name = kwargs.pop("messages_placeholder_name", "messages")
        optional_variables = kwargs.pop("optional_variables", [])
        add_messages_placeholder = kwargs.pop("add_messages_placeholder", True)

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

        # Create parser if instructions should be included
        parser = PydanticToolsParser(tools=tool_models)

        # Prepare partial variables with format instructions if needed
        partial_variables = kwargs.pop("partial_variables", {})
        if include_instructions and hasattr(parser, "get_format_instructions"):
            partial_variables["format_instructions"] = parser.get_format_instructions()

        return cls(
            pydantic_tools=tool_models,
            parser_type="pydantic_tools",
            output_parser=parser,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,
            messages_placeholder_name=messages_placeholder_name,
            partial_variables=partial_variables,
            include_format_instructions=include_instructions,
            optional_variables=optional_variables,
            add_messages_placeholder=add_messages_placeholder,
            **kwargs,
        )

    def instantiate_llm(self) -> Any:
        """Instantiate the LLM based on the configuration."""
        return self.llm_config.instantiate()
