"""
AugLLM configuration system for enhanced LLM chains.

Provides a structured way to configure and create LLM chains with prompts,
tools, output parsers, and structured output models.
"""

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
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
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
    Configuration for creating enhanced LLM chains.

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

    # Output handling
    structured_output_model: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured output"
    )
    output_parser: Optional[BaseOutputParser] = Field(
        default=None, description="Parser for LLM output"
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

        # Apply partial variables to the prompt template if needed
        if self.prompt_template and self.partial_variables:
            if hasattr(self.prompt_template, "partial_variables"):
                # Create a new partial_variables dict to avoid modifying the original
                existing_partials = (
                    getattr(self.prompt_template, "partial_variables", {}) or {}
                )
                self.prompt_template.partial_variables = {
                    **existing_partials,
                    **self.partial_variables,
                }

        # Create a FewShotPromptTemplate if examples and example_prompt are provided but no prompt_template
        if (
            self.examples
            and self.example_prompt
            and not self.prompt_template
            and self.prefix
            and self.suffix
        ):
            if self.input_variables:
                self.prompt_template = FewShotPromptTemplate(
                    examples=self.examples,
                    example_prompt=self.example_prompt,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    input_variables=self.input_variables,
                    example_separator=self.example_separator,
                    partial_variables=self.partial_variables,
                )
                self.uses_messages_field = (
                    False  # Few-shot prompts typically don't use messages
                )

        # Create a simple chat template with system message if only system_message is provided
        if self.system_message and not self.prompt_template:
            self.prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=self.system_message),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            self.uses_messages_field = True

        return self

    def _detect_uses_messages_field(self) -> bool:
        """Detect if this LLM configuration uses a messages field.

        Returns:
            True if messages field is detected, False otherwise
        """
        # Default to True for tools and system_message
        if self.tools or self.system_message:
            return True

        # Check if prompt_template contains MessagesPlaceholder
        if self.prompt_template:
            # Handle different template types differently
            if isinstance(self.prompt_template, ChatPromptTemplate):
                # Check if any message is a MessagesPlaceholder
                for msg in self.prompt_template.messages:
                    if isinstance(msg, MessagesPlaceholder) or (
                        hasattr(msg, "variable_name")
                        and msg.variable_name == "messages"
                    ):
                        return True

                # If it's a chat template but no messages placeholder, check input variables
                if hasattr(self.prompt_template, "input_variables"):
                    return "messages" in self.prompt_template.input_variables

                # Default to True for chat templates
                return True

            if isinstance(self.prompt_template, FewShotPromptTemplate):
                # FewShotPromptTemplate does not typically use messages
                return False

            # Check prompt template input variables
            if hasattr(self.prompt_template, "input_variables"):
                if "messages" in self.prompt_template.input_variables:
                    return True

        # Default to True - safer to assume we use messages
        return True

    def _get_input_variables(self) -> Set[str]:
        """Get all input variables required by the prompt template, excluding partials.

        Returns:
            Set of required input variable names
        """
        if not self.prompt_template:
            return {"messages"} if self.uses_messages_field else set()

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
                    all_vars.add(msg.variable_name)

        # Few-shot template variables
        if isinstance(self.prompt_template, FewShotPromptTemplate):
            if hasattr(self.prompt_template, "input_variables"):
                all_vars.update(self.prompt_template.input_variables)

        # Remove partial variables
        partial_vars = set()
        if hasattr(self.prompt_template, "partial_variables"):
            partial_vars.update(self.prompt_template.partial_variables.keys())
        partial_vars.update(self.partial_variables.keys())

        # Return variables that aren't partials
        result = all_vars - partial_vars

        # If empty, default to messages for safety based on uses_messages_field
        if not result and self.uses_messages_field:
            return {"messages"}

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

        # Get required input variables, excluding partials
        required_vars = self._get_input_variables()

        # Add messages field if needed
        if self.uses_messages_field or "messages" in required_vars:
            fields["messages"] = (ListType[BaseMessage], Field(default_factory=list))

        # Add fields for all required variables
        for var in required_vars:
            if var != "messages" and var not in fields:
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

        # Use structured_output_model if available
        if self.structured_output_model:
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
                model_name = self.structured_output_model.__name__.lower()
                fields[model_name] = (self.structured_output_model, None)

        # Handle output parser types
        elif self.output_parser:
            # Try to detect output type from parser
            parser_name = type(self.output_parser).__name__

            # String-based parsers
            if parser_name in ["StrOutputParser", "StringOutputParser"]:
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

            # List-based parsers
            elif parser_name in ["ListOutputParser", "CSVOutputParser"]:
                fields["items"] = (ListType[Any], None)

            # Default parser output
            else:
                fields["output"] = (Any, None)

        # Default output fields
        if not fields:
            fields["content"] = (OptionalType[str], None)
            fields["messages"] = (ListType[BaseMessage], Field(default_factory=list))

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
            if "messages" in required_vars or self.uses_messages_field:
                result["messages"] = [HumanMessage(content=input_data)]

            # For other variables, use the string directly
            for var in required_vars:
                if var != "messages":
                    result[var] = input_data

            return result

        # Handle list of messages
        if isinstance(input_data, list) and all(
            isinstance(item, BaseMessage) for item in input_data
        ):
            result = {"messages": input_data}
            return result

        # Default case - convert to human message
        return {"messages": [HumanMessage(content=str(input_data))]}

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
    def from_prompt(cls, prompt: BasePromptTemplate, llm_config: LLMConfig, **kwargs):
        """Create from a prompt template.

        Args:
            prompt: Prompt template to use
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Handle partial variables if provided in kwargs
        partial_variables = kwargs.pop("partial_variables", {})

        # Detect if this is a messages-based prompt
        uses_messages = kwargs.pop("uses_messages_field", None)
        if uses_messages is None:
            # Auto-detect based on prompt type
            if isinstance(prompt, ChatPromptTemplate):
                # Check if any message is a MessagesPlaceholder
                uses_messages = any(
                    isinstance(msg, MessagesPlaceholder)
                    or (
                        hasattr(msg, "variable_name")
                        and msg.variable_name == "messages"
                    )
                    for msg in prompt.messages
                )
            else:
                uses_messages = False

        config = cls(
            prompt_template=prompt,
            llm_config=llm_config,
            partial_variables=partial_variables,
            uses_messages_field=uses_messages,
            **kwargs,
        )
        return config

    @classmethod
    def from_system_prompt(cls, system_prompt: str, llm_config: LLMConfig, **kwargs):
        """Create from a system prompt string.

        Args:
            system_prompt: System prompt string
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Create a simple chat prompt template with system message and messages placeholder
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        return cls(
            prompt_template=prompt,
            system_message=system_prompt,
            llm_config=llm_config,
            uses_messages_field=True,
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
        llm_config: LLMConfig,
        **kwargs,
    ):
        """Create with few-shot examples.

        Args:
            examples: List of examples as dictionaries
            example_prompt: Template for formatting examples
            prefix: Text before examples
            suffix: Text after examples
            input_variables: Input variables for the prompt
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")

        # Create few-shot prompt template
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            partial_variables=partial_variables,
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
        llm_config: LLMConfig,
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
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})
        example_separator = kwargs.pop("example_separator", "\n\n")

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
        # Extract partial_variables from kwargs if provided
        partial_variables = kwargs.pop("partial_variables", {})

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
                            human_content = human_content.replace(f"{{{key}}}", value)
                    messages.append(HumanMessage(content=human_content))

                if ai_template:
                    # Create a formatted AI message
                    ai_content = ai_template
                    for key, value in example.items():
                        if isinstance(value, str):
                            ai_content = ai_content.replace(f"{{{key}}}", value)
                    messages.append(AIMessage(content=ai_content))

        # Add messages placeholder for user input
        messages.append(MessagesPlaceholder(variable_name="messages"))

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(messages)

        return cls(
            prompt_template=prompt,
            examples=examples,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,
            partial_variables=partial_variables,
            **kwargs,
        )

    @classmethod
    def from_tools(
        cls,
        tools: List[Union[BaseTool, Type[BaseTool], str]],
        system_message: Optional[str] = None,
        llm_config: LLMConfig = None,
        **kwargs,
    ):
        """Create with specified tools.

        Args:
            tools: List of tools to make available
            system_message: Optional system message
            llm_config: LLM configuration
            **kwargs: Additional parameters

        Returns:
            AugLLMConfig instance
        """
        # Create prompt template if system message is provided
        prompt_template = None
        if system_message:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_message),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )

        return cls(
            tools=tools,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config,
            uses_messages_field=True,  # Tool-using LLMs always use messages
            **kwargs,
        )

    def instatiate_llm(self) -> LLMConfig:
        """Instatiate the LLM based on the configuration."""
        return self.llm_config.instantiate()
