# src/haive/core/engine/aug_llm.py

import inspect
import logging
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Union, Type, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, chain
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model, model_validator

from haive.core.engine.base import EngineRegistry, EngineType, InvokableEngine
from haive.core.models.llm.base import AzureLLMConfig, LLMConfig

logger = logging.getLogger(__name__)

class AugLLMConfig(InvokableEngine[Union[str, Dict[str, Any], List[BaseMessage]], Union[BaseMessage, Dict[str, Any]]]):
    """Configuration for creating a structured runnable LLM pipeline.
    
    AugLLMConfig extends InvokableEngine to provide a powerful way to create
    LLM chains with prompts, tools, output parsers, and more.
    """
    engine_type: EngineType = Field(default=EngineType.LLM, description="The type of engine")

    # Use the existing LLMConfig system
    llm_config: Union[LLMConfig, Dict[str, Any]] = Field(
        default_factory=lambda: AzureLLMConfig(model="gpt-4o"),
        description="LLM provider configuration"
    )

    # Core components
    prompt_template: Optional[BasePromptTemplate] = Field(
        default=None,
        description="Prompt template for the LLM"
    )

    # System message for chat models
    system_message: Optional[str] = Field(
        default=None,
        description="System message for chat models"
    )

    # Few-shot components
    examples: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Examples for few-shot prompting"
    )
    
    example_prompt: Optional[PromptTemplate] = Field(
        default=None,
        description="Template for formatting few-shot examples"
    )
    
    prefix: Optional[str] = Field(
        default=None,
        description="Text before examples in few-shot prompting"
    )
    
    suffix: Optional[str] = Field(
        default=None,
        description="Text after examples in few-shot prompting"
    )
    
    example_separator: str = Field(
        default="\n\n",
        description="Separator between examples in few-shot prompting"
    )
    
    input_variables: Optional[List[str]] = Field(
        default=None,
        description="Input variables for the prompt template"
    )

    # Tools
    tools: List[Union[BaseTool, StructuredTool, Type[BaseTool], str]] = Field(
        default_factory=list,
        description="List of tools to make available to the LLM"
    )

    # Output handling
    structured_output_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model for structured output"
    )
    output_parser: Optional[BaseOutputParser] = Field(
        default=None,
        description="Parser for LLM output"
    )

    # Tool binding options
    tool_kwargs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters for tool instantiation"
    )
    bind_tools_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for binding tools to the LLM"
    )
    bind_tools_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for bind_tools"
    )
    force_tool_choice: Optional[str] = Field(
        default=None,
        description="Force the LLM to use this specific tool"
    )

    # Pre/post processing
    preprocess: Optional[Callable[[Any], Any]] = Field(
        default=None,
        description="Function to preprocess input before sending to LLM",
        exclude=True  # Exclude from serialization
    )
    postprocess: Optional[Callable[[Any], Any]] = Field(
        default=None,
        description="Function to postprocess output from LLM",
        exclude=True  # Exclude from serialization
    )

    # Runtime options
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature parameter for the LLM"
    )

    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )

    runtime_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional runtime options for the LLM"
    )

    # Custom runnables to chain
    custom_runnables: Optional[List[Runnable]] = Field(
        default=None,
        description="Custom runnables to add to the chain",
        exclude=True  # Exclude from serialization
    )

    # Schema definitions for explicit control
    input_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Explicit input schema definition",
        exclude=True
    )

    output_schema: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Explicit output schema definition",
        exclude=True
    )

    # Override messages field detection
    uses_messages_field: Optional[bool] = Field(
        default=None,
        description="Explicitly specify if this engine uses a messages field. If None, auto-detected."
    )

    # Partial variables for templates
    partial_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Partial variables for the prompt template"
    )

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_and_setup(self):
        """Perform validation and setup after initialization."""
        # Convert dict to LLMConfig if needed
        if isinstance(self.llm_config, dict):
            from haive.core.models.llm.base import AzureLLMConfig
            # Get provider if specified
            provider = self.llm_config.get("provider", "azure")

            if provider.lower() == "azure":
                self.llm_config = AzureLLMConfig(**self.llm_config)
            else:
                # Default to Azure if provider not recognized
                self.llm_config = AzureLLMConfig(**self.llm_config)

        # Auto-detect uses_messages_field if not explicitly set
        if self.uses_messages_field is None:
            self.uses_messages_field = self._detect_uses_messages_field()

        # Apply partial variables to the prompt template if needed
        if self.prompt_template and self.partial_variables:
            if not hasattr(self.prompt_template, "partial_variables"):
                # Create a partial variables dictionary if it doesn't exist
                self.prompt_template.partial_variables = {}

            # Update with our partial variables
            self.prompt_template.partial_variables.update(self.partial_variables)
            
        # Create a FewShotPromptTemplate if examples and example_prompt are provided but no prompt_template
        if self.examples and self.example_prompt and not self.prompt_template and self.prefix and self.suffix:
            if self.input_variables:
                self.prompt_template = FewShotPromptTemplate(
                    examples=self.examples,
                    example_prompt=self.example_prompt,
                    prefix=self.prefix,
                    suffix=self.suffix,
                    input_variables=self.input_variables,
                    example_separator=self.example_separator,
                    partial_variables=self.partial_variables
                )
                self.uses_messages_field = False  # Few-shot prompts typically don't use messages

        # Create a simple chat template with system message if only system_message is provided
        if self.system_message and not self.prompt_template:
            self.prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_message),
                MessagesPlaceholder(variable_name="messages")
            ])
            self.uses_messages_field = True

        return self

    def _detect_uses_messages_field(self) -> bool:
        """Detect if this LLM configuration uses a messages field.
        
        Returns:
            True if messages field is detected, False otherwise
        """
        # Default to True for safety
        if self.tools or self.system_message:
            return True

        # Check if prompt_template contains MessagesPlaceholder
        if self.prompt_template:
            # Handle different template types differently
            if isinstance(self.prompt_template, ChatPromptTemplate):
                # Check if any message is a MessagesPlaceholder
                for msg in self.prompt_template.messages:
                    if (isinstance(msg, MessagesPlaceholder) or
                        (hasattr(msg, "variable_name") and msg.variable_name == "messages")):
                        return True

                # If it's a chat template but no messages placeholder, check further
                # Look at input variables - if messages is in there, use it
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

    def _get_input_variables(self) -> set[str]:
        """Get all input variables required by the prompt template, excluding partials.
        
        Returns:
            Set of required input variable names
        """
        if not self.prompt_template:
            return {"messages"}  # Default to messages if no template

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
            if hasattr(self.prompt_template, "example_prompt") and hasattr(self.prompt_template.example_prompt, "input_variables"):
                # Don't add example variables as they're handled internally
                pass
            if hasattr(self.prompt_template, "input_variables"):
                all_vars.update(self.prompt_template.input_variables)

        # Remove partial variables
        partial_vars = set()
        if hasattr(self.prompt_template, "partial_variables"):
            partial_vars.update(self.prompt_template.partial_variables.keys())
        partial_vars.update(self.partial_variables.keys())

        # Return variables that aren't partials
        result = all_vars - partial_vars

        # If empty, default to messages for safety
        if not result and self.uses_messages_field:
            return {"messages"}

        return result

    def create_runnable(self, runnable_config: Optional[RunnableConfig] = None) -> Runnable:
        """Create a runnable LLM chain.
        
        Args:
            runnable_config: Optional runtime configuration
            
        Returns:
            A runnable LLM chain
        """
        # Extract config parameters from runnable_config
        config_params = {}
        if runnable_config:
            config_params = self.apply_runnable_config(runnable_config)

        # Create factory with config params
        factory = AugLLMFactory(self, config_params)

        # Build the runnable chain
        return factory.create_runnable()

    def invoke(self, input_data: Union[str, Dict[str, Any], List[BaseMessage]], runnable_config: Optional[RunnableConfig] = None) -> Union[BaseMessage, Dict[str, Any]]:
        """Invoke the LLM with input data.
        
        Args:
            input_data: Input as string, dict, or messages
            runnable_config: Optional runtime configuration
            
        Returns:
            LLM response
        """
        # Create runnable
        runnable = self.create_runnable(runnable_config)
        logger.debug(f"Input data: {input_data}")
        # Process input
        processed_input = self._process_input(input_data)
        logger.debug(f"Processed input: {processed_input}")
        # Invoke the runnable
        return runnable.invoke(processed_input, config=runnable_config)

    def _process_input(self, input_data: Union[str, Dict[str, Any], List[BaseMessage]]) -> Dict[str, Any]:
        """Process input into a format usable by the runnable."""
        # Find input variables required by the prompt template
        required_vars = set()
        if self.prompt_template:
            if hasattr(self.prompt_template, "input_variables"):
                required_vars.update(self.prompt_template.input_variables)
        
        # If no variables required, default to messages
        if not required_vars:
            required_vars = {"messages"}
        
        # Remove partial variables
        partial_vars = set()
        if hasattr(self.prompt_template, "partial_variables"):
            partial_vars.update(self.prompt_template.partial_variables.keys())
        partial_vars.update(self.partial_variables.keys())
        required_vars = required_vars - partial_vars
        
        # Handle dictionary input
        if isinstance(input_data, dict):
            # Simply return the input dict - all needed fields should be there
            return input_data
        
        # Handle string input
        if isinstance(input_data, str):
            result = {}
            for var in required_vars:
                result[var] = input_data
            return result
        
        # Handle list of messages
        if isinstance(input_data, list) and all(isinstance(item, BaseMessage) for item in input_data):
            result = {"messages": input_data}
            return result
        
        # Default case
        return {"messages": [HumanMessage(content=str(input_data))]}

    def derive_input_schema(self) -> Type[BaseModel]:
        """Derive input schema based on the prompt template and partial variables.
        
        Returns:
            Pydantic model for input schema
        """
        # Use provided schema if available
        if self.input_schema:
            return self.input_schema

        from typing import Optional as OptionalType, Dict, List as ListType, Any
        schema_fields = {}

        # Get required input variables, excluding partials
        required_vars = self._get_input_variables()

        # Always include messages field for safety
        schema_fields["messages"] = (List[BaseMessage], Field(default_factory=list))

        # Add content as a common field
        schema_fields["content"] = (OptionalType[str], None)

        # Add question field for QA contexts
        schema_fields["question"] = (OptionalType[str], None)

        # Add fields for all required variables
        for var in required_vars:
            if var not in schema_fields:
                # Try to get type information from prompt template
                var_type = str  # Default type
                if hasattr(self.prompt_template, "input_types") and var in self.prompt_template.input_types:
                    var_type = self.prompt_template.input_types[var]

                schema_fields[var] = (OptionalType[var_type], None)

        # Add fields from structured output model if available
        if self.structured_output_model:
            if hasattr(self.structured_output_model, "model_fields"):
                # Pydantic v2
                for field_name, field_info in self.structured_output_model.model_fields.items():
                    if field_name not in schema_fields:
                        schema_fields[field_name] = (field_info.annotation, field_info.default)
            elif hasattr(self.structured_output_model, "__fields__"):
                # Pydantic v1
                for field_name, field_info in self.structured_output_model.__fields__.items():
                    if field_name not in schema_fields:
                        schema_fields[field_name] = (field_info.type_, field_info.default)
        
        # Add fields based on output parser type
        if self.output_parser:
            # Try to detect output type from parser
            parser_type = None
            parser_output_type = str  # Default to string
            
            # Check for common parser types
            parser_name = type(self.output_parser).__name__
            
            # String-based parsers
            if parser_name in ["StrOutputParser", "StringOutputParser"]:
                parser_output_type = str
            
            # JSON-based parsers
            elif parser_name in ["JsonOutputParser", "JSONLinesOutputParser", "PydanticOutputParser"]:
                parser_output_type = Dict[str, Any]
            
            # List-based parsers
            elif parser_name in ["ListOutputParser", "CSVOutputParser"]:
                parser_output_type = ListType[Any]
                
            # Try to get more specific types from parser if possible
            if hasattr(self.output_parser, "pydantic_object") and self.output_parser.pydantic_object:
                # Extract from PydanticOutputParser
                pydantic_model = self.output_parser.pydantic_object
                if hasattr(pydantic_model, "model_fields"):  # Pydantic v2
                    for field_name, field_info in pydantic_model.model_fields.items():
                        if field_name not in schema_fields:
                            schema_fields[field_name] = (field_info.annotation, field_info.default)
                elif hasattr(pydantic_model, "__fields__"):  # Pydantic v1
                    for field_name, field_info in pydantic_model.__fields__.items():
                        if field_name not in schema_fields:
                            schema_fields[field_name] = (field_info.type_, field_info.default)
            
            # Add output field if not already present
            if "output" not in schema_fields:
                schema_fields["output"] = (OptionalType[parser_output_type], None)

        # Create and return the model
        return create_model(f"{self.__class__.__name__}Input", **schema_fields)

    def derive_output_schema(self) -> Type[BaseModel]:
        """Derive output schema based on structured_output_model.
        
        Returns:
            Pydantic model for output schema
        """
        # Use provided schema if available
        if self.output_schema:
            return self.output_schema

        schema_fields = {}

        # Use structured_output_model if available
        if self.structured_output_model:
            # Extract fields from the model
            if hasattr(self.structured_output_model, "model_fields"):
                # Pydantic v2
                for field_name, field_info in self.structured_output_model.model_fields.items():
                    schema_fields[field_name] = (field_info.annotation, field_info.default)
            elif hasattr(self.structured_output_model, "__fields__"):
                # Pydantic v1
                for field_name, field_info in self.structured_output_model.__fields__.items():
                    schema_fields[field_name] = (field_info.type_, field_info.default)
            else:
                # Fallback to using the model as a single field
                model_name = self.structured_output_model.__name__.lower()
                schema_fields[model_name] = (self.structured_output_model, None)

        # Always include content field for all outputs
        from typing import Optional as OptionalType
        schema_fields["content"] = (OptionalType[str], None)

        # Always include messages field for all outputs
        schema_fields["messages"] = (List[BaseMessage], Field(default_factory=list))

        # Create schema model
        return create_model(f"{self.__class__.__name__}Output", **schema_fields)

    def get_schema_fields(self) -> Dict[str, tuple[type, Any]]:
        """Get schema fields based on prompt template and default fields.
        
        Returns:
            Dictionary mapping field names to (type, default) tuples
        """
        from typing import Optional as OptionalType
        fields = {}

        # Always include content field
        fields["content"] = (OptionalType[str], None)

        # Always include question field
        fields["question"] = (OptionalType[str], None)

        # Always include messages field
        fields["messages"] = (List[BaseMessage], Field(default_factory=list))

        # Add variables from prompt template
        required_vars = self._get_input_variables()
        for var in required_vars:
            # Skip if already added
            if var in fields:
                continue

            # Add field with appropriate type
            if hasattr(self.prompt_template, "input_types") and var in self.prompt_template.input_types:
                var_type = self.prompt_template.input_types[var]
                fields[var] = (OptionalType[var_type], None)
            else:
                fields[var] = (OptionalType[str], None)

        # Add fields from structured output model if available
        if self.structured_output_model:
            if hasattr(self.structured_output_model, "model_fields"):
                # Pydantic v2
                for field_name, field_info in self.structured_output_model.model_fields.items():
                    if field_name not in fields:
                        fields[field_name] = (field_info.annotation, field_info.default)
            elif hasattr(self.structured_output_model, "__fields__"):
                # Pydantic v1
                for field_name, field_info in self.structured_output_model.__fields__.items():
                    if field_name not in fields:
                        fields[field_name] = (field_info.type_, field_info.default)

        return fields

    def apply_runnable_config(self, runnable_config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Extract parameters from runnable_config relevant to this engine.
        
        Args:
            runnable_config: Runtime configuration
            
        Returns:
            Dictionary of relevant parameters
        """
        # Start with common parameters
        params = super().apply_runnable_config(runnable_config)

        if runnable_config and "configurable" in runnable_config:
            configurable = runnable_config["configurable"]

            # Extract AugLLM-specific parameters
            aug_llm_params = [
                "tools", "force_tool_choice", "top_p", "top_k", "temperature", "max_tokens",
                "system_message", "stop_sequences", "stop", "frequency_penalty", "presence_penalty"
            ]
            for param in aug_llm_params:
                if param in configurable:
                    params[param] = configurable[param]

            # Check for partial_variables updates in config
            if "partial_variables" in configurable:
                params["partial_variables"] = configurable["partial_variables"]

        return params

    def _resolve_tools(self) -> List[BaseTool]:
        """Resolve tool references to actual tool objects.
        
        Returns:
            List of instantiated tool objects
        """
        resolved_tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                # Lookup tool from registry
                tool_engine = EngineRegistry.get_instance().get(EngineType.TOOL, tool)
                if tool_engine:
                    resolved_tools.append(tool_engine.instantiate())
                else:
                    raise ValueError(f"Tool not found in registry: {tool}")
            elif inspect.isclass(tool) and issubclass(tool, BaseTool):
                # Instantiate class
                tool_name = tool.__name__
                tool_kwargs = self.tool_kwargs.get(tool_name, {})
                resolved_tools.append(tool(**tool_kwargs) if tool_kwargs else tool())
            else:
                # Already an instance
                resolved_tools.append(tool)
        return resolved_tools

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
    def from_prompt(cls, prompt: BasePromptTemplate, llm_config: Optional[LLMConfig] = None, **kwargs):
        """Create from a prompt template.
        
        Args:
            prompt: Prompt template to use
            llm_config: Optional LLM configuration
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
                    isinstance(msg, MessagesPlaceholder) or
                    (hasattr(msg, "variable_name") and msg.variable_name == "messages")
                    for msg in prompt.messages
                )
            else:
                uses_messages = False

        config = cls(
            prompt_template=prompt,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            partial_variables=partial_variables,
            uses_messages_field=uses_messages,
            **kwargs
        )
        return config

    @classmethod
    def from_system_prompt(cls, system_prompt: str, llm_config: Optional[LLMConfig] = None, **kwargs):
        """Create from a system prompt string.
        
        Args:
            system_prompt: System prompt string
            llm_config: Optional LLM configuration
            **kwargs: Additional parameters
            
        Returns:
            AugLLMConfig instance
        """
        # Create a simple chat prompt template with system message and messages placeholder
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])

        return cls(
            prompt_template=prompt,
            system_message=system_prompt,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            uses_messages_field=True,
            **kwargs
        )

    @classmethod
    def from_few_shot(cls,
                     examples: List[Dict[str, Any]],
                     example_prompt: PromptTemplate,
                     prefix: str,
                     suffix: str,
                     input_variables: List[str],
                     llm_config: Optional[LLMConfig] = None,
                     **kwargs):
        """Create with few-shot examples.
        
        Args:
            examples: List of examples as dictionaries
            example_prompt: Template for formatting examples
            prefix: Text before examples
            suffix: Text after examples
            input_variables: Input variables for the prompt
            llm_config: Optional LLM configuration
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
            partial_variables=partial_variables
        )

        return cls(
            prompt_template=few_shot_prompt,
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            example_separator=example_separator,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            uses_messages_field=False,  # FewShotPromptTemplate typically doesn't use messages
            partial_variables=partial_variables,
            **kwargs
        )

    @classmethod
    def from_system_and_few_shot(cls,
                               system_message: str,
                               examples: List[Dict[str, Any]],
                               example_prompt: PromptTemplate,
                               prefix: str,
                               suffix: str,
                               input_variables: List[str],
                               llm_config: Optional[LLMConfig] = None,
                               **kwargs):
        """Create with system message and few-shot examples.
        
        Args:
            system_message: System message to use
            examples: List of examples as dictionaries
            example_prompt: Template for formatting examples
            prefix: Text before examples
            suffix: Text after examples
            input_variables: Input variables for the prompt
            llm_config: Optional LLM configuration
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
            partial_variables=partial_variables
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
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            uses_messages_field=False,  # FewShotPromptTemplate typically doesn't use messages
            partial_variables=partial_variables,
            **kwargs
        )

    @classmethod
    def from_few_shot_chat(cls,
                         examples: List[Dict[str, Any]],
                         system_message: Optional[str] = None,
                         human_template: Optional[str] = None,
                         ai_template: Optional[str] = None,
                         llm_config: Optional[LLMConfig] = None,
                         **kwargs):
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
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            uses_messages_field=True,
            partial_variables=partial_variables,
            **kwargs
        )

    @classmethod
    def from_tools(cls, tools: List[Union[BaseTool, StructuredTool, Type[BaseTool], str]],
                 system_message: Optional[str] = None,
                 llm_config: Optional[LLMConfig] = None,
                 **kwargs):
        """Create with specified tools.
        
        Args:
            tools: List of tools to make available
            system_message: Optional system message
            llm_config: Optional LLM configuration
            **kwargs: Additional parameters
            
        Returns:
            AugLLMConfig instance
        """
        # Create prompt template if system message is provided
        prompt_template = None
        if system_message:
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_message),
                MessagesPlaceholder(variable_name="messages")
            ])

        return cls(
            tools=tools,
            prompt_template=prompt_template,
            system_message=system_message,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            uses_messages_field=True,  # Tool-using LLMs always use messages
            **kwargs
        )


class AugLLMFactory:
    """Factory for creating structured LLM runnables.
    
    Handles the complexities of constructing LLM chains with various components.
    """

    def __init__(self, aug_config: AugLLMConfig, config_params: Optional[Dict[str, Any]] = None):
        """Initialize the factory with an AugLLMConfig.
        
        Args:
            aug_config: Configuration for the LLM chain
            config_params: Optional runtime parameters to override defaults
        """
        self.aug_config = aug_config
        self.config_params = config_params or {}

        # Initialize fields from config
        self.llm_config = self.aug_config.llm_config
        self.prompt_template = self.aug_config.prompt_template
        self.system_message = self.aug_config.system_message
        self.tools = self.aug_config.tools
        self.structured_output_model = self.aug_config.structured_output_model
        self.output_parser = self.aug_config.output_parser
        self.tool_kwargs = self.aug_config.tool_kwargs
        self.bind_tools_kwargs = self.aug_config.bind_tools_kwargs
        self.bind_tools_config = self.aug_config.bind_tools_config
        self.preprocess = self.aug_config.preprocess
        self.postprocess = self.aug_config.postprocess
        self.custom_runnables = getattr(self.aug_config, "custom_runnables", None)
        self.uses_messages_field = self.aug_config.uses_messages_field
        self.partial_variables = self.aug_config.partial_variables or {}

        # Few-shot components
        self.examples = self.aug_config.examples
        self.example_prompt = self.aug_config.example_prompt
        self.prefix = self.aug_config.prefix
        self.suffix = self.aug_config.suffix
        self.example_separator = self.aug_config.example_separator
        self.input_variables = self.aug_config.input_variables

        # Apply any runtime config overrides
        self._apply_config_params()

        # Build components
        self.runnable_llm = self.initialize_llm()

        if self.tools:
            self.runnable_llm = self.initialize_llm_with_tools()

        if self.structured_output_model:
            self.runnable_llm = self.initialize_llm_with_structured_output()

        if self.output_parser:
            self.runnable_llm = self.initialize_llm_with_output_parser()

    def _apply_config_params(self):
        """Apply runtime config parameters to the factory instance."""
        # Skip if no config params provided
        if not self.config_params:
            return

        # Apply model override if specified
        if "model" in self.config_params:
            if hasattr(self.llm_config, "model"):
                self.llm_config.model = self.config_params["model"]

        # Apply temperature override if specified
        if "temperature" in self.config_params:
            if hasattr(self.llm_config, "temperature"):
                self.llm_config.temperature = self.config_params["temperature"]

        # Apply max_tokens override if specified
        if "max_tokens" in self.config_params:
            if hasattr(self.llm_config, "max_tokens"):
                self.llm_config.max_tokens = self.config_params["max_tokens"]

        # Apply system_message override if specified
        if "system_message" in self.config_params:
            self.system_message = self.config_params["system_message"]

        # Apply tool selection if specified
        if "tools" in self.config_params:
            self.tools = self.config_params["tools"]

        # Apply partial variables override if specified
        if "partial_variables" in self.config_params:
            self.partial_variables.update(self.config_params["partial_variables"])

    def initialize_llm(self):
        """Initialize the base LLM.
        
        Returns:
            Instantiated LLM
        """
        return self.llm_config.instantiate()

    def initialize_llm_with_tools(self):
        """Bind tools to the LLM.
        
        Returns:
            LLM with tools bound
        """
        # Resolve tools
        resolved_tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                # Lookup tool from registry
                tool_engine = EngineRegistry.get_instance().get(EngineType.TOOL, tool)
                if tool_engine:
                    resolved_tools.append(tool_engine.instantiate())
                else:
                    raise ValueError(f"Tool not found in registry: {tool}")
            elif inspect.isclass(tool) and issubclass(tool, BaseTool):
                # Instantiate class
                tool_name = tool.__name__
                tool_kwargs = self.tool_kwargs.get(tool_name, {})
                resolved_tools.append(tool(**tool_kwargs) if tool_kwargs else tool())
            else:
                # Already an instance
                resolved_tools.append(tool)

        # Handle force_tool_choice if specified
        bind_tools_kwargs = dict(self.bind_tools_kwargs)
        if hasattr(self.aug_config, "force_tool_choice") and self.aug_config.force_tool_choice:
            for tool in resolved_tools:
                if tool.name == self.aug_config.force_tool_choice:
                    bind_tools_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
                    break

        # Bind tools to LLM
        return self.runnable_llm.bind_tools(
            resolved_tools,
            **bind_tools_kwargs
        ).with_config(**(self.bind_tools_config or {}))

    def initialize_llm_with_structured_output(self):
        """Add structured output capability.
        
        Returns:
            LLM with structured output
        """
        return self.runnable_llm.with_structured_output(
            self.structured_output_model,
            method="function_calling"
        )

    def initialize_llm_with_output_parser(self):
        """Add output parser.
        
        Returns:
            LLM with output parser
        """
        return self.runnable_llm | self.output_parser

    def apply_custom_runnables(self, runnable: Runnable) -> Runnable:
        """Add custom runnables to the chain
        
        Args:
            runnable: The runnable to extend
            
        Returns:
            Extended runnable
        """
        if not self.custom_runnables:
            return runnable

        for custom_runnable in self.custom_runnables:
            runnable = runnable | custom_runnable

        return runnable

    def create_runnable(self) -> Runnable:
        """Create the complete runnable chain.
        
        Returns:
            A complete runnable chain
        """
        # Start with the LLM
        runnable_chain = self.runnable_llm

        # Add preprocessing if provided
        if self.preprocess:
            runnable_chain = RunnableLambda(self.preprocess) | runnable_chain

        # Add prompt template if provided
        if self.prompt_template:
            # Create or update prompt template if needed
            prompt = self.prompt_template

            # Apply partial variables if any
            if self.partial_variables:
                # Different handling for different template types
                if isinstance(prompt, ChatPromptTemplate):
                    # For chat templates
                    # Create a new prompt with combined partial variables
                    existing_partials = getattr(prompt, "partial_variables", {}) or {}
                    combined_partials = {**existing_partials, **self.partial_variables}

                    if combined_partials:
                        # We need to create a new template with the combined partials
                        prompt = ChatPromptTemplate(
                            messages=prompt.messages,
                            partial_variables=combined_partials
                        )

                elif isinstance(prompt, FewShotPromptTemplate):
                    # For few-shot templates
                    # Create a new prompt with combined partial variables
                    existing_partials = getattr(prompt, "partial_variables", {}) or {}
                    combined_partials = {**existing_partials, **self.partial_variables}

                    if combined_partials:
                        # Create a new template with combined partials
                        prompt = FewShotPromptTemplate(
                            examples=prompt.examples,
                            example_prompt=prompt.example_prompt,
                            prefix=prompt.prefix,
                            suffix=prompt.suffix,
                            input_variables=prompt.input_variables,
                            example_separator=prompt.example_separator,
                            partial_variables=combined_partials
                        )

                else:
                    # For regular templates
                    existing_partials = getattr(prompt, "partial_variables", {}) or {}
                    combined_partials = {**existing_partials, **self.partial_variables}

                    if combined_partials:
                        # Create a new template with combined partials
                        prompt = PromptTemplate(
                            template=prompt.template,
                            input_variables=prompt.input_variables,
                            partial_variables=combined_partials
                        )

            # Update system message if needed
            if self.system_message and isinstance(prompt, ChatPromptTemplate):
                # Check if we need to update the system message in the prompt
                has_system = False
                new_messages = []

                for msg in prompt.messages:
                    if hasattr(msg, "role") and msg.role == "system":
                        # Replace with our system message
                        new_messages.append(SystemMessage(content=self.system_message))
                        has_system = True
                    else:
                        new_messages.append(msg)

                # Add system message if not present
                if not has_system:
                    new_messages.insert(0, SystemMessage(content=self.system_message))

                # Create new prompt with updated messages and partial variables
                prompt = ChatPromptTemplate.from_messages(
                    new_messages,
                    partial_variables=getattr(prompt, "partial_variables", None)
                )

            # Add prompt to chain
            runnable_chain = prompt | runnable_chain
        elif self.system_message:
            # No prompt template but system message is available - create a simple one
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_message),
                MessagesPlaceholder(variable_name="messages")
            ])
            runnable_chain = prompt | runnable_chain
        # Create few-shot template if components are available
        elif self.examples and self.example_prompt and self.prefix and self.suffix and self.input_variables:
            # Create few-shot prompt
            few_shot_prompt = FewShotPromptTemplate(
                examples=self.examples,
                example_prompt=self.example_prompt,
                prefix=self.prefix,
                suffix=self.suffix,
                input_variables=self.input_variables,
                example_separator=self.example_separator,
                partial_variables=self.partial_variables
            )
            runnable_chain = few_shot_prompt | runnable_chain

        # Add custom runnables if provided
        runnable_chain = self.apply_custom_runnables(runnable_chain)

        # Add postprocessing if provided
        if self.postprocess:
            # Custom handling for postprocessing to handle different message types
            def safe_postprocess(input_data):
                """Safely apply postprocessing to various input types."""
                if hasattr(input_data, "content"):
                    # Handle message objects
                    try:
                        processed = self.postprocess(input_data.content)
                        return processed
                    except Exception as e:
                        logger.warning(f"Postprocessing error: {e}")
                        return input_data
                elif isinstance(input_data, str):
                    # Handle strings directly
                    try:
                        return self.postprocess(input_data)
                    except Exception as e:
                        logger.warning(f"Postprocessing error: {e}")
                        return input_data
                else:
                    # Pass through other types
                    try:
                        return self.postprocess(input_data)
                    except Exception as e:
                        logger.warning(f"Postprocessing error: {e}")
                        return input_data

            # Use our safe postprocessing
            runnable_chain = runnable_chain | RunnableLambda(safe_postprocess)

        # Apply runtime configuration
        if self.aug_config.runtime_options:
            runnable_chain = runnable_chain.with_config(**self.aug_config.runtime_options)

        return runnable_chain


# Utility functions

def compose_runnable(aug_llm_config: AugLLMConfig, runnable_config: Optional[RunnableConfig] = None) -> Runnable:
    """Compose a runnable from an AugLLMConfig.
    
    Args:
        aug_llm_config: Configuration for the LLM chain
        runnable_config: Optional runtime configuration
        
    Returns:
        A runnable LLM chain
    """
    try:
        return aug_llm_config.create_runnable(runnable_config)
    except Exception as e:
        logger.error(f"Error composing runnable: {e}")
        raise e

def create_runnables_dict(runnables: List[AugLLMConfig]) -> Dict[str, AugLLMConfig]:
    """Create a dictionary mapping names to runnable configs.
    
    Args:
        runnables: List of AugLLMConfig objects
        
    Returns:
        Dictionary mapping names to configs
    """
    return {runnable.name: runnable for runnable in runnables}

def compose_runnables_from_dict(
    runnables: Dict[str, AugLLMConfig],
    runnable_config: Optional[RunnableConfig] = None
) -> Dict[str, Runnable]:
    """Compose and return a dictionary of runnables from configs.
    
    Args:
        runnables: Dictionary mapping names to configs
        runnable_config: Optional runtime configuration
        
    Returns:
        Dictionary mapping names to runnables
    """
    result = {}
    for key, aug_runnable_config in runnables.items():
        if isinstance(aug_runnable_config, AugLLMConfig):
            result[key] = compose_runnable(aug_runnable_config, runnable_config)
    return result