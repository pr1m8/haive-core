"""Enhanced prompt configuration with full feature parity to AugLLMConfig.

This module provides complete prompt management including few-shot learning,
messages placeholders, format instructions, and dynamic template creation.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.output_parsers import PydanticOutputParser

from haive.core.contracts.prompt_config import PromptContract, PromptVariable


class FewShotConfig(BaseModel):
    """Configuration for few-shot prompting.
    
    Attributes:
        examples: List of example input-output pairs.
        example_prompt: Template for formatting examples.
        prefix: Text before examples.
        suffix: Text after examples.
        example_separator: Separator between examples.
        example_selector: Optional selector for dynamic examples.
    """
    
    examples: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Few-shot examples"
    )
    example_prompt: Optional[Union[PromptTemplate, ChatPromptTemplate]] = Field(
        default=None,
        description="Template for examples"
    )
    prefix: Optional[str] = Field(
        default=None,
        description="Text before examples"
    )
    suffix: Optional[str] = Field(
        default=None,
        description="Text after examples"
    )
    example_separator: str = Field(
        default="\n\n",
        description="Separator between examples"
    )
    example_selector: Optional[Any] = Field(
        default=None,
        description="Dynamic example selector"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class MessagesConfig(BaseModel):
    """Configuration for messages placeholder handling.
    
    Attributes:
        add_messages_placeholder: Whether to add placeholder.
        messages_placeholder_name: Variable name for messages.
        force_messages_optional: Force optional messages.
        uses_messages_field: Whether template uses messages.
        messages_position: Where to insert placeholder.
    """
    
    add_messages_placeholder: bool = Field(
        default=True,
        description="Add messages placeholder"
    )
    messages_placeholder_name: str = Field(
        default="messages",
        description="Variable name for messages"
    )
    force_messages_optional: bool = Field(
        default=False,
        description="Force optional messages"
    )
    uses_messages_field: bool = Field(
        default=False,
        description="Template uses messages field"
    )
    messages_position: Literal["start", "before_human", "end"] = Field(
        default="before_human",
        description="Where to insert placeholder"
    )


class FormatInstructionsConfig(BaseModel):
    """Configuration for format instructions.
    
    Attributes:
        include_format_instructions: Whether to include instructions.
        format_instructions_key: Variable name for instructions.
        use_tool_for_format_instructions: Use tool-based approach.
        format_instructions_text: Cached instruction text.
        auto_generate: Auto-generate from structured model.
    """
    
    include_format_instructions: bool = Field(
        default=True,
        description="Include format instructions"
    )
    format_instructions_key: str = Field(
        default="format_instructions",
        description="Variable name"
    )
    use_tool_for_format_instructions: bool = Field(
        default=False,
        description="Use tool-based approach"
    )
    format_instructions_text: Optional[str] = Field(
        default=None,
        description="Cached instructions"
    )
    auto_generate: bool = Field(
        default=True,
        description="Auto-generate from model"
    )


class TemplateManager(BaseModel):
    """Manages multiple prompt templates.
    
    Attributes:
        stored_templates: Named templates.
        active_template: Currently active template.
        template_history: History of used templates.
        fallback_template: Fallback if main fails.
    """
    
    stored_templates: Dict[str, BasePromptTemplate] = Field(
        default_factory=dict,
        description="Named templates"
    )
    active_template: Optional[str] = Field(
        default=None,
        description="Active template name"
    )
    template_history: List[str] = Field(
        default_factory=list,
        description="Template usage history"
    )
    fallback_template: Optional[BasePromptTemplate] = Field(
        default=None,
        description="Fallback template"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True


class EnhancedPromptConfig(BaseModel):
    """Complete prompt configuration with full AugLLMConfig feature parity.
    
    This provides all prompt management features from AugLLMConfig:
    - Few-shot prompting with examples
    - Messages placeholder management
    - Dynamic template creation
    - Format instructions integration
    - Template storage and switching
    - Input variable management
    
    Attributes:
        prompt_template: Main prompt template.
        system_message: System message for chat models.
        few_shot: Few-shot configuration.
        messages: Messages configuration.
        format_instructions: Format instructions config.
        template_manager: Template management.
        input_variables: Required input variables.
        optional_variables: Optional input variables.
        partial_variables: Partial variables.
        contracts: Prompt contracts.
    """
    
    prompt_template: Optional[BasePromptTemplate] = Field(
        default=None,
        description="Main prompt template"
    )
    system_message: Optional[str] = Field(
        default=None,
        description="System message"
    )
    few_shot: FewShotConfig = Field(
        default_factory=FewShotConfig,
        description="Few-shot config"
    )
    messages: MessagesConfig = Field(
        default_factory=MessagesConfig,
        description="Messages config"
    )
    format_instructions: FormatInstructionsConfig = Field(
        default_factory=FormatInstructionsConfig,
        description="Format instructions"
    )
    template_manager: TemplateManager = Field(
        default_factory=TemplateManager,
        description="Template manager"
    )
    input_variables: List[str] = Field(
        default_factory=list,
        description="Required variables"
    )
    optional_variables: List[str] = Field(
        default_factory=list,
        description="Optional variables"
    )
    partial_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Partial variables"
    )
    contracts: Dict[str, PromptContract] = Field(
        default_factory=dict,
        description="Prompt contracts"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def create_template(self) -> Optional[BasePromptTemplate]:
        """Create appropriate template based on configuration.
        
        Returns:
            Created template or None.
        """
        # Return existing template if available
        if self.prompt_template:
            return self.prompt_template
        
        # Try few-shot template creation
        if self.few_shot.examples and self.few_shot.example_prompt:
            return self._create_few_shot_template()
        
        # Try chat template from system message
        if self.system_message:
            return self._create_chat_from_system()
        
        # Create default with messages placeholder
        if self.messages.add_messages_placeholder:
            return self._create_default_with_placeholder()
        
        # Fallback to simple template
        return PromptTemplate.from_template("{input}")
    
    def _create_few_shot_template(self) -> BasePromptTemplate:
        """Create few-shot template.
        
        Returns:
            FewShotPromptTemplate or FewShotChatMessagePromptTemplate.
        """
        # Check for chat-based few-shot
        if isinstance(self.few_shot.example_prompt, ChatPromptTemplate):
            return FewShotChatMessagePromptTemplate(
                examples=self.few_shot.examples,
                example_prompt=self.few_shot.example_prompt,
                input_variables=self.input_variables or ["input"]
            )
        
        # Standard few-shot with prefix/suffix
        if self.few_shot.prefix and self.few_shot.suffix:
            return FewShotPromptTemplate(
                examples=self.few_shot.examples,
                example_prompt=self.few_shot.example_prompt,
                prefix=self.few_shot.prefix,
                suffix=self.few_shot.suffix,
                example_separator=self.few_shot.example_separator,
                input_variables=self.input_variables or ["input"],
                example_selector=self.few_shot.example_selector
            )
        
        # Fallback to basic few-shot
        return FewShotPromptTemplate(
            examples=self.few_shot.examples,
            example_prompt=self.few_shot.example_prompt,
            input_variables=self.input_variables or ["input"]
        )
    
    def _create_chat_from_system(self) -> ChatPromptTemplate:
        """Create chat template from system message.
        
        Returns:
            ChatPromptTemplate with system message.
        """
        messages = [("system", self.system_message)]
        
        # Add messages placeholder if configured
        if self.messages.add_messages_placeholder:
            placeholder = MessagesPlaceholder(
                variable_name=self.messages.messages_placeholder_name,
                optional=self.messages.force_messages_optional or 
                        self.messages.messages_placeholder_name in self.optional_variables
            )
            
            if self.messages.messages_position == "start":
                messages.insert(0, placeholder)
            elif self.messages.messages_position == "before_human":
                messages.append(placeholder)
        
        # Add human message
        messages.append(("human", "{input}"))
        
        # Add after if position is end
        if (self.messages.add_messages_placeholder and 
            self.messages.messages_position == "end"):
            messages.append(placeholder)
        
        return ChatPromptTemplate.from_messages(messages)
    
    def _create_default_with_placeholder(self) -> ChatPromptTemplate:
        """Create default template with messages placeholder.
        
        Returns:
            Default ChatPromptTemplate.
        """
        messages = []
        
        # Add placeholder
        placeholder = MessagesPlaceholder(
            variable_name=self.messages.messages_placeholder_name,
            optional=True
        )
        messages.append(placeholder)
        
        # Add human message
        messages.append(("human", "{input}"))
        
        return ChatPromptTemplate.from_messages(messages)
    
    def ensure_messages_placeholder(self) -> None:
        """Ensure messages placeholder exists in template."""
        if not self.prompt_template:
            return
        
        if not isinstance(self.prompt_template, ChatPromptTemplate):
            return
        
        messages = list(self.prompt_template.messages)
        has_placeholder = any(
            isinstance(msg, MessagesPlaceholder) and
            getattr(msg, "variable_name", "") == self.messages.messages_placeholder_name
            for msg in messages
        )
        
        if not has_placeholder and self.messages.add_messages_placeholder:
            # Add placeholder
            placeholder = MessagesPlaceholder(
                variable_name=self.messages.messages_placeholder_name,
                optional=self.messages.force_messages_optional
            )
            
            # Insert at appropriate position
            if self.messages.messages_position == "start":
                messages.insert(0, placeholder)
            elif self.messages.messages_position == "before_human":
                # Find last human message
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i][0] == "human":
                        messages.insert(i, placeholder)
                        break
                else:
                    messages.append(placeholder)
            else:
                messages.append(placeholder)
            
            # Update template
            self.prompt_template = ChatPromptTemplate.from_messages(messages)
            self.messages.uses_messages_field = True
    
    def apply_partial_variables(self) -> None:
        """Apply partial variables to template."""
        if self.prompt_template and self.partial_variables:
            self.prompt_template = self.prompt_template.partial(**self.partial_variables)
    
    def add_format_instructions(
        self,
        model: type[BaseModel],
        as_tool: bool = False
    ) -> None:
        """Add format instructions for structured output.
        
        Args:
            model: Pydantic model for output.
            as_tool: Whether to use tool-based approach.
        """
        if not self.format_instructions.include_format_instructions:
            return
        
        # Generate instructions
        if as_tool or self.format_instructions.use_tool_for_format_instructions:
            instructions = f"Use the {model.__name__} tool to structure your response."
        else:
            parser = PydanticOutputParser(pydantic_object=model)
            instructions = parser.get_format_instructions()
        
        # Store instructions
        self.format_instructions.format_instructions_text = instructions
        self.partial_variables[self.format_instructions.format_instructions_key] = instructions
        
        # Apply to template
        self.apply_partial_variables()
    
    def store_template(self, name: str, template: BasePromptTemplate) -> None:
        """Store a template for later use.
        
        Args:
            name: Template name.
            template: Template to store.
        """
        self.template_manager.stored_templates[name] = template
        self.template_manager.template_history.append(name)
    
    def use_template(self, name: str) -> bool:
        """Switch to a stored template.
        
        Args:
            name: Template name to use.
            
        Returns:
            True if switched successfully.
        """
        if name in self.template_manager.stored_templates:
            self.prompt_template = self.template_manager.stored_templates[name]
            self.template_manager.active_template = name
            self.template_manager.template_history.append(name)
            return True
        return False
    
    def remove_template(self, name: Optional[str] = None) -> bool:
        """Remove a stored template.
        
        Args:
            name: Template to remove (current if None).
            
        Returns:
            True if removed successfully.
        """
        if name is None:
            name = self.template_manager.active_template
        
        if name and name in self.template_manager.stored_templates:
            del self.template_manager.stored_templates[name]
            if self.template_manager.active_template == name:
                self.template_manager.active_template = None
            return True
        return False
    
    def list_templates(self) -> List[str]:
        """List all stored template names.
        
        Returns:
            List of template names.
        """
        return list(self.template_manager.stored_templates.keys())
    
    def compute_input_variables(self) -> List[str]:
        """Compute required input variables.
        
        Returns:
            List of required variables.
        """
        if self.prompt_template:
            template_vars = list(self.prompt_template.input_variables)
            # Merge with configured variables
            all_vars = set(template_vars) | set(self.input_variables)
            # Remove partial and optional variables
            required = all_vars - set(self.partial_variables.keys()) - set(self.optional_variables)
            return list(required)
        return self.input_variables
    
    def validate_configuration(self) -> Dict[str, str]:
        """Validate the configuration.
        
        Returns:
            Dictionary of validation errors.
        """
        errors = {}
        
        # Check few-shot configuration
        if self.few_shot.examples and not self.few_shot.example_prompt:
            errors["few_shot"] = "Examples provided but no example_prompt"
        
        # Check format instructions
        if (self.format_instructions.include_format_instructions and
            self.format_instructions.format_instructions_key not in self.partial_variables and
            not self.format_instructions.auto_generate):
            errors["format_instructions"] = "Format instructions enabled but not provided"
        
        # Check input variables
        required_vars = self.compute_input_variables()
        if not required_vars and not self.optional_variables:
            errors["variables"] = "No input variables defined"
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return {
            "has_template": self.prompt_template is not None,
            "system_message": self.system_message,
            "few_shot": {
                "enabled": bool(self.few_shot.examples),
                "example_count": len(self.few_shot.examples)
            },
            "messages": self.messages.model_dump(),
            "format_instructions": {
                "enabled": self.format_instructions.include_format_instructions,
                "has_text": self.format_instructions.format_instructions_text is not None
            },
            "templates": {
                "stored_count": len(self.template_manager.stored_templates),
                "active": self.template_manager.active_template
            },
            "variables": {
                "input": self.input_variables,
                "optional": self.optional_variables,
                "partial": list(self.partial_variables.keys())
            }
        }