"""Prompt configuration system with contracts and composition.

This module provides focused prompt management extracted from AugLLMConfig,
reducing complexity while adding explicit contracts and composition patterns.
"""

from typing import Any, Dict, List, Literal, Optional, Set, Union
from pydantic import BaseModel, Field, field_validator
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, PromptTemplate


class PromptVariable(BaseModel):
    """Definition of a prompt variable with validation.
    
    Attributes:
        name: Variable name.
        type: Expected type of the variable.
        required: Whether variable is required.
        default: Default value if not provided.
        description: Human-readable description.
        validation_regex: Optional regex for validation.
        allowed_values: Optional list of allowed values.
    """
    
    name: str = Field(..., description="Variable name")
    type: Literal["string", "number", "boolean", "object", "array"] = Field(
        default="string",
        description="Expected type"
    )
    required: bool = Field(default=True, description="Whether required")
    default: Optional[Any] = Field(default=None, description="Default value")
    description: str = Field(default="", description="Human-readable description")
    validation_regex: Optional[str] = Field(
        default=None,
        description="Validation regex pattern"
    )
    allowed_values: Optional[List[Any]] = Field(
        default=None,
        description="Allowed values"
    )


class PromptContract(BaseModel):
    """Contract defining prompt's requirements and behavior.
    
    Attributes:
        name: Prompt identifier.
        description: What the prompt does.
        variables: Required variables.
        output_format: Expected output format.
        max_tokens: Maximum expected tokens.
        temperature_range: Recommended temperature range.
        examples: Example inputs and outputs.
        constraints: Behavioral constraints.
    """
    
    name: str = Field(..., description="Prompt identifier")
    description: str = Field(..., description="What the prompt does")
    variables: List[PromptVariable] = Field(
        default_factory=list,
        description="Required variables"
    )
    output_format: Literal["text", "json", "markdown", "code", "structured"] = Field(
        default="text",
        description="Expected output format"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum expected tokens"
    )
    temperature_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Recommended temperature range"
    )
    examples: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Example inputs and outputs"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Behavioral constraints"
    )
    
    @field_validator("temperature_range")
    @classmethod
    def validate_temperature_range(cls, v: tuple[float, float]) -> tuple[float, float]:
        """Validate temperature range.
        
        Args:
            v: Temperature range tuple.
            
        Returns:
            Validated temperature range.
            
        Raises:
            ValueError: If range is invalid.
        """
        if len(v) != 2:
            raise ValueError("Temperature range must be a tuple of (min, max)")
        if not (0.0 <= v[0] <= v[1] <= 2.0):
            raise ValueError("Temperature range must be between 0.0 and 2.0")
        return v


class PromptConfig(BaseModel):
    """Focused prompt configuration with contracts.
    
    This replaces scattered prompt management in AugLLMConfig (~150 lines)
    with a focused, contract-based approach.
    
    Attributes:
        prompt_template: Main prompt template.
        system_message: System message for the prompt.
        contracts: Prompt contracts by name.
        partial_variables: Partial variables to inject.
        format_instructions: Format instructions to include.
        composition_mode: How to compose multiple prompts.
        fallback_prompts: Fallback prompts if main fails.
        include_examples: Whether to include examples.
        max_prompt_length: Maximum prompt length in characters.
    
    Examples:
        Basic prompt configuration:
            >>> config = PromptConfig(
            ...     prompt_template=ChatPromptTemplate.from_template("Hello {name}"),
            ...     system_message="You are a helpful assistant"
            ... )
        
        With contracts:
            >>> config = PromptConfig(
            ...     prompt_template=analysis_prompt,
            ...     contracts={
            ...         "analysis": PromptContract(
            ...             name="analysis",
            ...             description="Analyze data",
            ...             variables=[
            ...                 PromptVariable(name="data", type="object")
            ...             ]
            ...         )
            ...     }
            ... )
    """
    
    prompt_template: Optional[BasePromptTemplate] = Field(
        default=None,
        description="Main prompt template"
    )
    system_message: Optional[str] = Field(
        default=None,
        description="System message"
    )
    contracts: Dict[str, PromptContract] = Field(
        default_factory=dict,
        description="Prompt contracts by name"
    )
    partial_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Partial variables to inject"
    )
    format_instructions: Optional[str] = Field(
        default=None,
        description="Format instructions"
    )
    composition_mode: Literal["sequential", "parallel", "hierarchical"] = Field(
        default="sequential",
        description="Prompt composition mode"
    )
    fallback_prompts: List[BasePromptTemplate] = Field(
        default_factory=list,
        description="Fallback prompts"
    )
    include_examples: bool = Field(
        default=True,
        description="Include examples in prompts"
    )
    max_prompt_length: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum prompt length"
    )
    
    def compose_with(
        self,
        other: "PromptConfig",
        mode: Optional[Literal["before", "after", "replace"]] = None
    ) -> "PromptConfig":
        """Compose with another prompt configuration.
        
        Args:
            other: Other prompt configuration.
            mode: Composition mode.
            
        Returns:
            New composed configuration.
        """
        mode = mode or "after"
        
        if mode == "replace":
            return other
        
        # Compose templates
        new_template = self._compose_templates(
            self.prompt_template,
            other.prompt_template,
            mode
        )
        
        # Merge contracts
        new_contracts = {**self.contracts, **other.contracts}
        
        # Merge partial variables
        new_partial = {**self.partial_variables, **other.partial_variables}
        
        return PromptConfig(
            prompt_template=new_template,
            system_message=other.system_message or self.system_message,
            contracts=new_contracts,
            partial_variables=new_partial,
            format_instructions=other.format_instructions or self.format_instructions,
            composition_mode=self.composition_mode,
            fallback_prompts=self.fallback_prompts + other.fallback_prompts,
            include_examples=self.include_examples and other.include_examples,
            max_prompt_length=min(
                self.max_prompt_length or float('inf'),
                other.max_prompt_length or float('inf')
            ) if self.max_prompt_length or other.max_prompt_length else None
        )
    
    def validate_variables(self, provided: Dict[str, Any]) -> Dict[str, str]:
        """Validate provided variables against contracts.
        
        Args:
            provided: Provided variables.
            
        Returns:
            Dictionary of validation errors (empty if valid).
        """
        errors = {}
        
        for contract in self.contracts.values():
            for var in contract.variables:
                if var.required and var.name not in provided:
                    if var.default is None:
                        errors[var.name] = f"Required variable '{var.name}' not provided"
                
                if var.name in provided:
                    value = provided[var.name]
                    
                    # Type validation
                    if not self._validate_type(value, var.type):
                        errors[var.name] = f"Variable '{var.name}' has wrong type"
                    
                    # Allowed values validation
                    if var.allowed_values and value not in var.allowed_values:
                        errors[var.name] = f"Variable '{var.name}' not in allowed values"
        
        return errors
    
    def apply_partial_variables(self) -> "PromptConfig":
        """Apply partial variables to the prompt template.
        
        Returns:
            Self for chaining.
        """
        if self.prompt_template and self.partial_variables:
            if isinstance(self.prompt_template, PromptTemplate):
                self.prompt_template = self.prompt_template.partial(**self.partial_variables)
            elif isinstance(self.prompt_template, ChatPromptTemplate):
                self.prompt_template = self.prompt_template.partial(**self.partial_variables)
        
        return self
    
    def add_example(
        self,
        input_vars: Dict[str, Any],
        expected_output: str
    ) -> "PromptConfig":
        """Add an example to the configuration.
        
        Args:
            input_vars: Example input variables.
            expected_output: Expected output.
            
        Returns:
            Self for chaining.
        """
        for contract in self.contracts.values():
            contract.examples.append({
                "input": input_vars,
                "output": expected_output
            })
        
        return self
    
    def get_required_variables(self) -> Set[str]:
        """Get all required variables from contracts.
        
        Returns:
            Set of required variable names.
        """
        required = set()
        for contract in self.contracts.values():
            for var in contract.variables:
                if var.required:
                    required.add(var.name)
        return required
    
    def _compose_templates(
        self,
        template1: Optional[BasePromptTemplate],
        template2: Optional[BasePromptTemplate],
        mode: str
    ) -> Optional[BasePromptTemplate]:
        """Compose two templates.
        
        Args:
            template1: First template.
            template2: Second template.
            mode: Composition mode.
            
        Returns:
            Composed template or None.
        """
        if not template1:
            return template2
        if not template2:
            return template1
        
        # For ChatPromptTemplate, we can compose messages
        if isinstance(template1, ChatPromptTemplate) and isinstance(template2, ChatPromptTemplate):
            if mode == "before":
                messages = template2.messages + template1.messages
            else:  # after
                messages = template1.messages + template2.messages
            
            return ChatPromptTemplate.from_messages(messages)
        
        # For other templates, convert to string and concatenate
        if mode == "before":
            combined = f"{template2.template}\n\n{template1.template}"
        else:  # after
            combined = f"{template1.template}\n\n{template2.template}"
        
        return PromptTemplate.from_template(combined)
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate value type.
        
        Args:
            value: Value to validate.
            expected_type: Expected type name.
            
        Returns:
            True if valid type.
        """
        type_map = {
            "string": str,
            "number": (int, float),
            "boolean": bool,
            "object": dict,
            "array": list
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Configuration as dictionary.
        """
        return {
            "prompt_template": str(self.prompt_template) if self.prompt_template else None,
            "system_message": self.system_message,
            "contracts": {
                name: contract.model_dump()
                for name, contract in self.contracts.items()
            },
            "partial_variables": self.partial_variables,
            "format_instructions": self.format_instructions,
            "composition_mode": self.composition_mode,
            "include_examples": self.include_examples,
            "max_prompt_length": self.max_prompt_length
        }