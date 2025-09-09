"""Adapter to integrate extracted components with AugLLMConfig.

This module shows how AugLLMConfig can be refactored to use the new
ToolConfig, PromptConfig, ToolRegistry, and PromptLibrary components.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseLLM

from haive.core.contracts.tool_config import ToolConfig, ToolContract
from haive.core.contracts.prompt_config import PromptConfig
from haive.core.contracts.tool_registry import ToolRegistry
from haive.core.contracts.prompt_library import PromptLibrary


class AugLLMAdapter(BaseModel):
    """Adapter showing how AugLLMConfig can use extracted components.
    
    This demonstrates the refactored architecture where:
    - Tool management is handled by ToolConfig and ToolRegistry
    - Prompt management is handled by PromptConfig and PromptLibrary
    - AugLLMConfig focuses on LLM configuration and orchestration
    
    Attributes:
        llm: The language model instance.
        tool_config: Tool configuration.
        prompt_config: Prompt configuration.
        tool_registry: Shared tool registry.
        prompt_library: Shared prompt library.
        temperature: LLM temperature.
        max_tokens: Maximum tokens.
        model_name: Model identifier.
    """
    
    llm: Optional[BaseLLM] = Field(default=None, description="Language model")
    tool_config: ToolConfig = Field(
        default_factory=ToolConfig,
        description="Tool configuration"
    )
    prompt_config: PromptConfig = Field(
        default_factory=PromptConfig,
        description="Prompt configuration"
    )
    tool_registry: Optional[ToolRegistry] = Field(
        default=None,
        description="Shared tool registry"
    )
    prompt_library: Optional[PromptLibrary] = Field(
        default=None,
        description="Shared prompt library"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Max tokens"
    )
    model_name: str = Field(
        default="gpt-4",
        description="Model name"
    )
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
    
    def with_tools(self, tools: List[Any]) -> "AugLLMAdapter":
        """Configure with tools using ToolConfig.
        
        Args:
            tools: List of tools.
            
        Returns:
            Self for chaining.
        """
        for tool in tools:
            self.tool_config.add_tool(tool)
            
            # Also register in shared registry if available
            if self.tool_registry:
                tool_name = self.tool_config._get_tool_name(tool)
                if tool_name:
                    self.tool_registry.register(tool_name, tool)
        
        return self
    
    def with_prompt_template(self, template: Any) -> "AugLLMAdapter":
        """Configure with prompt template using PromptConfig.
        
        Args:
            template: Prompt template.
            
        Returns:
            Self for chaining.
        """
        self.prompt_config.prompt_template = template
        
        # Also add to library if available
        if self.prompt_library:
            from haive.core.contracts.prompt_config import PromptContract
            contract = PromptContract(
                name="main",
                description="Main prompt template"
            )
            self.prompt_library.add_template(
                "main",
                template,
                contract
            )
        
        return self
    
    def get_safe_tools(self) -> List[Any]:
        """Get tools without side effects.
        
        Returns:
            List of safe tools.
        """
        # Delegate to ToolConfig
        return self.tool_config.get_safe_tools()
    
    def get_tools_by_capability(
        self,
        capability: str,
        value: bool = True
    ) -> List[Any]:
        """Get tools by capability.
        
        Args:
            capability: Capability name.
            value: Capability value.
            
        Returns:
            List of matching tools.
        """
        # Use registry if available, otherwise use config
        if self.tool_registry:
            return self.tool_registry.find_by_capability(capability, value)
        else:
            return self.tool_config.get_tools_by_capability(capability, value)
    
    def validate_permissions(
        self,
        available_permissions: set[str]
    ) -> Dict[str, bool]:
        """Validate all tool permissions.
        
        Args:
            available_permissions: Available permissions.
            
        Returns:
            Validation results by tool.
        """
        results = {}
        
        for tool in self.tool_config.tools:
            tool_name = self.tool_config._get_tool_name(tool)
            if tool_name:
                valid = self.tool_config.validate_permissions(
                    tool,
                    available_permissions
                )
                results[tool_name] = valid
        
        return results
    
    def get_prompt_from_library(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Any:
        """Get prompt from library.
        
        Args:
            name: Prompt name.
            version: Prompt version.
            
        Returns:
            Prompt template or None.
        """
        if self.prompt_library:
            return self.prompt_library.get_template(name, version)
        return None
    
    def compose_prompts(
        self,
        prompt_names: List[str]
    ) -> Any:
        """Compose multiple prompts.
        
        Args:
            prompt_names: Prompts to compose.
            
        Returns:
            Composed prompt.
        """
        if self.prompt_library:
            return self.prompt_library.compose_templates(
                prompt_names,
                "composed",
                mode="sequential"
            )
        return None
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary.
        
        Returns:
            Summary of configuration.
        """
        summary = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": {
                "count": len(self.tool_config.tools),
                "routing": self.tool_config.routing_strategy,
                "force_use": self.tool_config.force_tool_use
            },
            "prompt": {
                "has_template": self.prompt_config.prompt_template is not None,
                "has_system": self.prompt_config.system_message is not None,
                "composition_mode": self.prompt_config.composition_mode
            }
        }
        
        # Add registry stats if available
        if self.tool_registry:
            summary["registry"] = {
                "total_tools": len(self.tool_registry.tools),
                "capabilities": self.tool_registry.get_capability_summary()
            }
        
        # Add library stats if available
        if self.prompt_library:
            summary["library"] = {
                "total_templates": len(self.prompt_library.templates),
                "categories": len(self.prompt_library.categories)
            }
        
        return summary
    
    @classmethod
    def from_aug_llm_config(cls, config: Any) -> "AugLLMAdapter":
        """Create adapter from existing AugLLMConfig.
        
        This shows the migration path from monolithic AugLLMConfig
        to the new component-based architecture.
        
        Args:
            config: Existing AugLLMConfig.
            
        Returns:
            New adapter instance.
        """
        # Extract tool configuration
        tool_config = ToolConfig(
            tools=getattr(config, "tools", []),
            routing_strategy="auto",
            force_tool_use=getattr(config, "force_tool_use", False),
            specific_tool=getattr(config, "force_tool_choice", None),
            tool_choice_mode=getattr(config, "tool_choice_mode", "auto")
        )
        
        # Extract prompt configuration
        prompt_config = PromptConfig(
            prompt_template=getattr(config, "prompt_template", None),
            system_message=getattr(config, "system_message", None),
            partial_variables=getattr(config, "partial_variables", {}),
            format_instructions=getattr(config, "format_instructions", None)
        )
        
        # Create adapter
        return cls(
            llm=getattr(config, "llm", None),
            tool_config=tool_config,
            prompt_config=prompt_config,
            temperature=getattr(config, "temperature", 0.7),
            max_tokens=getattr(config, "max_tokens", None),
            model_name=getattr(config, "model", "gpt-4")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Configuration as dictionary.
        """
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tool_config": self.tool_config.to_dict(),
            "prompt_config": self.prompt_config.to_dict(),
            "summary": self.get_configuration_summary()
        }