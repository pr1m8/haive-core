"""Tool configuration system with contracts and capabilities.

This module provides a focused tool management system extracted from AugLLMConfig,
reducing complexity while adding explicit contracts and capability-based routing.
"""

from typing import Any, Dict, List, Literal, Optional, Set, Union, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool


class ToolCapability(BaseModel):
    """Defines what a tool can do at runtime.
    
    Attributes:
        can_read_state: Whether tool can read from state.
        can_write_state: Whether tool can write to state.
        can_call_external: Whether tool makes external calls.
        is_stateful: Whether tool maintains internal state.
        is_async: Whether tool supports async execution.
        requires_confirmation: Whether tool needs user confirmation.
        computational_cost: Relative cost of tool execution.
    """
    
    can_read_state: bool = Field(default=False, description="Can read from state")
    can_write_state: bool = Field(default=False, description="Can write to state")
    can_call_external: bool = Field(default=False, description="Makes external calls")
    is_stateful: bool = Field(default=False, description="Maintains internal state")
    is_async: bool = Field(default=False, description="Supports async execution")
    requires_confirmation: bool = Field(default=False, description="Needs user confirmation")
    computational_cost: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Relative computational cost"
    )


class ToolContract(BaseModel):
    """Contract defining tool's behavior and requirements.
    
    Attributes:
        name: Tool identifier.
        description: Human-readable description.
        capabilities: What the tool can do.
        input_schema: Expected input structure.
        output_schema: Expected output structure.
        side_effects: List of potential side effects.
        required_permissions: Permissions needed to execute.
    """
    
    name: str = Field(..., description="Tool identifier")
    description: str = Field(..., description="Human-readable description")
    capabilities: ToolCapability = Field(
        default_factory=ToolCapability,
        description="Tool capabilities"
    )
    input_schema: Optional[type[BaseModel]] = Field(
        default=None,
        description="Expected input structure"
    )
    output_schema: Optional[type[BaseModel]] = Field(
        default=None,
        description="Expected output structure"
    )
    side_effects: List[str] = Field(
        default_factory=list,
        description="Potential side effects"
    )
    required_permissions: Set[str] = Field(
        default_factory=set,
        description="Required permissions"
    )


class ToolConfig(BaseModel):
    """Focused tool configuration with contracts.
    
    This replaces the scattered tool management in AugLLMConfig (~266 lines)
    with a focused, contract-based approach.
    
    Attributes:
        tools: List of tools to configure.
        contracts: Tool contracts by name.
        routing_strategy: How to route tool calls.
        force_tool_use: Whether to force tool usage.
        specific_tool: Force specific tool selection.
        tool_choice_mode: Tool selection mode.
        allow_parallel: Whether to allow parallel tool execution.
        max_retries: Maximum retry attempts for failed tools.
        timeout_seconds: Tool execution timeout.
    
    Examples:
        Basic tool configuration:
            >>> config = ToolConfig(
            ...     tools=[calculator, web_search],
            ...     routing_strategy="capability"
            ... )
        
        With contracts:
            >>> config = ToolConfig(
            ...     tools=[data_processor],
            ...     contracts={
            ...         "data_processor": ToolContract(
            ...             name="data_processor",
            ...             description="Process data",
            ...             capabilities=ToolCapability(
            ...                 can_write_state=True,
            ...                 computational_cost="high"
            ...             )
            ...         )
            ...     }
            ... )
    """
    
    tools: List[Union[BaseTool, StructuredTool, type[BaseModel], Callable, str]] = Field(
        default_factory=list,
        description="List of tools to configure"
    )
    contracts: Dict[str, ToolContract] = Field(
        default_factory=dict,
        description="Tool contracts by name"
    )
    routing_strategy: Literal["auto", "capability", "priority", "manual"] = Field(
        default="auto",
        description="Tool routing strategy"
    )
    force_tool_use: bool = Field(
        default=False,
        description="Force tool usage"
    )
    specific_tool: Optional[str] = Field(
        default=None,
        description="Force specific tool"
    )
    tool_choice_mode: Literal["auto", "required", "none"] = Field(
        default="auto",
        description="Tool selection mode"
    )
    allow_parallel: bool = Field(
        default=False,
        description="Allow parallel tool execution"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    timeout_seconds: Optional[float] = Field(
        default=30.0,
        gt=0,
        description="Tool execution timeout"
    )
    
    def add_tool(
        self,
        tool: Any,
        contract: Optional[ToolContract] = None
    ) -> "ToolConfig":
        """Add a tool with optional contract.
        
        Args:
            tool: Tool to add.
            contract: Optional tool contract.
            
        Returns:
            Self for chaining.
        """
        if tool not in self.tools:
            self.tools.append(tool)
            
            # Get the tool name for consistent indexing
            tool_name = self._get_tool_name(tool)
            
            if contract:
                # Use the tool's actual name for indexing, not the contract name
                if tool_name:
                    self.contracts[tool_name] = contract
                else:
                    self.contracts[contract.name] = contract
            else:
                # Auto-generate basic contract
                if tool_name and tool_name not in self.contracts:
                    self.contracts[tool_name] = ToolContract(
                        name=tool_name,
                        description=self._get_tool_description(tool)
                    )
        
        return self
    
    def remove_tool(self, tool: Any) -> "ToolConfig":
        """Remove a tool and its contract.
        
        Args:
            tool: Tool to remove.
            
        Returns:
            Self for chaining.
        """
        if tool in self.tools:
            self.tools.remove(tool)
            tool_name = self._get_tool_name(tool)
            if tool_name in self.contracts:
                del self.contracts[tool_name]
        
        return self
    
    def get_tools_by_capability(
        self,
        capability: str,
        value: bool = True
    ) -> List[Any]:
        """Get tools matching a capability.
        
        Args:
            capability: Capability field name.
            value: Expected capability value.
            
        Returns:
            List of matching tools.
        """
        matching = []
        for tool in self.tools:
            tool_name = self._get_tool_name(tool)
            if tool_name in self.contracts:
                contract = self.contracts[tool_name]
                if getattr(contract.capabilities, capability, None) == value:
                    matching.append(tool)
        return matching
    
    def validate_permissions(
        self,
        tool: Any,
        available_permissions: Set[str]
    ) -> bool:
        """Check if tool has required permissions.
        
        Args:
            tool: Tool to validate.
            available_permissions: Available permissions.
            
        Returns:
            True if permissions are satisfied.
        """
        tool_name = self._get_tool_name(tool)
        if tool_name not in self.contracts:
            return True  # No contract means no restrictions
        
        contract = self.contracts[tool_name]
        return contract.required_permissions.issubset(available_permissions)
    
    def get_safe_tools(self) -> List[Any]:
        """Get tools that don't have side effects.
        
        Returns:
            List of safe tools.
        """
        safe = []
        for tool in self.tools:
            tool_name = self._get_tool_name(tool)
            if tool_name in self.contracts:
                contract = self.contracts[tool_name]
                if not contract.side_effects and not contract.capabilities.can_write_state:
                    safe.append(tool)
            else:
                # No contract means we can't guarantee safety
                continue
        return safe
    
    def _get_tool_name(self, tool: Any) -> Optional[str]:
        """Extract tool name from various tool types.
        
        Args:
            tool: Tool to get name from.
            
        Returns:
            Tool name or None.
        """
        if isinstance(tool, str):
            return tool
        elif hasattr(tool, "name"):
            return tool.name
        elif hasattr(tool, "__name__"):
            return tool.__name__
        elif isinstance(tool, type):
            return tool.__name__
        return None
    
    def _get_tool_description(self, tool: Any) -> str:
        """Extract tool description.
        
        Args:
            tool: Tool to get description from.
            
        Returns:
            Tool description or empty string.
        """
        if hasattr(tool, "description"):
            return tool.description
        elif hasattr(tool, "__doc__") and tool.__doc__:
            return tool.__doc__.split("\n")[0].strip()
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.
        
        Returns:
            Configuration as dictionary.
        """
        return {
            "tools": [self._get_tool_name(t) for t in self.tools],
            "contracts": {
                name: contract.model_dump()
                for name, contract in self.contracts.items()
            },
            "routing_strategy": self.routing_strategy,
            "force_tool_use": self.force_tool_use,
            "specific_tool": self.specific_tool,
            "tool_choice_mode": self.tool_choice_mode,
            "allow_parallel": self.allow_parallel,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds
        }