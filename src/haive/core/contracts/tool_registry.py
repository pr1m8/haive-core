"""Central tool registry with contract enforcement.

This module provides a centralized registry for tools with capability-based
lookup and contract enforcement, extracted from scattered tool management.
"""

from typing import Any, Dict, List, Optional, Set, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, StructuredTool

from haive.core.contracts.tool_config import ToolContract, ToolCapability


class ToolMetadata(BaseModel):
    """Metadata about a registered tool.
    
    Attributes:
        name: Tool identifier.
        contract: Tool contract.
        tags: Categorization tags.
        version: Tool version.
        registered_at: Registration timestamp.
        usage_count: Number of times used.
        last_used: Last usage timestamp.
        performance_metrics: Performance statistics.
    """
    
    name: str = Field(..., description="Tool identifier")
    contract: ToolContract = Field(..., description="Tool contract")
    tags: Set[str] = Field(default_factory=set, description="Categorization tags")
    version: str = Field(default="1.0.0", description="Tool version")
    registered_at: Optional[str] = Field(default=None, description="Registration timestamp")
    usage_count: int = Field(default=0, description="Usage count")
    last_used: Optional[str] = Field(default=None, description="Last usage timestamp")
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )


class ToolRegistry(BaseModel):
    """Central registry for tools with contract enforcement.
    
    Provides:
    - Tool registration with contracts
    - Capability-based tool discovery
    - Permission validation
    - Usage tracking
    - Performance monitoring
    
    Attributes:
        tools: Registered tools by name.
        metadata: Tool metadata by name.
        capability_index: Tools indexed by capability.
        tag_index: Tools indexed by tag.
        permission_requirements: Global permission requirements.
    
    Examples:
        Basic registration:
            >>> registry = ToolRegistry()
            >>> registry.register(
            ...     name="calculator",
            ...     tool=calculator_tool,
            ...     contract=calculator_contract
            ... )
        
        Capability-based lookup:
            >>> safe_tools = registry.find_by_capability(
            ...     "can_write_state", False
            ... )
    """
    
    tools: Dict[str, Any] = Field(
        default_factory=dict,
        description="Registered tools by name"
    )
    metadata: Dict[str, ToolMetadata] = Field(
        default_factory=dict,
        description="Tool metadata by name"
    )
    capability_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Tools indexed by capability"
    )
    tag_index: Dict[str, Set[str]] = Field(
        default_factory=dict,
        description="Tools indexed by tag"
    )
    permission_requirements: Set[str] = Field(
        default_factory=set,
        description="Global permission requirements"
    )
    
    def register(
        self,
        name: str,
        tool: Any,
        contract: Optional[ToolContract] = None,
        tags: Optional[Set[str]] = None,
        version: str = "1.0.0"
    ) -> "ToolRegistry":
        """Register a tool with contract.
        
        Args:
            name: Tool name.
            tool: Tool instance.
            contract: Tool contract.
            tags: Tool tags.
            version: Tool version.
            
        Returns:
            Self for chaining.
        """
        # Create default contract if not provided
        if not contract:
            contract = ToolContract(
                name=name,
                description=self._get_tool_description(tool)
            )
        
        # Store tool
        self.tools[name] = tool
        
        # Create and store metadata
        from datetime import datetime
        metadata = ToolMetadata(
            name=name,
            contract=contract,
            tags=tags or set(),
            version=version,
            registered_at=datetime.now().isoformat()
        )
        self.metadata[name] = metadata
        
        # Update indices
        self._update_capability_index(name, contract.capabilities)
        self._update_tag_index(name, tags or set())
        
        # Update global permissions
        self.permission_requirements.update(contract.required_permissions)
        
        return self
    
    def unregister(self, name: str) -> "ToolRegistry":
        """Unregister a tool.
        
        Args:
            name: Tool name to unregister.
            
        Returns:
            Self for chaining.
        """
        if name in self.tools:
            # Remove from main registry
            del self.tools[name]
            
            # Get metadata for cleanup
            metadata = self.metadata.get(name)
            if metadata:
                # Remove from capability index
                for capability_key in self._get_capability_keys(metadata.contract.capabilities):
                    if capability_key in self.capability_index:
                        self.capability_index[capability_key].discard(name)
                
                # Remove from tag index
                for tag in metadata.tags:
                    if tag in self.tag_index:
                        self.tag_index[tag].discard(name)
                
                # Remove metadata
                del self.metadata[name]
        
        return self
    
    def get_tool(self, name: str) -> Optional[Any]:
        """Get a tool by name.
        
        Args:
            name: Tool name.
            
        Returns:
            Tool instance or None.
        """
        return self.tools.get(name)
    
    def find_by_capability(
        self,
        capability: str,
        value: bool = True
    ) -> List[Any]:
        """Find tools by capability.
        
        Args:
            capability: Capability field name.
            value: Expected capability value.
            
        Returns:
            List of matching tools.
        """
        capability_key = f"{capability}={value}"
        tool_names = self.capability_index.get(capability_key, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def find_by_tag(self, tag: str) -> List[Any]:
        """Find tools by tag.
        
        Args:
            tag: Tag to search for.
            
        Returns:
            List of matching tools.
        """
        tool_names = self.tag_index.get(tag, set())
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def find_safe_tools(self) -> List[Any]:
        """Find tools without side effects.
        
        Returns:
            List of safe tools.
        """
        safe_tools = []
        for name, metadata in self.metadata.items():
            contract = metadata.contract
            if (not contract.side_effects and 
                not contract.capabilities.can_write_state and
                not contract.capabilities.can_call_external):
                safe_tools.append(self.tools[name])
        return safe_tools
    
    def find_stateful_tools(self) -> List[Any]:
        """Find tools that maintain state.
        
        Returns:
            List of stateful tools.
        """
        return self.find_by_capability("is_stateful", True)
    
    def find_async_tools(self) -> List[Any]:
        """Find async-capable tools.
        
        Returns:
            List of async tools.
        """
        return self.find_by_capability("is_async", True)
    
    def validate_permissions(
        self,
        tool_name: str,
        available_permissions: Set[str]
    ) -> tuple[bool, List[str]]:
        """Validate tool permissions.
        
        Args:
            tool_name: Tool to validate.
            available_permissions: Available permissions.
            
        Returns:
            Tuple of (is_valid, missing_permissions).
        """
        metadata = self.metadata.get(tool_name)
        if not metadata:
            return False, [f"Tool '{tool_name}' not found"]
        
        required = metadata.contract.required_permissions
        missing = required - available_permissions
        
        return len(missing) == 0, list(missing)
    
    def track_usage(self, tool_name: str, execution_time: float = 0.0) -> None:
        """Track tool usage.
        
        Args:
            tool_name: Tool that was used.
            execution_time: Execution time in seconds.
        """
        if tool_name in self.metadata:
            from datetime import datetime
            metadata = self.metadata[tool_name]
            metadata.usage_count += 1
            metadata.last_used = datetime.now().isoformat()
            
            # Update performance metrics
            if execution_time > 0:
                if "avg_execution_time" not in metadata.performance_metrics:
                    metadata.performance_metrics["avg_execution_time"] = execution_time
                else:
                    # Running average
                    current_avg = metadata.performance_metrics["avg_execution_time"]
                    count = metadata.usage_count
                    new_avg = ((current_avg * (count - 1)) + execution_time) / count
                    metadata.performance_metrics["avg_execution_time"] = new_avg
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all tools.
        
        Returns:
            Usage statistics by tool name.
        """
        stats = {}
        for name, metadata in self.metadata.items():
            stats[name] = {
                "usage_count": metadata.usage_count,
                "last_used": metadata.last_used,
                "performance": metadata.performance_metrics
            }
        return stats
    
    def get_capability_summary(self) -> Dict[str, int]:
        """Get summary of tool capabilities.
        
        Returns:
            Count of tools by capability.
        """
        summary = {}
        for capability_key, tool_names in self.capability_index.items():
            summary[capability_key] = len(tool_names)
        return summary
    
    def _update_capability_index(
        self,
        tool_name: str,
        capabilities: ToolCapability
    ) -> None:
        """Update capability index.
        
        Args:
            tool_name: Tool name.
            capabilities: Tool capabilities.
        """
        for capability_key in self._get_capability_keys(capabilities):
            if capability_key not in self.capability_index:
                self.capability_index[capability_key] = set()
            self.capability_index[capability_key].add(tool_name)
    
    def _update_tag_index(self, tool_name: str, tags: Set[str]) -> None:
        """Update tag index.
        
        Args:
            tool_name: Tool name.
            tags: Tool tags.
        """
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(tool_name)
    
    def _get_capability_keys(self, capabilities: ToolCapability) -> List[str]:
        """Get capability index keys.
        
        Args:
            capabilities: Tool capabilities.
            
        Returns:
            List of capability keys.
        """
        keys = []
        for field_name, field_value in capabilities.model_dump().items():
            if isinstance(field_value, bool):
                keys.append(f"{field_name}={field_value}")
            elif field_name == "computational_cost":
                keys.append(f"cost={field_value}")
        return keys
    
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
            Registry as dictionary.
        """
        return {
            "tools": list(self.tools.keys()),
            "metadata": {
                name: metadata.model_dump()
                for name, metadata in self.metadata.items()
            },
            "capability_summary": self.get_capability_summary(),
            "usage_stats": self.get_usage_stats(),
            "permission_requirements": list(self.permission_requirements)
        }