"""Tool analysis system leveraging Haive's existing utilities.

This module analyzes tools to determine their properties and capabilities
using the existing utilities in haive.core.common.
"""

import asyncio
import inspect
from collections.abc import Callable
from typing import get_args, get_origin, get_type_hints

from pydantic import BaseModel

from haive.core.utils.interrupt_utils import is_interruptible
from haive.core.utils.tools.tool_schema_generator import (
    extract_input_schema,
    extract_output_schema,
)

from haive.core.engine.tool.types import (
    InterruptibleTool,
    StateAwareTool,
    ToolCapability,
    ToolCategory,
    ToolLike,
    ToolProperties,
    ToolType,
)


class ToolAnalyzer:
    """Analyzes tools to determine their properties and capabilities.

    This analyzer uses existing Haive utilities to detect tool capabilities,
    schemas, and other properties needed for routing and execution.
    """

    def __init__(self):
        """Initialize the tool analyzer."""
        self._cache: dict[str, ToolProperties] = {}

    def analyze(self, tool: ToolLike, force: bool = False) -> ToolProperties:
        """Perform comprehensive tool analysis.

        Args:
            tool: Tool to analyze
            force: Force re-analysis even if cached

        Returns:
            ToolProperties with complete analysis
        """
        # Check cache
        tool_id = self._get_tool_id(tool)
        if not force and tool_id in self._cache:
            return self._cache[tool_id]

        # Get basic info
        name = self._get_tool_name(tool)
        tool_type = self._determine_tool_type(tool)

        # Create properties
        properties = ToolProperties(
            name=name,
            tool_type=tool_type,
            category=self._determine_category(tool),
            description=self._get_description(tool),
        )

        # Analyze capabilities
        self._analyze_capabilities(tool, properties)

        # Analyze state interaction
        self._analyze_state_interaction(tool, properties)

        # Extract schemas using existing utilities
        properties.input_schema = self._safe_extract_schema(extract_input_schema, tool)
        properties.output_schema = self._safe_extract_schema(
            extract_output_schema, tool
        )

        # Check for structured output model
        properties.structured_output_model = self._extract_structured_output_model(tool)
        properties.is_structured_output_model = (
            properties.structured_output_model is not None
        )

        # Analyze performance characteristics
        self._analyze_performance_hints(tool, properties)

        # Re-check structured output capability after schema extraction
        if (
            properties.output_schema or properties.structured_output_model
        ) and ToolCapability.STRUCTURED_OUTPUT not in properties.capabilities:
            properties.capabilities.add(ToolCapability.STRUCTURED_OUTPUT)

        # Also check for validated output if we have a structured output model
        if (
            properties.structured_output_model
            and ToolCapability.VALIDATED_OUTPUT not in properties.capabilities
        ):
            properties.capabilities.add(ToolCapability.VALIDATED_OUTPUT)

        # Cache result
        self._cache[tool_id] = properties

        return properties

    def _determine_tool_type(self, tool: ToolLike) -> ToolType:
        """Determine the tool implementation type.

        This follows the patterns from ToolRouteMixin._analyze_tool.
        """
        # Check for toolkit first
        from langchain_core.tools.base import BaseToolkit

        if isinstance(tool, BaseToolkit):
            return ToolType.TOOLKIT

        # Check if it's a class with BaseTool in MRO
        if hasattr(tool, "__bases__"):
            mro = inspect.getmro(tool)
            if any("BaseTool" in str(base) for base in mro):
                return ToolType.LANGCHAIN_TOOL

        # Check instances
        from langchain_core.tools import BaseTool, StructuredTool

        if isinstance(tool, StructuredTool):
            return ToolType.STRUCTURED_TOOL
        elif isinstance(tool, BaseTool):
            return ToolType.LANGCHAIN_TOOL

        # Check for Pydantic model with __call__
        if isinstance(tool, BaseModel):
            if callable(tool) and callable(tool.__call__):
                return ToolType.PYDANTIC_MODEL
            # Could still be a validation tool
            if "validat" in tool.__class__.__name__.lower():
                return ToolType.VALIDATION_TOOL

        # Check for retriever tool patterns
        if self._is_retriever(tool):
            return ToolType.RETRIEVER_TOOL

        # Default to function
        if callable(tool):
            return ToolType.FUNCTION

        return ToolType.FUNCTION

    def _determine_category(self, tool: ToolLike) -> ToolCategory:
        """Determine tool category from name, description, and type."""
        # Check explicit category marker
        if hasattr(tool, "__tool_category__"):
            return tool.__tool_category__

        name = self._get_tool_name(tool).lower()
        desc = self._get_description(tool).lower()

        # Category detection patterns
        patterns = {
            ToolCategory.RETRIEVAL: [
                "retriev",
                "fetch",
                "search",
                "query",
                "lookup",
                "find",
            ],
            ToolCategory.COMPUTATION: [
                "calculat",
                "comput",
                "math",
                "analyz",
                "process",
                "solve",
            ],
            ToolCategory.COMMUNICATION: [
                "send",
                "email",
                "notify",
                "api",
                "webhook",
                "message",
            ],
            ToolCategory.TRANSFORMATION: [
                "convert",
                "transform",
                "parse",
                "format",
                "encode",
                "decode",
            ],
            ToolCategory.VALIDATION: [
                "validat",
                "check",
                "verify",
                "test",
                "assert",
                "ensure",
            ],
            ToolCategory.MEMORY: [
                "remember",
                "store",
                "save",
                "persist",
                "cache",
                "recall",
            ],
            ToolCategory.SEARCH: [
                "search",
                "find",
                "google",
                "bing",
                "web",
                "internet",
            ],
            ToolCategory.GENERATION: [
                "generat",
                "create",
                "write",
                "compose",
                "build",
                "make",
            ],
            ToolCategory.COORDINATION: [
                "coordinat",
                "orchestrat",
                "manage",
                "control",
                "route",
            ],
        }

        # Check patterns
        for category, keywords in patterns.items():
            if any(kw in name or kw in desc for kw in keywords):
                return category

        return ToolCategory.UNKNOWN

    def _analyze_capabilities(self, tool: ToolLike, properties: ToolProperties) -> None:
        """Analyze and set tool capabilities."""
        capabilities = set()

        # Check interruptibility using existing util
        if is_interruptible(tool) or isinstance(tool, InterruptibleTool):
            capabilities.add(ToolCapability.INTERRUPTIBLE)
            properties.is_interruptible = True

        # Check if tool has explicit interruptible marker
        if hasattr(tool, "__interruptible__") and tool.__interruptible__:
            capabilities.add(ToolCapability.INTERRUPTIBLE)
            properties.is_interruptible = True

        # Check async capability
        if self._is_async(tool):
            capabilities.add(ToolCapability.ASYNC_CAPABLE)
            properties.is_async = True

        # Check structured output
        if properties.output_schema or properties.structured_output_model:
            capabilities.add(ToolCapability.STRUCTURED_OUTPUT)

        # Check if it's a retriever
        if self._is_retriever(tool) or properties.tool_type == ToolType.RETRIEVER_TOOL:
            capabilities.add(ToolCapability.RETRIEVER)

        # Check if it's a validator
        if (
            properties.tool_type == ToolType.VALIDATION_TOOL
            or "validat" in properties.name.lower()
        ):
            capabilities.add(ToolCapability.VALIDATOR)

        # Check for routed tool
        if hasattr(tool, "__tool_route__") or hasattr(tool, "route"):
            capabilities.add(ToolCapability.ROUTED)
            properties.is_routed = True

        # Check for transformer pattern
        if properties.category == ToolCategory.TRANSFORMATION:
            capabilities.add(ToolCapability.TRANSFORMER)

        # Check if tool has explicit capabilities
        if hasattr(tool, "__tool_capabilities__"):
            capabilities.update(tool.__tool_capabilities__)

        properties.capabilities = capabilities

    def _analyze_state_interaction(
        self, tool: ToolLike, properties: ToolProperties
    ) -> None:
        """Analyze how tool interacts with state."""
        # Check if implements StateAwareTool protocol
        if isinstance(tool, StateAwareTool):
            properties.is_state_tool = True
            if tool.reads_state:
                properties.from_state_tool = True
                properties.capabilities.add(ToolCapability.FROM_STATE)
                properties.capabilities.add(ToolCapability.READS_STATE)
            if tool.writes_state:
                properties.to_state_tool = True
                properties.capabilities.add(ToolCapability.TO_STATE)
                properties.capabilities.add(ToolCapability.WRITES_STATE)
            if hasattr(tool, "state_dependencies"):
                properties.state_dependencies = list(tool.state_dependencies)
            properties.capabilities.add(ToolCapability.STATE_AWARE)

        # Also check for InjectedState annotation
        if self._uses_injected_state(tool):
            properties.is_state_tool = True
            properties.from_state_tool = True
            properties.capabilities.add(ToolCapability.INJECTED_STATE)
            properties.capabilities.add(ToolCapability.READS_STATE)
            properties.capabilities.add(ToolCapability.STATE_AWARE)

        # Check parameter names for state interaction
        if callable(tool):
            try:
                sig = inspect.signature(tool)
                param_names = set(sig.parameters.keys())

                # State reading indicators
                state_read_params = {"state", "context", "graph_state", "agent_state"}
                if param_names.intersection(state_read_params):
                    properties.is_state_tool = True
                    properties.from_state_tool = True
                    properties.capabilities.add(ToolCapability.FROM_STATE)
                    properties.capabilities.add(ToolCapability.READS_STATE)
                    properties.capabilities.add(ToolCapability.STATE_AWARE)

                # Check return annotation for state writing
                if sig.return_annotation != sig.empty:
                    return_str = str(sig.return_annotation).lower()
                    if "state" in return_str or "dict" in return_str:
                        properties.to_state_tool = True
                        properties.capabilities.add(ToolCapability.TO_STATE)
                        properties.capabilities.add(ToolCapability.WRITES_STATE)
                        properties.capabilities.add(ToolCapability.STATE_AWARE)
            except:
                pass

        # Check docstring for state interaction hints (only positive indicators)
        doc = self._get_description(tool).lower()
        if "state" in doc:
            # Look for positive state interaction phrases, not just individual words
            positive_read_phrases = [
                "read state",
                "reads state",
                "get state",
                "gets state",
                "access state",
                "accesses state",
                "retrieve state",
                "retrieves state",
                "from state",
                "state data",
                "state information",
            ]
            positive_write_phrases = [
                "write state",
                "writes state",
                "update state",
                "updates state",
                "modify state",
                "modifies state",
                "set state",
                "sets state",
                "to state",
                "store state",
                "save state",
            ]

            # Only trigger if we find positive phrases, not negative ones
            if not any(
                neg in doc for neg in ["without state", "no state", "not state"]
            ):
                if any(phrase in doc for phrase in positive_read_phrases):
                    properties.from_state_tool = True
                    properties.capabilities.add(ToolCapability.FROM_STATE)
                    properties.capabilities.add(ToolCapability.READS_STATE)
                if any(phrase in doc for phrase in positive_write_phrases):
                    properties.to_state_tool = True
                    properties.capabilities.add(ToolCapability.TO_STATE)
                    properties.capabilities.add(ToolCapability.WRITES_STATE)
                if properties.from_state_tool or properties.to_state_tool:
                    properties.is_state_tool = True
                    properties.capabilities.add(ToolCapability.STATE_AWARE)

    def _uses_injected_state(self, tool: ToolLike) -> bool:
        """Check if tool uses InjectedState annotation."""
        # For StructuredTool/BaseTool, check the func attribute
        func_to_check = tool
        if hasattr(tool, "func") and callable(tool.func):
            func_to_check = tool.func
        elif not callable(tool):
            return False

        try:
            # Get type hints including extras for Annotated types
            hints = get_type_hints(func_to_check, include_extras=True)

            for param_name, param_type in hints.items():
                # Skip return type
                if param_name == "return":
                    continue

                # Check for Annotated type
                origin = get_origin(param_type)
                if origin is not None:
                    args = get_args(param_type)
                    # Look for InjectedState in the annotations
                    if any("InjectedState" in str(arg) for arg in args):
                        return True

                # Also check string representation as fallback
                if "InjectedState" in str(param_type):
                    return True
        except:
            pass

        return False

    def _is_async(self, tool: ToolLike) -> bool:
        """Check if tool supports async execution."""
        # Check if the tool itself is async
        if asyncio.iscoroutinefunction(tool):
            return True

        # Check if __call__ method is async
        if callable(tool):
            return asyncio.iscoroutinefunction(tool.__call__)

        # Check for async methods
        if hasattr(tool, "ainvoke") or hasattr(tool, "arun"):
            return True

        return False

    def _is_retriever(self, tool: ToolLike) -> bool:
        """Check if tool is a retriever."""
        # Check name/description patterns
        name = self._get_tool_name(tool).lower()
        desc = self._get_description(tool).lower()

        retriever_patterns = [
            "retriev",
            "fetch",
            "search",
            "query",
            "lookup",
            "find",
            "rag",
        ]
        if any(pattern in name or pattern in desc for pattern in retriever_patterns):
            return True

        # Check for retriever base classes or methods
        if hasattr(tool, "__class__"):
            class_name = tool.__class__.__name__.lower()
            if "retriever" in class_name:
                return True

        # Check for retriever-specific methods
        if hasattr(tool, "get_relevant_documents") or hasattr(tool, "retrieve"):
            return True

        return False

    def _extract_structured_output_model(
        self, tool: ToolLike
    ) -> type[BaseModel] | None:
        """Extract structured output model if present."""
        # Check for explicit structured_output_model attribute
        if hasattr(tool, "structured_output_model"):
            model = tool.structured_output_model
            if isinstance(model, type) and issubclass(model, BaseModel):
                return model

        # Check return type annotation
        if callable(tool):
            try:
                sig = inspect.signature(tool)
                if sig.return_annotation != sig.empty:
                    # Check if return type is a BaseModel subclass
                    return_type = sig.return_annotation
                    if isinstance(return_type, type) and issubclass(
                        return_type, BaseModel
                    ):
                        return return_type
            except:
                pass

        # Check if output schema is a BaseModel
        if hasattr(tool, "output_schema"):
            schema = tool.output_schema
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema

        return None

    def _analyze_performance_hints(
        self, tool: ToolLike, properties: ToolProperties
    ) -> None:
        """Analyze performance characteristics of the tool."""
        # Check for network requirements
        name = properties.name.lower()
        desc = properties.description or ""

        network_indicators = [
            "api",
            "http",
            "request",
            "fetch",
            "download",
            "web",
            "url",
        ]
        if any(
            indicator in name or indicator in desc.lower()
            for indicator in network_indicators
        ):
            properties.requires_network = True

        # Check for explicit performance hints
        if hasattr(tool, "__performance_hints__"):
            hints = tool.__performance_hints__
            if isinstance(hints, dict):
                properties.expected_duration = hints.get("expected_duration")
                properties.requires_network = hints.get(
                    "requires_network", properties.requires_network
                )

    def _safe_extract_schema(
        self, extractor: Callable, tool: ToolLike
    ) -> type[BaseModel] | None:
        """Safely extract schema using provided extractor."""
        try:
            schema = extractor(tool)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema
            return None
        except:
            return None

    def _get_tool_id(self, tool: ToolLike) -> str:
        """Get unique identifier for tool."""
        if hasattr(tool, "id"):
            return str(tool.id)
        elif hasattr(tool, "__name__"):
            return tool.__name__
        else:
            return f"{tool.__class__.__name__}_{id(tool)}"

    def _get_tool_name(self, tool: ToolLike) -> str:
        """Get tool name."""
        if hasattr(tool, "name"):
            return str(tool.name)
        elif hasattr(tool, "__name__"):
            return tool.__name__
        elif hasattr(tool, "__class__"):
            return tool.__class__.__name__
        return "unknown_tool"

    def _get_description(self, tool: ToolLike) -> str:
        """Get tool description."""
        if hasattr(tool, "description") and tool.description:
            return str(tool.description)
        elif hasattr(tool, "__doc__") and tool.__doc__:
            return tool.__doc__.strip()
        return ""
