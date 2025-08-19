"""Tool factory functions for creating specialized tools.

This module provides factory functions for creating different types of tools
including retriever tools, structured output tools, and validation tools.
"""

from collections.abc import Callable
from typing import Any, Literal

from langchain_core.prompts import BasePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field

from haive.core.engine.tool.types import ToolCapability, ToolCategory, ToolType


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: BasePromptTemplate | None = None,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> StructuredTool:
    """Create a tool to do retrieval of documents.

    This wraps the LangChain create_retriever_tool and adds proper
    tool capabilities metadata.

    Args:
        retriever: The retriever to use for document retrieval
        name: The name for the tool
        description: The description for the tool
        document_prompt: Optional prompt to format documents
        document_separator: Separator between documents
        response_format: Format for tool response

    Returns:
        A StructuredTool configured for retrieval
    """
    from langchain_core.tools import create_retriever_tool as lc_create_retriever_tool

    # Create the base retriever tool
    retriever_tool = lc_create_retriever_tool(
        retriever=retriever,
        name=name,
        description=description,
        document_prompt=document_prompt,
        document_separator=document_separator,
        response_format=response_format,
    )

    # Add tool metadata for our type system
    retriever_tool.__tool_type__ = ToolType.RETRIEVER_TOOL
    retriever_tool.__tool_category__ = ToolCategory.RETRIEVAL
    retriever_tool.__tool_capabilities__ = {
        ToolCapability.RETRIEVER,
        ToolCapability.READS_STATE,  # Retriever tools typically read query state
        ToolCapability.STRUCTURED_OUTPUT,  # Returns structured results
    }

    return retriever_tool


def create_structured_output_tool(
    func: Callable[..., Any],
    name: str,
    description: str,
    output_model: type[BaseModel],
    *,
    infer_schema: bool = True,
) -> StructuredTool:
    """Create a tool that produces structured output.

    This creates a tool from a function that guarantees structured output
    according to the provided Pydantic model.

    Args:
        func: The function to wrap as a tool
        name: Tool name
        description: Tool description
        output_model: Pydantic model for output validation
        infer_schema: Whether to infer input schema from function

    Returns:
        A StructuredTool with structured output guarantees
    """

    # Create wrapper that validates output
    def structured_wrapper(*args, **kwargs):
        """Structured Wrapper."""
        result = func(*args, **kwargs)

        # Validate output against model
        if isinstance(result, output_model):
            return result
        elif isinstance(result, dict):
            return output_model(**result)
        else:
            # Try to construct from result
            return output_model(result=result)

    # Copy function metadata
    structured_wrapper.__name__ = func.__name__
    structured_wrapper.__doc__ = func.__doc__

    # Create tool with proper schema
    if infer_schema:
        structured_tool = tool(
            name=name,
            description=description,
            return_direct=False,
        )(structured_wrapper)
    else:
        structured_tool = StructuredTool(
            name=name,
            description=description,
            func=structured_wrapper,
        )

    # Add metadata
    structured_tool.__tool_type__ = ToolType.STRUCTURED_TOOL
    structured_tool.__tool_capabilities__ = {
        ToolCapability.STRUCTURED_OUTPUT,
        ToolCapability.VALIDATED_OUTPUT,
    }
    structured_tool.structured_output_model = output_model

    return structured_tool


def create_validation_tool(
    validator_func: Callable[[Any], bool | tuple[bool, str]],
    name: str,
    description: str,
    *,
    input_model: type[BaseModel] | None = None,
    error_on_invalid: bool = False,
) -> StructuredTool:
    """Create a tool for validation.

    This creates a tool that validates input according to custom logic
    and returns validation results.

    Args:
        validator_func: Function that validates input, returns bool or (bool, message)
        name: Tool name
        description: Tool description
        input_model: Optional Pydantic model for input validation
        error_on_invalid: Whether to raise exception on invalid input

    Returns:
        A StructuredTool configured for validation
    """

    class ValidationResult(BaseModel):
        """Result of validation."""

        is_valid: bool = Field(description="Whether input is valid")
        message: str = Field(default="", description="Validation message")
        input_data: Any = Field(description="The input that was validated")

    def validation_wrapper(data: Any) -> ValidationResult:
        """Wrapper that returns structured validation results."""
        try:
            # Validate against input model if provided
            if input_model:
                if isinstance(data, dict):
                    data = input_model(**data)
                elif not isinstance(data, input_model):
                    return ValidationResult(
                        is_valid=False,
                        message=f"Input must be of type {input_model.__name__}",
                        input_data=data,
                    )

            # Run validation
            result = validator_func(data)

            # Handle different return types
            if isinstance(result, bool):
                is_valid = result
                message = "Valid" if is_valid else "Invalid"
            elif isinstance(result, tuple) and len(result) == 2:
                is_valid, message = result
            else:
                is_valid = bool(result)
                message = str(result)

            # Handle invalid case
            if not is_valid and error_on_invalid:
                raise ValueError(f"Validation failed: {message}")

            return ValidationResult(is_valid=is_valid, message=message, input_data=data)

        except Exception as e:
            if error_on_invalid:
                raise
            return ValidationResult(
                is_valid=False, message=f"Validation error: {e!s}", input_data=data
            )

    # Create the tool
    if input_model:
        validation_tool = StructuredTool(
            name=name,
            description=description,
            func=validation_wrapper,
            args_schema=input_model,
        )
    else:
        validation_tool = tool(
            name=name,
            description=description,
        )(validation_wrapper)

    # Add metadata
    validation_tool.__tool_type__ = ToolType.VALIDATION_TOOL
    validation_tool.__tool_category__ = ToolCategory.VALIDATION
    validation_tool.__tool_capabilities__ = {
        ToolCapability.VALIDATOR,
        ToolCapability.STRUCTURED_OUTPUT,
        ToolCapability.VALIDATED_OUTPUT,
    }
    validation_tool.structured_output_model = ValidationResult

    return validation_tool


def create_state_tool(
    func: Callable[..., Any] | StructuredTool | BaseTool,
    name: str | None = None,
    description: str | None = None,
    *,
    reads_state: bool = False,
    writes_state: bool = False,
    state_keys: list[str] | None = None,
) -> StructuredTool:
    """Create a tool that interacts with state.

    This creates a tool with explicit state interaction declarations.
    Can wrap existing tools or create new ones.

    Args:
        func: The function or existing tool to wrap
        name: Tool name (uses existing if wrapping)
        description: Tool description (uses existing if wrapping)
        reads_state: Whether tool reads from state
        writes_state: Whether tool writes to state
        state_keys: Specific state keys the tool interacts with

    Returns:
        A StructuredTool with state interaction metadata
    """
    # Handle wrapping existing tools
    if isinstance(func, (StructuredTool, BaseTool)):
        state_tool = func
        if name is None:
            name = func.name
        if description is None:
            description = func.description
    else:
        # Create new tool from function
        state_tool = tool(name=name, description=description)(func)

    # Add state metadata
    capabilities = set()

    if reads_state:
        capabilities.add(ToolCapability.READS_STATE)
        capabilities.add(ToolCapability.FROM_STATE)

    if writes_state:
        capabilities.add(ToolCapability.WRITES_STATE)
        capabilities.add(ToolCapability.TO_STATE)

    if reads_state or writes_state:
        capabilities.add(ToolCapability.STATE_AWARE)

    state_tool.__tool_capabilities__ = capabilities
    state_tool.__state_interaction__ = {
        "reads": reads_state,
        "writes": writes_state,
        "keys": state_keys or [],
    }

    # Implement StateAwareTool protocol
    state_tool.reads_state = reads_state
    state_tool.writes_state = writes_state
    state_tool.state_dependencies = state_keys or []

    return state_tool


def create_interruptible_tool(
    func: Callable[..., Any] | StructuredTool | BaseTool,
    name: str | None = None,
    description: str | None = None,
    *,
    interrupt_message: str = "Tool execution interrupted",
) -> StructuredTool:
    """Create a tool that can be interrupted.

    Can wrap existing tools to add interruption capability.

    Args:
        func: The function or existing tool to wrap
        name: Tool name (uses existing if wrapping)
        description: Tool description (uses existing if wrapping)
        interrupt_message: Message to show when interrupted

    Returns:
        A StructuredTool with interruption support
    """
    # Handle wrapping existing tools
    if isinstance(func, (StructuredTool, BaseTool)):
        interruptible = func
        if name is None:
            name = func.name
        if description is None:
            description = func.description
    else:
        # Create new tool from function
        interruptible = tool(name=name, description=description)(func)

    # Add interruption metadata
    interruptible.__tool_capabilities__ = {ToolCapability.INTERRUPTIBLE}
    interruptible.__interruptible__ = True
    interruptible.__interrupt_message__ = interrupt_message

    # Add interrupt method
    def interrupt():
        """Interrupt."""
        raise InterruptedError(interrupt_message)

    interruptible.interrupt = interrupt
    interruptible.is_interruptible = True

    return interruptible


# Note: The augment_tool functionality has been moved to ToolEngine as a class method
# This allows for better integration with the engine's tool management capabilities
