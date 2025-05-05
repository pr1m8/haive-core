"""
Factory for creating LLM chain runnables from AugLLMConfig.

Provides a clean separation between configuration and runnable creation,
with special focus on flexible message handling and output parsing.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import (
    BaseOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.engine.base import EngineRegistry, EngineType

logger = logging.getLogger(__name__)


class AugLLMFactory:
    """Factory for creating structured LLM runnables from AugLLMConfig with flexible message handling."""

    def __init__(
        self, aug_config: AugLLMConfig, config_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the factory with an AugLLMConfig.

        Args:
            aug_config: Configuration for the LLM chain
            config_params: Optional runtime parameters to override defaults
        """
        self.aug_config = aug_config
        self.config_params = config_params or {}

        # Apply runtime config overrides if any
        self._apply_config_params()

    def _apply_config_params(self):
        """Apply runtime config parameters to the factory instance."""
        # Skip if no config params provided
        if not self.config_params:
            return

        # Apply overrides to augLLMConfig for the factory instance
        for param in [
            "temperature",
            "max_tokens",
            "system_message",
            "tools",
            "parse_raw_output",
            "messages_placeholder_name",
            "force_tool_choice",
            "optional_variables",
            "include_format_instructions",
            "parser_type",
            "pydantic_tools",
            "add_messages_placeholder",
        ]:
            if param in self.config_params:
                setattr(self.aug_config, param, self.config_params[param])

        # Handle partial variables separately (update, don't replace)
        if "partial_variables" in self.config_params:
            self.aug_config.partial_variables.update(
                self.config_params["partial_variables"]
            )

        # Handle prompt modification if system_message was updated
        if "system_message" in self.config_params and self.aug_config.prompt_template:
            self._update_system_message_in_prompt()

        # Update format instructions if parser changed
        if any(
            param in self.config_params
            for param in [
                "structured_output_model",
                "pydantic_tools",
                "include_format_instructions",
            ]
        ):
            self._update_parser_instructions()

        # Apply optional variables if changed
        if "optional_variables" in self.config_params:
            self.aug_config._apply_optional_variables()

    def _update_system_message_in_prompt(self):
        """Update system message in prompt template if changed in config params."""
        if not isinstance(self.aug_config.prompt_template, ChatPromptTemplate):
            return

        new_system_message = self.aug_config.system_message
        if not new_system_message:
            return

        # Build new messages list with updated system message
        new_messages = []
        system_updated = False

        for msg in self.aug_config.prompt_template.messages:
            if hasattr(msg, "role") and msg.role == "system":
                new_messages.append(SystemMessage(content=new_system_message))
                system_updated = True
            else:
                new_messages.append(msg)

        # Add system message at the beginning if none was updated
        if not system_updated:
            new_messages.insert(0, SystemMessage(content=new_system_message))

        # Create new template with updated messages
        partial_vars = getattr(
            self.aug_config.prompt_template, "partial_variables", None
        )
        self.aug_config.prompt_template = ChatPromptTemplate.from_messages(
            new_messages, partial_variables=partial_vars
        )

    def _update_parser_instructions(self):
        """Update parser format instructions if relevant settings changed."""
        # Only update if instructions should be included
        if not self.aug_config.include_format_instructions:
            return

        # Handle Pydantic output parsing
        if (
            self.aug_config.parser_type == "pydantic"
            and self.aug_config.structured_output_model
        ):
            # Create/update parser
            self.aug_config.output_parser = PydanticOutputParser(
                pydantic_object=self.aug_config.structured_output_model
            )

            # Update format instructions
            if hasattr(self.aug_config.output_parser, "get_format_instructions"):
                format_instructions = (
                    self.aug_config.output_parser.get_format_instructions()
                )
                self.aug_config.partial_variables["format_instructions"] = (
                    format_instructions
                )

                # Apply to prompt template if it exists
                self.aug_config._apply_partial_variables()

        # Handle Pydantic tools parsing
        elif (
            self.aug_config.parser_type == "pydantic_tools"
            and self.aug_config.pydantic_tools
        ):
            # Create/update parser
            self.aug_config.output_parser = PydanticToolsParser(
                tools=self.aug_config.pydantic_tools
            )

            # Update format instructions
            if hasattr(self.aug_config.output_parser, "get_format_instructions"):
                format_instructions = (
                    self.aug_config.output_parser.get_format_instructions()
                )
                self.aug_config.partial_variables["format_instructions"] = (
                    format_instructions
                )

                # Apply to prompt template if it exists
                self.aug_config._apply_partial_variables()

    def create_runnable(self) -> Runnable:
        """
        Create the complete runnable chain with proper message handling.

        Returns:
            A complete runnable chain
        """
        # Initialize LLM with any runtime parameters
        llm_params = {}
        if self.aug_config.temperature is not None:
            llm_params["temperature"] = self.aug_config.temperature
        if self.aug_config.max_tokens is not None:
            llm_params["max_tokens"] = self.aug_config.max_tokens

        # Create base LLM
        runnable_llm = self.aug_config.llm_config.instantiate(**llm_params)

        # Make sure we have a valid LLM
        if runnable_llm is None:
            raise ValueError("Failed to instantiate LLM from llm_config")

        # Add tools if specified
        if self.aug_config.tools:
            runnable_llm = self._initialize_llm_with_tools(runnable_llm)

        # Add structured output handling
        runnable_llm = self._configure_structured_output(runnable_llm)

        # Build the complete chain with prompt template and pre/post processing
        runnable_chain = self._build_chain(runnable_llm)

        # Make sure we have a valid chain
        if runnable_chain is None:
            # If we have no prompt template, just use the LLM as the chain
            runnable_chain = runnable_llm

        # Apply runtime config if any
        if self.aug_config.runtime_options:
            runnable_chain = runnable_chain.with_config(
                **self.aug_config.runtime_options
            )

        return runnable_chain

    def _initialize_llm_with_tools(self, llm: Runnable) -> Runnable:
        """Configure LLM with tools based on configuration.

        Args:
            llm: Base LLM runnable

        Returns:
            LLM with tools configured
        """
        tools = self.aug_config.tools

        # Check if list is empty
        if not tools:
            return llm

        # Resolve tool instances if needed
        tool_instances = []
        for tool in tools:
            if isinstance(tool, str):
                # Look up tool by name
                try:
                    from haive.core.engine.tool import ToolRegistry

                    tool_instance = ToolRegistry.get(tool)
                    tool_instances.append(tool_instance)
                except (ImportError, AttributeError):
                    # Fallback - just skip this tool
                    continue
            elif isinstance(tool, type) and issubclass(tool, BaseTool):
                # Instantiate tool class
                kwargs = self.aug_config.tool_kwargs.get(tool.__name__, {})
                tool_instances.append(tool(**kwargs))
            else:
                # Assume it's already a tool instance
                tool_instances.append(tool)

        # Check if we found any valid tools
        if not tool_instances:
            return llm

        # Bind tools to the LLM
        bind_kwargs = self.aug_config.bind_tools_kwargs
        if self.aug_config.force_tool_choice:
            bind_kwargs["tool_choice"] = self.aug_config.force_tool_choice

        # Use bind_tools method if available
        if hasattr(llm, "bind_tools"):
            return llm.bind_tools(tool_instances, **bind_kwargs)

        # Fallback - try with_structured_output for OpenAI compatibility
        return llm.with_tools(tool_instances, **bind_kwargs)

    def _configure_structured_output(self, llm: Runnable) -> Runnable:
        """Configure structured output parsing based on configuration.

        Args:
            llm: Base LLM runnable

        Returns:
            LLM with structured output configuration
        """
        # Handle Pydantic output model with function calling
        if (
            self.aug_config.structured_output_model
            and not self.aug_config.parse_raw_output
            and self.aug_config.parser_type == "pydantic"
        ):

            # Use with_structured_output for best support
            try:
                return llm.with_structured_output(
                    self.aug_config.structured_output_model,
                    method="function_calling",  # Explicitly use function_calling
                )
            except Exception as e:
                # Fallback - chain with output parser
                if self.aug_config.output_parser:
                    return llm | self.aug_config.output_parser
                return llm

        # Handle Pydantic tools
        elif (
            self.aug_config.pydantic_tools
            and self.aug_config.parser_type == "pydantic_tools"
            and not self.aug_config.parse_raw_output
        ):

            # Using with_structured_output doesn't work well for tool parsing
            # Instead, we'll chain with the dedicated parser
            if isinstance(self.aug_config.output_parser, PydanticToolsParser):
                return llm | self.aug_config.output_parser

        # Handle custom output parser
        elif self.aug_config.output_parser and not isinstance(
            self.aug_config.output_parser, PydanticToolsParser
        ):
            return llm | self.aug_config.output_parser

        # If parse_raw_output is True, use StrOutputParser regardless
        elif self.aug_config.parse_raw_output:
            return llm | StrOutputParser()

        # Default - just return the LLM as is
        return llm

    def _build_chain(self, llm: Runnable) -> Runnable:
        """Build the complete chain with prompt template and pre/post processing.

        Args:
            llm: LLM runnable with tools/output handling already configured

        Returns:
            Complete runnable chain
        """
        # If no prompt template, just return the LLM
        if not self.aug_config.prompt_template:
            return llm

        # Create system message-based prompt if needed
        if not self.aug_config.prompt_template and self.aug_config.system_message:
            messages = [SystemMessage(content=self.aug_config.system_message)]

            # Add messages placeholder if needed
            if self.aug_config.add_messages_placeholder:
                messages.append(
                    MessagesPlaceholder(
                        variable_name=self.aug_config.messages_placeholder_name
                    )
                )

            self.aug_config.prompt_template = ChatPromptTemplate.from_messages(messages)

        # If still no prompt template, just return the LLM
        if not self.aug_config.prompt_template:
            return llm

        # Create full chain with prompt
        chain = self.aug_config.prompt_template | llm

        # Add preprocessing if specified
        if self.aug_config.preprocess:
            chain = RunnableLambda(self.aug_config.preprocess) | chain

        # Add postprocessing if specified
        if self.aug_config.postprocess:
            chain = chain | RunnableLambda(self.aug_config.postprocess)

        # Add custom runnables if specified
        if self.aug_config.custom_runnables:
            for runnable in self.aug_config.custom_runnables:
                chain = chain | runnable

        return chain
