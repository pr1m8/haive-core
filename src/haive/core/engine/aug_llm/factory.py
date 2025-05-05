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
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Local imports will be used in actual implementation
# from haive.core.engine.aug_llm.config import AugLLMConfig
# from haive.core.engine.base import EngineRegistry, EngineType

logger = logging.getLogger(__name__)
console = Console()


class AugLLMFactory:
    """Factory for creating structured LLM runnables from AugLLMConfig with flexible message handling."""

    def __init__(self, aug_config: Any, config_params: Optional[Dict[str, Any]] = None):
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

        self._debug_log(
            "AugLLMFactory Initialization",
            {
                "config_name": self.aug_config.name,
                "runtime_overrides": bool(config_params),
                "has_prompt_template": self.aug_config.prompt_template is not None,
                "has_tools": len(self.aug_config.tools) > 0,
                "has_structured_output": self.aug_config.structured_output_model
                is not None,
                "force_messages_optional": self.aug_config.force_messages_optional,
                "messages_in_optional_vars": self.aug_config.messages_placeholder_name
                in self.aug_config.optional_variables,
            },
        )

    def _debug_log(self, title: str, content: Dict[str, Any]):
        """Pretty print debug information."""
        table = Table(title=title, title_justify="left", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")

        for key, value in content.items():
            if value is not None:
                formatted_value = str(value)
                if len(formatted_value) > 100:
                    formatted_value = formatted_value[:97] + "..."
                table.add_row(key, formatted_value)

        console.print(Panel(table, expand=False))

    def _apply_config_params(self):
        """Apply runtime config parameters to the factory instance."""
        # Skip if no config params provided
        if not self.config_params:
            return

        rprint("[blue]Applying runtime config parameters[/blue]")

        # Track what we're overriding
        override_summary = {}

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
            "force_messages_optional",
        ]:
            if param in self.config_params:
                setattr(self.aug_config, param, self.config_params[param])
                override_summary[param] = self.config_params[param]
                rprint(f"[cyan]Overriding {param}: {self.config_params[param]}[/cyan]")

        # Handle partial variables separately (update, don't replace)
        if "partial_variables" in self.config_params:
            self.aug_config.partial_variables.update(
                self.config_params["partial_variables"]
            )
            override_summary["partial_variables"] = "updated"
            rprint("[cyan]Updated partial variables[/cyan]")

        # MODIFIED: Make sure messages is in optional variables
        if (
            self.aug_config.messages_placeholder_name
            not in self.aug_config.optional_variables
        ):
            self.aug_config.optional_variables.append(
                self.aug_config.messages_placeholder_name
            )
            rprint(
                f"[yellow]Added {self.aug_config.messages_placeholder_name} to optional_variables during config param application[/yellow]"
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

        # Apply optional messages placeholder handling if changed
        if any(
            param in self.config_params
            for param in [
                "force_messages_optional",
                "messages_placeholder_name",
                "add_messages_placeholder",
            ]
        ):
            self.aug_config._handle_chat_template_messages_placeholder()

        # Debug summary
        if override_summary:
            self._debug_log("Applied Runtime Overrides", override_summary)

    def _update_system_message_in_prompt(self):
        """Update system message in prompt template if changed in config params."""
        if not isinstance(self.aug_config.prompt_template, ChatPromptTemplate):
            rprint(
                "[yellow]Not a ChatPromptTemplate - skipping system message update[/yellow]"
            )
            return

        new_system_message = self.aug_config.system_message
        if not new_system_message:
            rprint("[yellow]No system message to update[/yellow]")
            return

        rprint("[blue]Updating system message in prompt template[/blue]")

        # Build new messages list with updated system message
        new_messages = []
        system_updated = False

        for msg in self.aug_config.prompt_template.messages:
            if hasattr(msg, "role") and msg.role == "system":
                new_messages.append(SystemMessage(content=new_system_message))
                system_updated = True
                rprint("[green]Replaced existing system message[/green]")
            else:
                new_messages.append(msg)

        # Add system message at the beginning if none was updated
        if not system_updated:
            new_messages.insert(0, SystemMessage(content=new_system_message))
            rprint("[green]Added new system message at beginning[/green]")

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
            rprint("[yellow]Format instructions disabled - skipping update[/yellow]")
            return

        rprint("[blue]Updating parser format instructions[/blue]")

        # Handle Pydantic output parsing
        if (
            self.aug_config.parser_type == "pydantic"
            and self.aug_config.structured_output_model
        ):
            rprint("[cyan]Updating PydanticOutputParser[/cyan]")
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
                rprint("[green]Updated format instructions for Pydantic parser[/green]")

                # Apply to prompt template if it exists
                if self.aug_config.prompt_template:
                    self.aug_config._apply_partial_variables()

        # Handle Pydantic tools parsing
        elif (
            self.aug_config.parser_type == "pydantic_tools"
            and self.aug_config.pydantic_tools
        ):
            rprint("[cyan]Updating PydanticToolsParser[/cyan]")
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
                rprint(
                    "[green]Updated format instructions for Pydantic Tools parser[/green]"
                )

                # Apply to prompt template if it exists
                if self.aug_config.prompt_template:
                    self.aug_config._apply_partial_variables()

    def create_runnable(self) -> Runnable:
        """
        Create the complete runnable chain with proper message handling.

        Returns:
            A complete runnable chain
        """
        rprint("[bold blue]Creating runnable chain[/bold blue]")

        # MODIFIED: Last check to ensure messages are optional
        if (
            self.aug_config.messages_placeholder_name
            not in self.aug_config.optional_variables
        ):
            self.aug_config.optional_variables.append(
                self.aug_config.messages_placeholder_name
            )
            rprint(
                f"[yellow]Added {self.aug_config.messages_placeholder_name} to optional_variables during runnable creation[/yellow]"
            )

        # Force chat templates to have optional messages placeholder
        if isinstance(self.aug_config.prompt_template, ChatPromptTemplate):
            self.aug_config._handle_chat_template_messages_placeholder()
            rprint("[green]Enforced optional messages in chat template[/green]")

        # Initialize LLM with any runtime parameters
        llm_params = {}
        if self.aug_config.temperature is not None:
            llm_params["temperature"] = self.aug_config.temperature
        if self.aug_config.max_tokens is not None:
            llm_params["max_tokens"] = self.aug_config.max_tokens

        # Debug LLM initialization
        self._debug_log(
            "LLM Initialization",
            {
                "model": self.aug_config.llm_config.model,
                "temperature": self.aug_config.temperature,
                "max_tokens": self.aug_config.max_tokens,
                "override_params": llm_params,
            },
        )

        # Create base LLM
        runnable_llm = self.aug_config.llm_config.instantiate(**llm_params)

        # Make sure we have a valid LLM
        if runnable_llm is None:
            error_msg = "Failed to instantiate LLM from llm_config"
            rprint(f"[red]{error_msg}[/red]")
            raise ValueError(error_msg)

        rprint("[green]Successfully instantiated base LLM[/green]")

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
            rprint("[yellow]No prompt template - using raw LLM as chain[/yellow]")

        # Apply runtime config if any
        if self.aug_config.runtime_options:
            runnable_chain = runnable_chain.with_config(
                **self.aug_config.runtime_options
            )
            rprint("[cyan]Applied runtime options to chain[/cyan]")

        rprint("[bold green]Successfully created runnable chain[/bold green]")
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
            rprint("[yellow]No tools to bind - returning LLM unchanged[/yellow]")
            return llm

        rprint(f"[blue]Binding {len(tools)} tools to LLM[/blue]")

        # Resolve tool instances if needed
        tool_instances = []
        for i, tool in enumerate(tools):
            if isinstance(tool, str):
                # Look up tool by name
                try:
                    # The import would be from haive.core.engine.tool import ToolRegistry in real code
                    # This is a placeholder - in actual implementation this would be proper registry lookup
                    tool_instance = {
                        "name": tool,
                        "description": f"Mock tool for {tool}",
                    }
                    tool_instances.append(tool_instance)
                    rprint(f"[green]Resolved tool {i+1}: {tool}[/green]")
                except (ImportError, AttributeError):
                    # Fallback - just skip this tool
                    rprint(f"[red]Failed to resolve tool {i+1}: {tool}[/red]")
                    continue
            elif isinstance(tool, type) and hasattr(tool, "description"):
                # Instantiate tool class
                kwargs = self.aug_config.tool_kwargs.get(
                    getattr(tool, "__name__", "Tool"), {}
                )
                tool_instances.append(tool(**kwargs))
                rprint(
                    f"[green]Instantiated tool {i+1}: {getattr(tool, '__name__', 'Unknown')}[/green]"
                )
            else:
                # Assume it's already a tool instance
                tool_instances.append(tool)
                tool_class_name = (
                    tool.__class__.__name__ if hasattr(tool, "__class__") else "Unknown"
                )
                rprint(f"[green]Using tool instance {i+1}: {tool_class_name}[/green]")

        # Check if we found any valid tools
        if not tool_instances:
            rprint("[yellow]No valid tools found - returning LLM unchanged[/yellow]")
            return llm

        # Bind tools to the LLM
        bind_kwargs = self.aug_config.bind_tools_kwargs.copy()
        if self.aug_config.force_tool_choice:
            bind_kwargs["tool_choice"] = self.aug_config.force_tool_choice
            rprint(
                f"[cyan]Forcing tool choice: {self.aug_config.force_tool_choice}[/cyan]"
            )

        # Use bind_tools method if available
        if hasattr(llm, "bind_tools"):
            rprint("[cyan]Using bind_tools method[/cyan]")
            return llm.bind_tools(tool_instances, **bind_kwargs)

        # Fallback - try with_tools for OpenAI compatibility
        rprint("[yellow]Falling back to with_tools method[/yellow]")
        if hasattr(llm, "with_tools"):
            return llm.with_tools(tool_instances, **bind_kwargs)

        # If no tool binding method available, return original LLM with warning
        rprint("[red]No tool binding method available on LLM[/red]")
        return llm

    def _configure_structured_output(self, llm: Runnable) -> Runnable:
        """Configure structured output parsing based on configuration.

        Args:
            llm: Base LLM runnable

        Returns:
            LLM with structured output configuration
        """
        rprint("[blue]Configuring structured output[/blue]")

        # Handle Pydantic output model with function calling
        if (
            self.aug_config.structured_output_model
            and not self.aug_config.parse_raw_output
            and self.aug_config.parser_type == "pydantic"
        ):

            rprint(
                "[cyan]Using Pydantic structured output with function calling[/cyan]"
            )

            # Use with_structured_output for best support
            try:
                if hasattr(llm, "with_structured_output"):
                    configured_llm = llm.with_structured_output(
                        self.aug_config.structured_output_model,
                        method="function_calling",  # Explicitly use function_calling
                    )
                    rprint("[green]Successfully configured structured output[/green]")
                    return configured_llm
                else:
                    rprint(
                        "[yellow]with_structured_output not available - falling back to chain with parser[/yellow]"
                    )
            except Exception as e:
                rprint(f"[red]Failed to configure structured output: {e}[/red]")
                # Fallback - chain with output parser
                if self.aug_config.output_parser:
                    rprint("[yellow]Falling back to output parser chain[/yellow]")
                    return llm | self.aug_config.output_parser
                return llm

        # Handle Pydantic tools
        elif (
            self.aug_config.pydantic_tools
            and self.aug_config.parser_type == "pydantic_tools"
            and not self.aug_config.parse_raw_output
        ):

            rprint("[cyan]Using Pydantic tools parser[/cyan]")
            # Using with_structured_output doesn't work well for tool parsing
            # Instead, we'll chain with the dedicated parser
            if isinstance(self.aug_config.output_parser, PydanticToolsParser):
                return llm | self.aug_config.output_parser

        # Handle custom output parser
        elif self.aug_config.output_parser and not isinstance(
            self.aug_config.output_parser, PydanticToolsParser
        ):
            rprint(
                f"[cyan]Using custom output parser: {type(self.aug_config.output_parser).__name__}[/cyan]"
            )
            return llm | self.aug_config.output_parser

        # If parse_raw_output is True, use StrOutputParser regardless
        elif self.aug_config.parse_raw_output:
            rprint("[cyan]Using StrOutputParser for raw output[/cyan]")
            return llm | StrOutputParser()

        # Default - just return the LLM as is
        rprint(
            "[yellow]No output parsing configuration - returning LLM unchanged[/yellow]"
        )
        return llm

    def _build_chain(self, llm: Runnable) -> Runnable:
        """Build the complete chain with prompt template and pre/post processing.

        Args:
            llm: LLM runnable with tools/output handling already configured

        Returns:
            Complete runnable chain
        """
        rprint("[blue]Building complete chain[/blue]")

        # If no prompt template, just return the LLM
        if not self.aug_config.prompt_template:
            rprint("[yellow]No prompt template - returning LLM as chain[/yellow]")
            return llm

        # Ensure we have a proper prompt template
        if not self.aug_config.prompt_template and self.aug_config.system_message:
            rprint("[cyan]Creating prompt template from system message[/cyan]")
            messages = [SystemMessage(content=self.aug_config.system_message)]

            # Add messages placeholder if needed
            if self.aug_config.add_messages_placeholder:
                # MODIFIED: Always make messages optional
                is_optional = True
                messages.append(
                    MessagesPlaceholder(
                        variable_name=self.aug_config.messages_placeholder_name,
                        optional=is_optional,
                    )
                )
                rprint(
                    f"[green]Added messages placeholder (optional={is_optional})[/green]"
                )

            self.aug_config.prompt_template = ChatPromptTemplate.from_messages(messages)

        # If still no prompt template, just return the LLM
        if not self.aug_config.prompt_template:
            rprint(
                "[yellow]Still no prompt template - returning LLM unchanged[/yellow]"
            )
            return llm

        # Create full chain with prompt
        chain = self.aug_config.prompt_template | llm
        rprint("[green]Created base chain with prompt template[/green]")

        # Add preprocessing if specified
        if self.aug_config.preprocess:
            chain = RunnableLambda(self.aug_config.preprocess) | chain
            rprint("[cyan]Added preprocessing to chain[/cyan]")

        # Add postprocessing if specified
        if self.aug_config.postprocess:
            chain = chain | RunnableLambda(self.aug_config.postprocess)
            rprint("[cyan]Added postprocessing to chain[/cyan]")

        # Add custom runnables if specified
        if self.aug_config.custom_runnables:
            for i, runnable in enumerate(self.aug_config.custom_runnables):
                chain = chain | runnable
                rprint(f"[cyan]Added custom runnable {i+1}[/cyan]")

        # Debug final chain composition
        self._debug_log(
            "Chain Composition",
            {
                "has_prompt_template": bool(self.aug_config.prompt_template),
                "has_preprocess": bool(self.aug_config.preprocess),
                "has_postprocess": bool(self.aug_config.postprocess),
                "custom_runnables": len(self.aug_config.custom_runnables or []),
                "messages_optional": True,  # Always true with our changes
            },
        )

        return chain
