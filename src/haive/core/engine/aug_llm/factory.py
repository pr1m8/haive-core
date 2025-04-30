"""
Factory for creating LLM chain runnables from AugLLMConfig.

Provides a clean separation between configuration and runnable creation.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from haive.core.engine.aug_llm.config import AugLLMConfig
from haive.core.engine.base import EngineRegistry, EngineType

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate, 
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, chain
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class AugLLMFactory:
    """Factory for creating structured LLM runnables from AugLLMConfig."""

    def __init__(self, aug_config: AugLLMConfig, config_params: Optional[Dict[str, Any]] = None):
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
        if "temperature" in self.config_params:
            self.aug_config.temperature = self.config_params["temperature"]
            
        if "max_tokens" in self.config_params:
            self.aug_config.max_tokens = self.config_params["max_tokens"]
            
        if "system_message" in self.config_params:
            self.aug_config.system_message = self.config_params["system_message"]
            
        if "tools" in self.config_params:
            self.aug_config.tools = self.config_params["tools"]
            
        if "partial_variables" in self.config_params:
            # Update, don't replace, to preserve existing partial variables
            self.aug_config.partial_variables.update(self.config_params["partial_variables"])

    def create_runnable(self) -> Runnable:
        """
        Create the complete runnable chain.
        
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
        
        # Add tools if specified
        if self.aug_config.tools:
            runnable_llm = self._initialize_llm_with_tools(runnable_llm)
            
        # Add structured output if specified
        if self.aug_config.structured_output_model:
            runnable_llm = runnable_llm.with_structured_output(
                self.aug_config.structured_output_model,
                method="function_calling"  # Explicitly use function_calling to avoid warnings
            )
            
        # Add output parser if specified
        if self.aug_config.output_parser:
            runnable_llm = runnable_llm | self.aug_config.output_parser
            
        # Build the complete chain
        runnable_chain = self._build_chain(runnable_llm)
        
        # Apply runtime config if any
        if self.aug_config.runtime_options:
            runnable_chain = runnable_chain.with_config(**self.aug_config.runtime_options)
            
        return runnable_chain
        
    def _initialize_llm_with_tools(self, llm: Runnable) -> Runnable:
        """
        Bind tools to the LLM.
        
        Args:
            llm: The base LLM runnable
            
        Returns:
            LLM with tools bound
        """
        # Resolve tools
        resolved_tools = self._resolve_tools()
        
        # No tools to add
        if not resolved_tools:
            return llm
            
        # Handle force_tool_choice if specified
        bind_tools_kwargs = dict(self.aug_config.bind_tools_kwargs)
        if self.aug_config.force_tool_choice:
            for tool in resolved_tools:
                if tool.name == self.aug_config.force_tool_choice:
                    bind_tools_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
                    break
                    
        # Bind tools to LLM
        llm_with_tools = llm.bind_tools(
            resolved_tools,
            **bind_tools_kwargs
        )
        
        # Apply any bind_tools_config if specified
        if self.aug_config.bind_tools_config:
            llm_with_tools = llm_with_tools.with_config(**(self.aug_config.bind_tools_config or {}))
            
        return llm_with_tools
        
    def _resolve_tools(self) -> List[BaseTool]:
        """
        Resolve tool references to actual tool objects.
        
        Returns:
            List of instantiated tool objects
        """
        import inspect
        from haive.core.engine.base import EngineRegistry, EngineType
        
        resolved_tools = []
        for tool in self.aug_config.tools:
            if isinstance(tool, str):
                # Lookup tool from registry
                tool_engine = EngineRegistry.get_instance().get(EngineType.TOOL, tool)
                if tool_engine:
                    resolved_tools.append(tool_engine.instantiate())
                else:
                    logger.warning(f"Tool not found in registry: {tool}")
            elif inspect.isclass(tool) and issubclass(tool, BaseTool):
                # Instantiate class
                tool_name = tool.__name__
                tool_kwargs = self.aug_config.tool_kwargs.get(tool_name, {})
                resolved_tools.append(tool(**tool_kwargs) if tool_kwargs else tool())
            elif isinstance(tool, BaseTool):
                # Already an instance
                resolved_tools.append(tool)
            else:
                logger.warning(f"Unsupported tool type: {type(tool)}")
                
        return resolved_tools
        
    def _build_chain(self, llm: Runnable) -> Runnable:
        """
        Build the complete chain with prompt, preprocessing, postprocessing, etc.
        
        Args:
            llm: The configured LLM runnable
            
        Returns:
            Complete chain with all components
        """
        chain_components = []
        
        # Add preprocessing if provided
        if self.aug_config.preprocess:
            chain_components.append(RunnableLambda(self.aug_config.preprocess))
            
        # Add prompt template if provided
        if self.aug_config.prompt_template:
            # Handle different prompt template types
            prompt = self._prepare_prompt_template()
            chain_components.append(prompt)
            
        # Add LLM
        chain_components.append(llm)
        
        # Add custom runnables if provided
        if self.aug_config.custom_runnables:
            for custom_runnable in self.aug_config.custom_runnables:
                chain_components.append(custom_runnable)
                
        # Add postprocessing if provided
        if self.aug_config.postprocess:
            chain_components.append(RunnableLambda(self._create_safe_postprocessor()))
            
        # Build the chain
        if len(chain_components) == 1:
            # Just one component
            return chain_components[0]
        else:
            # Manually chain components using the | operator
            result = chain_components[0]
            for component in chain_components[1:]:
                result = result | component
            return result
            
    def _prepare_prompt_template(self) -> BasePromptTemplate:
        """
        Prepare and potentially update the prompt template.
        
        Returns:
            Updated prompt template
        """
        prompt = self.aug_config.prompt_template
        
        # Update system message if needed
        if self.aug_config.system_message and isinstance(prompt, ChatPromptTemplate):
            # Check if we need to update the system message
            has_system = False
            new_messages = []
            
            for msg in prompt.messages:
                if hasattr(msg, "role") and msg.role == "system":
                    # Replace with our system message
                    new_messages.append(SystemMessagePromptTemplate.from_template(self.aug_config.system_message))
                    has_system = True
                else:
                    new_messages.append(msg)
                    
            # Add system message if not present
            if not has_system:
                new_messages.insert(0, SystemMessagePromptTemplate.from_template(self.aug_config.system_message))
                
            # Create updated prompt
            prompt = ChatPromptTemplate.from_messages(
                new_messages,
                partial_variables=getattr(prompt, "partial_variables", None)
            )
            
        return prompt
        
    def _create_safe_postprocessor(self) -> Callable[[Any], Any]:
        """
        Create a safe postprocessing function that handles different input types.
        
        Returns:
            Safe postprocessing function
        """
        def safe_postprocess(input_data):
            """Safely apply postprocessing to various input types."""
            try:
                if hasattr(input_data, "content"):
                    # Handle message objects
                    processed = self.aug_config.postprocess(input_data.content)
                    return processed
                else:
                    # Handle other types directly
                    return self.aug_config.postprocess(input_data)
            except Exception as e:
                logger.warning(f"Postprocessing error: {e}")
                return input_data
                
        return safe_postprocess