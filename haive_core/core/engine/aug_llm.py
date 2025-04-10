from typing import Optional, List, Type, Dict, Any, Union, Callable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from src.haive.core.models.llm.base import LLMConfig, AzureLLMConfig
import inspect
from pydantic import BaseModel, Field, field_validator
from src.haive.core.engine.base import Engine, EngineType, EngineRegistry

class AugLLMConfig(Engine):
    """Configuration for creating a structured runnable LLM pipeline."""
    engine_type: EngineType = Field(default=EngineType.LLM, description="The type of engine")
    
    # Use the existing LLMConfig system
    llm_config: Union[LLMConfig, Dict[str, Any]] = Field(
        default_factory=lambda: AzureLLMConfig(model="gpt-4o"),
        description="LLM provider configuration"
    )
    
    prompt_template: Optional[BasePromptTemplate] = None
    tools: List[Union[BaseTool, StructuredTool, Type[BaseTool], str]] = Field(default_factory=list)
    structured_output_model: Optional[Type[BaseModel]] = None
    output_parser: Optional[BaseOutputParser] = None
    
    # Tool binding options
    tool_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    bind_tools_kwargs: Dict[str, Any] = Field(default_factory=dict)
    bind_tools_config: Dict[str, Any] = Field(default_factory=dict)
    force_tool_choice: Optional[str] = Field(default=None, description="Force the LLM to use this specific tool")
    
    # Pre/post processing
    preprocess: Optional[Callable[[Any], Any]] = None
    postprocess: Optional[Callable[[Any], Any]] = None
    
    # Runtime options
    runtime_options: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('llm_config')
    @classmethod
    def validate_llm_config(cls, value):
        """Validate and convert llm_config."""
        from src.haive.core.models.llm.base import LLMConfig, AzureLLMConfig
        
        # If it's a dictionary, return as is
        if isinstance(value, dict):
            return value
            
        # If it's already an LLMConfig instance, convert to dict
        if isinstance(value, LLMConfig):
            return value.model_dump()
            
        # If it's specifically an AzureLLMConfig, convert to dict
        if isinstance(value, AzureLLMConfig):
            return value.model_dump()
            
        # Default case - try to convert to dict if possible
        try:
            if hasattr(value, 'model_dump'):
                return value.model_dump()
        except Exception:
            pass
            
        # Return as is if we can't convert
        return value
    
    def create_runnable(self) -> Runnable:
        """Create a runnable LLM chain."""
        factory = AugLLMFactory(self)
        return factory.create_runnable()
    
    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig, **kwargs):
        """Create from an existing LLMConfig."""
        return cls(llm_config=llm_config, **kwargs)
    
    @classmethod
    def from_prompt(cls, prompt: BasePromptTemplate, llm_config: Optional[LLMConfig] = None, **kwargs):
        """Create from a prompt template."""
        config = cls(
            prompt_template=prompt,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            **kwargs
        )
        return config
    
    @classmethod
    def from_system_prompt(cls, system_prompt: str, llm_config: Optional[LLMConfig] = None, **kwargs):
        """Create from a system prompt string."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        return cls.from_prompt(prompt, llm_config, **kwargs)
    
    @classmethod
    def from_tools(cls, tools: List[Union[BaseTool, StructuredTool, Type[BaseTool]]], 
                 llm_config: Optional[LLMConfig] = None, **kwargs):
        """Create with specified tools."""
        return cls(
            tools=tools,
            llm_config=llm_config or AzureLLMConfig(model="gpt-4o"),
            **kwargs
        )
    
    def _resolve_tools(self):
        """Resolve tool references to actual tool objects."""
        resolved_tools = []
        for tool in self.tools:
            if isinstance(tool, str):
                # Lookup tool from registry
                tool_engine = EngineRegistry.get_instance().get(EngineType.TOOL, tool)
                if tool_engine:
                    resolved_tools.append(tool_engine)
                else:
                    raise ValueError(f"Tool not found in registry: {tool}")
            else:
                resolved_tools.append(tool)
        return resolved_tools

class AugLLMFactory:
    """Factory for creating structured LLM runnables."""
    
    def __init__(self, aug_config: AugLLMConfig):
        self.aug_config = aug_config
        self.llm_config = aug_config.llm_config
        self.prompt_template = aug_config.prompt_template
        self.tools = aug_config.tools
        self.structured_output_model = aug_config.structured_output_model
        self.output_parser = aug_config.output_parser
        self.tool_kwargs = aug_config.tool_kwargs
        self.bind_tools_kwargs = aug_config.bind_tools_kwargs
        self.bind_tools_config = aug_config.bind_tools_config
        self.runtime_options = aug_config.runtime_options
        self.preprocess = aug_config.preprocess
        self.postprocess = aug_config.postprocess
        
        # Build the runnable
        self.runnable_llm = self.initialize_llm()
        
        if self.tools:
            self.runnable_llm = self.initialize_llm_with_tools()
            
        if self.structured_output_model:
            self.runnable_llm = self.initialize_llm_with_structured_output()
            
        if self.output_parser:
            self.runnable_llm = self.initialize_llm_with_output_parser()
            
        self.runnable = self.create_runnable()
    
    def initialize_llm(self):
        """Initialize the base LLM."""
        from src.haive.core.models.llm.base import LLMConfig, AzureLLMConfig
        
        # Check if llm_config is a dictionary or LLMConfig
        if isinstance(self.llm_config, dict):
            # Determine provider to create the right config type
            provider = self.llm_config.get('provider', 'azure')
            
            # Convert back to LLMConfig based on provider
            if provider == 'azure' or provider == 'AZURE':
                config = AzureLLMConfig(**self.llm_config) 
            else:
                # Default to AzureLLMConfig for now
                config = AzureLLMConfig(**self.llm_config)
                
            return config.instantiate_llm()
        else:
            # It's already a LLMConfig object
            return self.llm_config.instantiate_llm()
    
    def initialize_llm_with_tools(self):
        """Bind tools with proper configuration and tool choice."""
        # Resolve tools
        resolved_tools = self.aug_config._resolve_tools()
        
        # Instantiate tools if needed
        instantiated_tools = []
        for tool in resolved_tools:
            if inspect.isclass(tool):  # It's a class (uninstantiated)
                tool_name = tool.__name__
                tool_kwargs = self.tool_kwargs.get(tool_name, {})
                instantiated_tool = tool(**tool_kwargs) if tool_kwargs else tool()
            else:  # It's already an instance
                instantiated_tool = tool
            instantiated_tools.append(instantiated_tool)
        
        # Handle force_tool_choice if specified
        bind_tools_kwargs = dict(self.bind_tools_kwargs)
        if self.aug_config.force_tool_choice:
            for tool in instantiated_tools:
                if tool.name == self.aug_config.force_tool_choice:
                    bind_tools_kwargs["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
                    break
        
        return self.runnable_llm.bind_tools(
            instantiated_tools,
            **bind_tools_kwargs
        ).with_config(**self.bind_tools_config)
    
    def initialize_llm_with_structured_output(self):
        """Add structured output capability."""
        return self.runnable_llm.with_structured_output(
            self.structured_output_model,
            method="function_calling"
        )
    
    def initialize_llm_with_output_parser(self):
        """Add output parser."""
        return self.runnable_llm | self.output_parser
    
    def create_runnable(self):
        """Create the final runnable chain."""
        runnable_chain = self.runnable_llm
        
        # Add preprocessing if provided
        if self.preprocess:
            runnable_chain = RunnableLambda(self.preprocess) | runnable_chain
        
        # Add prompt template if provided
        if self.prompt_template:
            runnable_chain = self.prompt_template | runnable_chain
        
        # Add postprocessing if provided
        if self.postprocess:
            runnable_chain = runnable_chain | RunnableLambda(self.postprocess)
        
        # Apply runtime configuration
        return runnable_chain.with_config(**self.runtime_options)


def compose_runnable(AugLLMConfig: AugLLMConfig) -> Runnable:
    """Compose a runnable from an AugLLMConfig."""
    factory = AugLLMFactory(AugLLMConfig)
    return factory.create_runnable()