from pydantic import BaseModel, Field
from typing import Optional, List, Type, Dict, Any,Union,Callable
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from haive_core.models.llm.base import LLMConfig, AzureLLMConfig
from langchain_core.tools import StructuredTool
from langchain_core.runnables import RunnableLambda
import inspect
class AugLLMConfig(BaseModel):
    """
    Configuration for creating a runnable pipeline.
    """
    name: Optional[str] = Field(default="runnable", description="The name of the runnable")
    llm_config: Optional[LLMConfig] = Field(
        default=AzureLLMConfig(),
        description="Configuration for the LLM",
    )  
    prompt_template: Optional[BasePromptTemplate] = None  
    tools: Optional[List[Union[BaseTool, StructuredTool]]] = None  
    structured_output_model: Optional[Type[BaseModel]] = None  
    output_parser: Optional[BaseOutputParser] = None  
    tool_kwargs: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict, description="Tool instantiation kwargs")
    bind_tools_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Kwargs for binding tools")
    bind_tools_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configuration for bind_tools")
    runtime_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution settings for the runnable chain")

    # Additional Callables
    preprocess: Optional[Callable[[Any], Any]] = None
    postprocess: Optional[Callable[[Any], Any]] = None
    #custom_runnables: Optional[List[Runnable]] = None  # Allows chaining additional runnables

    def create_runnable(self) -> Runnable:
        return AugLLMFactory(self).runnable

class AugLLMFactory:
    """
    Factory for creating runnables.
    """
    def __init__(self, aug_config: AugLLMConfig,):
        self.aug_config = aug_config
        self.llm_config = aug_config.llm_config
        self.prompt_template = aug_config.prompt_template
        self.tools = aug_config.tools or []
        self.structured_output_model = aug_config.structured_output_model
        self.output_parser = aug_config.output_parser
        self.tool_kwargs = aug_config.tool_kwargs
        self.bind_tools_kwargs = aug_config.bind_tools_kwargs
        self.bind_tools_config = aug_config.bind_tools_config
        self.runtime_options = aug_config.runtime_options
        self.preprocess = aug_config.preprocess
        self.postprocess = aug_config.postprocess
        # Custom runnables in create or compose. 
        #self.custom_runnables = aug_config.custom_runnables or []

        self.runnable_llm = self.initialize_llm()
        self.runnable_llm = self.initialize_llm_with_tools() if self.tools else self.runnable_llm
        self.runnable_llm = self.initialize_llm_with_structured_output() if self.structured_output_model else self.runnable_llm
        self.runnable_llm = self.initialize_llm_with_output_parser() if self.output_parser else self.runnable_llm
        self.runnable = self.create_runnable()

    def initialize_llm(self) -> Runnable:
        return self.llm_config.instantiate_llm()


    def initialize_llm_with_tools(self) -> Runnable:
        """
        Bind tools with instantiation kwargs, bind_tools kwargs, and bind_tools config.
        """
        instantiated_tools = []
        for tool in self.tools:
            if inspect.isclass(tool):  # It's a class (uninstantiated)
                tool_name = tool.__name__
                tool_kwargs = self.tool_kwargs.get(tool_name, {})
                instantiated_tool = tool(**tool_kwargs) if tool_kwargs else tool()
            else:  # It's already an instance
                instantiated_tool = tool
            instantiated_tools.append(instantiated_tool)

        return self.runnable_llm.bind_tools(
            instantiated_tools,
            **self.bind_tools_kwargs
        ).with_config(**self.bind_tools_config)


    def initialize_llm_with_structured_output(self) -> Runnable:
        return self.runnable_llm.with_structured_output(self.structured_output_model,method="function_calling")

    def initialize_llm_with_output_parser(self) -> Runnable:
        return self.runnable_llm | self.output_parser

    def apply_custom_runnables(self, runnable: Runnable,custom_runnables:Optional[List[Runnable]]=None) -> Runnable:
        """
        Apply additional custom runnables, if provided.
        """
        for custom_runnable in custom_runnables:
            runnable = runnable | custom_runnable
        return runnable

    def create_runnable(self,custom_runnables:Optional[List[Runnable]]=None) -> Runnable:
        runnable_chain = self.runnable_llm

        # Apply preprocessing step if provided
        if self.preprocess:
            runnable_chain = RunnableLambda(self.preprocess) | runnable_chain

        # If a prompt template exists, prepend it
        if self.prompt_template:
            runnable_chain = self.prompt_template | runnable_chain
        if custom_runnables:
            runnable_chain = self.apply_custom_runnables(runnable_chain,custom_runnables)

        # Apply postprocessing step if provided
        if self.postprocess:
            runnable_chain = runnable_chain | RunnableLambda(self.postprocess)

        return runnable_chain.with_config(**self.runtime_options)


def compose_runnable(aug_llm_config: AugLLMConfig) -> Optional[Runnable]:
    """
    Create and return a runnable pipeline based on the provided config.
    """
    try:
        return AugLLMFactory(aug_llm_config).runnable
    except Exception as e:
        print("Error composing runnable:", e)
        return None


def create_runnables_dict(runnables: List[AugLLMConfig]) -> Dict[str, AugLLMConfig]:
    """
    Create a dictionary mapping names to runnable configs.
    """
    return {runnable.name: runnable for runnable in runnables}


def compose_runnables_from_dict(runnables: Dict[str, AugLLMConfig]) -> Dict[str, AugLLMConfig]:
    """
    Compose and return a dictionary of runnables from src.config mappings.
    """
    for key, aug_runnable_config in runnables.items():
        if isinstance(aug_runnable_config, AugLLMConfig):
            runnables[key] = compose_runnable(aug_runnable_config)
    return runnables

