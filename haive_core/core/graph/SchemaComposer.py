# src/haive/core/graph/SchemaComposer.py

from typing import Dict, List, Optional, Union, Tuple, Any, Type
from pydantic import BaseModel
from src.haive.core.engine.aug_llm import AugLLMConfig
from src.haive.core.graph.StateSchemaManager import StateSchemaManager
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence
from langgraph.graph import add_messages
import logging
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
# Set up logging
logger = logging.getLogger(__name__)

class SchemaComposer:
    """
    A utility for dynamically composing state schemas from AugLLM and Agent configurations.
    Uses the merge capabilities of StateSchemaManager to combine schemas.
    """
    #input_mapping = {}  
    #output_mapping = {}
    @staticmethod
    def derive_schema_from_aug_llm(aug_llm_config: AugLLMConfig, schema_manager=None):
        """
        Creates or extends a schema manager with fields derived from an AugLLM configuration.
        
        Args:
            aug_llm_config: The AugLLMConfig to derive schema from
            schema_manager: Existing schema manager to extend, or None to create new
            
        Returns:
            StateSchemaManager with fields added from the AugLLMConfig
        """
        if schema_manager is None:
            # Create a new schema manager with a name based on the AugLLMConfig
            schema_manager = StateSchemaManager(name=f"{aug_llm_config.name}_Schema")
            logger.debug(f"Created new schema manager for {aug_llm_config.name}")
        
        # Add required fields based on prompt template
        if aug_llm_config.prompt_template:
            # Get input variables and variable mapping
            template_has_messages = False
            input_vars = []
            
            # Extract input variables from the prompt template
            if hasattr(aug_llm_config.prompt_template, 'input_variables'):
                input_vars = aug_llm_config.prompt_template.input_variables
                logger.debug(f"Input variables from prompt: {input_vars}")
                
                # Check if 'messages' is in the input variables
                if 'messages' in input_vars:
                    template_has_messages = True
            if hasattr(aug_llm_config.prompt_template, 'partial_variables'):
                partial_input_vars = list(aug_llm_config.prompt_template.partial_variables.keys())
                logger.debug(f"Input variables from prompt: {partial_input_vars}")
                if 'messages' in partial_input_vars:
                    template_has_messages = True
                print(f"DEBUG: partial_variables: {aug_llm_config.prompt_template.partial_variables}")
            # Check for message placeholders (another way to use messages)
            if hasattr(aug_llm_config.prompt_template, 'messages'):
                for message in aug_llm_config.prompt_template.messages:
                    if hasattr(message, 'variable_name') and message.variable_name == 'messages':
                        template_has_messages = True
                        # Make sure 'messages' is in input_vars
                        #if 'messages' not in input_vars:
                            #input_vars.append('messages')
            
            # Always add messages field for compatibility and standard access pattern
            # Only add messages field if explicitly required by prompt template - we want to tweak this up
            if 'messages' in schema_manager.fields and not aug_llm_config.structured_output_model:
                logger.debug("✅ `messages` already exists in schema, skipping auto-add.")
            elif template_has_messages:
                logger.warning(f"⚠️ `messages` detected in {aug_llm_config.name}, adding to schema.")
                schema_manager.add_field(
                    "messages", 
                    Annotated[Sequence[BaseMessage], add_messages], 
                    default=[]
                )
                logger.debug("Added messages field to schema")
            elif 'contents' in schema_manager.fields:
                schema_manager.add_field('contents', Union[str, List[str],List[Document]], default=None)
            #logger.debug("Added messages field to schema")
            if 'contents' not in schema_manager.fields:
                from langchain_core.messages import AnyMessage
                from langchain_core.documents import Document
                from typing import Union, List
                schema_manager.add_field('contents', Union[str, List[str],List[Document],AnyMessage], default=None)
                logger.debug("Added contents field to schema")
            # Add all input variables from the prompt
            for var in input_vars:
                if var != 'messages' and var not in schema_manager.fields:
                    # For variables like 'context', add them to the schema
                    schema_manager.add_field(var, str, default=None)
                    logger.debug(f"Added input variable field: {var}")
                    
                    # If this is a context-like variable and we didn't find messages in the template,
                    # add a method to convert messages to this variable
                    #if var in ['context', 'query', 'input', 'text'] and not template_has_messages:
                    #    logger.debug(f"Adding message_to_{var} method to schema")
                        
                        # Add a computed property or method to derive context from messages
                        # Note: This would be implemented in the actual model creation
                        # We're just flagging it here for now
                    schema_manager._add_message_conversion_flag(var)
        
        # Ensure structured output model is added
        if aug_llm_config.structured_output_model:
            model_class = aug_llm_config.structured_output_model
            field_name = model_class.__name__.lower()
            logger.debug(f"Adding structured output model: {field_name} -> {model_class}")
    
            # Add structured output model if it's missing
            if field_name not in schema_manager.fields:
                from typing import Optional as OptionalType
                schema_manager.add_field(field_name, OptionalType[model_class], default=None)

        # Ensure output field exists if output parser is present
        if aug_llm_config.output_parser and 'output' not in schema_manager.fields:
            schema_manager.add_field('output', str, default="")
            logger.debug("Added output field for output parser")
            
            # Add parsed_output field for storing parsed results
            if 'parsed_output' not in schema_manager.fields:
                from typing import Any
                schema_manager.add_field('parsed_output', Any, default=None)
                logger.debug("Added parsed_output field for output parser results")
    
        # Ensure tool_results field exists if tools are present
        if aug_llm_config.tools and 'tool_results' not in schema_manager.fields:
            from typing import Dict, Any, List
            from langchain_core.tools import BaseTool
            schema_manager.add_field('tool_results', List[Dict[str, Any]], default_factory=list)
            logger.debug("Added tool_results field for tools")
    
        return schema_manager.get_model()

    @staticmethod
    def compose_schema(components: List[Union[AugLLMConfig, BaseModel]], name: str="ComposedSchema"):
        """
        Creates a Pydantic model by composing from multiple components.
        
        Args:
            components: List of configs to derive schema from
            name: Name for the resulting schema
            
        Returns:
            A Pydantic BaseModel class for the composed schema
        """
        schema_manager = SchemaComposer.create_schema_for_components(components, name)
        
        # Add methods for message conversion if needed
        model = schema_manager.get_model()
        
        # Add message conversion methods if flagged in the schema manager
        if hasattr(schema_manager, '_message_conversion_vars') and schema_manager._message_conversion_vars:
            model = SchemaComposer._add_message_conversion_methods(model, schema_manager._message_conversion_vars)
            
        return model
    
    @staticmethod
    def _add_message_conversion_methods(model_class, conversion_vars):
        """
        Add methods to convert messages to specific variables.
        
        Args:
            model_class: The Pydantic model class to enhance
            conversion_vars: List of variable names to generate conversion methods for
            
        Returns:
            Enhanced model class
        """
        for var in conversion_vars:
            # Create a method to convert messages to the variable
            method_name = f"_get_{var}_from_messages"
            
            def create_conversion_method(var_name):
                def conversion_method(self):
                    """Convert messages to the required variable."""
                    if not self.messages:
                        return None
                    
                    # Extract content from messages and join
                    texts = []
                    for msg in self.messages:
                        if hasattr(msg, 'content') and msg.content:
                            texts.append(msg.content)
                        elif isinstance(msg, tuple) and len(msg) >= 2:
                            texts.append(str(msg[1]))
                    
                    # Join all message content
                    return "\n".join(texts)
                
                # Set the method's name (for prettier debugging)
                conversion_method.__name__ = method_name
                return conversion_method
            
            # Add the method to the model class
            setattr(model_class, method_name, create_conversion_method(var))
            
            # Add a property to use the method
            def create_property(method_name, var_name):
                def getter(self):
                    # Get value if explicitly set
                    explicit_value = getattr(self, f"_{var_name}", None)
                    if explicit_value is not None:
                        return explicit_value
                    
                    # Otherwise, compute from messages
                    method = getattr(self, method_name)
                    return method()
                
                def setter(self, value):
                    setattr(self, f"_{var_name}", value)
                
                return property(getter, setter)
            
            # Add private backing field
            setattr(model_class, f"_{var}", None)
            
            # Add the property
            setattr(model_class, var, create_property(method_name, var))
        
        return model_class
    
    @staticmethod
    def compose_schema_from_dict(schema_dict: Dict[str, Any], name: str="ComposedSchema"):
        """
        Creates a Pydantic model from a dictionary of field definitions.
        
        Args:
            schema_dict: Dictionary mapping field names to (type, default) tuples
            name: Name for the resulting schema
            
        Returns:
            A Pydantic BaseModel class
        """
        schema_manager = StateSchemaManager(name=name)
        
        for field_name, field_info in schema_dict.items():
            if isinstance(field_info, tuple) and len(field_info) == 2:
                field_type, default_value = field_info
                schema_manager.add_field(field_name, field_type, default=default_value)
            else:
                # Handle case where it's just a type
                schema_manager.add_field(field_name, field_info, default=None)
        
        return schema_manager.get_model()

    @staticmethod
    def create_schema_for_components(components: List[Union[AugLLMConfig, BaseModel]], name: str="ComposedSchema"):
        """
        Creates a schema manager based on multiple components.
        
        Args:
            components: List of configs to derive schema from
            name: Name for the resulting schema
            
        Returns:
            A StateSchemaManager with fields for all components
        """
        # Create base schema manager with the provided name
        schema_manager = StateSchemaManager(name=name)
        
        # Add method to track variables that need message conversion
        schema_manager._message_conversion_vars = []
        schema_manager._add_message_conversion_flag = lambda var: schema_manager._message_conversion_vars.append(var)
        
        logger.debug(f"Creating schema for {len(components)} components with name '{name}'")
        
        # Process each component
        for i, component in enumerate(components):
            component_name = getattr(component, 'name', f'component_{i}')
            logger.debug(f"Processing component {i+1}/{len(components)}: {component_name}")
            
            if isinstance(component, AugLLMConfig):
                schema_manager = SchemaComposer.derive_schema_from_aug_llm(component, schema_manager)
            elif isinstance(component, BaseModel):
                # Convert Pydantic model to StateSchemaManager and merge
                component_manager = StateSchemaManager(component)
                schema_manager = schema_manager.merge(component_manager)
            elif isinstance(component, dict):
                # Convert dict to StateSchemaManager and merge
                component_manager = StateSchemaManager(component, name=f"Component_{i}")
                schema_manager = schema_manager.merge(component_manager)
            else:
                logger.warning(f"Unsupported component type: {type(component)}")
                
        # Ensure messages field exists as a default
        print(f"DEBUG: schema_manager: {schema_manager}")
        if 'messages' not in schema_manager.model_fields and not schema_manager.model_fields:
            print(f"DEBUG: adding messages field to schema")
            print('adding messages field to schema')
            schema_manager = StateSchemaManager(schema_manager).add_field(
                "messages", 
                Annotated[Sequence[BaseMessage], add_messages], 
                default=[]
            ).get_model()
            logger.debug("Added default messages field to schema")
        if 'contents' not in schema_manager.model_fields and not schema_manager.model_fields:
            print(f"DEBUG: adding contents field to schema")
            print('adding contents field to schema')
            schema_manager = StateSchemaManager(schema_manager).add_field(
                "contents", 
                Union[str, List[str],List[Document],AnyMessage], 
                default=[]
            ).get_model()
            logger.debug("Added default contents field to schema")
                
        return schema_manager