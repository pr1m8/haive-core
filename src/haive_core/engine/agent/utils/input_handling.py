# src/haive/core/engine/agent/utils/input_handling.py

from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage

import logging

logger = logging.getLogger(__name__)

def process_input(
    input_data: Union[str, List[str], Dict[str, Any], BaseModel], 
    input_schema=None
) -> Dict[str, Any]:
    """
    Process input for the agent based on the input schema.
    
    Args:
        input_data: Input in various formats
        input_schema: Schema for validation
        
    Returns:
        Processed input compatible with the graph
    """
    # Extract schema field information if available
    schema_fields = {}
    schema_requires_runnable_config = False
    
    if input_schema:
        if hasattr(input_schema, 'model_fields'):
            schema_fields = input_schema.model_fields
            # Check if runnable_config is in the schema fields
            schema_requires_runnable_config = 'runnable_config' in schema_fields
        elif hasattr(input_schema, '__fields__'):
            schema_fields = input_schema.__fields__
            schema_requires_runnable_config = 'runnable_config' in schema_fields

    # Handle string input
    if isinstance(input_data, str):
        # Initialize with messages
        prepared_input = {"messages": [HumanMessage(content=input_data)]}
        
        # Add dummy runnable_config if the schema requires it
        if schema_requires_runnable_config:
            prepared_input["runnable_config"] = {}
        
        # Add to other input fields based on schema
        for field_name, field_info in schema_fields.items():
            if field_name not in ['messages', 'runnable_config', '__runnable_config__']:
                # Only add to text fields
                field_type = field_info.annotation if hasattr(field_info, 'annotation') else field_info.type_
                type_name = str(field_type)
                if 'str' in type_name or 'String' in type_name:
                    prepared_input[field_name] = input_data
        
        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**prepared_input)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
        
        return prepared_input
    
    # Handle list of strings
    elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
        # Create messages list
        messages = [HumanMessage(content=item) for item in input_data]
        prepared_input = {'messages': messages}
        
        # Add dummy runnable_config if the schema requires it
        if schema_requires_runnable_config:
            prepared_input["runnable_config"] = {}
        
        # Join strings for other text fields
        joined_text = "\n".join(input_data)
        for field_name, field_info in schema_fields.items():
            if field_name not in ['messages', 'runnable_config', '__runnable_config__']:
                # Only add to text fields
                field_type = field_info.annotation if hasattr(field_info, 'annotation') else field_info.type_
                type_name = str(field_type)
                if 'str' in type_name or 'String' in type_name:
                    prepared_input[field_name] = joined_text
        
        # Validate against schema
        if input_schema:
            try:
                return input_schema(**prepared_input)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
                
        return prepared_input
    
    # Handle dictionary input
    elif isinstance(input_data, dict):
        # Create a copy to avoid modifying the original
        input_dict = input_data.copy()
        
        # Add dummy runnable_config if the schema requires it and it's not already present
        if schema_requires_runnable_config and 'runnable_config' not in input_dict:
            input_dict["runnable_config"] = {}
        
        # Ensure there's a messages field if not present and required by schema
        if "messages" not in input_dict and 'messages' in schema_fields:
            # Try to create messages from other fields
            for field in ['input', 'query', 'content', 'text']:
                if field in input_dict and isinstance(input_dict[field], str):
                    input_dict["messages"] = [HumanMessage(content=input_dict[field])]
                    break
        
        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**input_dict)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
                
        return input_dict
    
    # Handle Pydantic model input
    elif isinstance(input_data, BaseModel):
        # Convert to dict
        if hasattr(input_data, "model_dump"):
            # Pydantic v2
            model_dict = input_data.model_dump()
        elif hasattr(input_data, "dict"):
            # Pydantic v1
            model_dict = input_data.dict()
        else:
            # Manual extraction
            model_dict = {}
            for field in input_data.__annotations__:
                if hasattr(input_data, field):
                    model_dict[field] = getattr(input_data, field)
        
        # Add dummy runnable_config if the schema requires it and it's not already present
        if schema_requires_runnable_config and 'runnable_config' not in model_dict:
            model_dict["runnable_config"] = {}
        
        # Ensure there's a messages field if needed by schema
        if "messages" not in model_dict and 'messages' in schema_fields:
            # Try to create messages from other fields
            for field in ['input', 'query', 'content', 'text']:
                if field in model_dict and isinstance(model_dict[field], str):
                    model_dict["messages"] = [HumanMessage(content=model_dict[field])]
                    break
        
        # Validate against schema if available
        if input_schema:
            try:
                return input_schema(**model_dict)
            except Exception as e:
                logger.warning(f"Schema validation failed: {e}")
                
        return model_dict
    
    # Fallback for other types - convert to string message
    fallback_input = {
        "messages": [HumanMessage(content=str(input_data))]
    }
    
    # Add dummy runnable_config if the schema requires it
    if schema_requires_runnable_config:
        fallback_input["runnable_config"] = {}
    
    # Validate against schema if available
    if input_schema:
        try:
            return input_schema(**fallback_input)
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
    
    return fallback_input

def prepare_merged_input(
    input_data: Union[str, List[str], Dict[str, Any], BaseModel],
    previous_state: Optional[Any] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
    input_schema=None,
    state_schema=None
) -> Any:
    """
    Process input data and merge with previous state if available.
    
    Args:
        input_data: Input data in various formats
        previous_state: Previous state from checkpointer
        runtime_config: Runtime configuration
        input_schema: Schema for input validation
        state_schema: Schema for state validation
        
    Returns:
        Processed input data merged with previous state
    """
    logger.debug(f"Preparing merged input with schemas - Input: {input_schema.__name__ if hasattr(input_schema, '__name__') else type(input_schema).__name__}, State: {state_schema.__name__ if hasattr(state_schema, '__name__') else type(state_schema).__name__}")
    
    # Process the input based on schema
    processed_input = process_input(input_data, input_schema)
    
    # Return as is if no previous state
    if not previous_state:
        return processed_input
    
    # Extract values from StateSnapshot if needed
    previous_values = None
    
    if hasattr(previous_state, 'values'):
        # For StateSnapshot objects
        previous_values = previous_state.values
    elif hasattr(previous_state, 'channel_values') and previous_state.channel_values:
        # Alternative attribute name
        previous_values = previous_state.channel_values
    elif isinstance(previous_state, dict):
        # Dictionary state
        previous_values = previous_state
        
    # Return processed input if no valid previous values
    if not previous_values:
        return processed_input
    
    # Merge with previous state
    
    # Special handling for messages to append rather than replace
    if "messages" in processed_input and "messages" in previous_values:
        # Start with all previous messages
        merged_messages = list(previous_values["messages"])
        
        # Add new messages
        new_messages = processed_input["messages"]
        merged_messages.extend(new_messages)
        
        # Update processed input with merged messages
        processed_input["messages"] = merged_messages
        
        # Log the message count
        logger.debug(f"Merged messages: {len(merged_messages)} total")
    
    # Make sure to preserve runnable_config from previous state to avoid validation issues
    if "runnable_config" in previous_values and "runnable_config" not in processed_input:
        processed_input["runnable_config"] = previous_values["runnable_config"]
    elif "runnable_config" not in processed_input:
        # Add empty dict as fallback
        processed_input["runnable_config"] = {}
        
    # Merge other fields, keeping processed_input as priority
    merged_input = dict(previous_values)
    
    # Update with new input values
    for key, value in processed_input.items():
        merged_input[key] = value
        
    # Handle shared fields and reducers if using StateSchema
    if state_schema:
        if hasattr(state_schema, '__shared_fields__'):
            for field in state_schema.__shared_fields__:
                if field in previous_values and field not in processed_input:
                    merged_input[field] = previous_values[field]
    
        if hasattr(state_schema, '__reducer_fields__'):
            for field, reducer in state_schema.__reducer_fields__.items():
                if field in merged_input and field in previous_values:
                    try:
                        merged_input[field] = reducer(previous_values[field], merged_input[field])
                    except Exception as e:
                        logger.warning(f"Reducer for {field} failed: {e}")
    
    # Validate against state schema if available
    if state_schema:
        try:
            return state_schema(**merged_input)
        except Exception as e:
            logger.warning(f"Schema validation failed: {e}")
    
    return merged_input