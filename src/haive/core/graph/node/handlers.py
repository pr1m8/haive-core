# src/haive/core/graph/node/handlers.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langgraph.types import Command, Send
from haive.core.graph.node.registry import register_command_handler, register_input_processor, register_output_processor
from langchain_core.messages import BaseMessage
import logging

logger = logging.getLogger(__name__)

@register_command_handler("standard")
class StandardCommandHandler:
    """Standard handler for Command/Send pattern."""
    
    def process_result(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Any:
        """Process result to handle Command/Send pattern."""
        logger.debug(f"Processing result type: {type(result)}")
        
        # Already using Command/Send pattern
        if isinstance(result, Command):
            logger.debug(f"Command object detected: {result}")
            
            # Only modify if it has no goto but config does
            if result.goto is None and config.command_goto is not None:
                logger.debug(f"Modifying Command to add goto: {config.command_goto}")
                
                # Handle the update attribute carefully
                if hasattr(result, 'update') and not callable(result.update):
                    update_data = result.update
                else:
                    logger.warning(f"Command.update is callable or missing, creating empty dict")
                    update_data = {}
                
                new_command = Command(
                    update=update_data,
                    goto=config.command_goto,
                    resume=result.resume if hasattr(result, 'resume') else None,
                    graph=result.graph if hasattr(result, 'graph') else None
                )
                
                logger.debug(f"Created new Command: {new_command}")
                return new_command
            return result
        
        elif isinstance(result, Send):
            logger.debug(f"Send object detected: {result}")
            return result
            
        elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
            logger.debug(f"List of Send objects detected: {result}")
            return result
        
        # Not Command/Send - apply command_goto if specified
        if config.command_goto is not None:
            logger.debug(f"Creating new Command with goto: {config.command_goto}")
            
            new_command = Command(update=result, goto=config.command_goto)
            logger.debug(f"Created new Command: {new_command}")
            
            return new_command
        
        # Return as-is
        logger.debug(f"Returning result as-is: {result}")
        return result

@register_input_processor("direct")
class DirectInputProcessor:
    """Processor for direct input (no mapping)."""
    
    def extract_input(self, state: Dict[str, Any], config: Any) -> Any:
        """Extract input without mapping."""
        logger.debug(f"Direct input processor for {config.name}")
        
        # If using direct messages and they exist, return them
        if config.use_direct_messages and "messages" in state:
            logger.debug(f"Using direct messages: {len(state['messages'])} messages")
            return state["messages"]
        
        # Otherwise return the full state
        logger.debug(f"Returning full state with keys: {list(state.keys())}")
        return state


@register_input_processor("mapped")
class MappedInputProcessor:
    """Processor for mapped input."""
    
    def extract_input(self, state: Dict[str, Any], config: Any) -> Any:
        """Extract input using mapping."""
        logger.debug(f"Mapped input processor for {config.name}")
        
        # No mapping - fallback to direct
        if not config.input_mapping:
            logger.debug("No input mapping, using full state")
            return state
        
        # Apply mapping
        mapped_input = {}
        for state_key, input_key in config.input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]
                logger.debug(f"Mapped {state_key} → {input_key}")
        
        # If only one field was mapped and we have that value, return it directly
        if len(config.input_mapping) == 1 and len(mapped_input) == 1:
            result = list(mapped_input.values())[0]
            logger.debug(f"Returning single mapped value: {type(result).__name__}")
            return result
        
        logger.debug(f"Returning mapped input with keys: {list(mapped_input.keys())}")
        return mapped_input


@register_output_processor("standard")
class StandardOutputProcessor:
    """Standard processor for output."""
    
    def process_output(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process standard output."""
        logger.debug(f"Processing output for {config.name}")
        logger.debug(f"Result type: {type(result).__name__}")
        
        updates = {}
        
        # Start with original state if preserving
        if config.preserve_state:
            logger.debug("Preserving original state")
            updates = original_state.copy()
        
        # Handle BaseMessage result
        if isinstance(result, BaseMessage):
            logger.debug(f"Processing BaseMessage result: {type(result).__name__}")
            
            # Add message to existing messages if present
            if "messages" in updates and isinstance(updates["messages"], list):
                updates["messages"].append(result)
                logger.debug(f"Added message to existing messages list (now {len(updates['messages'])})")
            else:
                updates["messages"] = [result]
                logger.debug("Created new messages list with message")
                
            # Save as specific message type (e.g., aimessage for AIMessage)
            message_type = result.__class__.__name__.lower()
            updates[message_type] = result
            logger.debug(f"Added message as {message_type}")
                
            # Extract content if needed
            if config.extract_content:
                updates["content"] = result.content
                logger.debug(f"Extracted content from message")
                
            return updates
        
        # Handle dictionary result
        if isinstance(result, dict):
            logger.debug(f"Processing dict result with keys: {list(result.keys())}")
            
            # Apply output mapping if exists
            if config.output_mapping:
                logger.debug(f"Applying output mapping: {config.output_mapping}")
                for output_key, state_key in config.output_mapping.items():
                    # Handle nested keys with dot notation
                    if "." in output_key:
                        parts = output_key.split(".")
                        current = result
                        valid_path = True
                        
                        # Navigate through nested structure
                        for part in parts[:-1]:
                            if part in current and isinstance(current, dict):
                                current = current[part]
                            elif hasattr(current, part):
                                current = getattr(current, part)
                            else:
                                valid_path = False
                                break
                        
                        # Set value if path exists
                        if valid_path:
                            last_part = parts[-1]
                            if isinstance(current, dict) and last_part in current:
                                updates[state_key] = current[last_part]
                                logger.debug(f"Mapped nested {output_key} → {state_key}")
                            elif hasattr(current, last_part):
                                updates[state_key] = getattr(current, last_part)
                                logger.debug(f"Mapped nested attr {output_key} → {state_key}")
                    
                    # Direct key mapping
                    elif output_key in result:
                        updates[state_key] = result[output_key]
                        logger.debug(f"Mapped {output_key} → {state_key}")
            else:
                # No mapping - update with all result keys
                logger.debug("No mapping, updating with all result keys")
                updates.update(result)
                
            return updates
        
        # Fallback - store as result
        logger.debug(f"Using fallback for type {type(result).__name__}")
        updates["result"] = result
        return updates


@register_output_processor("structured")
class StructuredOutputProcessor:
    """Processor for structured output models."""
    
    def process_output(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process output from structured output models."""
        logger.debug(f"Processing structured output for {config.name}")
        logger.debug(f"Result type: {type(result).__name__}")
        
        updates = {}
        
        # Start with original state if preserving
        if config.preserve_state:
            logger.debug("Preserving original state")
            updates = original_state.copy()
        
        # Handle Pydantic model result
        if isinstance(result, BaseModel):
            # Get model name (lowercase)
            model_name = result.__class__.__name__.lower()
            logger.debug(f"Processing BaseModel: {model_name}")
            
            # Add the full model to updates
            updates[model_name] = result
            logger.debug(f"Added model as {model_name}")
            
            # Handle specific case for BaseMessage subtypes
            if isinstance(result, BaseMessage):
                # If we have messages field in state, append the message
                if "messages" in updates and isinstance(updates["messages"], list):
                    updates["messages"].append(result)
                    logger.debug(f"Added message to existing messages list (now {len(updates['messages'])})")
                else:
                    # Create a new messages list
                    updates["messages"] = [result]
                    logger.debug("Created new messages list with message")
            
            # Extract fields if output mapping exists
            if config.output_mapping:
                logger.debug(f"Applying output mapping: {config.output_mapping}")
                for output_key, state_key in config.output_mapping.items():
                    # Handle nested attributes with dot notation
                    if "." in output_key:
                        parts = output_key.split(".")
                        if parts[0] == model_name:
                            # Navigate to nested attribute
                            current = result
                            for part in parts[1:]:
                                if hasattr(current, part):
                                    current = getattr(current, part)
                                else:
                                    current = None
                                    break
                            
                            # Set the value if path exists
                            if current is not None:
                                updates[state_key] = current
                                logger.debug(f"Mapped nested attr {output_key} → {state_key}")
                    # Direct field mapping
                    elif hasattr(result, output_key):
                        updates[state_key] = getattr(result, output_key)
                        logger.debug(f"Mapped {output_key} → {state_key}")
            
            return updates
        
        # Fallback to standard processing for non-BaseModel results
        logger.debug("Falling back to standard processing")
        return StandardOutputProcessor().process_output(result, config, original_state)