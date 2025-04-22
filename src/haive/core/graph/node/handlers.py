# src/haive/core/graph/node/handlers.py

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from langgraph.types import Command, Send
from haive.core.graph.node.registry import register_command_handler, register_input_processor, register_output_processor

@register_command_handler("standard")
class StandardCommandHandler:
    """Standard handler for Command/Send pattern."""
    
    def process_result(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Any:
        """Process result to handle Command/Send pattern."""
        # Already using Command/Send pattern
        if isinstance(result, Command):
            # Only modify if it has no goto but config does
            if result.goto is None and config.command_goto is not None:
                return Command(
                    update=result.update,
                    goto=config.command_goto,
                    resume=result.resume,
                    graph=result.graph
                )
            return result
        
        elif isinstance(result, Send):
            # Return Send as-is
            return result
            
        elif isinstance(result, list) and all(isinstance(item, Send) for item in result):
            # Return list of Send objects as-is
            return result
        
        # Not Command/Send - apply command_goto if specified
        if config.command_goto is not None:
            return Command(update=result, goto=config.command_goto)
        
        # Return as-is
        return result


@register_input_processor("direct")
class DirectInputProcessor:
    """Processor for direct input (no mapping)."""
    
    def extract_input(self, state: Dict[str, Any], config: Any) -> Any:
        """Extract input without mapping."""
        # If using direct messages and they exist, return them
        if config.use_direct_messages and "messages" in state:
            return state["messages"]
        
        # Otherwise return the full state
        return state


@register_input_processor("mapped")
class MappedInputProcessor:
    """Processor for mapped input."""
    
    def extract_input(self, state: Dict[str, Any], config: Any) -> Any:
        """Extract input using mapping."""
        # No mapping - fallback to direct
        if not config.input_mapping:
            return state
        
        # Apply mapping
        mapped_input = {}
        for state_key, input_key in config.input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]
        
        # If only one field was mapped and we have that value, return it directly
        if len(config.input_mapping) == 1 and len(mapped_input) == 1:
            return list(mapped_input.values())[0]
        
        return mapped_input


@register_output_processor("standard")
class StandardOutputProcessor:
    """Standard processor for output."""
    
    def process_output(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process standard output."""
        updates = {}
        
        # Start with original state if preserving
        if config.preserve_state:
            updates = original_state.copy()
        
        # Handle BaseMessage result
        if isinstance(result, BaseMessage):
            # Add message to existing messages if present
            if "messages" in original_state and isinstance(original_state["messages"], list):
                updates["messages"] = original_state["messages"] + [result]
            else:
                updates["messages"] = [result]
                
            # Extract content if needed
            if config.extract_content:
                updates["content"] = result.content
                
            return updates
        
        # Handle dictionary result
        if isinstance(result, dict):
            # Apply output mapping if exists
            if config.output_mapping:
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
                            elif hasattr(current, last_part):
                                updates[state_key] = getattr(current, last_part)
                    
                    # Direct key mapping
                    elif output_key in result:
                        updates[state_key] = result[output_key]
            else:
                # No mapping - update with all result keys
                updates.update(result)
                
            return updates
        
        # Fallback - store as result
        updates["result"] = result
        return updates


@register_output_processor("structured")
class StructuredOutputProcessor:
    """Processor for structured output models."""
    
    def process_output(self, result: Any, config: Any, original_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process output from structured output models."""
        updates = {}
        
        # Start with original state if preserving
        if config.preserve_state:
            updates = original_state.copy()
        
        # Handle Pydantic model result
        if isinstance(result, BaseModel):
            # Get model name (lowercase)
            model_name = result.__class__.__name__.lower()
            
            # Add the full model to updates
            updates[model_name] = result
            
            # Extract fields if output mapping exists
            if config.output_mapping:
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
                    # Direct field mapping
                    elif output_key in result.__dict__:
                        updates[state_key] = getattr(result, output_key)
            
            return updates
        
        # Fallback to standard processing for non-BaseModel results
        return StandardOutputProcessor().process_output(result, config, original_state)