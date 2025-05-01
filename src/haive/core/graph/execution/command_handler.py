# execution/command_handler.py
from typing import Any, Optional, List
from langgraph.types import Command, Send
from haive.core.schema.state_schema import StateSchema

def process_engine_result(result: Any, command_goto: Optional[Any] = None) -> Any:
    """Process engine result according to Command pattern."""
    # If result is already a Command or Send, pass it through
    if isinstance(result, Command):
        return result
        
    if isinstance(result, Send) or (
        isinstance(result, list) and all(isinstance(x, Send) for x in result)
    ):
        return result
    
    # Otherwise, wrap in Command if goto specified
    if command_goto is not None:
        return Command(update=result, goto=command_goto)
    
    # Just return result directly
    return result

def process_parent_command(command: Command, parent_state: StateSchema, 
                           shared_fields: Optional[List[str]] = None) -> Command:
    """Process Command with PARENT routing."""
    if command.graph != Command.PARENT:
        return command
    
    # Apply shared field updates to parent state
    if command.update and shared_fields:
        updated_state = apply_shared_fields(parent_state, command.update, shared_fields)
    else:
        updated_state = parent_state
    
    # Return new Command with parent context
    return Command(
        update=updated_state,
        goto=command.goto,
        resume=command.resume
    )