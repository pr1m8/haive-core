# execution/interrupt_handler.py
from langgraph.types import interrupt, Command
from langgraph.errors import GraphInterrupt
from typing import Any, Callable, Optional

def handle_interrupt(state: Any, resume_data: Optional[Any] = None) -> Command:
    """Create a Command for resuming from an interrupt."""
    return Command(update=state, resume=resume_data)

def create_interruptible_node(node_func: Callable) -> Callable:
    """Make a node function interruptible."""
    def wrapped_node(state, config):
        # Check if we're resuming from interruption
        if "__interrupt__" in state:
            resume_data = state.get("__resume_data__")
            clean_state = remove_interrupt_markers(state)
            
            try:
                # Try with resume data
                return node_func(clean_state, config)
            except GraphInterrupt as e:
                # New interruption
                return Command(update={"__interrupt__": e.interrupts})
        
        # Normal execution
        try:
            return node_func(state, config)
        except GraphInterrupt as e:
            # Handle interruption
            return Command(update={"__interrupt__": e.interrupts})
    
    return wrapped_node