from typing import Any, Callable, Dict, List, Optional

from haive.core.schema.state_schema import StateSchema


# execution/state_handler.py
def extract_engine_inputs(state: StateSchema, engine: Any) -> Any:
    """Extract engine inputs from state based on schema metadata."""
    # Use engine_io_mappings from schema
    if hasattr(state.__class__, "__engine_io_mappings__"):
        mappings = state.__class__.__engine_io_mappings__
        if hasattr(engine, "name") and engine.name in mappings:
            input_fields = mappings[engine.name].get("inputs", [])

            # Return single field directly if only one
            if len(input_fields) == 1:
                return getattr(state, input_fields[0])

            # Return dict of fields
            return {
                field: getattr(state, field)
                for field in input_fields
                if hasattr(state, field)
            }

    # Default: return entire state
    return state


def apply_shared_fields(
    parent_state: StateSchema,
    child_state: Any,
    shared_fields: Optional[List[str]] = None,
    reducers: Optional[Dict[str, Callable]] = None,
) -> StateSchema:
    """Apply shared field updates from child to parent."""
    # Get shared fields from schema if not provided
    if shared_fields is None and hasattr(parent_state.__class__, "__shared_fields__"):
        shared_fields = parent_state.__class__.__shared_fields__

    # Get reducers from schema if not provided
    schema_reducers = {}
    if hasattr(parent_state.__class__, "__reducer_fields__"):
        schema_reducers = parent_state.__class__.__reducer_fields__

    # Use provided reducers as override
    active_reducers = reducers or schema_reducers

    # Create copy of parent state
    updated_parent = copy.deepcopy(parent_state)

    # Update shared fields
    if shared_fields:
        for field in shared_fields:
            # Extract child value
            child_value = None
            if isinstance(child_state, dict) and field in child_state:
                child_value = child_state[field]
            elif hasattr(child_state, field):
                child_value = getattr(child_state, field)

            # Skip if no value found
            if child_value is None:
                continue

            # Apply reducer if available
            if field in active_reducers:
                current = getattr(updated_parent, field)
                updated = active_reducers[field](current, child_value)
                setattr(updated_parent, field, updated)
            else:
                # Direct replacement
                setattr(updated_parent, field, child_value)

    return updated_parent
