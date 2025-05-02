"""
State schema base class for the Haive framework.

This module provides the StateSchema base class that adds field sharing,
reducers, and other enhancements to Pydantic's BaseModel.
"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

T = TypeVar("T", bound=BaseModel)


class StateSchema(BaseModel, Generic[T]):
    """
    Enhanced base class for state schemas in the Haive framework.

    StateSchema extends Pydantic's BaseModel with features for:
    - Field sharing between parent and child graphs
    - Reducer functions for combining field values
    - Input/output tracking for engines
    - Message handling utilities
    - Serialization and deserialization support
    - State manipulation utilities
    - Pretty printing and visualization

    Field sharing and reducers are critical for proper state handling in nested graphs,
    enabling parent and child graphs to share and update state properly.
    """

    # Class variables to track field sharing and reducers
    __shared_fields__: List[str] = []
    __serializable_reducers__: Dict[str, str] = {}
    __engine_io_mappings__: Dict[str, Dict[str, List[str]]] = {}
    __input_fields__: Dict[str, List[str]] = {}
    __output_fields__: Dict[str, List[str]] = {}
    __structured_models__: Dict[str, str] = {}
    __structured_model_fields__: Dict[str, List[str]] = {}

    # Note: __reducer_fields__ is created dynamically and not part of instance properties

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """
        Override model_dump to exclude internal fields and handle special types.

        Args:
            **kwargs: Keyword arguments for model_dump

        Returns:
            Dictionary representation of the state
        """
        # Get the base model_dump result from Pydantic v2
        data = super().model_dump(**kwargs)

        # Filter out internal fields
        internal_fields = [
            "__shared_fields__",
            "__serializable_reducers__",
            "__reducer_fields__",
            "__engine_io_mappings__",
            "__input_fields__",
            "__output_fields__",
            "__structured_models__",
            "__structured_model_fields__",
        ]
        for field in internal_fields:
            if field in data:
                data.pop(field)

        return data

    def dict(self, **kwargs) -> Dict[str, Any]:
        """
        Backwards compatibility alias for model_dump.

        Args:
            **kwargs: Keyword arguments for model_dump

        Returns:
            Dictionary representation of the state
        """
        return self.model_dump(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a clean dictionary.

        Returns:
            Dictionary representation of the state
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Convert state to JSON string.

        Returns:
            JSON string representation of the state
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "StateSchema":
        """
        Create state from JSON string.

        Args:
            json_str: JSON string to parse

        Returns:
            New StateSchema instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSchema":
        """
        Create a state from a dictionary.

        Args:
            data: Dictionary with field values

        Returns:
            New StateSchema instance
        """
        # Filter out internal fields if present
        internal_fields = [
            "__shared_fields__",
            "__serializable_reducers__",
            "__reducer_fields__",
            "__engine_io_mappings__",
            "__input_fields__",
            "__output_fields__",
            "__structured_models__",
            "__structured_model_fields__",
        ]
        clean_data = {k: v for k, v in data.items() if k not in internal_fields}

        # Use Pydantic v2 method for validation
        return cls.model_validate(clean_data)

    @classmethod
    def from_partial_dict(cls, data: Dict[str, Any]) -> "StateSchema":
        """
        Create a state from a partial dictionary, filling in defaults.

        Args:
            data: Partial dictionary with field values

        Returns:
            New StateSchema instance with defaults applied
        """
        # Get defaults from model fields
        full_data = {}
        for field_name, field_info in cls.model_fields.items():
            # Get default and default_factory
            default = field_info.default
            default_factory = field_info.default_factory

            # Apply defaults
            if default is not ...:
                full_data[field_name] = default
            elif default_factory is not None:
                full_data[field_name] = default_factory()

        # Update with provided data
        full_data.update(data)

        # Create instance with Pydantic v2 method
        return cls.model_validate(full_data)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Safely get a field value with a default.

        Args:
            key: Field name to get
            default: Default value if field doesn't exist

        Returns:
            Field value or default
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def update(self, other: Union[Dict[str, Any], "StateSchema"]) -> "StateSchema":
        """
        Update the state with values from another state or dictionary.

        This method performs a simple update without applying reducers.

        Args:
            other: Dictionary or StateSchema with update values

        Returns:
            Self for chaining
        """
        if isinstance(other, StateSchema):
            data = other.model_dump()
        else:
            data = other

        # Simple update without attempting to apply reducers
        for key, value in data.items():
            setattr(self, key, value)

        return self

    def apply_reducers(
        self, other: Union[Dict[str, Any], "StateSchema"]
    ) -> "StateSchema":
        """
        Update state applying reducer functions where defined.

        This method processes updates with special handling for fields
        that have reducer functions defined.

        Args:
            other: Dictionary or StateSchema with update values

        Returns:
            Self for chaining
        """
        if isinstance(other, StateSchema):
            data = other.model_dump()
        else:
            data = other

        # Get reducer functions
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})

        # Apply updates with reducers where defined
        for key, value in data.items():
            # Skip if the field doesn't exist in this state
            if not hasattr(self, key):
                # Just add the field with simple assignment
                setattr(self, key, value)
                continue

            # Get current value
            current_value = getattr(self, key)

            # Apply reducer if available for this field
            if key in reducer_fields:
                reducer = reducer_fields[key]
                try:
                    # Apply reducer and set the result
                    reduced_value = reducer(current_value, value)
                    setattr(self, key, reduced_value)
                    continue  # Skip to next field after successful reduction
                except Exception as e:
                    logger.warning(f"Error applying reducer for {key}: {e}")
                    # Fall through to special handling or simple assignment

            # Special handling for list values - concat them when both are lists
            if isinstance(current_value, list) and isinstance(value, list):
                merged_list = current_value + value
                setattr(self, key, merged_list)
                continue

            # Special handling for dictionary values - merge them instead of replacing
            if isinstance(current_value, dict) and isinstance(value, dict):
                merged_dict = current_value.copy()
                merged_dict.update(value)
                setattr(self, key, merged_dict)
                continue

            # Simple assignment (no reducer or reducer failed)
            setattr(self, key, value)

        return self

    def add_message(self, message: BaseMessage) -> "StateSchema":
        """
        Add a single message to the messages field.

        Args:
            message: BaseMessage to add

        Returns:
            Self for chaining
        """
        if not hasattr(self, "messages"):
            # Create messages field if it doesn't exist
            self.messages = [message]
            return self

        # Check if we're using a reducer
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})
        if "messages" in reducer_fields:
            # Use the reducer with a single-item list
            self.messages = reducer_fields["messages"](self.messages, [message])
        else:
            # Simple append
            if isinstance(self.messages, list):
                self.messages.append(message)
            else:
                self.messages = [message]

        return self

    def add_messages(self, new_messages: List[BaseMessage]) -> "StateSchema":
        """
        Add multiple messages to the messages field.

        Args:
            new_messages: List of messages to add

        Returns:
            Self for chaining
        """
        if not hasattr(self, "messages"):
            # Create messages field if it doesn't exist
            self.messages = list(new_messages)  # Create a copy
            return self

        # Check if we're using a reducer
        reducer_fields = getattr(self.__class__, "__reducer_fields__", {})
        if "messages" in reducer_fields:
            # Use the reducer
            self.messages = reducer_fields["messages"](self.messages, new_messages)
        else:
            # Simple extend
            if isinstance(self.messages, list):
                self.messages.extend(new_messages)
            else:
                self.messages = list(new_messages)

        return self

    def merge_messages(self, new_messages: List[BaseMessage]) -> "StateSchema":
        """
        Merge new messages with existing messages using appropriate reducer.

        Args:
            new_messages: New messages to add

        Returns:
            Self for chaining
        """
        return self.add_messages(new_messages)

    def clear_messages(self) -> "StateSchema":
        """
        Clear all messages in the messages field.

        Returns:
            Self for chaining
        """
        if hasattr(self, "messages"):
            self.messages = []
        return self

    def get_last_message(self) -> Optional[BaseMessage]:
        """
        Get the last message in the messages field.

        Returns:
            Last message or None if no messages exist
        """
        if hasattr(self, "messages") and self.messages:
            return self.messages[-1]
        return None

    def copy(self, **updates) -> "StateSchema":
        """
        Create a copy of this state, optionally with updates.

        Args:
            **updates: Field values to update in the copy

        Returns:
            New StateSchema instance
        """
        # Use Pydantic v2 model_copy
        return self.model_copy(update=updates)

    def deep_copy(self) -> "StateSchema":
        """
        Create a deep copy of this state object.

        Returns:
            New StateSchema instance with deep-copied values
        """
        return copy.deepcopy(self)

    @classmethod
    def _get_reducer_registry(cls) -> Dict[str, Callable]:
        """
        Get a registry of reducer functions mapped to their names.

        Returns:
            Dictionary mapping reducer names to functions
        """
        registry = {}

        # Add standard reducers
        try:
            from langgraph.graph import add_messages

            registry["add_messages"] = add_messages
        except ImportError:
            # Create a simple concat function as fallback
            def concat_lists(a, b):
                return (a or []) + (b or [])

            registry["concat_lists"] = concat_lists

        # Add common reducer functions
        def concat_strings(a, b):
            return (a or "") + (b or "")

        registry["concat_strings"] = concat_strings

        def sum_values(a, b):
            return (a or 0) + (b or 0)

        registry["sum_values"] = sum_values

        # Add common functions
        registry["max"] = max
        registry["min"] = min

        # Add operator module reducers
        import operator

        for op_name in dir(operator):
            if not op_name.startswith("_"):
                op_func = getattr(operator, op_name)
                if callable(op_func):
                    registry[f"operator.{op_name}"] = op_func
                    registry[op_name] = (
                        op_func  # Also store without prefix for backward compatibility
                    )

        # Try to get reducer functions from class if they exist
        if hasattr(cls, "__reducer_fields__"):
            registry.update(cls.__reducer_fields__)

        # Handle lambda functions
        if "<lambda>" in cls.__serializable_reducers__.values():
            # Can't restore lambdas from name, but we can provide a generic reducer
            def generic_lambda_reducer(a, b):
                # Simple fallback implementation
                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    return a + b
                elif isinstance(a, dict) and isinstance(b, dict):
                    result = a.copy()
                    result.update(b)
                    return result
                else:
                    # Default to returning the newer value
                    return b

            registry["<lambda>"] = generic_lambda_reducer

        return registry

    @classmethod
    def shared_fields(cls) -> List[str]:
        """
        Get the list of fields shared with parent graphs.

        Returns:
            List of shared field names
        """
        return cls.__shared_fields__

    @classmethod
    def is_shared(cls, field_name: str) -> bool:
        """
        Check if a field is shared with parent graphs.

        Args:
            field_name: Field name to check

        Returns:
            True if field is shared, False otherwise
        """
        return field_name in cls.__shared_fields__

    @classmethod
    def to_manager(cls, name: Optional[str] = None) -> "StateSchemaManager":
        """
        Convert schema class to a StateSchemaManager for further manipulation.

        Args:
            name: Optional name for the resulting manager

        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager

        return StateSchemaManager(cls, name=name or cls.__name__)

    @classmethod
    def manager(cls) -> "StateSchemaManager":
        """
        Get a manager for this schema (shorthand for to_manager()).

        Returns:
            StateSchemaManager instance
        """
        return cls.to_manager()

    @classmethod
    def derive_input_schema(
        cls, engine_name: Optional[str] = None, name: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Derive an input schema for the given engine from this state schema.

        Args:
            engine_name: Optional name of the engine to target (default: all inputs)
            name: Optional name for the schema class

        Returns:
            A BaseModel subclass for input validation
        """
        fields = {}

        # Get input field names
        if engine_name is not None and hasattr(cls, "__engine_io_mappings__"):
            if engine_name in cls.__engine_io_mappings__:
                input_fields = cls.__engine_io_mappings__[engine_name].get("inputs", [])
            else:
                input_fields = []
        elif hasattr(cls, "__input_fields__"):
            # Collect input fields across all engines
            input_fields = []
            for engine_inputs in cls.__input_fields__.values():
                input_fields.extend(engine_inputs)
        else:
            input_fields = []

        # Add input fields to schema
        for field_name in input_fields:
            if field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]
                fields[field_name] = (field_info.annotation, field_info)

        # Create model
        schema_name = name or f"{cls.__name__}Input"
        return create_model(schema_name, **fields)

    @classmethod
    def derive_output_schema(
        cls, engine_name: Optional[str] = None, name: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Derive an output schema for the given engine from this state schema.

        Args:
            engine_name: Optional name of the engine to target (default: all outputs)
            name: Optional name for the schema class

        Returns:
            A BaseModel subclass for output validation
        """
        fields = {}

        # Get output field names
        if engine_name is not None and hasattr(cls, "__engine_io_mappings__"):
            if engine_name in cls.__engine_io_mappings__:
                output_fields = cls.__engine_io_mappings__[engine_name].get(
                    "outputs", []
                )
            else:
                output_fields = []
        elif hasattr(cls, "__output_fields__"):
            # Collect output fields across all engines
            output_fields = []
            for engine_outputs in cls.__output_fields__.values():
                output_fields.extend(engine_outputs)
        else:
            output_fields = []

        # Add output fields to schema
        for field_name in output_fields:
            if field_name in cls.model_fields:
                field_info = cls.model_fields[field_name]
                fields[field_name] = (field_info.annotation, field_info)

        # Create model
        schema_name = name or f"{cls.__name__}Output"
        return create_model(schema_name, **fields)

    @classmethod
    def with_shared_fields(cls, fields: List[str]) -> Type[StateSchema]:
        """
        Create a copy of this schema with specified shared fields.

        Args:
            fields: List of field names to be marked as shared

        Returns:
            New StateSchema subclass with updated shared fields
        """
        # Create schema with same fields
        schema = create_model(f"{cls.__name__}WithShared", __base__=cls)

        # Update shared fields
        schema.__shared_fields__ = list(fields)

        return schema

    def patch(
        self, update_data: Dict[str, Any], apply_reducers: bool = True
    ) -> "StateSchema":
        """
        Update specific fields in the state.

        Args:
            update_data: Dictionary of field updates
            apply_reducers: Whether to apply reducer functions

        Returns:
            Self for chaining
        """
        if apply_reducers:
            return self.apply_reducers(update_data)
        else:
            return self.update(update_data)

    def combine_with(
        self, other: Union["StateSchema", Dict[str, Any]]
    ) -> "StateSchema":
        """
        Combine this state with another, applying reducers for shared fields.

        This is more sophisticated than update() or apply_reducers() as it
        properly handles StateSchema-specific metadata and shared fields.

        Args:
            other: Other state to combine with

        Returns:
            New combined state instance
        """
        # Convert to dict if StateSchema
        if isinstance(other, StateSchema):
            other_data = other.model_dump()
        else:
            other_data = other

        # Create a copy of self
        combined = self.model_copy()

        # Apply reducers to the copy
        combined.apply_reducers(other_data)

        return combined

    def differences_from(
        self, other: Union["StateSchema", Dict[str, Any]]
    ) -> Dict[str, Tuple[Any, Any]]:
        """
        Compare this state with another and return differences.

        Args:
            other: Other state to compare with

        Returns:
            Dictionary mapping field names to (self_value, other_value) tuples
        """
        # Convert to dict if StateSchema
        if isinstance(other, StateSchema):
            other_data = other.model_dump()
        else:
            other_data = other

        # Get self data
        self_data = self.model_dump()

        # Find differences
        differences = {}

        # Check fields in self
        for field_name, self_value in self_data.items():
            if field_name in other_data:
                other_value = other_data[field_name]
                if self_value != other_value:
                    differences[field_name] = (self_value, other_value)
            else:
                # Field not in other
                differences[field_name] = (self_value, None)

        # Check fields in other but not in self
        for field_name, other_value in other_data.items():
            if field_name not in self_data:
                differences[field_name] = (None, other_value)

        return differences

    # LangGraph integration methods

    def to_command(
        self, goto: Optional[str] = None, graph: Optional[str] = None
    ) -> Any:
        """
        Convert state to a Command object for LangGraph control flow.

        Args:
            goto: Optional next node to go to
            graph: Optional graph to target (None for current, PARENT for parent)

        Returns:
            Command object with state update
        """
        try:
            from langgraph.types import Command

            # Convert state to dictionary for update
            update = self.model_dump()

            # Create and return command
            return Command(update=update, goto=goto, graph=graph)
        except ImportError:
            logger.warning("LangGraph not available, cannot create Command")
            return {"state": self.model_dump(), "goto": goto, "graph": graph}

    @classmethod
    def from_snapshot(cls, snapshot: Any) -> "StateSchema":
        """
        Create a state from a LangGraph StateSnapshot.

        Args:
            snapshot: StateSnapshot from LangGraph

        Returns:
            New StateSchema instance
        """
        if snapshot is None:
            return cls()

        # Extract values based on object type
        if hasattr(snapshot, "values"):
            # Standard StateSnapshot
            return cls.from_dict(snapshot.values)
        elif hasattr(snapshot, "channel_values") and snapshot.channel_values:
            # Alternative attribute name in some versions
            return cls.from_dict(snapshot.channel_values)
        elif isinstance(snapshot, dict):
            # Dictionary state
            return cls.from_dict(snapshot)

        # Last resort - empty state
        logger.warning(f"Couldn't extract state from snapshot of type {type(snapshot)}")
        return cls()

    # Engine integration methods

    def prepare_for_engine(self, engine_name: str) -> Dict[str, Any]:
        """
        Prepare state data for a specific engine.

        Extracts only fields that are inputs for the specified engine.

        Args:
            engine_name: Name of the engine to prepare for

        Returns:
            Dictionary with engine-specific inputs
        """
        # Extract input field names for this engine
        input_fields = []

        if hasattr(self.__class__, "__engine_io_mappings__"):
            if engine_name in self.__class__.__engine_io_mappings__:
                input_fields = self.__class__.__engine_io_mappings__[engine_name].get(
                    "inputs", []
                )
        elif hasattr(self.__class__, "__input_fields__"):
            if engine_name in self.__class__.__input_fields__:
                input_fields = self.__class__.__input_fields__[engine_name]

        # If no input fields specified, return empty dict
        if not input_fields:
            return {}

        # Extract values for input fields
        result = {}
        for field_name in input_fields:
            if hasattr(self, field_name):
                result[field_name] = getattr(self, field_name)

        return result

    def merge_engine_output(
        self, engine_name: str, output: Dict[str, Any], apply_reducers: bool = True
    ) -> "StateSchema":
        """
        Merge output from an engine into this state.

        Args:
            engine_name: Name of the engine
            output: Output data from the engine
            apply_reducers: Whether to apply reducers during merge

        Returns:
            Self for chaining
        """
        # Filter output to include only fields that are outputs from this engine
        filtered_output = {}

        if hasattr(self.__class__, "__engine_io_mappings__"):
            if engine_name in self.__class__.__engine_io_mappings__:
                output_fields = self.__class__.__engine_io_mappings__[engine_name].get(
                    "outputs", []
                )
                for field_name in output_fields:
                    if field_name in output:
                        filtered_output[field_name] = output[field_name]
        elif hasattr(self.__class__, "__output_fields__"):
            if engine_name in self.__class__.__output_fields__:
                output_fields = self.__class__.__output_fields__[engine_name]
                for field_name in output_fields:
                    if field_name in output:
                        filtered_output[field_name] = output[field_name]

        # Apply update with or without reducers
        if apply_reducers:
            return self.apply_reducers(filtered_output)
        else:
            return self.update(filtered_output)

    # Configuration integration

    def to_runnable_config(
        self, thread_id: Optional[str] = None, **kwargs
    ) -> RunnableConfig:
        """
        Convert state to a RunnableConfig.

        Args:
            thread_id: Optional thread ID for the configuration
            **kwargs: Additional configuration parameters

        Returns:
            RunnableConfig containing state data
        """
        # Create base configuration
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id or str(uuid.uuid4()),
                "state": self.model_dump(),
            }
        }

        # Add additional parameters
        for key, value in kwargs.items():
            config["configurable"][key] = value

        return config

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Optional["StateSchema"]:
        """
        Extract state from a RunnableConfig.

        Args:
            config: RunnableConfig to extract from

        Returns:
            StateSchema instance or None if no state found
        """
        if config and "configurable" in config and "state" in config["configurable"]:
            state_data = config["configurable"]["state"]
            return cls.from_dict(state_data)
        return None

    # Visualization and pretty printing methods

    def pretty_print(self, title: Optional[str] = None) -> None:
        """
        Print state with rich formatting for easy inspection.

        Args:
            title: Optional title for the display
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.tree import Tree

            console = Console()
            display_title = title or f"{self.__class__.__name__} Instance"

            # Create tree representation
            tree = Tree(f"{self.__class__.__name__}:")

            # Add fields
            for field_name, field_value in self.model_dump().items():
                # Format field value
                formatted_value = self._format_field_value(field_value)
                # Add to tree
                tree.add(f"{field_name}: {formatted_value}")

            # Create panel with tree
            panel = Panel(tree, title=display_title, border_style="blue")

            # Print to console
            console.print(panel)

        except ImportError:
            # Fall back to simple print if rich is not available
            print(f"--- {title or self.__class__.__name__} ---")
            for field_name, field_value in self.model_dump().items():
                print(f"{field_name}: {field_value}")

    @staticmethod
    def _format_field_value(value: Any) -> str:
        """
        Format a field value for display.

        Args:
            value: Value to format

        Returns:
            Formatted string representation
        """
        if value is None:
            return "None"
        elif isinstance(value, str):
            if len(value) > 100:
                return f'"{value[:97]}..."'
            return f'"{value}"'
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, list):
            if not value:
                return "[]"
            if len(value) > 5:
                items_str = ", ".join(str(v)[:20] for v in value[:3])
                return f"[{items_str}, ... ({len(value)} items)]"
            return f"[{', '.join(str(v)[:50] for v in value)}]"
        elif isinstance(value, dict):
            if not value:
                return "{}"
            if len(value) > 3:
                items = list(value.items())[:3]
                items_str = ", ".join(f"{k}: {str(v)[:20]}" for k, v in items)
                return f"{{{items_str}, ... ({len(value)} items)}}"
            return f"{{{', '.join(f'{k}: {str(v)[:50]}' for k, v in value.items())}}}"
        elif hasattr(value, "__class__"):
            class_name = value.__class__.__name__
            if hasattr(value, "model_dump"):
                return f"{class_name}(...)"
            return f"<{class_name}>"
        return str(value)

        # Add these methods to the StateSchema class

    @classmethod
    def create_input_schema(
        cls, engine_name: Optional[str] = None, name: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Alias for derive_input_schema for backward compatibility.

        Args:
            engine_name: Optional name of the engine to target
            name: Optional name for the schema class

        Returns:
            A BaseModel subclass for input validation
        """
        return cls.derive_input_schema(engine_name, name)

    @classmethod
    def create_output_schema(
        cls, engine_name: Optional[str] = None, name: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Alias for derive_output_schema for backward compatibility.

        Args:
            engine_name: Optional name of the engine to target
            name: Optional name for the schema class

        Returns:
            A BaseModel subclass for output validation
        """
        return cls.derive_output_schema(engine_name, name)

    # Enhance the display_schema method to better handle structured output models
    @classmethod
    def display_schema(cls, title: Optional[str] = None) -> None:
        """
        Display schema information in a rich format.

        Args:
            title: Optional title for the display
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            from rich.tree import Tree

            console = Console()
            schema_name = cls.__name__
            display_title = title or f"{schema_name} Schema"

            # Create main tree
            tree = Tree(f"class {schema_name}({cls.__base__.__name__}):")

            # Add fields
            fields_node = tree.add("Fields:")
            for field_name, field_info in cls.model_fields.items():
                # Skip special fields
                if field_name.startswith("__"):
                    continue

                # Format field type
                field_type = field_info.annotation
                type_str = str(field_type).replace("typing.", "")

                # Format default value
                if field_info.default_factory is not None:
                    factory_name = getattr(
                        field_info.default_factory, "__name__", "factory"
                    )
                    default_str = f"default_factory={factory_name}"
                else:
                    default = field_info.default
                    if default is ...:
                        default_str = "required"
                    else:
                        default_str = f"default={repr(default)}"

                # Add description if available
                if field_info.description:
                    desc_str = f" # {field_info.description}"
                else:
                    desc_str = ""

                # Add to tree
                fields_node.add(f"{field_name}: {type_str} ({default_str}){desc_str}")

            # Add shared fields
            if cls.__shared_fields__:
                shared_node = tree.add("Shared Fields:")
                for field in cls.__shared_fields__:
                    shared_node.add(field)

            # Add reducers
            if cls.__serializable_reducers__:
                reducers_node = tree.add("Reducers:")
                for field, reducer in cls.__serializable_reducers__.items():
                    reducers_node.add(f"{field}: {reducer}")

            # Add engine I/O mappings
            if cls.__engine_io_mappings__:
                io_node = tree.add("Engine I/O Mappings:")
                for engine, mapping in cls.__engine_io_mappings__.items():
                    engine_node = io_node.add(f"{engine}:")
                    if mapping.get("inputs"):
                        engine_node.add(f"Inputs: {mapping['inputs']}")
                    if mapping.get("outputs"):
                        engine_node.add(f"Outputs: {mapping['outputs']}")

            # Add structured output models information
            if hasattr(cls, "__structured_models__") and cls.__structured_models__:
                structured_node = tree.add("Structured Models:")
                for model_name, model_path in cls.__structured_models__.items():
                    structured_node.add(f"{model_name}: {model_path}")

                    # Add fields if we have them
                    if (
                        hasattr(cls, "__structured_model_fields__")
                        and model_name in cls.__structured_model_fields__
                    ):
                        fields = cls.__structured_model_fields__[model_name]
                        fields_str = ", ".join(fields)
                        structured_node.add(f"  Fields: {fields_str}")

            # Create panel with tree
            panel = Panel(tree, title=display_title, border_style="green")

            # Print to console
            console.print(panel)

        except ImportError:
            # Fall back to simple print if rich is not available
            print(f"--- {title or cls.__name__} Schema ---")
            print(f"class {cls.__name__}({cls.__base__.__name__}):")
            print("  Fields:")
            for field_name, field_info in cls.model_fields.items():
                if not field_name.startswith("__"):
                    print(f"    {field_name}: {field_info.annotation}")

            if cls.__shared_fields__:
                print("  Shared Fields:")
                for field in cls.__shared_fields__:
                    print(f"    {field}")

            if cls.__serializable_reducers__:
                print("  Reducers:")
                for field, reducer in cls.__serializable_reducers__.items():
                    print(f"    {field}: {reducer}")

            if hasattr(cls, "__structured_models__") and cls.__structured_models__:
                print("  Structured Models:")
                for model_name, model_path in cls.__structured_models__.items():
                    print(f"    {model_name}: {model_path}")

    @classmethod
    def to_python_code(cls) -> str:
        """
        Convert schema to Python code representation.

        Returns:
            String containing Python code representation
        """
        lines = [f"class {cls.__name__}(StateSchema):"]
        lines.append('    """')
        lines.append(f"    {cls.__name__} schema")
        lines.append('    """')
        lines.append("")

        # Add fields
        for field_name, field_info in cls.model_fields.items():
            # Skip special fields
            if field_name.startswith("__"):
                continue

            # Format field type
            field_type = field_info.annotation
            type_str = str(field_type).replace("typing.", "")

            # Format default value
            if field_info.default_factory is not None:
                factory_name = getattr(
                    field_info.default_factory, "__name__", "factory"
                )
                default_str = f"Field(default_factory={factory_name}"
            else:
                default = field_info.default
                if default is ...:
                    default_str = "Field(..."
                else:
                    default_str = f"Field(default={repr(default)}"

            # Add description if available
            if field_info.description:
                default_str += f', description="{field_info.description}"'

            default_str += ")"

            # Add field line
            lines.append(f"    {field_name}: {type_str} = {default_str}")

        # Add empty line
        lines.append("")

        # Add class variables
        if cls.__shared_fields__:
            lines.append(f"    __shared_fields__ = {cls.__shared_fields__}")

        if cls.__serializable_reducers__:
            lines.append(
                f"    __serializable_reducers__ = {cls.__serializable_reducers__}"
            )

        if cls.__engine_io_mappings__:
            lines.append(f"    __engine_io_mappings__ = {cls.__engine_io_mappings__}")

        # Add structured models if available
        if hasattr(cls, "__structured_models__") and cls.__structured_models__:
            lines.append(f"    __structured_models__ = {cls.__structured_models__}")

        if (
            hasattr(cls, "__structured_model_fields__")
            and cls.__structured_model_fields__
        ):
            lines.append(
                f"    __structured_model_fields__ = {cls.__structured_model_fields__}"
            )

        return "\n".join(lines)

    @classmethod
    def get_structured_model(cls, model_name: str) -> Optional[Type[BaseModel]]:
        """
        Get a structured output model class by name.

        Args:
            model_name: Name of the structured model

        Returns:
            Model class if found, None otherwise
        """
        if (
            not hasattr(cls, "__structured_models__")
            or model_name not in cls.__structured_models__
        ):
            return None

        # Get the model path
        model_path = cls.__structured_models__[model_name]

        # Try to import the model
        try:
            module_path, class_name = model_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            logger.warning(f"Could not load structured model {model_name}: {e}")
            return None

    @classmethod
    def list_structured_models(cls) -> List[str]:
        """
        List all structured output models in this schema.

        Returns:
            List of structured model names
        """
        if hasattr(cls, "__structured_models__"):
            return list(cls.__structured_models__.keys())
        return []

    @classmethod
    def display_code(cls, title: Optional[str] = None) -> None:
        """
        Display Python code representation of the schema.

        Args:
            title: Optional title for the display
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax

            console = Console()
            code = cls.to_python_code()

            # Create syntax highlighted code
            syntax = Syntax(code, "python", theme="monokai", line_numbers=True)

            # Create panel with syntax
            panel = Panel(
                syntax, title=title or f"{cls.__name__} Code", border_style="yellow"
            )

            # Print to console
            console.print(panel)
        except ImportError:
            # Fall back to simple print if rich is not available
            print(f"--- {title or cls.__name__} Code ---")
            print(cls.to_python_code())

    @classmethod
    def compare_with(
        cls, other: Type["StateSchema"], title: Optional[str] = None
    ) -> None:
        """
        Compare this schema with another in a side-by-side display.

        Args:
            other: Other schema to compare with
            title: Optional title for the comparison
        """
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=title or "Schema Comparison")

            # Add columns
            table.add_column("Field", style="cyan")
            table.add_column(cls.__name__, style="green")
            table.add_column(other.__name__, style="blue")

            # Get all field names
            all_fields = set(cls.model_fields.keys()) | set(other.model_fields.keys())
            all_fields = {field for field in all_fields if not field.startswith("__")}

            # Add rows for each field
            for field_name in sorted(all_fields):
                cls_field = cls.model_fields.get(field_name)
                other_field = other.model_fields.get(field_name)

                # Format fields
                cls_str = (
                    cls._format_field_info(cls_field) if cls_field else "Not present"
                )
                other_str = (
                    cls._format_field_info(other_field)
                    if other_field
                    else "Not present"
                )

                # Add row
                table.add_row(field_name, cls_str, other_str)

            # Add metadata comparison
            table.add_section()

            # Compare shared fields
            cls_shared = cls.__shared_fields__
            other_shared = other.__shared_fields__
            table.add_row("Shared Fields", str(cls_shared), str(other_shared))

            # Compare reducers
            cls_reducers = cls.__serializable_reducers__
            other_reducers = other.__serializable_reducers__
            table.add_row("Reducers", str(cls_reducers), str(other_reducers))

            # Print table
            console.print(table)

        except ImportError:
            # Fall back to simple print if rich is not available
            print(f"--- Schema Comparison: {cls.__name__} vs {other.__name__} ---")
            print("Fields in both:")
            for field in set(cls.model_fields.keys()) & set(other.model_fields.keys()):
                if not field.startswith("__"):
                    print(f"  {field}")

            print(f"Fields only in {cls.__name__}:")
            for field in set(cls.model_fields.keys()) - set(other.model_fields.keys()):
                if not field.startswith("__"):
                    print(f"  {field}")

            print(f"Fields only in {other.__name__}:")
            for field in set(other.model_fields.keys()) - set(cls.model_fields.keys()):
                if not field.startswith("__"):
                    print(f"  {field}")

    @staticmethod
    def _format_field_info(field_info: Any) -> str:
        """
        Format field info for display.

        Args:
            field_info: Field info to format

        Returns:
            Formatted string representation
        """
        if field_info is None:
            return "None"

        # Extract type
        type_str = str(field_info.annotation).replace("typing.", "")

        # Extract default
        if field_info.default_factory is not None:
            factory_name = getattr(field_info.default_factory, "__name__", "factory")
            default_str = f"default_factory={factory_name}"
        else:
            default = field_info.default
            if default is ...:
                default_str = "required"
            else:
                default_str = f"default={repr(default)}"

        return f"{type_str} ({default_str})"
