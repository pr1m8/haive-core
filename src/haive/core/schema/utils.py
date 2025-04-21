import logging
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo

# Type variables
T = TypeVar("T", bound=BaseModel)
FieldType = TypeVar("FieldType")
DefaultType = TypeVar("DefaultType")

# Logger setup
logger = logging.getLogger(__name__)

class SchemaUtils:
    """Utility functions for schema manipulation and formatting."""

    @staticmethod
    def format_type_annotation(type_hint: Any) -> str:
        """Format a type hint for pretty printing.
        
        Args:
            type_hint: The type hint to format
            
        Returns:
            A clean string representation of the type
        """
        # Handle primitive types
        if type_hint is str:
            return "str"
        if type_hint is int:
            return "int"
        if type_hint is float:
            return "float"
        if type_hint is bool:
            return "bool"
        if type_hint is list:
            return "list"
        if type_hint is dict:
            return "dict"
        if type_hint is None or type_hint is type(None):
            return "None"

        # Handle typing annotations
        type_str = str(type_hint)

        # Remove 'typing.' prefix
        type_str = type_str.replace("typing.", "")

        # Handle special case of Optional and Union
        if type_str.startswith("Optional[") or type_str.startswith("Union["):
            return type_str

        # Handle nested annotations
        if type_str.startswith("<class '"):
            # Extract the actual type name from <class 'type'>
            type_name = type_str.strip("<class '").strip("'>")

            # Handle built-in types
            if type_name.startswith("builtins."):
                return type_name[9:]  # Remove builtins. prefix

            return type_name

        return type_str

    @staticmethod
    def extract_field_info(
        field_info: FieldInfo
    ) -> tuple[Any, str, str | None]:
        """Extract useful information from a Pydantic FieldInfo.
        
        Args:
            field_info: Pydantic field info object
            
        Returns:
            Tuple of (default_value, default_string_representation, description)
        """
        description = getattr(field_info, "description", None)

        # Handle default_factory
        if hasattr(field_info, "default_factory") and field_info.default_factory is not None:
            factory = field_info.default_factory
            if factory == list:
                default_str = " = Field(default_factory=list)"
            elif hasattr(factory, "__name__"):
                default_str = f" = Field(default_factory={factory.__name__})"
            else:
                default_str = f" = Field(default_factory=lambda: {factory()})"
            return factory, default_str, description

        # Handle normal default
        if hasattr(field_info, "default") and field_info.default != ...:
            default = field_info.default
            if default is None:
                default_str = " = None"
            elif isinstance(default, str):
                default_str = f' = "{default}"'
            elif isinstance(default, (int, float, bool)):
                default_str = f" = {default}"
            else:
                default_str = f" = {default!r}"
            return default, default_str, description

        # Handle required field
        return ..., "", description

    @staticmethod
    def format_schema_as_python(
        schema_name: str,
        fields: dict[str, tuple[Any, FieldInfo]],
        properties: dict[str, Any] = None,
        computed_properties: dict[str, Any] = None,
        class_methods: dict[str, Any] = None,
        static_methods: dict[str, Any] = None,
        field_descriptions: dict[str, str] = None,
        shared_fields: set[str] | None = None,
        reducer_fields: dict[str, Callable] | None = None,
        base_class: str = "StateSchema"
    ) -> str:
        """Format a schema definition as Python code.
        
        Args:
            schema_name: Name of the schema class
            fields: Dictionary of field names to (type, field_info) tuples
            properties: Optional dictionary of property names to property methods
            computed_properties: Optional dictionary of computed property definitions
            class_methods: Optional dictionary of class method names to methods
            static_methods: Optional dictionary of static method names to methods
            field_descriptions: Optional dictionary of field descriptions
            shared_fields: Optional set of field names that are shared with parent
            reducer_fields: Optional dictionary of fields with reducer functions
            base_class: Base class name for the schema
        
        Returns:
            String containing the Python code representation
        """
        properties = properties or {}
        computed_properties = computed_properties or {}
        class_methods = class_methods or {}
        static_methods = static_methods or {}
        field_descriptions = field_descriptions or {}
        shared_fields = shared_fields or set()
        reducer_fields = reducer_fields or {}

        # Generate the schema representation
        output = []
        output.append(f"class {schema_name}({base_class}):")

        if not any([fields, properties, computed_properties, class_methods, static_methods]):
            output.append("    pass  # No fields defined\n")
            return "\n".join(output)

        # Add field definitions
        for field_name, (field_type, field_info) in fields.items():
            # Format the type string
            type_str = SchemaUtils.format_type_annotation(field_type)

            # Extract default info
            _, default_str, _ = SchemaUtils.extract_field_info(field_info)

            # Add description as comment if present
            description = field_descriptions.get(field_name, "")
            if description:
                output.append(f"    # {description}")

            # Add special annotations for shared or reducer fields
            annotations = []
            if field_name in shared_fields:
                annotations.append("shared")
            if field_name in reducer_fields:
                reducer_name = getattr(reducer_fields[field_name], "__name__", "reducer")
                annotations.append(f"reducer={reducer_name}")

            if annotations:
                output.append(f"    # {', '.join(annotations)}")

            # Add the field definition
            output.append(f"    {field_name}: {type_str}{default_str}")

        # Add properties
        for prop_name in properties.keys():
            output.append("\n    @property")
            output.append(f"    def {prop_name}(self): ...")

        # Add computed properties
        for prop_name, (getter, setter) in computed_properties.items():
            output.append("\n    @property")
            output.append(f"    def {prop_name}(self): ...")
            if setter:
                output.append(f"    @{prop_name}.setter")
                output.append(f"    def {prop_name}(self, value): ...")

        # Add class methods
        for method_name in class_methods.keys():
            output.append("\n    @classmethod")
            output.append(f"    def {method_name}(cls): ...")

        # Add static methods
        for method_name in static_methods.keys():
            output.append("\n    @staticmethod")
            output.append(f"    def {method_name}(): ...")

        # Return the output string
        return "\n".join(output)

    @staticmethod
    def build_state_schema(
        name: str,
        fields: dict[str, tuple[type, Any]],
        shared_fields: list[str] | None = None,
        reducers: dict[str, Callable] | None = None,
        base_class: type[BaseModel] | None = None
    ) -> type[BaseModel]:
        """Build a state schema from field definitions.
        
        Args:
            name: Name for the schema class
            fields: Dictionary mapping field names to (type, default) tuples
            shared_fields: Optional list of fields shared with parent
            reducers: Optional dictionary mapping field names to reducer functions
            base_class: Optional base class (defaults to StateSchema)
            
        Returns:
            A new schema class
        """
        # Import StateSchema if no base class provided
        if base_class is None:
            from haive.core.schema.state_schema import StateSchema
            base_class = StateSchema

        # Create the model with fields
        model = create_model(
            name,
            __base__=base_class,
            **fields
        )

        # Add shared fields
        if shared_fields:
            model.__shared_fields__ = shared_fields

        # Add reducers
        if reducers:
            # Create serializable_reducers dict
            serializable_reducers = {}
            for field, reducer in reducers.items():
                reducer_name = getattr(reducer, "__name__", str(reducer))
                serializable_reducers[field] = reducer_name

            # Store both the serializable names and the actual reducer functions
            model.__serializable_reducers__ = serializable_reducers
            model.__reducer_fields__ = reducers

        return model

    @staticmethod
    def add_field_to_schema(
        schema: type[BaseModel],
        name: str,
        field_type: type,
        default: Any = None,
        description: str | None = None,
        shared: bool = False,
        reducer: Callable | None = None
    ) -> type[BaseModel]:
        """Add a field to an existing schema class.
        
        Args:
            schema: Existing schema class
            name: Field name to add
            field_type: Type of the field
            default: Default value
            description: Optional field description
            shared: Whether the field is shared with parent
            reducer: Optional reducer function
            
        Returns:
            Updated schema class with the new field
        """
        # Create field dict for the new model
        field_dict = {}

        # Copy existing fields
        for field_name, field_info in schema.model_fields.items():
            field_dict[field_name] = (field_info.annotation, field_info)

        # Add the new field
        field_info = Field(default=default, description=description)
        field_dict[name] = (field_type, field_info)

        # Create new model with all fields
        new_model = create_model(
            schema.__name__,
            __base__=schema.__base__,
            **field_dict
        )

        # Copy shared fields
        if hasattr(schema, "__shared_fields__"):
            new_model.__shared_fields__ = list(schema.__shared_fields__)
            # Add new field to shared fields if needed
            if shared and name not in new_model.__shared_fields__:
                new_model.__shared_fields__.append(name)

        # Copy reducers
        if hasattr(schema, "__serializable_reducers__"):
            new_model.__serializable_reducers__ = dict(schema.__serializable_reducers__)

        if hasattr(schema, "__reducer_fields__"):
            new_model.__reducer_fields__ = dict(schema.__reducer_fields__)

        # Add reducer if provided
        if reducer:
            if not hasattr(new_model, "__serializable_reducers__"):
                new_model.__serializable_reducers__ = {}
            if not hasattr(new_model, "__reducer_fields__"):
                new_model.__reducer_fields__ = {}

            reducer_name = getattr(reducer, "__name__", str(reducer))
            new_model.__serializable_reducers__[name] = reducer_name
            new_model.__reducer_fields__[name] = reducer

        return new_model
