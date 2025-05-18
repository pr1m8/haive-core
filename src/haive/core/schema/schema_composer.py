"""
SchemaComposer for the Haive framework.

Provides a clean implementation focused on properly handling structured output models
without duplicating their fields in the main schema.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field, create_model

from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.state_schema import StateSchema

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager
logger = logging.getLogger(__name__)


class SchemaComposer:
    """
    Utility for extracting field information from components and composing schemas.

    The SchemaComposer provides a high-level API for:
    - Dynamically extracting fields from various components (engines, models, dictionaries)
    - Composing schemas from field definitions
    - Tracking field relationships and metadata
    - Building optimized schema classes with proper configuration
    """

    def __init__(self, name: str = "ComposedSchema"):
        """Initialize a new SchemaComposer."""
        self.name = name
        self.fields = {}
        self.shared_fields = set()
        self.field_sources = defaultdict(set)

        # Track input/output mappings for engines
        self.input_fields = defaultdict(set)
        self.output_fields = defaultdict(set)
        self.engine_io_mappings = {}

        # Track structured models separately
        self.structured_models = {}
        self.structured_model_fields = defaultdict(set)

    def add_field(
        self,
        name: str,
        field_type: Type,
        default: Any = None,
        default_factory: Optional[Callable[[], Any]] = None,
        description: Optional[str] = None,
        shared: bool = False,
        reducer: Optional[Callable] = None,
        source: Optional[str] = None,
    ) -> "SchemaComposer":
        """
        Add a field definition to the schema.

        Args:
            name: Field name
            field_type: Type of the field
            default: Default value for the field
            default_factory: Optional factory function for default value
            description: Optional field description
            shared: Whether field is shared with parent graph
            reducer: Optional reducer function for this field
            source: Optional source identifier (component name, etc.)

        Returns:
            Self for chaining
        """
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self

        # Create field definition
        field_def = FieldDefinition(
            name=name,
            field_type=field_type,
            default=default,
            default_factory=default_factory,
            description=description,
            shared=shared,
            reducer=reducer,
        )

        # Store the field
        self.fields[name] = field_def

        # Track additional metadata
        if shared:
            self.shared_fields.add(name)

        if source:
            self.field_sources[name].add(source)

        return self

    def add_fields_from_dict(self, fields_dict: Dict[str, Any]) -> "SchemaComposer":
        """
        Add fields from a dictionary definition.

        Args:
            fields_dict: Dictionary mapping field names to type/value information

        Returns:
            Self for chaining
        """
        for field_name, field_info in fields_dict.items():
            # Skip special fields
            if field_name == "__runnable_config__" or field_name == "runnable_config":
                logger.warning(f"Skipping special field {field_name}")
                continue

            # Handle different field info formats
            if isinstance(field_info, tuple) and len(field_info) >= 2:
                # (type, default) format
                field_type, default = field_info[0], field_info[1]

                # Look for extra params in a third item
                description = None
                shared = False
                reducer = None
                if len(field_info) >= 3 and isinstance(field_info[2], dict):
                    extra_params = field_info[2]
                    description = extra_params.get("description")
                    shared = extra_params.get("shared", False)
                    reducer = extra_params.get("reducer")

                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    description=description,
                    shared=shared,
                    reducer=reducer,
                )
            elif isinstance(field_info, dict) and "type" in field_info:
                # Dictionary with type key
                field_type = field_info.pop("type")
                default = field_info.pop("default", None)
                default_factory = field_info.pop("default_factory", None)
                description = field_info.pop("description", None)
                shared = field_info.pop("shared", False)
                reducer = field_info.pop("reducer", None)

                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=description,
                    shared=shared,
                    reducer=reducer,
                )
            else:
                # Assume it's a type with no default
                self.add_field(name=field_name, field_type=field_info, default=None)

        return self

    def add_fields_from_model(self, model: Type[BaseModel]) -> "SchemaComposer":
        """
        Extract fields from a Pydantic model.

        Args:
            model: Pydantic model to extract fields from

        Returns:
            Self for chaining
        """
        source = model.__name__

        # Extract fields differently based on Pydantic version
        if hasattr(model, "model_fields"):
            # Pydantic v2
            for field_name, field_info in model.model_fields.items():
                # Skip special fields and private fields
                if field_name.startswith("__") or field_name.startswith("_"):
                    continue

                # Get field type and defaults
                field_type = field_info.annotation

                # Handle default vs default_factory
                if field_info.default_factory is not None:
                    default_factory = field_info.default_factory
                    default = None
                else:
                    default_factory = None
                    default = field_info.default

                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=field_info.description,
                    source=source,
                )
        elif hasattr(model, "__fields__"):
            # Pydantic v1
            for field_name, field_info in model.__fields__.items():
                # Skip special fields and private fields
                if field_name.startswith("__") or field_name.startswith("_"):
                    continue

                # Get field type and defaults
                field_type = field_info.type_

                # Handle default vs default_factory
                if field_info.default_factory is not None:
                    default_factory = field_info.default_factory
                    default = None
                else:
                    default_factory = None
                    default = field_info.default

                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=field_info.description,
                    source=source,
                )

        return self

    def add_fields_from_engine(self, engine: Any) -> "SchemaComposer":
        """Extract fields from an Engine object."""
        source = getattr(engine, "name", str(engine))
        logger.debug(f"Extracting fields from engine: {source}")

        # Get engine name for tracking
        engine_name = getattr(engine, "name", str(engine))

        # Initialize engine IO mapping
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}

        # Check for structured output model first
        if (
            hasattr(engine, "structured_output_model")
            and engine.structured_output_model is not None
        ):
            model = engine.structured_output_model
            model_name = model.__name__.lower()

            logger.debug(f"Found structured_output_model in {source}: {model.__name__}")

            # Store structured model
            self.structured_models[model_name] = model

            # Track model fields for reference
            if hasattr(model, "model_fields"):
                structured_model_fields = set(model.model_fields.keys())
                logger.debug(
                    f"Found structured model fields: {structured_model_fields}"
                )

                # Store these fields for reference
                for field_name in structured_model_fields:
                    self.structured_model_fields[model_name].add(field_name)
                    field_info = model.model_fields[field_name]
                    logger.debug(
                        f"  - Model field: {field_name}: {field_info.annotation}"
                    )

            # Add a single field for the entire model
            if model_name not in self.fields:
                from typing import Optional

                field_type = Optional[model]

                self.add_field(
                    name=model_name,
                    field_type=field_type,
                    default=None,
                    description=f"Output in {model.__name__} format",
                    source=f"{source}.structured_output_model",
                )

                # Mark as output field
                self.output_fields[engine_name].add(model_name)

                # Update engine mapping
                if model_name not in self.engine_io_mappings[engine_name]["outputs"]:
                    self.engine_io_mappings[engine_name]["outputs"].append(model_name)

        # Extract input fields (unchanged)
        if hasattr(engine, "get_input_fields") and callable(engine.get_input_fields):
            try:
                input_fields = engine.get_input_fields()

                for field_name, (field_type, field_info) in input_fields.items():
                    # Skip if already has this field
                    if field_name in self.fields:
                        continue

                    # Get default and default_factory
                    if hasattr(field_info, "default") and field_info.default is not ...:
                        default = field_info.default
                    else:
                        default = None

                    default_factory = getattr(field_info, "default_factory", None)
                    description = getattr(field_info, "description", None)

                    # Add the field
                    self.add_field(
                        name=field_name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory,
                        description=description,
                        source=source,
                    )

                    # Track as input field
                    self.input_fields[engine_name].add(field_name)

                # Update engine IO mapping
                self.engine_io_mappings[engine_name]["inputs"] = list(
                    self.input_fields[engine_name]
                )

                logger.debug(
                    f"Added {len(input_fields)} input fields from engine {engine_name}"
                )
            except Exception as e:
                logger.warning(f"Error getting input_fields from {engine_name}: {e}")

        # Extract output fields EXCEPT for structured model fields
        if hasattr(engine, "get_output_fields") and callable(engine.get_output_fields):
            try:
                # CRITICAL FIX: Don't extract output fields if we have a structured model
                # This prevents the duplication problem
                if (
                    not hasattr(engine, "structured_output_model")
                    or engine.structured_output_model is None
                ):
                    output_fields = engine.get_output_fields()

                    for field_name, (field_type, field_info) in output_fields.items():
                        # Skip if already has this field
                        if field_name in self.fields:
                            # Just mark as output field if already exists
                            self.output_fields[engine_name].add(field_name)
                            continue

                        # Get default and default_factory
                        if (
                            hasattr(field_info, "default")
                            and field_info.default is not ...
                        ):
                            default = field_info.default
                        else:
                            default = None

                        default_factory = getattr(field_info, "default_factory", None)
                        description = getattr(field_info, "description", None)

                        # Add the field
                        self.add_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=description,
                            source=source,
                        )

                        # Track as output field
                        self.output_fields[engine_name].add(field_name)

                    logger.debug(
                        f"Added {len(output_fields)} output fields from engine {engine_name}"
                    )
            except Exception as e:
                logger.warning(f"Error getting output_fields from {engine_name}: {e}")

        # Update engine IO mapping outputs
        self.engine_io_mappings[engine_name]["outputs"] = list(
            self.output_fields[engine_name]
        )

        return self

    def mark_as_input_field(
        self, field_name: str, engine_name: str
    ) -> "SchemaComposer":
        """
        Mark a field as input field for a specific engine.

        Args:
            field_name: Name of the field
            engine_name: Name of the engine

        Returns:
            Self for chaining
        """
        # Initialize engine mapping if not exists
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}

        # Add field to inputs for this engine
        self.input_fields[engine_name].add(field_name)

        # Make sure field is in engine mapping inputs
        if field_name not in self.engine_io_mappings[engine_name]["inputs"]:
            self.engine_io_mappings[engine_name]["inputs"].append(field_name)

        return self

    def mark_as_output_field(
        self, field_name: str, engine_name: str
    ) -> "SchemaComposer":
        """
        Mark a field as output field for a specific engine.

        Args:
            field_name: Name of the field
            engine_name: Name of the engine

        Returns:
            Self for chaining
        """
        # Initialize engine mapping if not exists
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}

        # Add field to outputs for this engine
        self.output_fields[engine_name].add(field_name)

        # Make sure field is in engine mapping outputs
        if field_name not in self.engine_io_mappings[engine_name]["outputs"]:
            self.engine_io_mappings[engine_name]["outputs"].append(field_name)

        return self

    def build(self) -> Type[StateSchema]:
        """
        Build a StateSchema directly from the composer with proper handling of structured models.

        Returns:
            Subclass of StateSchema with configured fields and metadata
        """
        # Create field definitions for the model
        field_defs = {}
        for name, field_def in self.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)

        # Create the base schema
        schema = create_model(self.name, __base__=StateSchema, **field_defs)

        # Add shared fields
        schema.__shared_fields__ = list(self.shared_fields)

        # Add reducers
        schema.__serializable_reducers__ = {}
        schema.__reducer_fields__ = {}

        for name, field_def in self.fields.items():
            if field_def.reducer:
                reducer_name = field_def.get_reducer_name()
                schema.__serializable_reducers__[name] = reducer_name
                schema.__reducer_fields__[name] = field_def.reducer

        # Make sure to deep copy the engine I/O mappings to avoid reference issues
        schema.__engine_io_mappings__ = {}
        for engine_name, mapping in self.engine_io_mappings.items():
            schema.__engine_io_mappings__[engine_name] = mapping.copy()

        # Same for input/output fields - convert sets to lists and deep copy
        schema.__input_fields__ = {}
        for engine_name, fields in self.input_fields.items():
            schema.__input_fields__[engine_name] = list(fields)

        schema.__output_fields__ = {}
        for engine_name, fields in self.output_fields.items():
            schema.__output_fields__[engine_name] = list(fields)

        # Add structured model fields metadata safely - use field names instead of class references
        if self.structured_model_fields:
            schema.__structured_model_fields__ = {
                k: list(v) for k, v in self.structured_model_fields.items()
            }

        # Add structured models safely - use string identifiers instead of class references
        if self.structured_models:
            schema.__structured_models__ = {
                k: f"{v.__module__}.{v.__name__}"
                for k, v in self.structured_models.items()
            }

        return schema

    def configure_messages_field(
        self, with_reducer: bool = True, force_add: bool = False
    ) -> "SchemaComposer":
        """
        Configure a messages field with appropriate settings if it exists or if requested.

        Args:
            with_reducer: Whether to add a reducer for the messages field
            force_add: Whether to add the messages field if it doesn't exist

        Returns:
            Self for chaining
        """
        # Only proceed if the field exists or we're forcing its addition
        if "messages" in self.fields or force_add:
            from typing import List

            # Try to use langgraph's add_messages if requested
            if with_reducer:
                try:
                    from langchain_core.messages import BaseMessage
                    from langgraph.graph import add_messages

                    # If force_add is True and the field doesn't exist, add it
                    if force_add and "messages" not in self.fields:
                        self.add_field(
                            name="messages",
                            field_type=List[BaseMessage],
                            default_factory=list,
                            description="Messages for agent conversation",
                            reducer=add_messages,
                        )
                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = add_messages

                except ImportError:
                    # Fallback if add_messages is not available
                    from typing import Any

                    # Create simple concat lists reducer
                    def concat_lists(a, b):
                        return (a or []) + (b or [])

                    # If force_add is True and the field doesn't exist, add it
                    if force_add and "messages" not in self.fields:
                        self.add_field(
                            name="messages",
                            field_type=List[Any],
                            default_factory=list,
                            description="Messages for agent conversation",
                            reducer=concat_lists,
                        )
                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = concat_lists

        return self

    def to_manager(self) -> "StateSchemaManager":
        """
        Convert to a StateSchemaManager for further manipulation.

        Returns:
            StateSchemaManager instance
        """
        from haive.core.schema.schema_manager import StateSchemaManager

        return StateSchemaManager(self)

    @classmethod
    def merge(
        cls,
        first: "SchemaComposer",
        second: "SchemaComposer",
        name: str = "MergedSchema",
    ) -> "SchemaComposer":
        """
        Merge two SchemaComposer instances.

        Args:
            first: First composer
            second: Second composer
            name: Name for merged composer

        Returns:
            New merged SchemaComposer
        """
        merged = cls(name=name)

        # Add fields from first composer
        for field_name, field_def in first.fields.items():
            merged.add_field(
                name=field_name,
                field_type=field_def.field_type,
                default=field_def.default,
                default_factory=field_def.default_factory,
                description=field_def.description,
                shared=field_def.shared,
                reducer=field_def.reducer,
                source=f"first_composer_{field_def.name}",
            )

        # Add fields from second composer (overwriting if they exist)
        for field_name, field_def in second.fields.items():
            merged.add_field(
                name=field_name,
                field_type=field_def.field_type,
                default=field_def.default,
                default_factory=field_def.default_factory,
                description=field_def.description,
                shared=field_def.shared,
                reducer=field_def.reducer,
                source=f"second_composer_{field_def.name}",
            )

        # Merge shared fields
        merged.shared_fields.update(first.shared_fields)
        merged.shared_fields.update(second.shared_fields)

        # Merge field sources
        for field_name, sources in first.field_sources.items():
            merged.field_sources[field_name].update(sources)
        for field_name, sources in second.field_sources.items():
            merged.field_sources[field_name].update(sources)

        # Merge input/output tracking
        for engine, fields in first.input_fields.items():
            merged.input_fields[engine].update(fields)
        for engine, fields in second.input_fields.items():
            merged.input_fields[engine].update(fields)

        for engine, fields in first.output_fields.items():
            merged.output_fields[engine].update(fields)
        for engine, fields in second.output_fields.items():
            merged.output_fields[engine].update(fields)

        # Merge engine mappings
        merged.engine_io_mappings.update(first.engine_io_mappings)
        merged.engine_io_mappings.update(second.engine_io_mappings)

        # Merge structured models
        merged.structured_models.update(first.structured_models)
        merged.structured_models.update(second.structured_models)

        # Merge structured model fields
        for model_name, fields in first.structured_model_fields.items():
            merged.structured_model_fields[model_name].update(fields)
        for model_name, fields in second.structured_model_fields.items():
            merged.structured_model_fields[model_name].update(fields)

        return merged

    @classmethod
    def from_components(
        cls, components: List[Any], name: str = "ComposedSchema"
    ) -> Type[StateSchema]:
        """
        Create a schema from components.

        Args:
            components: List of components to extract fields from
            name: Name for the schema

        Returns:
            StateSchema subclass
        """
        composer = cls(name=name)

        # Process each component
        for component in components:
            if component is None:
                continue

            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                composer.add_fields_from_engine(component)
            elif isinstance(component, BaseModel):
                # BaseModel instance
                composer.add_fields_from_model(component.__class__)
            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                composer.add_fields_from_model(component)
            elif isinstance(component, dict):
                # Dictionary
                composer.add_fields_from_dict(component)
            else:
                logger.debug(f"Skipping unsupported component: {type(component)}")

        # Build the schema
        return composer.build()

    @classmethod
    def compose_input_schema(
        cls, components: List[Any], name: str = "InputSchema"
    ) -> Type[BaseModel]:
        """
        Create an input schema from components, focusing on input fields.

        Args:
            components: List of components to extract fields from
            name: Name for the schema

        Returns:
            BaseModel subclass optimized for input
        """
        composer = cls(name=name)

        # Process each component
        for component in components:
            if component is None:
                continue

            # Only extract input fields from engines
            if hasattr(component, "engine_type") and hasattr(
                component, "get_input_fields"
            ):
                try:
                    # Extract input fields
                    input_fields = component.get_input_fields()
                    engine_name = getattr(component, "name", str(component))

                    for field_name, (field_type, field_info) in input_fields.items():
                        # Skip if already has this field
                        if field_name in composer.fields:
                            continue

                        # Skip special fields
                        if (
                            field_name == "__runnable_config__"
                            or field_name == "runnable_config"
                        ):
                            continue

                        # Get default and default_factory
                        if (
                            hasattr(field_info, "default")
                            and field_info.default is not ...
                        ):
                            default = field_info.default
                        else:
                            default = None

                        default_factory = getattr(field_info, "default_factory", None)
                        description = getattr(field_info, "description", None)

                        # Add the field
                        composer.add_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=description,
                            source=engine_name,
                        )

                        # Track as input field
                        composer.input_fields[engine_name].add(field_name)

                    # Update engine IO mapping
                    composer.engine_io_mappings[engine_name] = {
                        "inputs": list(composer.input_fields[engine_name]),
                        "outputs": [],
                    }
                except Exception as e:
                    logger.warning(
                        f"Error extracting input fields from {component}: {e}"
                    )

            # Handle BaseModel components differently - only extract specific input-related fields
            elif isinstance(component, BaseModel) or (
                isinstance(component, type) and issubclass(component, BaseModel)
            ):
                model = (
                    component if isinstance(component, type) else component.__class__
                )
                source = model.__name__

                # Focus on common input field names
                input_field_names = [
                    "input",
                    "query",
                    "question",
                    "messages",
                    "text",
                    "content",
                ]

                # Extract differently based on Pydantic version
                if hasattr(model, "model_fields"):
                    # Pydantic v2
                    for field_name, field_info in model.model_fields.items():
                        # Only include common input fields and skip special fields
                        if (
                            field_name not in input_field_names
                            or field_name.startswith("__")
                        ):
                            continue

                        # Skip runnable_config
                        if (
                            field_name == "__runnable_config__"
                            or field_name == "runnable_config"
                        ):
                            continue

                        # Get field type and defaults
                        field_type = field_info.annotation

                        # Handle default vs default_factory
                        if field_info.default_factory is not None:
                            default_factory = field_info.default_factory
                            default = None
                        else:
                            default_factory = None
                            default = field_info.default

                        # Add the field
                        composer.add_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=field_info.description,
                            source=source,
                        )
                elif hasattr(model, "__fields__"):
                    # Pydantic v1
                    for field_name, field_info in model.__fields__.items():
                        # Only include common input fields and skip special fields
                        if (
                            field_name not in input_field_names
                            or field_name.startswith("__")
                        ):
                            continue

                        # Skip runnable_config
                        if (
                            field_name == "__runnable_config__"
                            or field_name == "runnable_config"
                        ):
                            continue

                        # Get field type and defaults
                        field_type = field_info.type_

                        # Handle default vs default_factory
                        if field_info.default_factory is not None:
                            default_factory = field_info.default_factory
                            default = None
                        else:
                            default_factory = None
                            default = field_info.default

                        # Add the field
                        composer.add_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=field_info.description,
                            source=source,
                        )

        # Add standard input fields if not already present
        from typing import List, Optional

        from langchain_core.messages import BaseMessage

        # Always ensure we have a messages field
        if "messages" not in composer.fields:
            composer.add_field(
                name="messages",
                field_type=List[BaseMessage],
                default_factory=list,
                description="Messages for agent conversation",
            )

        # Create model directly instead of using StateSchema as base
        field_defs = {}
        for name, field_def in composer.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)

        # Create the input schema
        return create_model(name, **field_defs)

    @classmethod
    def compose_output_schema(
        cls, components: List[Any], name: str = "OutputSchema"
    ) -> Type[BaseModel]:
        """
        Create an output schema from components, focusing on output fields.

        Args:
            components: List of components to extract fields from
            name: Name for the schema

        Returns:
            BaseModel subclass optimized for output
        """
        composer = cls(name=name)

        # Process each component
        for component in components:
            if component is None:
                continue

            # Only extract output fields from engines
            if hasattr(component, "engine_type") and hasattr(
                component, "get_output_fields"
            ):
                try:
                    # Extract output fields
                    output_fields = component.get_output_fields()
                    engine_name = getattr(component, "name", str(component))

                    for field_name, (field_type, field_info) in output_fields.items():
                        # Skip if already has this field
                        if field_name in composer.fields:
                            continue

                        # Skip special fields
                        if (
                            field_name == "__runnable_config__"
                            or field_name == "runnable_config"
                        ):
                            continue

                        # Get default and default_factory
                        if (
                            hasattr(field_info, "default")
                            and field_info.default is not ...
                        ):
                            default = field_info.default
                        else:
                            default = None

                        default_factory = getattr(field_info, "default_factory", None)
                        description = getattr(field_info, "description", None)

                        # Add the field
                        composer.add_field(
                            name=field_name,
                            field_type=field_type,
                            default=default,
                            default_factory=default_factory,
                            description=description,
                            source=engine_name,
                        )

                        # Track as output field
                        composer.output_fields[engine_name].add(field_name)

                    # Add structured output model if available
                    if (
                        hasattr(component, "structured_output_model")
                        and component.structured_output_model
                    ):
                        model = component.structured_output_model
                        model_name = model.__name__.lower()

                        # Store the model
                        composer.structured_models[model_name] = model

                        # Add field for the model
                        from typing import Optional

                        composer.add_field(
                            name=model_name,
                            field_type=Optional[model],
                            default=None,
                            description=f"Output in {model.__name__} format",
                            source=engine_name,
                        )

                        # Track as output field
                        composer.output_fields[engine_name].add(model_name)

                    # Update engine IO mapping
                    composer.engine_io_mappings[engine_name] = {
                        "inputs": [],
                        "outputs": list(composer.output_fields[engine_name]),
                    }
                except Exception as e:
                    logger.warning(
                        f"Error extracting output fields from {component}: {e}"
                    )

        # Add standard output fields if not already present
        from typing import Any, Dict, List, Optional

        from langchain_core.messages import BaseMessage

        # Always ensure we have a messages field
        if "messages" not in composer.fields:
            composer.add_field(
                name="messages",
                field_type=List[BaseMessage],
                default_factory=list,
                description="Messages from agent conversation",
            )

        # Add a content field if no structured output model is present
        has_structured_output = any(composer.structured_models)
        if not has_structured_output:
            composer.add_field(
                name="content",
                field_type=str,
                default="",
                description="Agent output content",
            )

        # Create model directly instead of using StateSchema as base
        field_defs = {}
        for name, field_def in composer.fields.items():
            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)

        # Create the output schema
        return create_model(name, **field_defs)

    @classmethod
    def create_message_state(
        cls,
        additional_fields: Optional[Dict[str, Any]] = None,
        name: str = "MessageState",
    ) -> Type[StateSchema]:
        """
        Create a schema with messages field and additional fields.

        Args:
            additional_fields: Optional dictionary of additional fields to add
            name: Name for the schema

        Returns:
            StateSchema subclass with messages field
        """
        # Create composer
        composer = cls(name=name)

        # Add messages field with reducer
        from typing import List, Sequence

        try:
            from langchain_core.messages import BaseMessage
            from langgraph.graph import add_messages

            # Add messages field with reducer
            composer.add_field(
                name="messages",
                field_type=Sequence[BaseMessage],
                default_factory=list,
                description="Messages for conversation",
                reducer=add_messages,
            )
        except ImportError:
            # Fallback if add_messages is not available
            from typing import Any

            # Create simple concat lists reducer
            def concat_lists(a, b):
                return (a or []) + (b or [])

            composer.add_field(
                name="messages",
                field_type=List[Any],
                default_factory=list,
                description="Messages for conversation",
                reducer=concat_lists,
            )

        # Add additional fields
        if additional_fields:
            for name, value in additional_fields.items():
                if isinstance(value, tuple) and len(value) >= 2:
                    field_type, default = value[0], value[1]

                    # Check if default is a factory
                    default_factory = None
                    if callable(default) and not isinstance(default, type):
                        default_factory = default
                        default = None

                    composer.add_field(
                        name=name,
                        field_type=field_type,
                        default=default,
                        default_factory=default_factory,
                    )
                else:
                    # Infer type from value
                    composer.add_field(name=name, field_type=type(value), default=value)

        # Build schema
        return composer.build()

    @classmethod
    def create_state_from_io_schemas(
        cls,
        input_schema: Type[BaseModel],
        output_schema: Type[BaseModel],
        name: str = "ComposedStateSchema",
    ) -> Type[StateSchema]:
        """
        Create a state schema that combines input and output schemas.

        Args:
            input_schema: Input schema class
            output_schema: Output schema class
            name: Name for the composed schema

        Returns:
            StateSchema subclass that inherits from both input and output schemas
        """
        from typing import List

        from langchain_core.messages import BaseMessage

        # Create composer
        composer = cls(name=name)

        # Add a messages field with reducer first
        try:
            from langgraph.graph import add_messages

            composer.add_field(
                name="messages",
                field_type=List[BaseMessage],
                default_factory=list,
                description="Messages for conversation",
                reducer=add_messages,
                shared=True,
            )
        except ImportError:
            # Fallback if add_messages is not available
            def concat_lists(a, b):
                return (a or []) + (b or [])

            composer.add_field(
                name="messages",
                field_type=List[BaseMessage],
                default_factory=list,
                description="Messages for conversation",
                reducer=concat_lists,
                shared=True,
            )

        # Add fields from input schema
        composer.add_fields_from_model(input_schema)

        # Add fields from output schema
        composer.add_fields_from_model(output_schema)

        # Create field definitions for the model including the base classes
        field_defs = {}
        for name, field_def in composer.fields.items():
            # Skip if field is already in a base class
            if (
                hasattr(input_schema, "model_fields")
                and name in input_schema.model_fields
            ):
                continue
            if (
                hasattr(output_schema, "model_fields")
                and name in output_schema.model_fields
            ):
                continue

            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)

        # Create the schema that inherits from both input and output schemas
        schema = create_model(
            name, __base__=(StateSchema, input_schema, output_schema), **field_defs
        )

        # Configure StateSchema features
        schema.__shared_fields__ = list(composer.shared_fields)

        # Add reducers
        schema.__serializable_reducers__ = {}
        schema.__reducer_fields__ = {}

        for name, field_def in composer.fields.items():
            if field_def.reducer:
                reducer_name = field_def.get_reducer_name()
                schema.__serializable_reducers__[name] = reducer_name
                schema.__reducer_fields__[name] = field_def.reducer

        # Deep copy the engine I/O mappings
        schema.__engine_io_mappings__ = {}
        for engine_name, mapping in composer.engine_io_mappings.items():
            schema.__engine_io_mappings__[engine_name] = mapping.copy()

        # Convert sets to lists for input/output fields
        schema.__input_fields__ = {}
        for engine_name, fields in composer.input_fields.items():
            schema.__input_fields__[engine_name] = list(fields)

        schema.__output_fields__ = {}
        for engine_name, fields in composer.output_fields.items():
            schema.__output_fields__[engine_name] = list(fields)

        # Set structured model info
        if composer.structured_model_fields:
            schema.__structured_model_fields__ = {
                k: list(v) for k, v in composer.structured_model_fields.items()
            }

        if composer.structured_models:
            schema.__structured_models__ = {
                k: f"{v.__module__}.{v.__name__}"
                for k, v in composer.structured_models.items()
            }

        return schema

    def compose_state_from_io(
        self, input_schema: Type[BaseModel], output_schema: Type[BaseModel]
    ) -> Type[StateSchema]:
        """
        Compose a state schema from input and output schemas using this composer.

        Args:
            input_schema: Input schema class
            output_schema: Output schema class

        Returns:
            StateSchema subclass
        """
        # Add fields from input schema if not already present
        for field_name, field_def in self.fields.items():
            if (
                field_name not in input_schema.model_fields
                and field_name not in output_schema.model_fields
            ):
                continue

        # Add remaining fields from input schema
        self.add_fields_from_model(input_schema)

        # Add remaining fields from output schema
        self.add_fields_from_model(output_schema)

        # Build the final schema
        return self.build()

    # src/haive/core/schema/schema_composer.py

    # Add this method to the SchemaComposer class

    def extract_tool_schemas(self, tools: List[Any]) -> None:
        """
        Extract input and output schemas from tools.

        Args:
            tools: List of tools to analyze
        """
        for tool in tools:
            # Get tool name
            tool_name = getattr(tool, "name", None)
            if not tool_name and hasattr(tool, "__name__"):
                tool_name = tool.__name__

            if not tool_name:
                continue

            # Extract input schema
            input_schema = None

            # Check for args_schema on instance or class
            if hasattr(tool, "args_schema"):
                input_schema = tool.args_schema

            # For class types, try to instantiate
            elif isinstance(tool, type):
                if issubclass(tool, BaseTool):
                    try:
                        instance = tool()
                        if hasattr(instance, "args_schema"):
                            input_schema = instance.args_schema
                    except Exception:
                        pass

            # For BaseModel types
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                input_schema = tool

            # Add input schema field if found
            if (
                input_schema
                and isinstance(input_schema, type)
                and issubclass(input_schema, BaseModel)
            ):
                # Add to tool_schemas dictionary in state
                self.add_field(
                    name=f"tool_schemas.{tool_name}",
                    field_type=Type[BaseModel],
                    default=input_schema,
                    description=f"Schema for {tool_name}",
                )

                # Try to find matching output schema
                output_class_name = None
                input_class_name = input_schema.__name__

                # Common naming patterns for output schemas
                possible_names = [
                    f"{tool_name.capitalize()}Result",
                    f"{tool_name.capitalize()}Output",
                    f"{input_class_name}Result",
                    f"{input_class_name}Output",
                    f"{input_class_name.replace('Input', '')}Result",
                    f"{input_class_name.replace('Query', '')}Result",
                ]

                # Look in surrounding module
                if hasattr(input_schema, "__module__"):
                    module = sys.modules.get(input_schema.__module__)
                    if module:
                        for name in possible_names:
                            if hasattr(module, name):
                                output_class = getattr(module, name)
                                if isinstance(output_class, type) and issubclass(
                                    output_class, BaseModel
                                ):
                                    # Add to output_schemas
                                    self.add_field(
                                        name=f"output_schemas.{name}",
                                        field_type=Type[BaseModel],
                                        default=output_class,
                                        description=f"Output schema for {tool_name}",
                                    )

                                    # Add tool_name attribute to schema
                                    setattr(output_class, "tool_name", tool_name)
                                    break

        # Add tool field to track tool instances
        self.add_field(
            name="tools",
            field_type=Dict[str, Any],
            default_factory=dict,
            description="Tool instances indexed by name",
        )

    # Also add this method to SchemaComposer

    @classmethod
    def from_tools(
        cls, tools: List[Any], name: str = "ToolsState", **kwargs
    ) -> "SchemaComposer":
        """
        Create a schema composer from a list of tools.

        Args:
            tools: List of tools to analyze
            name: Name for the schema
            **kwargs: Additional parameters for SchemaComposer

        Returns:
            SchemaComposer with tool schemas
        """
        # Create composer with standard message capabilities
        composer = cls(name=name, **kwargs)

        # Configure for tools
        composer.configure_messages_field(with_reducer=True)

        # Add fields for tool state
        composer.add_field(
            name="tools",
            field_type=Dict[str, Any],
            default_factory=dict,
            description="Tool instances indexed by name",
        )

        composer.add_field(
            name="tool_schemas",
            field_type=Dict[str, Type[BaseModel]],
            default_factory=dict,
            description="Tool schemas indexed by name",
        )

        composer.add_field(
            name="output_schemas",
            field_type=Dict[str, Type[BaseModel]],
            default_factory=dict,
            description="Output schemas indexed by name",
        )

        composer.add_field(
            name="tool_calls",
            field_type=List[Dict[str, Any]],
            default_factory=list,
            description="Current tool calls",
            reducer=operator.add,
        )

        composer.add_field(
            name="validated_tool_calls",
            field_type=List[Dict[str, Any]],
            default_factory=list,
            description="Validated tool calls",
            reducer=operator.add,
        )

        composer.add_field(
            name="completed_tool_calls",
            field_type=List[Dict[str, Any]],
            default_factory=list,
            description="Completed tool calls",
            reducer=operator.add,
        )

        # Extract schemas from tools
        composer.extract_tool_schemas(tools)

        return composer
