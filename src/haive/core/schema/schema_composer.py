"""
SchemaComposer for the Haive framework.

Provides a comprehensive implementation for schema composition with special focus
on structured output models, recursive annotations, and rich visualization.
"""

from __future__ import annotations

import inspect
import logging
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

# Rich UI components for visualization and debugging
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from rich.tree import Tree

from haive.core.schema.field_definition import FieldDefinition
from haive.core.schema.state_schema import StateSchema

if TYPE_CHECKING:
    from haive.core.schema.schema_manager import StateSchemaManager

# Configure rich logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)
console = Console()


class SchemaComposer:
    """
    Utility for extracting field information from components and composing schemas.

    The SchemaComposer provides a high-level API for:
    - Dynamically extracting fields from various components (engines, models, dictionaries)
    - Composing schemas from field definitions
    - Tracking field relationships and metadata
    - Building optimized schema classes with proper configuration
    - Support for recursive annotations and nested state schemas
    - Managing engines and updating their configurations
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

        # Track nested state schemas
        self.nested_schemas = {}

        # Track if we found tools or messages
        self.has_tools = False
        self.has_messages = False

        # Store engines for later reference and updates
        self.engines = {}  # name -> engine mapping
        self.engines_by_type = defaultdict(list)  # type -> [engine names]

        # Base class detection (determined early)
        self.detected_base_class = None
        self.base_class_fields = set()

        # Debug tracking
        self.processing_history = []

        logger.debug(f"Created SchemaComposer for '{name}'")
        self._visualize_creation()

    def _visualize_creation(self):
        """Display creation information using Rich."""
        tree = Tree(
            f"[bold green]SchemaComposer[/bold green]: [blue]{self.name}[/blue]"
        )
        tree.add("[yellow]Ready to compose schema[/yellow]")
        console.print(tree)

    def _detect_base_class_requirements(
        self, components: Optional[List[Any]] = None
    ) -> None:
        """
        Detect which base class should be used based on components and current fields.
        Must be called before adding fields to avoid duplicates.
        """
        logger.debug("Detecting base class requirements")

        # Check current fields
        if "messages" in self.fields or self.has_messages:
            self.has_messages = True

        if "tools" in self.fields or self.has_tools:
            self.has_tools = True

        # Check components if provided
        if components:
            for component in components:
                if component is None:
                    continue

                # Check for messages in engine I/O
                if hasattr(component, "get_input_fields") and callable(
                    component.get_input_fields
                ):
                    try:
                        input_fields = component.get_input_fields()
                        if "messages" in input_fields:
                            self.has_messages = True
                    except:
                        pass

                if hasattr(component, "get_output_fields") and callable(
                    component.get_output_fields
                ):
                    try:
                        output_fields = component.get_output_fields()
                        if "messages" in output_fields:
                            self.has_messages = True
                    except:
                        pass

                # Check for tools
                if hasattr(component, "tools") and component.tools:
                    self.has_tools = True

        # Determine base class
        base_class = None

        if self.has_tools:
            try:
                from haive.core.schema.prebuilt.tool_state import ToolState

                base_class = ToolState
                logger.debug("Detected need for ToolState base class")
            except ImportError:
                logger.warning(
                    "Could not import ToolState, falling back to MessagesState"
                )
                self.has_tools = False

        if not base_class and self.has_messages:
            try:
                from haive.core.schema.prebuilt.messages_state import MessagesState

                base_class = MessagesState
                logger.debug("Detected need for MessagesState base class")
            except ImportError:
                logger.warning(
                    "Could not import MessagesState, falling back to StateSchema"
                )

        if not base_class:
            from haive.core.schema.state_schema import StateSchema

            base_class = StateSchema
            logger.debug("Using StateSchema as base class")

        self.detected_base_class = base_class

        # Extract fields from base class to avoid duplicates
        if hasattr(base_class, "model_fields"):
            self.base_class_fields = set(base_class.model_fields.keys())
            logger.debug(f"Base class provides fields: {self.base_class_fields}")

    def add_engine(self, engine: Any) -> "SchemaComposer":
        """
        Add an engine to the composer for tracking and later updates.

        Args:
            engine: Engine to add

        Returns:
            Self for chaining
        """
        if engine is None:
            return self

        # Get engine name and type
        engine_name = getattr(engine, "name", f"engine_{id(engine)}")
        engine_type = getattr(engine, "engine_type", None)

        # Store engine
        self.engines[engine_name] = engine

        # Track by type if available
        if engine_type:
            engine_type_str = (
                engine_type.value if hasattr(engine_type, "value") else str(engine_type)
            )
            self.engines_by_type[engine_type_str].append(engine_name)
            logger.debug(f"Added engine '{engine_name}' of type '{engine_type_str}'")

        # Add tracking entry
        self.processing_history.append(
            {
                "action": "add_engine",
                "engine_name": engine_name,
                "engine_type": engine_type,
            }
        )

        return self

    def update_engine_provider(
        self, engine_type: str, updates: Dict[str, Any]
    ) -> "SchemaComposer":
        """
        Update configuration for all engines of a specific type.

        Args:
            engine_type: Type of engines to update (e.g., "llm", "retriever")
            updates: Dictionary of updates to apply

        Returns:
            Self for chaining
        """
        logger.debug(f"Updating all {engine_type} engines with: {updates}")

        updated_count = 0
        for engine_name in self.engines_by_type.get(engine_type, []):
            if engine_name in self.engines:
                engine = self.engines[engine_name]

                # For LLM engines, update llm_config field
                if engine_type == "llm" and hasattr(engine, "llm_config"):
                    for key, value in updates.items():
                        if hasattr(engine.llm_config, key):
                            setattr(engine.llm_config, key, value)
                            logger.debug(
                                f"Updated {engine_name}.llm_config.{key} = {value}"
                            )
                            updated_count += 1

                # For other engine types, update directly
                else:
                    for key, value in updates.items():
                        if hasattr(engine, key):
                            setattr(engine, key, value)
                            logger.debug(f"Updated {engine_name}.{key} = {value}")
                            updated_count += 1

        logger.info(
            f"Updated {updated_count} fields across {len(self.engines_by_type.get(engine_type, []))} {engine_type} engines"
        )

        # Add tracking entry
        self.processing_history.append(
            {
                "action": "update_engine_provider",
                "engine_type": engine_type,
                "updates": updates,
                "affected_engines": updated_count,
            }
        )

        return self

    def get_engines_by_type(self, engine_type: str) -> List[Any]:
        """
        Get all engines of a specific type.

        Args:
            engine_type: Type of engines to retrieve

        Returns:
            List of engines of the specified type
        """
        engine_names = self.engines_by_type.get(engine_type, [])
        return [self.engines[name] for name in engine_names if name in self.engines]

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
        input_for: Optional[List[str]] = None,
        output_from: Optional[List[str]] = None,
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
            input_for: Optional list of engines this field is input for
            output_from: Optional list of engines this field is output from

        Returns:
            Self for chaining
        """
        # Skip special fields
        if name == "__runnable_config__" or name == "runnable_config":
            logger.warning(f"Skipping special field {name}")
            return self

        # Check if field is provided by base class
        if self.detected_base_class and name in self.base_class_fields:
            logger.debug(
                f"Field '{name}' is provided by base class, updating metadata only"
            )

            # Still track metadata for the field
            if shared:
                self.shared_fields.add(name)
            if source:
                self.field_sources[name].add(source)
            if input_for:
                for engine in input_for:
                    self.input_fields[engine].add(name)
                    if engine not in self.engine_io_mappings:
                        self.engine_io_mappings[engine] = {"inputs": [], "outputs": []}
                    if name not in self.engine_io_mappings[engine]["inputs"]:
                        self.engine_io_mappings[engine]["inputs"].append(name)
            if output_from:
                for engine in output_from:
                    self.output_fields[engine].add(name)
                    if engine not in self.engine_io_mappings:
                        self.engine_io_mappings[engine] = {"inputs": [], "outputs": []}
                    if name not in self.engine_io_mappings[engine]["outputs"]:
                        self.engine_io_mappings[engine]["outputs"].append(name)

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
            input_for=input_for or [],
            output_from=output_from or [],
        )

        # Check if field is a StateSchema type - mark for special handling
        if inspect.isclass(field_type) and issubclass(field_type, StateSchema):
            logger.debug(f"Field '{name}' is a StateSchema type: {field_type.__name__}")
            self.nested_schemas[name] = field_type
        elif getattr(field_type, "__origin__", None) is Union and any(
            inspect.isclass(arg) and issubclass(arg, StateSchema)
            for arg in field_type.__args__
            if inspect.isclass(arg)
        ):
            # Handle Optional[StateSchema] and Union types containing StateSchema
            for arg in field_type.__args__:
                if inspect.isclass(arg) and issubclass(arg, StateSchema):
                    logger.debug(
                        f"Field '{name}' contains StateSchema type: {arg.__name__}"
                    )
                    self.nested_schemas[name] = arg
                    break

        # Check if we're adding messages field
        if name == "messages":
            self.has_messages = True
            logger.debug(
                "Added 'messages' field - will use MessagesState as base class"
            )

        # Check if we're adding tools field
        if name == "tools":
            self.has_tools = True
            logger.debug("Added 'tools' field - will use ToolState as base class")

        # Store the field
        self.fields[name] = field_def

        # Track input/output relationships
        if input_for:
            for engine in input_for:
                self.input_fields[engine].add(name)
                # Make sure engine io mapping exists
                if engine not in self.engine_io_mappings:
                    self.engine_io_mappings[engine] = {"inputs": [], "outputs": []}
                # Add to inputs list if not already there
                if name not in self.engine_io_mappings[engine]["inputs"]:
                    self.engine_io_mappings[engine]["inputs"].append(name)

        if output_from:
            for engine in output_from:
                self.output_fields[engine].add(name)
                # Make sure engine io mapping exists
                if engine not in self.engine_io_mappings:
                    self.engine_io_mappings[engine] = {"inputs": [], "outputs": []}
                # Add to outputs list if not already there
                if name not in self.engine_io_mappings[engine]["outputs"]:
                    self.engine_io_mappings[engine]["outputs"].append(name)

        # Track additional metadata
        if shared:
            self.shared_fields.add(name)
            logger.debug(f"Marked field '{name}' as shared")

        if source:
            self.field_sources[name].add(source)
            logger.debug(f"Field '{name}' source: {source}")

        # Add tracking entry
        self.processing_history.append(
            {
                "action": "add_field",
                "field_name": name,
                "field_type": str(field_type),
                "shared": shared,
                "has_reducer": reducer is not None,
            }
        )

        logger.debug(f"Added field '{name}' of type {field_type}")

        return self

    def add_fields_from_dict(self, fields_dict: Dict[str, Any]) -> "SchemaComposer":
        """
        Add fields from a dictionary definition.

        Args:
            fields_dict: Dictionary mapping field names to type/value information

        Returns:
            Self for chaining
        """
        logger.debug(f"Adding fields from dictionary with {len(fields_dict)} entries")

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
                    source="dict",
                )
                logger.debug(f"Added field '{field_name}' from dict tuple format")

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
                    source="dict",
                )
                logger.debug(f"Added field '{field_name}' from dict with type key")

            else:
                # Assume it's a type with no default
                self.add_field(
                    name=field_name, field_type=field_info, default=None, source="dict"
                )
                logger.debug(
                    f"Added field '{field_name}' from dict with type-only format"
                )

        return self

    def add_fields_from_model(self, model: Type[BaseModel]) -> "SchemaComposer":
        """
        Extract fields from a Pydantic model with improved handling of nested schemas.

        Args:
            model: Pydantic model to extract fields from

        Returns:
            Self for chaining
        """
        source = model.__name__
        logger.debug(f"Extracting fields from model: {source}")

        # Keep track of field metadata for schema
        shared_fields = []
        reducers = {}
        engine_io = {}

        # Check if model is a StateSchema
        is_state_schema = (
            issubclass(model, StateSchema) if inspect.isclass(model) else False
        )

        if is_state_schema:
            logger.debug(f"Model {source} is a StateSchema - will extract metadata")

            # Extract shared fields
            if hasattr(model, "__shared_fields__"):
                shared_fields = model.__shared_fields__
                logger.debug(f"Found shared fields: {shared_fields}")

            # Extract reducer info
            if hasattr(model, "__serializable_reducers__"):
                reducers = model.__serializable_reducers__
                logger.debug(f"Found reducers: {reducers}")

            # Extract engine IO mappings
            if hasattr(model, "__engine_io_mappings__"):
                engine_io = model.__engine_io_mappings__
                logger.debug(f"Found engine IO mappings: {len(engine_io)} engines")

                # Update our engine mappings
                for engine_name, mapping in engine_io.items():
                    if engine_name not in self.engine_io_mappings:
                        self.engine_io_mappings[engine_name] = {
                            "inputs": [],
                            "outputs": [],
                        }

                    # Add inputs
                    for field_name in mapping.get("inputs", []):
                        if (
                            field_name
                            not in self.engine_io_mappings[engine_name]["inputs"]
                        ):
                            self.engine_io_mappings[engine_name]["inputs"].append(
                                field_name
                            )
                            self.input_fields[engine_name].add(field_name)

                    # Add outputs
                    for field_name in mapping.get("outputs", []):
                        if (
                            field_name
                            not in self.engine_io_mappings[engine_name]["outputs"]
                        ):
                            self.engine_io_mappings[engine_name]["outputs"].append(
                                field_name
                            )
                            self.output_fields[engine_name].add(field_name)

        # Extract fields based on Pydantic version
        if hasattr(model, "model_fields"):
            # Pydantic v2
            logger.debug(
                f"Processing Pydantic v2 model: {len(model.model_fields)} fields"
            )

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

                # Check if this field is shared
                is_shared = field_name in shared_fields if is_state_schema else False

                # Check if this field has a reducer
                reducer_name = reducers.get(field_name)
                reducer = None
                if reducer_name:
                    try:
                        # Try to resolve the reducer from common places
                        if reducer_name == "add_messages":
                            from langgraph.graph import add_messages

                            reducer = add_messages
                        elif reducer_name.startswith("operator."):
                            import operator

                            op_name = reducer_name.split(".", 1)[1]
                            reducer = getattr(operator, op_name, None)
                        logger.debug(
                            f"Resolved reducer '{reducer_name}' for field '{field_name}'"
                        )
                    except ImportError:
                        logger.warning(f"Could not import reducer {reducer_name}")

                # Check if field is a nested StateSchema
                if inspect.isclass(field_type) and issubclass(field_type, StateSchema):
                    logger.debug(
                        f"Field '{field_name}' is a StateSchema: {field_type.__name__}"
                    )
                    self.nested_schemas[field_name] = field_type

                # Get input/output information
                input_for = []
                output_from = []
                for engine_name, mapping in engine_io.items():
                    if field_name in mapping.get("inputs", []):
                        input_for.append(engine_name)
                    if field_name in mapping.get("outputs", []):
                        output_from.append(engine_name)

                # Add the field
                self.add_field(
                    name=field_name,
                    field_type=field_type,
                    default=default,
                    default_factory=default_factory,
                    description=field_info.description,
                    shared=is_shared,
                    reducer=reducer,
                    source=source,
                    input_for=input_for,
                    output_from=output_from,
                )

        # If a StateSchema, check for structured_models
        if is_state_schema and hasattr(model, "__structured_models__"):
            for model_name, model_path in model.__structured_models__.items():
                logger.debug(f"Found structured model: {model_name} -> {model_path}")

                # Try to import the model
                try:
                    module_path, class_name = model_path.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    structured_model = getattr(module, class_name)

                    # Add to our structured models
                    self.structured_models[model_name] = structured_model

                    # Get field names if available
                    if (
                        hasattr(model, "__structured_model_fields__")
                        and model_name in model.__structured_model_fields__
                    ):
                        fields = model.__structured_model_fields__[model_name]
                        for field in fields:
                            self.structured_model_fields[model_name].add(field)

                    logger.debug(
                        f"Imported structured model {model_name}: {structured_model.__name__}"
                    )
                except (ImportError, AttributeError, ValueError) as e:
                    logger.warning(
                        f"Could not import structured model {model_name}: {e}"
                    )

        return self

    def add_fields_from_engine(self, engine: Any) -> "SchemaComposer":
        """
        Extract fields from an Engine object with enhanced nested schema handling.

        Args:
            engine: Engine to extract fields from

        Returns:
            Self for chaining
        """
        source = getattr(engine, "name", str(engine))
        logger.debug(f"Extracting fields from engine: {source}")

        # Add the engine for tracking
        self.add_engine(engine)

        # Add tracking entry
        self.processing_history.append(
            {"action": "add_fields_from_engine", "engine": source}
        )

        # Get engine name for tracking
        engine_name = getattr(engine, "name", str(engine))

        # Initialize engine IO mapping
        if engine_name not in self.engine_io_mappings:
            self.engine_io_mappings[engine_name] = {"inputs": [], "outputs": []}

        # Process steps:
        # 1. First, check for input/output schemas
        # 2. Then check for structured output model
        # 3. Fall back to get_input_fields/get_output_fields methods
        # 4. Handle tools

        # 1. Check for input/output schema properties or methods
        input_schema = None
        output_schema = None

        # 1.1 Try to get input schema
        if hasattr(engine, "input_schema") and engine.input_schema:
            input_schema = engine.input_schema
            logger.debug(f"Using input_schema from engine: {input_schema.__name__}")
        elif hasattr(engine, "derive_input_schema") and callable(
            engine.derive_input_schema
        ):
            try:
                input_schema = engine.derive_input_schema()
                logger.debug(f"Using derived input schema: {input_schema.__name__}")
            except Exception as e:
                logger.warning(f"Error deriving input schema from {engine_name}: {e}")

        # 1.2 Try to get output schema
        if hasattr(engine, "output_schema") and engine.output_schema:
            output_schema = engine.output_schema
            logger.debug(f"Using output_schema from engine: {output_schema.__name__}")
        elif hasattr(engine, "derive_output_schema") and callable(
            engine.derive_output_schema
        ):
            try:
                output_schema = engine.derive_output_schema()
                logger.debug(f"Using derived output schema: {output_schema.__name__}")
            except Exception as e:
                logger.warning(f"Error deriving output schema from {engine_name}: {e}")

        # 1.3 Process input schema if found
        if input_schema:
            logger.debug(f"Processing input schema for {engine_name}")

            # Extract fields from the schema
            self.add_fields_from_model(input_schema)

            # Mark fields as inputs for this engine
            if hasattr(input_schema, "model_fields"):
                for field_name in input_schema.model_fields:
                    if (
                        field_name not in self.fields
                        and field_name not in self.base_class_fields
                    ):
                        logger.warning(
                            f"Field {field_name} from {engine_name} input schema not found in fields"
                        )
                        continue

                    # Add to input fields
                    self.input_fields[engine_name].add(field_name)

                    # Add to engine IO mapping
                    if field_name not in self.engine_io_mappings[engine_name]["inputs"]:
                        self.engine_io_mappings[engine_name]["inputs"].append(
                            field_name
                        )
                        logger.debug(
                            f"Marked field '{field_name}' as input for engine '{engine_name}'"
                        )

        # 1.4 Process output schema if found
        if output_schema:
            logger.debug(f"Processing output schema for {engine_name}")

            # Extract fields from the schema
            self.add_fields_from_model(output_schema)

            # Mark fields as outputs for this engine
            if hasattr(output_schema, "model_fields"):
                for field_name in output_schema.model_fields:
                    if (
                        field_name not in self.fields
                        and field_name not in self.base_class_fields
                    ):
                        logger.warning(
                            f"Field {field_name} from {engine_name} output schema not found in fields"
                        )
                        continue

                    # Add to output fields
                    self.output_fields[engine_name].add(field_name)

                    # Add to engine IO mapping
                    if (
                        field_name
                        not in self.engine_io_mappings[engine_name]["outputs"]
                    ):
                        self.engine_io_mappings[engine_name]["outputs"].append(
                            field_name
                        )
                        logger.debug(
                            f"Marked field '{field_name}' as output for engine '{engine_name}'"
                        )

        # 2. Process structured output model if available
        content_field_exists = False

        if (
            hasattr(engine, "structured_output_model")
            and engine.structured_output_model
        ):
            model = engine.structured_output_model
            model_name = model.__name__.lower()
            logger.debug(f"Found structured_output_model in {source}: {model.__name__}")

            # Check if model contains content/contents field
            model_has_content = False
            content_field_name = None

            if hasattr(model, "model_fields"):
                for content_field in [
                    "content",
                    "contents",
                    "response",
                    "result",
                    "output",
                ]:
                    if content_field in model.model_fields:
                        model_has_content = True
                        content_field_name = content_field
                        content_field_exists = True
                        break

                if model_has_content:
                    logger.debug(
                        f"Model {model.__name__} has content field: {content_field_name}"
                    )
                else:
                    logger.debug(
                        f"Model {model.__name__} has no explicit content field"
                    )

            # Store structured model
            self.structured_models[model_name] = model

            # Track model fields for reference
            if hasattr(model, "model_fields"):
                for field_name in model.model_fields:
                    self.structured_model_fields[model_name].add(field_name)

                logger.debug(
                    f"Added {len(model.model_fields)} fields to structured_model_fields for {model_name}"
                )

            # Add a single field for the entire model
            if (
                model_name not in self.fields
                and model_name not in self.base_class_fields
            ):
                from typing import Optional

                field_type = Optional[model]

                self.add_field(
                    name=model_name,
                    field_type=field_type,
                    default=None,
                    description=f"Output in {model.__name__} format",
                    source=f"{source}.structured_output_model",
                    output_from=[engine_name],
                )

                logger.debug(f"Added field '{model_name}' for structured output model")

        # 3. Use get_input_fields/get_output_fields methods if needed

        # 3.1 Extract input fields if needed
        if (
            not input_schema
            and hasattr(engine, "get_input_fields")
            and callable(engine.get_input_fields)
        ):
            try:
                input_fields = engine.get_input_fields()
                logger.debug(
                    f"Got {len(input_fields)} input fields from get_input_fields()"
                )

                for field_name, (field_type, field_info) in input_fields.items():
                    # Skip if already has this field or in base class
                    if (
                        field_name in self.fields
                        or field_name in self.base_class_fields
                    ):
                        # Just mark as input if it exists
                        self.input_fields[engine_name].add(field_name)
                        if (
                            field_name
                            not in self.engine_io_mappings[engine_name]["inputs"]
                        ):
                            self.engine_io_mappings[engine_name]["inputs"].append(
                                field_name
                            )
                        logger.debug(
                            f"Field '{field_name}' already exists, marked as input for '{engine_name}'"
                        )
                        continue

                    # Skip special fields
                    if (
                        field_name == "__runnable_config__"
                        or field_name == "runnable_config"
                    ):
                        logger.warning(f"Skipping special field {field_name}")
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
                        input_for=[engine_name],
                    )

                    logger.debug(
                        f"Added input field '{field_name}' from engine '{engine_name}'"
                    )

            except Exception as e:
                logger.warning(
                    f"Error getting input_fields from {engine_name}: {e}", exc_info=True
                )

        # 3.2 Extract output fields if needed
        if (
            not output_schema
            and hasattr(engine, "get_output_fields")
            and callable(engine.get_output_fields)
        ):
            try:
                # Skip extracting output fields if content field exists in structured model
                if not content_field_exists:
                    output_fields = engine.get_output_fields()
                    logger.debug(
                        f"Got {len(output_fields)} output fields from get_output_fields()"
                    )

                    for field_name, (field_type, field_info) in output_fields.items():
                        # Skip if already has this field or in base class
                        if (
                            field_name in self.fields
                            or field_name in self.base_class_fields
                        ):
                            # Just mark as output if it exists
                            self.output_fields[engine_name].add(field_name)
                            if (
                                field_name
                                not in self.engine_io_mappings[engine_name]["outputs"]
                            ):
                                self.engine_io_mappings[engine_name]["outputs"].append(
                                    field_name
                                )
                            logger.debug(
                                f"Field '{field_name}' already exists, marked as output for '{engine_name}'"
                            )
                            continue

                        # Skip special fields
                        if (
                            field_name == "__runnable_config__"
                            or field_name == "runnable_config"
                        ):
                            logger.warning(f"Skipping special field {field_name}")
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
                            output_from=[engine_name],
                        )

                        logger.debug(
                            f"Added output field '{field_name}' from engine '{engine_name}'"
                        )
                else:
                    logger.debug(
                        f"Skipping get_output_fields for {engine_name} - structured model has content field"
                    )
            except Exception as e:
                logger.warning(
                    f"Error getting output_fields from {engine_name}: {e}",
                    exc_info=True,
                )

        # 4. Check for tools
        if hasattr(engine, "tools") and engine.tools:
            tools = engine.tools
            logger.debug(f"Engine {engine_name} has {len(tools)} tools")

            # Extract tool schemas
            self.extract_tool_schemas(tools)

            # Mark that we found tools
            self.has_tools = True

            # Add tools field if not present and not in base class
            if "tools" not in self.fields and "tools" not in self.base_class_fields:
                from typing import List

                self.add_field(
                    name="tools",
                    field_type=List[Any],
                    default_factory=list,
                    description="Tools for this engine",
                    source=source,
                )

                logger.debug(f"Added 'tools' field for engine '{engine_name}'")

        # Update engine IO mapping
        if self.input_fields[engine_name]:
            self.engine_io_mappings[engine_name]["inputs"] = list(
                self.input_fields[engine_name]
            )

        if self.output_fields[engine_name]:
            self.engine_io_mappings[engine_name]["outputs"] = list(
                self.output_fields[engine_name]
            )

        logger.debug(
            f"Engine '{engine_name}' IO mappings: {len(self.input_fields[engine_name])} inputs, {len(self.output_fields[engine_name])} outputs"
        )

        return self

    def extract_tool_schemas(self, tools: List[Any]) -> None:
        """
        Extract input and output schemas from tools with improved parsing detection.

        Args:
            tools: List of tools to analyze
        """
        logger.debug(f"Extracting schemas from {len(tools)} tools")

        # Track tools with structured schemas
        tools_with_parse_output = []

        for tool in tools:
            # Get tool name
            tool_name = getattr(tool, "name", None)
            if not tool_name and hasattr(tool, "__name__"):
                tool_name = tool.__name__
            if not tool_name:
                tool_name = f"tool_{id(tool)}"

            logger.debug(f"Processing tool: {tool_name}")

            # Check for structured_output_model - tools with this go to parse_output
            if (
                hasattr(tool, "structured_output_model")
                and tool.structured_output_model
            ):
                tools_with_parse_output.append(tool_name)
                logger.debug(
                    f"Tool {tool_name} has structured_output_model - marked for parse_output"
                )

            # Extract input schema
            input_schema = None

            # Check for args_schema on instance or class
            if hasattr(tool, "args_schema"):
                input_schema = tool.args_schema
                logger.debug(f"Using args_schema from tool {tool_name}")

            # For class types, try to instantiate
            elif isinstance(tool, type):
                if issubclass(tool, BaseTool):
                    try:
                        instance = tool()
                        if hasattr(instance, "args_schema"):
                            input_schema = instance.args_schema
                            logger.debug(
                                f"Using args_schema from instantiated {tool_name}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not instantiate tool class {tool_name}: {e}"
                        )

            # For BaseModel types, use directly
            if isinstance(tool, type) and issubclass(tool, BaseModel):
                input_schema = tool
                logger.debug(f"Using tool class as schema for {tool_name}")

            # Add input schema field if found
            if (
                input_schema
                and isinstance(input_schema, type)
                and issubclass(input_schema, BaseModel)
            ):
                # Create field path for tools dictionary
                field_name = f"tool_schemas.{tool_name}"
                logger.debug(f"Adding schema for tool {tool_name}")

                # Add to tool_schemas dictionary in state
                self.add_field(
                    name=field_name,
                    field_type=Type[BaseModel],
                    default=input_schema,
                    description=f"Schema for {tool_name}",
                )

                # Try to find matching output schema
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
                                    output_field_name = f"output_schemas.{name}"
                                    self.add_field(
                                        name=output_field_name,
                                        field_type=Type[BaseModel],
                                        default=output_class,
                                        description=f"Output schema for {tool_name}",
                                    )

                                    # Mark this tool for parse_output
                                    if tool_name not in tools_with_parse_output:
                                        tools_with_parse_output.append(tool_name)
                                        logger.debug(
                                            f"Tool {tool_name} has output schema - marked for parse_output"
                                        )

                                    # Add tool_name attribute to schema
                                    output_class.tool_name = tool_name
                                    logger.debug(
                                        f"Found output schema {name} for {tool_name}"
                                    )
                                    break

        # Add tool type fields
        logger.debug("Adding tool_types_dict field")
        self.add_field(
            name="tool_types_dict",
            field_type=Dict[str, str],
            default_factory=dict,
            description="Dictionary mapping tool names to their routing destinations",
        )

        # If we have tools marked for parse_output, add them to the schema
        if tools_with_parse_output:
            # Initialize tool_types_dict
            for tool_name in tools_with_parse_output:
                # Create routing entries
                logger.debug(f"Setting routing for {tool_name} -> parse_output")

                # Use this for demo purposes, it will be rebuilt at runtime in ToolState
                # but this gives visibility into what will happen
                if "tool_types_dict" in self.fields:
                    tool_field = self.fields["tool_types_dict"]
                    if tool_field.default is None:
                        tool_field.default = {}

                    tool_field.default[tool_name] = "parse_output"

        # Add tools field if not present and not in base class
        if "tools" not in self.fields and "tools" not in self.base_class_fields:
            self.add_field(
                name="tools",
                field_type=List[Any],
                default_factory=list,
                description="Tool instances indexed by name",
            )
            self.has_tools = True

        # Add tool_schemas field if not present
        if not any(name.startswith("tool_schemas.") for name in self.fields):
            self.add_field(
                name="tool_schemas",
                field_type=Dict[str, Type[BaseModel]],
                default_factory=dict,
                description="Tool schemas indexed by name",
            )

        # Add output_schemas field if not present
        if not any(name.startswith("output_schemas.") for name in self.fields):
            self.add_field(
                name="output_schemas",
                field_type=Dict[str, Type[BaseModel]],
                default_factory=dict,
                description="Output schemas indexed by name",
            )

    def add_fields_from_components(self, components: List[Any]) -> "SchemaComposer":
        """
        Add fields from multiple components.

        This method processes a list of components, extracting fields from each based on
        its type (Engine, BaseModel, dict, etc).

        Args:
            components: List of components to extract fields from

        Returns:
            Self for chaining
        """
        logger.debug(f"Extracting fields from {len(components)} components")

        # Detect base class requirements early
        self._detect_base_class_requirements(components)

        # Track number of fields before
        field_count_before = len(self.fields)

        # Process each component
        for i, component in enumerate(components):
            if component is None:
                continue

            component_type = component.__class__.__name__
            component_id = getattr(component, "name", f"component_{i}")

            logger.debug(f"Processing component {i}: {component_id} ({component_type})")

            # Add tracking entry
            self.processing_history.append(
                {
                    "action": "process_component",
                    "component_type": component_type,
                    "component_id": component_id,
                }
            )

            # Process based on type
            if hasattr(component, "engine_type"):
                # Looks like an Engine
                logger.debug(f"Component {component_id} is an Engine")
                self.add_fields_from_engine(component)

            elif isinstance(component, BaseModel):
                # BaseModel instance
                logger.debug(f"Component {component_id} is a BaseModel instance")
                self.add_fields_from_model(component.__class__)

            elif isinstance(component, type) and issubclass(component, BaseModel):
                # BaseModel class
                logger.debug(f"Component {component_id} is a BaseModel class")
                self.add_fields_from_model(component)

            elif isinstance(component, dict):
                # Dictionary
                logger.debug(f"Component {component_id} is a dictionary")
                self.add_fields_from_dict(component)

            else:
                logger.debug(f"Skipping unsupported component type: {component_type}")

        # Report results
        field_count_after = len(self.fields)
        field_count_added = field_count_after - field_count_before

        logger.debug(
            f"Added {field_count_added} fields from components (total: {field_count_after})"
        )

        # Always ensure we have 'messages' and 'runnable_config' if we found engines
        self._ensure_standard_fields()

        return self

    def _ensure_standard_fields(self) -> None:
        """
        Ensure standard fields are present if not already added.
        """
        # Ensure runnable_config if we found engines
        has_engines = any(
            entry.get("action") == "process_component"
            and hasattr(entry.get("component_type"), "engine_type")
            for entry in self.processing_history
        )

        if (
            has_engines
            and "runnable_config" not in self.fields
            and "runnable_config" not in self.base_class_fields
        ):
            from typing import Any, Dict, Optional

            self.add_field(
                name="runnable_config",
                field_type=Optional[Dict[str, Any]],
                default=None,
                description="Runtime configuration for engines",
                source="auto_added",
            )
            logger.debug("Added standard field 'runnable_config'")

        # Ensure we have messages field for chat agents
        engines_with_messages = [
            entry
            for entry in self.processing_history
            if entry.get("action") == "process_component"
            and entry.get("component_type") == "AugLLMConfig"
        ]

        if (
            engines_with_messages
            and "messages" not in self.fields
            and "messages" not in self.base_class_fields
        ):
            self.configure_messages_field(with_reducer=True, force_add=True)
            logger.debug("Added standard field 'messages' with reducer")

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
        logger.debug(
            f"Configuring messages field (with_reducer={with_reducer}, force_add={force_add})"
        )

        # Check if messages will be provided by base class
        if self.detected_base_class and "messages" in self.base_class_fields:
            logger.debug("Messages field will be provided by base class")
            # Still mark as shared if requested
            self.shared_fields.add("messages")
            return self

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
                            shared=True,  # Usually want messages to be shared
                        )
                        self.has_messages = True
                        logger.debug(
                            "Added 'messages' field with add_messages reducer and shared=True"
                        )

                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = add_messages
                        # Also mark as shared if not already
                        if "messages" not in self.shared_fields:
                            self.shared_fields.add("messages")
                            self.fields["messages"].shared = True
                            logger.debug(
                                "Added add_messages reducer to existing messages field and marked as shared"
                            )

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
                            shared=True,
                        )
                        self.has_messages = True
                        logger.debug(
                            "Added 'messages' field with concat_lists reducer (fallback) and shared=True"
                        )

                    # Otherwise, just set the reducer if the field exists
                    elif "messages" in self.fields:
                        self.fields["messages"].reducer = concat_lists
                        # Also mark as shared if not already
                        if "messages" not in self.shared_fields:
                            self.shared_fields.add("messages")
                            self.fields["messages"].shared = True
                            logger.debug(
                                "Added concat_lists reducer to existing messages field and marked as shared"
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
        # Check field exists
        if field_name not in self.fields and field_name not in self.base_class_fields:
            logger.warning(f"Cannot mark non-existent field '{field_name}' as input")
            return self

        logger.debug(
            f"Marking field '{field_name}' as input for engine '{engine_name}'"
        )

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
        # Check field exists
        if field_name not in self.fields and field_name not in self.base_class_fields:
            logger.warning(f"Cannot mark non-existent field '{field_name}' as output")
            return self

        logger.debug(
            f"Marking field '{field_name}' as output for engine '{engine_name}'"
        )

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
        Build a StateSchema with the right base class and all metadata.

        Returns:
            Subclass of StateSchema with appropriate base class and field attributes
        """
        # Make sure we've detected base class requirements
        if self.detected_base_class is None:
            self._detect_base_class_requirements()

        base_class = self.detected_base_class

        # Show what we're building
        logger.debug(
            f"Building {self.name} with {len(self.fields)} fields using base class {base_class.__name__}"
        )

        # Create field definitions for the model (excluding base class fields)
        field_defs = {}
        for name, field_def in self.fields.items():
            # Skip schema dictionary entries for now (handle nested dictionaries later)
            if "." in name:
                logger.debug(f"Skipping nested field {name} for now")
                continue

            # Skip fields that the base class provides
            if name in self.base_class_fields:
                logger.debug(f"Skipping field {name} - provided by base class")
                continue

            field_type, field_info = field_def.to_field_info()
            field_defs[name] = (field_type, field_info)

        # Create the base schema
        schema = create_model(self.name, __base__=base_class, **field_defs)

        # Copy attributes from base class if they exist
        if hasattr(base_class, "__shared_fields__"):
            # Merge with our shared fields
            base_shared = set(base_class.__shared_fields__)
            schema.__shared_fields__ = list(base_shared | self.shared_fields)
        else:
            schema.__shared_fields__ = list(self.shared_fields)

        logger.debug(f"Shared fields: {schema.__shared_fields__}")

        # Handle reducers - merge base class reducers with ours
        schema.__serializable_reducers__ = {}
        schema.__reducer_fields__ = {}

        # Copy base class reducers first
        if hasattr(base_class, "__serializable_reducers__"):
            schema.__serializable_reducers__.update(
                base_class.__serializable_reducers__
            )
        if hasattr(base_class, "__reducer_fields__"):
            schema.__reducer_fields__.update(base_class.__reducer_fields__)

        # Add our reducers (potentially overriding base class)
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

        # Store engines in the schema
        schema.__engines__ = self.engines
        schema.__engines_by_type__ = dict(self.engines_by_type)

        # Now handle nested fields like tool_schemas.xyz
        # We need to build nested dictionaries for these

        tool_schemas = {}
        output_schemas = {}

        for name, field_def in self.fields.items():
            # Look for dot notation
            if "." in name:
                parts = name.split(".", 1)

                if parts[0] == "tool_schemas":
                    logger.debug(f"Adding tool schema: {parts[1]}")
                    tool_schemas[parts[1]] = field_def.default

                elif parts[0] == "output_schemas":
                    logger.debug(f"Adding output schema: {parts[1]}")
                    output_schemas[parts[1]] = field_def.default

        # Create post_init method to handle nested fields and engine setup
        def schema_post_init(self, *args, **kwargs):
            # Call original post_init if it exists
            original_post_init = getattr(
                super(self.__class__, self), "model_post_init", None
            )
            if original_post_init:
                original_post_init(*args, **kwargs)

            # Initialize tool_schemas
            if hasattr(self, "tool_schemas") and tool_schemas:
                for name, schema in tool_schemas.items():
                    self.tool_schemas[name] = schema

            # Initialize output_schemas
            if hasattr(self, "output_schemas") and output_schemas:
                for name, schema in output_schemas.items():
                    self.output_schemas[name] = schema

            # Return self for chaining
            return self

        # Add post_init to the schema
        schema.model_post_init = schema_post_init

        # Print summary
        logger.debug(f"Created schema {schema.__name__} with {len(field_defs)} fields")
        logger.debug(f"Engine mappings: {len(schema.__engine_io_mappings__)} engines")
        if schema.__serializable_reducers__:
            logger.debug(
                f"Reducers: {len(schema.__serializable_reducers__)} fields have reducers"
            )

        # Create rich table for field display
        self._display_schema_summary(schema)

        return schema

    def _display_schema_summary(self, schema: Type[StateSchema]) -> None:
        """Display a visual summary of the created schema."""
        # Only display if debug logging is enabled
        if logger.level > logging.DEBUG:
            return

        # Create a table for field display
        table = Table(title=f"{schema.__name__} Schema Fields", show_header=True)
        table.add_column("Field Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Description", style="blue")
        table.add_column("Metadata", style="magenta")

        for field_name, field_info in schema.model_fields.items():
            if field_name.startswith("__"):
                continue

            field_type = str(field_info.annotation).replace("typing.", "")
            desc = field_info.description or ""

            # Build metadata string
            metadata = []
            if field_name in schema.__shared_fields__:
                metadata.append("[green]shared[/green]")
            if field_name in schema.__serializable_reducers__:
                metadata.append(
                    f"[yellow]reducer={schema.__serializable_reducers__[field_name]}[/yellow]"
                )

            # Check if field is input or output for any engine
            for engine_name, mapping in schema.__engine_io_mappings__.items():
                if field_name in mapping["inputs"]:
                    metadata.append(f"[cyan]input({engine_name})[/cyan]")
                if field_name in mapping["outputs"]:
                    metadata.append(f"[blue]output({engine_name})[/blue]")

            table.add_row(field_name, field_type, desc, ", ".join(metadata))

        console.print(table)

        if schema.__engine_io_mappings__:
            console.print("\n[bold green]Engine I/O Mappings:[/bold green]")
            for engine_name, mapping in schema.__engine_io_mappings__.items():
                inputs = ", ".join(mapping["inputs"]) if mapping["inputs"] else "-"
                outputs = ", ".join(mapping["outputs"]) if mapping["outputs"] else "-"
                console.print(f"  [bold]{engine_name}[/bold]:")
                console.print(f"    [cyan]Inputs[/cyan]: {inputs}")
                console.print(f"    [blue]Outputs[/blue]: {outputs}")

        # Display engines if any
        if hasattr(schema, "__engines__") and schema.__engines__:
            console.print("\n[bold yellow]Registered Engines:[/bold yellow]")
            engine_table = Table(show_header=True)
            engine_table.add_column("Name", style="cyan")
            engine_table.add_column("Type", style="yellow")
            engine_table.add_column("Config", style="green")

            for engine_name, engine in schema.__engines__.items():
                engine_type = getattr(engine, "engine_type", "unknown")
                if hasattr(engine_type, "value"):
                    engine_type = engine_type.value

                # Get key config info
                config_info = []
                if engine_type == "llm" and hasattr(engine, "llm_config"):
                    if hasattr(engine.llm_config, "model"):
                        config_info.append(f"model={engine.llm_config.model}")
                elif hasattr(engine, "model"):
                    config_info.append(f"model={engine.model}")

                engine_table.add_row(
                    engine_name,
                    str(engine_type),
                    ", ".join(config_info) if config_info else "-",
                )

            console.print(engine_table)

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
        logger.debug(f"Merging composers: {first.name} + {second.name} -> {name}")
        merged = cls(name=name)

        # Detect base class requirements from both composers
        if first.has_messages or second.has_messages:
            merged.has_messages = True
        if first.has_tools or second.has_tools:
            merged.has_tools = True
        merged._detect_base_class_requirements()

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

        # Merge engines
        merged.engines.update(first.engines)
        merged.engines.update(second.engines)

        # Merge engines by type
        for engine_type, engine_names in first.engines_by_type.items():
            merged.engines_by_type[engine_type].extend(engine_names)
        for engine_type, engine_names in second.engines_by_type.items():
            merged.engines_by_type[engine_type].extend(engine_names)

        # Remove duplicates from engines_by_type
        for engine_type in merged.engines_by_type:
            merged.engines_by_type[engine_type] = list(
                set(merged.engines_by_type[engine_type])
            )

        # Merge metadata
        merged.has_messages = first.has_messages or second.has_messages
        merged.has_tools = first.has_tools or second.has_tools

        # Merge tracking
        merged.processing_history = first.processing_history + second.processing_history

        logger.debug(f"Merged composers: {len(merged.fields)} total fields")

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
        logger.debug(f"Creating schema {name} from {len(components)} components")
        composer = cls(name=name)

        # Detect base class requirements early
        composer._detect_base_class_requirements(components)

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

        # Ensure standard fields if needed
        composer._ensure_standard_fields()

        # Build the schema
        schema = composer.build()

        return schema

    def show_engines(self) -> None:
        """Display a summary of all registered engines."""
        if not self.engines:
            console.print("[yellow]No engines registered[/yellow]")
            return

        tree = Tree("[bold green]Registered Engines[/bold green]")

        for engine_type, engine_names in self.engines_by_type.items():
            type_node = tree.add(
                f"[bold cyan]{engine_type}[/bold cyan] ({len(engine_names)} engines)"
            )

            for engine_name in engine_names:
                if engine_name in self.engines:
                    engine = self.engines[engine_name]

                    # Build info string
                    info_parts = []
                    if engine_type == "llm" and hasattr(engine, "llm_config"):
                        if hasattr(engine.llm_config, "model"):
                            info_parts.append(f"model={engine.llm_config.model}")
                        if hasattr(engine.llm_config, "temperature"):
                            info_parts.append(f"temp={engine.llm_config.temperature}")
                    elif hasattr(engine, "model"):
                        info_parts.append(f"model={engine.model}")

                    info_str = (
                        f" [dim]({', '.join(info_parts)})[/dim]" if info_parts else ""
                    )
                    type_node.add(f"[yellow]{engine_name}[/yellow]{info_str}")

        console.print(tree)
