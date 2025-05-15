import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langgraph.types import Command, RetryPolicy
from pydantic import BaseModel, Field, model_validator

from haive.core.engine.base import Engine
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

logger = logging.getLogger(__name__)


class EngineNodeConfig(NodeConfig):
    """
    Configuration for an engine-based node.
    """

    # Override node_type for this specific node type
    node_type: NodeType = Field(
        default=NodeType.ENGINE, description="Type of node (always ENGINE)"
    )

    # Engine reference
    engine: Optional[Engine] = Field(
        default=None, description="Engine instance to use for this node"
    )
    engine_name: Optional[str] = Field(
        default=None, description="Name of engine to look up in registry"
    )

    # Schema integration
    state_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="State schema class for this node"
    )
    input_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Input schema for this node"
    )
    output_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Output schema for this node"
    )

    # Input/Output mapping
    input_fields: Optional[Union[List[str], Dict[str, str]]] = Field(
        default=None,
        description="List of input fields or mapping from state keys to node input keys",
    )
    output_fields: Optional[Union[List[str], Dict[str, str]]] = Field(
        default=None,
        description="List of output fields or mapping from node output keys to state keys",
    )

    # Execution options
    retry_policy: Optional[RetryPolicy] = Field(
        default=None, description="Retry policy for node execution"
    )

    @model_validator(mode="after")
    def validate_engine_node(self) -> "EngineNodeConfig":
        """Ensure engine is specified properly."""
        if self.engine is None and self.engine_name is None:
            raise ValueError(
                "Either engine or engine_name must be specified for EngineNodeConfig"
            )

        # Normalize input/output fields to dictionaries
        if isinstance(self.input_fields, list):
            self.input_fields = {field: field for field in self.input_fields}

        if isinstance(self.output_fields, list):
            self.output_fields = {field: field for field in self.output_fields}

        # Apply auto schema generation if possible
        if self.engine and not self.state_schema:
            self._auto_generate_schema()

        # Ensure node_type is ENGINE
        self.node_type = NodeType.ENGINE

        return self

    def _auto_generate_schema(self) -> None:
        """Auto-generate schema from engine if available."""
        try:
            from haive.core.schema.schema_composer import SchemaComposer

            schema = SchemaComposer.from_components([self.engine])
            self.state_schema = schema.build()

            # Also derive input/output schemas if not set
            if not self.input_schema:
                self.input_schema = schema.create_input_schema()
            if not self.output_schema:
                self.output_schema = schema.create_output_schema()

            self._extract_io_mappings()

        except Exception as e:
            logger.warning(f"Could not auto-generate schema from engine: {e}")

    def _extract_io_mappings(self) -> None:
        """Extract input/output mappings from engine I/O schema."""
        if self.engine and not self.input_fields and not self.output_fields:
            try:
                engine_name = getattr(self.engine, "name", "default")

                # Check if state schema has engine I/O mappings
                if hasattr(self.state_schema, "__engine_io_mappings__"):
                    io_mappings = getattr(
                        self.state_schema, "__engine_io_mappings__", {}
                    )
                    if engine_name in io_mappings:
                        mapping = io_mappings[engine_name]

                        # Extract input fields
                        if "inputs" in mapping and not self.input_fields:
                            input_fields = mapping["inputs"]
                            # Create identity mapping
                            self.input_fields = {field: field for field in input_fields}

                        # Extract output fields
                        if "outputs" in mapping and not self.output_fields:
                            output_fields = mapping["outputs"]
                            # Create identity mapping
                            self.output_fields = {
                                field: field for field in output_fields
                            }
            except Exception as e:
                logger.warning(f"Could not extract I/O mappings from schema: {e}")

    def get_engine(self) -> Tuple[Optional[Engine], Optional[str]]:
        """
        Get the engine for this node, resolving from registry if needed.

        Returns:
            Tuple of (engine_instance, engine_id)
        """
        # Return direct engine if already set
        if self.engine is not None:
            engine_id = getattr(self.engine, "id", None)
            return self.engine, engine_id

        # Lookup engine by name in registry
        try:
            from haive.core.engine.base import EngineRegistry

            registry = EngineRegistry.get_instance()

            engine = registry.find(self.engine_name)
            if engine:
                engine_id = getattr(engine, "id", None)
                self.engine = engine  # Cache for future use
                return engine, engine_id
        except ImportError:
            logger.warning(
                f"Could not import EngineRegistry to resolve engine: {self.engine_name}"
            )

        # Not found - return None
        return None, None

    def get_input_mapping(self) -> Dict[str, str]:
        """Get the input mapping for this node."""
        if self.input_fields:
            return dict(self.input_fields)
        return {}

    def get_output_mapping(self) -> Dict[str, str]:
        """Get the output mapping for this node."""
        if self.output_fields:
            return dict(self.output_fields)
        return {}

    def __call__(self, state: StateLike, config: Optional[ConfigLike] = None) -> Any:
        """
        Execute the engine with the given state and configuration.
        """
        # Resolve the engine
        engine, engine_id = self.get_engine()
        if engine is None:
            raise ValueError(f"Could not resolve engine for node {self.name}")

        # Extract input based on mapping
        input_mapping = self.get_input_mapping()
        input_data = self._extract_input(state, input_mapping)

        # Apply config overrides
        merged_config = self._merge_configs(config, engine_id)

        try:
            # Invoke the engine
            result = engine.invoke(input_data, merged_config)

            # Process output
            return self._process_output(result)
        except Exception as e:
            # Handle errors according to retry policy
            logger.error(f"Error executing engine node {self.name}: {e}")
            raise

    def _extract_input(
        self, state: Dict[str, Any], input_mapping: Dict[str, str]
    ) -> Any:
        """Extract input for the engine from state based on mapping."""
        if not input_mapping:
            # If no mapping, pass through the entire state
            return state

        # Apply mapping
        mapped_input = {}
        for state_key, input_key in input_mapping.items():
            if state_key in state:
                mapped_input[input_key] = state[state_key]

        # If only one field is mapped, return the value directly
        if len(input_mapping) == 1:
            return next(iter(mapped_input.values()), None)

        return mapped_input

    def _process_output(self, result: Any) -> Any:
        """Process the output from the engine."""
        # If result is already a Command, just return it
        if isinstance(result, Command):
            # If goto not specified in result but is in config, add it
            if result.goto is None and self.command_goto is not None:
                return Command(
                    update=result.update,
                    goto=self.command_goto,
                    resume=result.resume,
                    graph=result.graph,
                )
            return result

        # If command_goto is specified, wrap in Command
        if self.command_goto is not None:
            # Apply output mapping if needed
            if self.output_fields:
                # Convert result to appropriate format based on output mapping
                mapped_output = self._apply_output_mapping(result)
                return Command(update=mapped_output, goto=self.command_goto)

            # Otherwise use result directly
            return Command(update=result, goto=self.command_goto)

        # No command_goto, return result directly
        if self.output_fields:
            return self._apply_output_mapping(result)

        return result

    def _apply_output_mapping(self, result: Any) -> Dict[str, Any]:
        """Apply output mapping to result."""
        output_mapping = self.get_output_mapping()

        # Handle non-dict results
        if not isinstance(result, dict):
            # If we have exactly one output mapping, use it
            if len(output_mapping) == 1:
                return {next(iter(output_mapping.values())): result}
            # Otherwise put in a default field
            return {"result": result}

        # Apply mapping for dict results
        mapped_output = {}
        for engine_key, state_key in output_mapping.items():
            if engine_key in result:
                mapped_output[state_key] = result[engine_key]

        # If no mappings were applied, use the result as is
        return mapped_output if mapped_output else result

    def _merge_configs(
        self,
        runtime_config: Optional[Dict[str, Any]] = None,
        engine_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Merge base config overrides with runtime config."""
        if not runtime_config and not self.config_overrides:
            return None

        # Start with a copy of runtime_config or empty dict
        merged = dict(runtime_config or {})

        # Add configurable section if not present
        if "configurable" not in merged:
            merged["configurable"] = {}

        # Apply config_overrides
        for key, value in self.config_overrides.items():
            merged["configurable"][key] = value

        # Add engine_id targeting if available
        if engine_id:
            if "engine_configs" not in merged["configurable"]:
                merged["configurable"]["engine_configs"] = {}

            # Ensure this engine's config exists
            if engine_id not in merged["configurable"]["engine_configs"]:
                merged["configurable"]["engine_configs"][engine_id] = {}

            # Apply any engine-specific overrides
            if self.config_overrides:
                merged["configurable"]["engine_configs"][engine_id].update(
                    self.config_overrides
                )

        return merged
