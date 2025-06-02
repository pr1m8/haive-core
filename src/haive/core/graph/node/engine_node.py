# src/haive/core/graph/node/engine_node.py

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field, model_validator
from rich.console import Console

from haive.core.engine.base import Engine
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

logger = logging.getLogger(__name__)
console = Console()


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

    # Routing options
    use_send: bool = Field(
        default=False, description="Whether to use Send instead of Command for routing"
    )

    # Debug option
    debug: bool = Field(default=False, description="Enable debug output")

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

        return self

    def __call__(self, state: StateLike, config: Optional[ConfigLike] = None) -> Any:
        """Execute the engine with the given state and configuration."""
        # First extract inputs and execute the engine
        raw_result = self.execute_engine(state, config)

        # Then wrap the results in Command or Send
        return self.wrap_result(raw_result, state)

    def execute_engine(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Any:
        """Execute the engine and return raw result."""
        # Resolve the engine
        engine, engine_id = self.get_engine()
        if engine is None:
            raise ValueError(f"Could not resolve engine for node {self.name}")

        # Extract input based on mapping
        input_data = self.extract_input(state)

        if self.debug:
            console.print(
                f"[cyan]Node {self.name}[/] extracted input: {list(input_data.keys() if isinstance(input_data, dict) else ['<non-dict>'])}"
            )

        # Apply config overrides
        merged_config = self._merge_configs(config, engine_id)

        try:
            # Invoke the engine
            if self.debug:
                console.print(
                    f"[cyan]Node {self.name}[/] invoking engine {getattr(engine, 'name', 'unknown')}"
                )
            print(f"[cyan]input_data: {input_data}")
            if engine.input_schema:
                input_data = engine.input_schema.model_validate(input_data)
            result = engine.invoke(input_data, merged_config)

            if self.debug:
                result_type = type(result).__name__
                console.print(f"[green]Engine returned result type:[/] {result_type}")

            return result
        except Exception as e:
            # Handle errors according to retry policy
            logger.error(f"Error executing engine node {self.name}: {e}")
            raise

    def wrap_result(self, result: Any, state: StateLike) -> Union[Command, Send]:
        """Wrap the engine result in Command or Send."""
        # If result is already a Command or Send, return it
        if isinstance(result, Command) or isinstance(result, Send):
            return result

        # Get engine output fields from schema if available
        engine_name = getattr(self.engine, "name", None)
        engine_outputs = []

        if hasattr(state, "__engine_io_mappings__") and engine_name:
            mappings = getattr(state, "__engine_io_mappings__", {})
            if engine_name in mappings:
                engine_outputs = mappings[engine_name].get("outputs", [])
                if self.debug:
                    console.print(
                        f"[cyan]Found engine outputs in schema:[/] {engine_outputs}"
                    )

        # Prepare update dictionary based on output fields and schema
        update_dict = {}

        # Case 1: Message-like result for an LLM - add to messages list
        if self._is_message_result(result) and (
            "messages" in engine_outputs or not engine_outputs
        ):
            if hasattr(state, "messages"):
                messages = list(getattr(state, "messages", []))
                if isinstance(result, list):
                    messages.extend(result)
                else:
                    messages.append(result)
                update_dict["messages"] = messages
                if self.debug:
                    console.print("[green]Added message result to messages list[/]")

        # Case 2: Dictionary result - apply mapping or use schema outputs
        elif isinstance(result, dict):
            output_mapping = self.get_output_mapping()

            # If we have explicit output mapping, use it
            if output_mapping:
                for result_key, state_key in output_mapping.items():
                    if result_key in result:
                        update_dict[state_key] = result[result_key]
                        if self.debug:
                            console.print(
                                f"[green]Mapped output:[/] {result_key} → {state_key}"
                            )

            # If engine outputs from schema, use those (if not already mapped)
            elif engine_outputs:
                for output_field in engine_outputs:
                    if output_field in result:
                        update_dict[output_field] = result[output_field]
                        if self.debug:
                            console.print(
                                f"[green]Used schema output field:[/] {output_field}"
                            )

            # If nothing mapped, use the full result
            if not update_dict:
                update_dict = result
                if self.debug:
                    console.print(
                        "[yellow]No mapping found - using full result dict[/]"
                    )

        # Case 3: Other result types - map to default or schema-defined field
        else:
            # Use first schema output field if available
            if engine_outputs:
                update_dict[engine_outputs[0]] = result
                if self.debug:
                    console.print(
                        f"[green]Mapped result to schema output:[/] {engine_outputs[0]}"
                    )

            # Otherwise use result field
            else:
                update_dict["result"] = result
                if self.debug:
                    console.print("[yellow]Using default 'result' field[/]")

        # Create Command or Send with the update dictionary
        if self.use_send:
            return Send(node=self.command_goto, arg=update_dict)
        else:
            return Command(update=update_dict, goto=self.command_goto)

    def _is_message_result(self, result: Any) -> bool:
        """Check if result is a message type that should be added to messages."""
        # Import here to avoid circular imports
        try:
            from langchain_core.messages import AIMessage, BaseMessage

            return isinstance(result, (AIMessage, BaseMessage))
        except ImportError:
            # If langchain isn't available, use duck typing
            return hasattr(result, "content") and (
                hasattr(result, "type") or hasattr(result, "message_type")
            )

    def extract_input(self, state: StateLike) -> Any:
        """Extract input for the engine from state."""
        # Check for engine I/O mappings in schema
        engine_name = getattr(self.engine, "name", None)
        engine_inputs = []

        if hasattr(state, "__engine_io_mappings__") and engine_name:
            mappings = getattr(state, "__engine_io_mappings__", {})
            if engine_name in mappings:
                engine_inputs = mappings[engine_name].get("inputs", [])
                if self.debug:
                    console.print(
                        f"[cyan]Found engine inputs in schema:[/] {engine_inputs}"
                    )

        # Use input_fields mapping if available
        input_mapping = self.get_input_mapping()

        # Prioritize explicitly configured input mapping
        if input_mapping:
            return self._extract_with_mapping(state, input_mapping)

        # If we have engine inputs from schema, use those
        elif engine_inputs:
            # Create input dictionary with just those fields
            input_data = {}
            for field in engine_inputs:
                if hasattr(state, field):
                    input_data[field] = getattr(state, field)
                elif isinstance(state, dict) and field in state:
                    input_data[field] = state[field]
            return input_data

        # Special case for messages - most engines expect this
        elif hasattr(state, "messages"):
            return {"messages": state.messages}
        elif isinstance(state, dict) and "messages" in state:
            return {"messages": state["messages"]}

        # Last resort - return state as is
        if hasattr(state, "__dict__"):
            return state.__dict__
        return state

    def _extract_with_mapping(
        self, state: Any, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract input fields using a mapping."""
        input_data = {}

        for state_key, input_key in mapping.items():
            # Try attribute access first
            if hasattr(state, state_key):
                input_data[input_key] = getattr(state, state_key)
            # Then try dictionary access
            elif isinstance(state, dict) and state_key in state:
                input_data[input_key] = state[state_key]
            # Log missing fields in debug mode
            elif self.debug:
                logger.debug(f"Missing input field {state_key} in state")

        return input_data

    def get_engine(self) -> Tuple[Optional[Engine], Optional[str]]:
        """Get the engine for this node, resolving from registry if needed."""
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
                )  # src/haive/core/graph/node/engine_node.py


import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field
from rich.console import Console

from haive.core.engine.base import Engine, EngineType
from haive.core.graph.common.types import ConfigLike, StateLike
from haive.core.graph.node.base_config import NodeConfig
from haive.core.graph.node.types import NodeType

logger = logging.getLogger(__name__)
console = Console()


class EngineNodeConfig(NodeConfig):
    """Elegant engine-based node with intelligent I/O handling."""

    # Core identity
    node_type: NodeType = Field(default=NodeType.ENGINE)
    engine: Optional[Engine] = Field(default=None)
    engine_name: Optional[str] = Field(default=None)

    # Field mappings (auto-normalized)
    input_fields: Optional[Union[List[str], Dict[str, str]]] = Field(default=None)
    output_fields: Optional[Union[List[str], Dict[str, str]]] = Field(default=None)

    # Options
    retry_policy: Optional[RetryPolicy] = Field(default=None)
    use_send: bool = Field(default=False)
    debug: bool = Field(default=False)

    def __call__(
        self, state: StateLike, config: Optional[ConfigLike] = None
    ) -> Union[Command, Send]:
        """Execute engine node with intelligent I/O handling."""
        try:
            # Get engine and validate
            engine = self._get_engine()
            if not engine:
                raise ValueError(f"No engine available for node '{self.name}'")

            # Extract input intelligently
            input_data = self._extract_smart_input(state, engine)

            # Execute with merged config
            result = self._execute_with_config(engine, input_data, config)

            # Wrap result intelligently
            return self._wrap_smart_result(result, state, engine)

        except Exception as e:
            self._debug_error(e)
            raise

    def _get_engine(self) -> Optional[Engine]:
        """Get engine from direct reference or registry."""
        if self.engine:
            return self.engine

        if self.engine_name:
            engine = self._lookup_engine(self.engine_name)
            if engine:
                self.engine = engine  # Cache it
                return engine

        return None

    def _extract_smart_input(self, state: StateLike, engine: Engine) -> Any:
        """Extract input using the most appropriate strategy."""
        # Strategy 1: Explicit mapping
        if self.input_fields:
            return self._extract_mapped_input(
                state, self._normalize_mapping(self.input_fields)
            )

        # Strategy 2: Schema-defined inputs
        schema_inputs = self._get_schema_inputs(state, engine.name)
        if schema_inputs:
            return self._extract_typed_input(state, schema_inputs, engine.engine_type)

        # Strategy 3: Engine-defined inputs
        engine_inputs = self._get_engine_inputs(engine)
        if engine_inputs:
            return self._extract_typed_input(state, engine_inputs, engine.engine_type)

        # Strategy 4: Type-based defaults
        return self._extract_default_input(state, engine.engine_type)

    def _extract_typed_input(
        self, state: StateLike, fields: List[str], engine_type: EngineType
    ) -> Dict[str, Any]:
        """Extract fields with type-specific intelligence."""
        extractors = {
            EngineType.RETRIEVER: self._extract_retriever_fields,
            EngineType.LLM: self._extract_llm_fields,
            EngineType.VECTOR_STORE: self._extract_vectorstore_fields,
            EngineType.EMBEDDINGS: self._extract_embeddings_fields,
        }

        extractor = extractors.get(engine_type, self._extract_generic_fields)
        return extractor(state, fields)

    def _extract_retriever_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Retriever-specific extraction: always include query, filter None values."""
        input_data = {}

        for field in fields:
            value = self._get_state_value(state, field)

            if field == "query":
                # Always include query, even if empty

                input_data[field] = value or ""
                self._debug(f"Retriever query: '{value or ''}'")
            elif value is not None:
                # Only include other fields if they have values
                input_data[field] = value
                self._debug(f"Retriever {field}: {value}")
            else:
                self._debug(f"Skipping None value for {field}")

        return input_data

    def _extract_llm_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """LLM-specific extraction: include all fields."""
        return {field: self._get_state_value(state, field) for field in fields}

    def _extract_vectorstore_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Vector store extraction: filter None values except query."""
        input_data = {}
        for field in fields:
            value = self._get_state_value(state, field)
            if field == "query" or value is not None:
                input_data[field] = value
        return input_data

    def _extract_embeddings_fields(self, state: StateLike, fields: List[str]) -> Any:
        """Embeddings extraction: often just needs text."""
        # Try to get text/query field first
        for field in ["query", "text", "content"]:
            if field in fields:
                value = self._get_state_value(state, field)
                if value:
                    return value

        # Fall back to all fields as dict
        return {field: self._get_state_value(state, field) for field in fields}

    def _extract_generic_fields(
        self, state: StateLike, fields: List[str]
    ) -> Dict[str, Any]:
        """Generic extraction: include non-None values."""
        return {
            field: value
            for field in fields
            if (value := self._get_state_value(state, field)) is not None
        }

    def _extract_default_input(self, state: StateLike, engine_type: EngineType) -> Any:
        """Default extraction when no fields are specified."""
        defaults = {
            EngineType.RETRIEVER: lambda: self._extract_retriever_fields(
                state, ["query", "k", "filter", "search_type", "score_threshold"]
            ),
            EngineType.LLM: lambda: {
                "messages": self._get_state_value(state, "messages", [])
            },
            EngineType.VECTOR_STORE: lambda: self._extract_vectorstore_fields(
                state, ["query", "k", "filter"]
            ),
            EngineType.EMBEDDINGS: lambda: self._get_state_value(state, "query", ""),
        }

        extractor = defaults.get(engine_type, lambda: self._state_as_dict(state))
        return extractor()

    def _wrap_smart_result(
        self, result: Any, state: StateLike, engine: Engine
    ) -> Union[Command, Send]:
        """Intelligently wrap result based on type and configuration."""
        # Already wrapped? Return as-is
        if isinstance(result, (Command, Send)):
            return result

        # Generate update dictionary
        update = self._create_update_dict(result, state, engine)

        # Return appropriate wrapper
        if self.use_send and self.command_goto:
            return Send(node=self.command_goto, arg=update)
        else:
            return Command(update=update, goto=self.command_goto)

    def _create_update_dict(
        self, result: Any, state: StateLike, engine: Engine
    ) -> Dict[str, Any]:
        """Create state update dictionary from result."""
        # Strategy 1: Explicit output mapping
        if self.output_fields:
            return self._apply_output_mapping(result)

        # Strategy 2: Schema-defined outputs
        schema_outputs = self._get_schema_outputs(state, engine.name)
        if schema_outputs:
            return self._map_to_outputs(result, schema_outputs)

        # Strategy 3: Smart type-based mapping
        return self._smart_result_mapping(result, state, engine.engine_type)

    def _smart_result_mapping(
        self, result: Any, state: StateLike, engine_type: EngineType
    ) -> Dict[str, Any]:
        """Smart result mapping based on result type and engine type."""
        # Message results go to messages
        if self._is_message_like(result):
            return self._update_messages(result, state)

        # Dictionary results
        if isinstance(result, dict):
            return result

        # Engine-specific single value mapping
        field_map = {
            EngineType.RETRIEVER: "documents",
            EngineType.LLM: "response",
            EngineType.EMBEDDINGS: "embeddings",
            EngineType.VECTOR_STORE: "documents",
        }

        field = field_map.get(engine_type, "result")
        return {field: result}

    def _update_messages(self, result: Any, state: StateLike) -> Dict[str, Any]:
        """Update messages list with new message(s)."""
        existing = self._get_state_value(state, "messages", [])
        messages = list(existing) if existing else []

        if isinstance(result, list):
            messages.extend(result)
        else:
            messages.append(result)

        return {"messages": messages}

    # Utility methods
    def _get_state_value(self, state: StateLike, key: str, default: Any = None) -> Any:
        """Get value from state with fallback."""
        if hasattr(state, key):
            return getattr(state, key)
        elif isinstance(state, dict):
            return state.get(key, default)
        return default

    def _get_schema_inputs(
        self, state: StateLike, engine_name: str
    ) -> Optional[List[str]]:
        """Get engine inputs from schema."""
        if not hasattr(state, "__engine_io_mappings__") or not engine_name:
            return None
        return (
            getattr(state, "__engine_io_mappings__", {})
            .get(engine_name, {})
            .get("inputs")
        )

    def _get_schema_outputs(
        self, state: StateLike, engine_name: str
    ) -> Optional[List[str]]:
        """Get engine outputs from schema."""
        if not hasattr(state, "__engine_io_mappings__") or not engine_name:
            return None
        return (
            getattr(state, "__engine_io_mappings__", {})
            .get(engine_name, {})
            .get("outputs")
        )

    def _get_engine_inputs(self, engine: Engine) -> Optional[List[str]]:
        """Get input fields from engine definition."""
        if hasattr(engine, "get_input_fields"):
            return list(engine.get_input_fields().keys())
        return None

    def _extract_mapped_input(
        self, state: StateLike, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """Extract using explicit state->input mapping."""
        return {
            input_key: self._get_state_value(state, state_key)
            for state_key, input_key in mapping.items()
            if self._get_state_value(state, state_key) is not None
        }

    def _apply_output_mapping(self, result: Any) -> Dict[str, Any]:
        """Apply explicit output mapping."""
        mapping = self._normalize_mapping(self.output_fields)

        if isinstance(result, dict):
            return {
                state_key: result.get(result_key)
                for result_key, state_key in mapping.items()
                if result_key in result
            }
        else:
            # Single value to first mapped field
            first_state_key = next(iter(mapping.values()))
            return {first_state_key: result}

    def _map_to_outputs(self, result: Any, output_fields: List[str]) -> Dict[str, Any]:
        """Map result to schema output fields."""
        if isinstance(result, dict):
            return {
                field: result.get(field) for field in output_fields if field in result
            }
        else:
            return {output_fields[0]: result} if output_fields else {"result": result}

    def _execute_with_config(
        self, engine: Engine, input_data: Any, config: Optional[ConfigLike]
    ) -> Any:
        """Execute engine with merged configuration."""
        merged_config = self._build_merged_config(config, engine)

        # DEBUG: Show what we're about to invoke
        console.print("[bold red]DEBUG _execute_with_config:[/bold red]")
        console.print(
            f"  Engine: {engine.name} (type: {getattr(engine, 'engine_type', 'unknown')})"
        )
        console.print(f"  Input data type: {type(input_data)}")
        console.print(f"  Input data: {input_data}")
        console.print(f"  Config: {merged_config}")

        # Special handling for retrievers - they need string queries
        if hasattr(engine, "engine_type") and engine.engine_type.value == "retriever":
            console.print("[yellow]RETRIEVER DETECTED - Special handling[/yellow]")

            if isinstance(input_data, dict):
                if "query" in input_data:
                    query_str = str(input_data["query"])
                    console.print(f"  Extracting query string: '{query_str}'")
                    console.print(
                        f"  Other params: {[k for k in input_data.keys() if k != 'query']}"
                    )
                    return engine.invoke(query_str, merged_config)
                else:
                    console.print(
                        "  No 'query' key in dict, using whole dict as string"
                    )
                    return engine.invoke(str(input_data), merged_config)
            else:
                console.print(
                    f"  Input is not dict, converting to string: '{str(input_data)}'"
                )
                return engine.invoke(str(input_data), merged_config)

        console.print("[green]Standard engine invoke[/green]")
        return engine.invoke(input_data, merged_config)

    def _build_merged_config(
        self, runtime_config: Optional[ConfigLike], engine: Engine
    ) -> Optional[Dict[str, Any]]:
        """Build merged configuration."""
        if not runtime_config and not self.config_overrides:
            return None

        config = dict(runtime_config or {})

        # Ensure configurable section
        config.setdefault("configurable", {})

        # Apply node-level overrides
        config["configurable"].update(self.config_overrides)

        # Apply engine-specific overrides
        engine_id = getattr(engine, "id", None)
        if engine_id and self.config_overrides:
            config["configurable"].setdefault("engine_configs", {})
            config["configurable"]["engine_configs"].setdefault(engine_id, {})
            config["configurable"]["engine_configs"][engine_id].update(
                self.config_overrides
            )

        return config

    def _lookup_engine(self, name: str) -> Optional[Engine]:
        """Lookup engine in registry."""
        try:
            from haive.core.engine.base import EngineRegistry

            return EngineRegistry.get_instance().find(name)
        except ImportError:
            logger.warning(f"Could not import EngineRegistry for {name}")
            return None

    def _normalize_mapping(
        self, fields: Optional[Union[List[str], Dict[str, str]]]
    ) -> Dict[str, str]:
        """Normalize field mapping to dict."""
        if isinstance(fields, list):
            return {field: field for field in fields}
        return fields or {}

    def _state_as_dict(self, state: StateLike) -> Dict[str, Any]:
        """Convert state to dictionary."""
        if isinstance(state, dict):
            return state
        elif hasattr(state, "model_dump"):
            return state.model_dump()
        elif hasattr(state, "__dict__"):
            return state.__dict__
        else:
            return {"value": state}

    def _is_message_like(self, obj: Any) -> bool:
        """Check if object is message-like."""
        try:
            from langchain_core.messages import BaseMessage

            return isinstance(obj, BaseMessage)
        except ImportError:
            return hasattr(obj, "content") and hasattr(obj, "type")

    def _debug(self, msg: str) -> None:
        """Debug output if enabled."""
        if self.debug:
            console.print(f"[dim]{self.name}:[/] {msg}")

    def _debug_error(self, error: Exception) -> None:
        """Debug error output."""
        if self.debug:
            console.print(f"[red]Error in {self.name}:[/] {error}")
        logger.error(f"Engine node '{self.name}' failed: {error}")

    def __repr__(self) -> str:
        """Clean string representation."""
        engine_ref = self.engine.name if self.engine else self.engine_name or "None"
        return f"EngineNode(name='{self.name}', engine='{engine_ref}')"
        return merged
