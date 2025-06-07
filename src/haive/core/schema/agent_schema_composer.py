# haive/core/schema/agent_schema_composer.py
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from haive.agents.base.agent import Agent
from langchain_core.messages import BaseMessage

from haive.core.schema.schema_composer import SchemaComposer
from haive.core.schema.state_schema import StateSchema


class AgentSchemaComposer(SchemaComposer):
    """Enhanced schema composer that understands agents."""

    @classmethod
    def from_agents(
        cls,
        agents: List[Agent],
        name: Optional[str] = None,
        include_meta: bool = None,  # Auto-detect if None
        separation: str = "smart",  # "smart", "shared", "namespaced"
    ) -> Type[StateSchema]:
        """Compose schema from agents with smart defaults."""

        # Auto-detect meta state need
        if include_meta is None:
            include_meta = any(hasattr(a, "meta_state") for a in agents)

        composer = cls(name=name or cls._generate_name(agents))

        # Collect all fields from all agents
        all_fields = cls._collect_all_fields(agents)

        # Apply separation strategy
        if separation == "smart":
            cls._apply_smart_separation(composer, all_fields, agents)
        elif separation == "shared":
            cls._apply_shared_separation(composer, all_fields)
        elif separation == "namespaced":
            cls._apply_namespaced_separation(composer, all_fields, agents)

        # Add implicit fields
        cls._add_implicit_fields(composer, agents, include_meta)

        return composer.build()

    @staticmethod
    def _collect_all_fields(
        agents: List[Agent],
    ) -> Dict[str, List[Tuple[str, str, Type, Any]]]:
        """Collect all fields from all agents. Returns {field_name: [(agent_id, field_name, type, info)]}"""
        fields = {}

        for agent in agents:
            agent_id = agent.id  # Use ID instead of object for hashability

            # From state schema
            if agent.state_schema and hasattr(agent.state_schema, "model_fields"):
                for fname, finfo in agent.state_schema.model_fields.items():
                    if fname not in fields:
                        fields[fname] = []
                    fields[fname].append((agent_id, fname, finfo.annotation, finfo))

            # From engines - using get_input_fields and get_output_fields
            for engine in agent.engines.values():
                # Get input fields
                if hasattr(engine, "get_input_fields"):
                    try:
                        input_fields = engine.get_input_fields()
                        for fname, (ftype, fdefault) in input_fields.items():
                            if fname not in fields:
                                fields[fname] = []
                            # Create a mock field info
                            field_info = type(
                                "FieldInfo",
                                (),
                                {"annotation": ftype, "default": fdefault},
                            )()
                            fields[fname].append((agent_id, fname, ftype, field_info))
                    except:
                        pass

                # Get output fields
                if hasattr(engine, "get_output_fields"):
                    try:
                        output_fields = engine.get_output_fields()
                        for fname, (ftype, fdefault) in output_fields.items():
                            if fname not in fields:
                                fields[fname] = []
                            # Create a mock field info
                            field_info = type(
                                "FieldInfo",
                                (),
                                {"annotation": ftype, "default": fdefault},
                            )()
                            fields[fname].append((agent_id, fname, ftype, field_info))
                    except:
                        pass

        return fields

    @staticmethod
    def _apply_smart_separation(
        composer: SchemaComposer, all_fields: Dict, agents: List[Agent]
    ):
        """Apply smart separation logic."""

        # Create agent ID to name mapping
        agent_id_to_name = {agent.id: agent.name for agent in agents}

        for field_name, field_sources in all_fields.items():
            # Check if field appears in multiple agents (using agent IDs)
            unique_agent_ids = set(source[0] for source in field_sources)

            if len(unique_agent_ids) > 1:
                # Field used by multiple agents - share it
                # Use the first occurrence for type info
                _, _, ftype, finfo = field_sources[0]

                # Determine if it should be shared
                should_share = (
                    field_name in ["messages", "meta_state"]
                    or field_name.startswith("shared_")
                    or len(unique_agent_ids) == len(agents)  # All agents use it
                )

                # Handle field info properly
                if hasattr(finfo, "default"):
                    default = finfo.default
                elif hasattr(finfo, "default_factory"):
                    composer.add_field(
                        field_name,
                        ftype,
                        default_factory=finfo.default_factory,
                        shared=should_share,
                    )
                    continue
                else:
                    default = None

                composer.add_field(
                    field_name, ftype, default=default, shared=should_share
                )
            else:
                # Field used by single agent
                agent_id = field_sources[0][0]
                agent_name = agent_id_to_name.get(agent_id, agent_id)
                _, _, ftype, finfo = field_sources[0]

                # Could namespace it
                if len(agents) > 1:
                    namespaced_name = (
                        f"{agent_name.lower().replace(' ', '_')}_{field_name}"
                    )

                    # Handle field info properly
                    if hasattr(finfo, "default_factory"):
                        composer.add_field(
                            namespaced_name,
                            ftype,
                            default_factory=finfo.default_factory,
                            shared=False,
                        )
                    else:
                        composer.add_field(
                            namespaced_name,
                            ftype,
                            default=(
                                finfo.default if hasattr(finfo, "default") else None
                            ),
                            shared=False,
                        )
                else:
                    # Single agent, use as is
                    if hasattr(finfo, "default_factory"):
                        composer.add_field(
                            field_name, ftype, default_factory=finfo.default_factory
                        )
                    else:
                        composer.add_field(
                            field_name,
                            ftype,
                            default=(
                                finfo.default if hasattr(finfo, "default") else None
                            ),
                        )

    @staticmethod
    def _add_implicit_fields(
        composer: SchemaComposer, agents: List[Agent], include_meta: bool
    ):
        """Add fields that should be implicitly included."""

        # Check if field exists using has_field method or by checking the fields dict
        def field_exists(name: str) -> bool:
            if hasattr(composer, "has_field"):
                return composer.has_field(name)
            elif hasattr(composer, "fields"):
                return name in composer.fields
            elif hasattr(composer, "_field_definitions"):
                return name in composer._field_definitions
            else:
                # Try to access through the model fields if already built
                try:
                    temp_model = composer.build()
                    return name in temp_model.model_fields
                except:
                    return False

        # Always ensure messages field exists
        if not field_exists("messages"):
            composer.add_field(
                "messages",
                List[BaseMessage],
                default_factory=list,
                shared=True,
                reducer=add_messages,
            )

        # Add meta state if needed
        if include_meta and not field_exists("meta_state"):
            # Import here to avoid circular imports
            # from haive.core.schema.meta_state import MetaState

            composer.add_field(
                "meta_state",
                MetaState,
                default_factory=MetaState,
                shared=True,
                description="Shared meta state for agent coordination",
            )

        # Multi-agent coordination fields
        if len(agents) > 1:
            if not field_exists("active_agent_id"):
                composer.add_field(
                    "active_agent_id", Optional[str], default=None, shared=True
                )

            if not field_exists("agent_outputs"):
                composer.add_field(
                    "agent_outputs",
                    Dict[str, Any],
                    default_factory=dict,
                    shared=True,
                    description="Collected outputs from each agent",
                )

    @staticmethod
    def _generate_name(agents: List[Agent]) -> str:
        """Generate a name for the composed schema."""
        if len(agents) == 1:
            return f"{agents[0].__class__.__name__}State"
        else:
            return "MultiAgentState"

    @staticmethod
    def _apply_shared_separation(composer: SchemaComposer, all_fields: Dict):
        """Apply shared separation - all fields are shared."""
        for field_name, field_sources in all_fields.items():
            # Use first occurrence for type info
            _, _, ftype, finfo = field_sources[0]

            if hasattr(finfo, "default_factory"):
                composer.add_field(
                    field_name,
                    ftype,
                    default_factory=finfo.default_factory,
                    shared=True,
                )
            else:
                composer.add_field(
                    field_name,
                    ftype,
                    default=finfo.default if hasattr(finfo, "default") else None,
                    shared=True,
                )

    @staticmethod
    def _apply_namespaced_separation(
        composer: SchemaComposer, all_fields: Dict, agents: List[Agent]
    ):
        """Apply namespaced separation - each agent gets its own namespace."""
        # Create agent ID to name mapping
        agent_id_to_name = {agent.id: agent.name for agent in agents}

        for field_name, field_sources in all_fields.items():
            for agent_id, fname, ftype, finfo in field_sources:
                agent_name = agent_id_to_name.get(agent_id, agent_id)
                namespaced_name = f"{agent_name.lower().replace(' ', '_')}_{fname}"

                if hasattr(finfo, "default_factory"):
                    composer.add_field(
                        namespaced_name,
                        ftype,
                        default_factory=finfo.default_factory,
                        shared=False,
                    )
                else:
                    composer.add_field(
                        namespaced_name,
                        ftype,
                        default=finfo.default if hasattr(finfo, "default") else None,
                        shared=False,
                    )
