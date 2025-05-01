# node/engine_node.py
def create_engine_node(
    engine: Any, command_goto: Optional[Any] = None, node_config: Optional[dict] = None
) -> NodeFunction:
    """
    Create a node function from an engine.

    Args:
        engine: Engine with create_runnable method
        command_goto: Optional destination for Command routing
        node_config: Optional node configuration defaults

    Returns:
        Node function
    """

    def node_function(state, config=None):
        # Use node_config as fallback
        effective_config = config or node_config

        # Extract inputs based on engine mappings or schema
        input_data = extract_engine_inputs(state, engine)

        # Create runnable with config (class method)
        runnable = engine.create_runnable(effective_config)

        # Invoke runnable
        result = runnable(input_data)

        # Wrap in Command if needed
        if command_goto is not None and not isinstance(result, Command):
            return Command(update=result, goto=command_goto)

        return result

    # Add metadata
    node_function.__engine__ = engine
    node_function.__node_config__ = node_config
    node_function.__command_goto__ = command_goto

    return node_function
