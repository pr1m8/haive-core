from pydantic import BaseModel, Field, field_validator, model_serializer


class StateInterfaceMixin(BaseModel):

    use_state: bool = Field(
        default=False, description="Whether to use the state for the tool node"
    )
    state_key: str = Field(default="state", description="The key to use for the state")
