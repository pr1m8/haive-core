from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, RetryPolicy, Send
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from haive.core.schema.state_schema import StateSchema

NodeLike = TypeVar("NodeLike", bound=Any)
NodeOutput = TypeVar(
    "NodeOutput", bound=[Union[Command, Send, List[Send], str, BaseModel, Dict]]
)
StateLike = TypeVar("StateLike", bound=Union[Dict, BaseModel, StateSchema])
ConfigLike = TypeVar("ConfigLike", bound=Union[RunnableConfig, Dict, BaseModel])
NodeCallable = TypeVar(
    "NodeCallable", bound=Callable[[StateLike, Optional[ConfigLike]], NodeOutput]
)
BaseEdge = TypeVar("Base_Edge", bound=Union[Tuple[str, str], Tuple[NodeLike, NodeLike]])
# Branch = Typ
# Edge_Type = TypeVar("Node_Callable",bound=Union[Node_Callable,Base_Edge])
