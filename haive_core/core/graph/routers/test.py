from src.haive.core.graph.routers.base import RoutingConfig, Route
from src.haive.core.graph.routers.conditions import *


should_continue = StateValueCondition(
    state_key="should_continue",
    condition_map={
        "True": "end",
        "False": "end",
    },
)