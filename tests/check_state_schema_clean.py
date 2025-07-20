"""Check what state schema is being created - clean version."""

import logging
import sys

from haive.agents.reasoning_and_critique.self_discover.v2.agent import self_discovery

logging.getLogger().setLevel(logging.CRITICAL)

sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


# Check the fields in the state schema
if hasattr(self_discovery.state_schema, "model_fields"):
    for _field_name, _field_info in self_discovery.state_schema.model_fields.items():
        pass
else:
    pass

# Check what engines are available
for _engine_name in self_discovery.engines:
    pass

# Check individual agents
for _agent_name, agent in self_discovery.agents.items():
    if hasattr(agent, "state_schema") and hasattr(agent.state_schema, "model_fields"):
        for _field_name, _field_info in agent.state_schema.model_fields.items():
            pass
