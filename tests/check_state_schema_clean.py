"""Check what state schema is being created - clean version."""

import logging
import sys

from haive.agents.reasoning_and_critique.self_discover.v2.agent import self_discovery

logging.getLogger().setLevel(logging.CRITICAL)

sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-agents/src")
sys.path.insert(0, "/home/will/Projects/haive/backend/haive/packages/haive-core/src")


print("🔍 Checking state schema...")
print(f"Agent name: {self_discovery.name}")
print(f"State schema: {self_discovery.state_schema}")
print(f"State schema name: {self_discovery.state_schema.__name__}")

# Check the fields in the state schema
print("\nState schema fields:")
if hasattr(self_discovery.state_schema, "model_fields"):
    for field_name, field_info in self_discovery.state_schema.model_fields.items():
        print(f"  {field_name}: {field_info.annotation}")
else:
    print("  No model_fields found")

# Check what engines are available
print(f"\nEngines ({len(self_discovery.engines)}):")
for engine_name in self_discovery.engines.keys():
    print(f"  {engine_name}")

# Check individual agents
print(f"\nIndividual agents ({len(self_discovery.agents)}):")
for agent_name, agent in self_discovery.agents.items():
    print(f"  {agent_name}: {type(agent).__name__}")
    if hasattr(agent, "state_schema"):
        print(f"    State schema: {agent.state_schema.__name__}")
        if hasattr(agent.state_schema, "model_fields"):
            for field_name, field_info in agent.state_schema.model_fields.items():
                print(f"      {field_name}: {field_info.annotation}")
    print()
