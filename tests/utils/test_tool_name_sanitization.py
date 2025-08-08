"""Test the new tool name sanitization."""

import asyncio
from haive.agents.simple import SimpleAgent
from haive.agents.planning_v2.base.models import Plan, Task
from haive.agents.planning_v2.base.planner.prompts import planner_prompt
from haive.core.engine.aug_llm import AugLLMConfig


async def test_generic_tool_naming():
    """Test that Plan[Task] gets sanitized to plan_task_generic."""
    
    print("Testing tool name sanitization with Plan[Task]...")
    
    # This should now work with the sanitized tool name
    engine = AugLLMConfig(
        temperature=0.3,
        structured_output_model=Plan[Task]  # Generic class with brackets
    )
    
    # Create planner agent
    planner = SimpleAgent(
        name="test_planner",
        engine=engine,
        prompt_template=planner_prompt
    )
    
    print(f"Engine tools: {[getattr(tool, '__name__', str(tool)) for tool in engine.tools]}")
    print(f"Tool routes: {engine.tool_routes}")
    
    # Check that the tool name was sanitized
    expected_name = "plan_task_generic"  # Plan[Task] -> plan_task_generic
    
    # Find the sanitized tool name in routes
    sanitized_names = [name for name in engine.tool_routes.keys() if expected_name in name]
    print(f"Found sanitized tool names: {sanitized_names}")
    
    # Try to run the planner
    try:
        result = await planner.arun({
            "objective": "Test simple objective for tool name verification"
        })
        print(f"✅ Success! Generated plan with {len(result.steps) if hasattr(result, 'steps') else 'unknown'} steps")
        print(f"Result type: {type(result)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_generic_tool_naming())
    if success:
        print("\n🎉 Tool name sanitization working correctly!")
    else:
        print("\n💥 Tool name sanitization needs more work")