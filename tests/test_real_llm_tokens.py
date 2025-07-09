"""Test if real LLM calls provide token usage metadata."""

import asyncio

from haive.agents.simple.agent import SimpleAgent

from haive.core.engine.aug_llm import AugLLMConfig


async def test_real_llm_token_metadata():
    """Test what metadata real LLM calls provide."""
    print("🧪 Testing real LLM token metadata...")

    # Create agent with real LLM
    config = AugLLMConfig(
        temperature=0.1,
        max_tokens=50,
    )

    agent = SimpleAgent(name="llm_token_test", engine=config)

    # Make a real call and examine the raw response
    print("   Making real LLM call...")

    # Try to get the raw LLM response
    response = await agent.arun("What is 2 + 2?")
    print(f"   Response type: {type(response)}")
    print(f"   Response: {response}")

    # Check if the response has messages
    if hasattr(response, "messages"):
        print(f"   Messages in response: {len(response.messages)}")
        for i, msg in enumerate(response.messages):
            print(f"     Message {i}: {type(msg)} - {msg.content[:50]}...")
            if hasattr(msg, "response_metadata"):
                print(f"       Response metadata: {msg.response_metadata}")
            if hasattr(msg, "usage_metadata"):
                print(f"       Usage metadata: {msg.usage_metadata}")

    # Try to access the agent's graph state
    if hasattr(agent, "graph") and hasattr(agent.graph, "get_state"):
        try:
            print("   Checking agent graph state...")
            # This might require a thread_id or config
            thread_config = {"configurable": {"thread_id": "test-thread"}}
            state = agent.graph.get_state(thread_config)
            print(f"   Graph state: {state}")
        except Exception as e:
            print(f"   Could not get graph state: {e}")

    # Try to access the engine directly
    print(f"   Engine type: {type(agent.engine)}")
    if hasattr(agent.engine, "llm_config"):
        print(f"   Engine LLM config: {agent.engine.llm_config}")


async def test_direct_engine_call():
    """Test calling the engine directly to see token metadata."""
    print("\n🧪 Testing direct engine call...")

    from langchain_core.messages import HumanMessage

    config = AugLLMConfig(
        temperature=0.1,
        max_tokens=50,
    )

    # Try to call the engine directly
    try:
        # Create a runnable from the engine
        runnable = config.create_runnable()
        print(f"   Created runnable: {type(runnable)}")

        # Make a direct call
        messages = [HumanMessage(content="What is 2 + 2?")]
        result = await runnable.ainvoke({"messages": messages})

        print(f"   Direct result type: {type(result)}")
        print(f"   Direct result: {result}")

        # Check if result has token usage
        if hasattr(result, "response_metadata"):
            print(f"   Response metadata: {result.response_metadata}")
        if hasattr(result, "usage_metadata"):
            print(f"   Usage metadata: {result.usage_metadata}")

    except Exception as e:
        print(f"   Direct engine call failed: {e}")


async def main():
    """Run all tests."""
    print("🚀 Testing Real LLM Token Metadata\n")

    try:
        await test_real_llm_token_metadata()
        await test_direct_engine_call()

        print("\n🎉 Real LLM token metadata tests completed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
