#!/usr/bin/env python3
"""Example: Agent with custom store memory tools.

This example demonstrates how to create an agent with custom store memory
tools similar to LangMem, using our PostgreSQL store infrastructure.
"""

import asyncio
import logging
import os
from typing import List, Optional

from haive.agents.simple import SimpleAgent

from haive.core.engine.aug_llm import AugLLMConfig
from haive.core.persistence.store.factory import create_store
from haive.core.persistence.store.types import StoreType
from haive.core.tools.store_manager import StoreManager
from haive.core.tools.store_tools import create_memory_tools_suite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryAgent:
    """An agent with advanced memory capabilities using our store system."""

    def __init__(
        self,
        name: str = "memory_agent",
        user_id: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_postgres: bool = True,
    ):
        """Initialize the memory agent.

        Args:
            name: Agent name
            user_id: User ID for namespace isolation
            connection_string: PostgreSQL connection string
            use_postgres: Whether to use PostgreSQL or memory store
        """
        self.name = name
        self.user_id = user_id

        # Create store manager
        if use_postgres and connection_string:
            logger.info("Creating agent with PostgreSQL store")
            store = create_store(
                store_type=StoreType.POSTGRES_SYNC, connection_string=connection_string
            )
        else:
            logger.info("Creating agent with memory store")
            store = create_store(store_type=StoreType.MEMORY)

        # Create namespace first
        if self.user_id:
            namespace = (
                "haive",
                "users",
                self.user_id,
                "agents",
                self.name,
                "memories",
            )
        else:
            namespace = ("haive", "agents", self.name, "memories")

        self.store_manager = StoreManager(store=store, default_namespace=namespace)

        # Create memory tools
        memory_tools = create_memory_tools_suite(
            store_manager=self.store_manager, namespace=namespace
        )

        # Create agent with memory tools
        self.agent = SimpleAgent(
            name=name,
            engine=AugLLMConfig(
                system_message=self._get_system_message(),
                tools=memory_tools,
                temperature=0.7,
            ),
        )

        logger.info(
            f"Created memory agent '{name}' with {len(memory_tools)} memory tools"
        )

    def _get_namespace(self) -> tuple:
        """Get the namespace for this agent."""
        return self.store_manager.default_namespace

    def _get_system_message(self) -> str:
        """Get the system message for the agent."""
        return """You are an intelligent assistant with advanced memory capabilities.

You have access to the following memory tools:
- store_memory: Store important information for later retrieval
- search_memory: Search through your stored memories
- retrieve_memory: Get a specific memory by ID
- update_memory: Update existing memories
- delete_memory: Remove memories (use carefully)

Use these tools to:
1. Remember user preferences, facts, and important information
2. Recall relevant information from past conversations
3. Update your knowledge when information changes
4. Maintain context across conversations

Categories for memories:
- user_preference: User likes, dislikes, preferences
- fact: Factual information worth remembering
- event: Important events or interactions
- context: Conversation context and background
- task: Tasks or requests from the user

Always be helpful and use your memory tools wisely to provide personalized assistance."""

    async def chat(self, message: str) -> str:
        """Chat with the agent.

        Args:
            message: User message

        Returns:
            Agent response
        """
        response = await self.agent.arun(message)
        return response

    def get_memory_stats(self) -> dict:
        """Get statistics about the agent's memory."""
        return self.store_manager.get_memory_stats()


async def demo_memory_agent():
    """Demonstrate the memory agent capabilities."""

    print("=== Haive Memory Agent Demo ===\n")

    # Get PostgreSQL connection if available
    postgres_connection = os.getenv("POSTGRES_CONNECTION_STRING")

    # Create memory agent
    agent = MemoryAgent(
        name="demo_agent",
        user_id="demo_user",
        connection_string=postgres_connection,
        use_postgres=bool(postgres_connection),
    )

    print(f"Created agent with store: {type(agent.store_manager.store).__name__}")
    print(f"Namespace: {agent._get_namespace()}\n")

    # Conversation scenarios
    scenarios = [
        # Learning about user
        "Hi! My name is Alice and I love pizza and hiking. I'm a software developer.",
        # Ask agent to remember something specific
        "Please remember that I have a meeting with Bob tomorrow at 3 PM about the new project.",
        # Test recall
        "What do you know about me?",
        # Test memory search
        "What did I tell you about meetings?",
        # Update preference
        "Actually, I prefer Thai food over pizza now.",
        # Test updated memory
        "What are my food preferences?",
        # Complex scenario
        "I'm planning a team outing. What would be good activities based on what you know about me?",
    ]

    for i, message in enumerate(scenarios, 1):
        print(f"User: {message}")

        try:
            response = await agent.chat(message)
            print(f"Agent: {response}\n")

            # Add a small delay to make it easier to follow
            await asyncio.sleep(1)

        except Exception as e:
            print(f"Error: {e}\n")

    # Show memory stats
    print("=== Memory Statistics ===")
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


async def interactive_memory_agent():
    """Interactive session with the memory agent."""

    print("=== Interactive Memory Agent ===")
    print("Type 'quit' to exit\n")

    # Get PostgreSQL connection if available
    postgres_connection = os.getenv("POSTGRES_CONNECTION_STRING")

    # Create memory agent
    agent = MemoryAgent(
        name="interactive_agent",
        user_id="interactive_user",
        connection_string=postgres_connection,
        use_postgres=bool(postgres_connection),
    )

    print(f"Agent ready with store: {type(agent.store_manager.store).__name__}")
    print("The agent can remember information across our conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            response = await agent.chat(user_input)
            print(f"Agent: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_memory_agent())
    else:
        asyncio.run(demo_memory_agent())
