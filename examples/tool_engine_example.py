"""Example usage of the enhanced ToolEngine with universal typing.

This example demonstrates:
1. Creating various types of tools
2. Tool analysis and capability detection
3. Routing strategies
4. Store/memory tools integration
5. Retriever tools
6. Human interrupt workflows
"""

import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from haive.core.engine.tool import (
    ToolCapability,
    ToolEngine,
)
from haive.core.tools.store_manager import StoreManager


# Example tool functions
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        # In production, use a safe math parser
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error calculating: {e!s}"


@tool
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information."""
    # Mock search results
    results = []
    for i in range(max_results):
        results.append(
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet about {query}...",
            }
        )
    return results


# Structured output example
class EmailContent(BaseModel):
    """Structured email content."""

    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body content")
    cc: list[str] | None = Field(default=None, description="CC recipients")


def compose_email(
    recipient: str, topic: str, tone: str = "professional"
) -> EmailContent:
    """Compose an email with structured output."""
    return EmailContent(
        to=recipient,
        subject=f"{tone.title()} email about {topic}",
        body=f"Dear recipient,\n\nThis is a {tone} email regarding {topic}.\n\nBest regards",
        cc=None,
    )


# State-aware tool example
def update_conversation_state(message: str, state: dict) -> dict:
    """Update conversation state with new message."""
    if "messages" not in state:
        state["messages"] = []

    state["messages"].append({"role": "assistant", "content": message})

    state["last_update"] = "now"
    return state


async def main():
    """Demonstrate ToolEngine capabilities."""

    print("=== Enhanced ToolEngine Example ===\n")

    # 1. Create basic ToolEngine with mixed tools
    print("1. Creating ToolEngine with various tools...")

    # Create structured output tool
    email_tool = ToolEngine.create_structured_output_tool(
        func=compose_email,
        name="email_composer",
        description="Compose structured emails",
        output_model=EmailContent,
    )

    # Create state-aware tool
    state_tool = ToolEngine.create_state_tool(
        func=update_conversation_state,
        name="state_updater",
        description="Update conversation state",
        reads_state=True,
        writes_state=True,
        state_keys=["messages", "last_update"],
    )

    # Create interruptible tool
    approval_tool = ToolEngine.create_human_interrupt_tool(
        name="get_approval",
        description="Get human approval for actions",
        allow_edit=True,
        interrupt_message="Please review and approve this action",
    )

    # Initialize engine
    engine = ToolEngine(
        tools=[calculator, web_search, email_tool, state_tool, approval_tool],
        enable_analysis=True,
        routing_strategy="capability",
    )

    print(f"Created engine with {len(engine._tool_properties)} tools\n")

    # 2. Analyze tool properties
    print("2. Tool Analysis Results:")
    for tool_name, props in engine._tool_properties.items():
        print(f"\n  Tool: {tool_name}")
        print(f"  - Type: {props.tool_type}")
        print(f"  - Category: {props.category}")
        print(f"  - Capabilities: {', '.join(cap.value for cap in props.capabilities)}")
        print(
            f"  - State interaction: reads={props.from_state_tool}, writes={props.to_state_tool}"
        )

    print("\n")

    # 3. Query tools by capability
    print("3. Querying tools by capability:")

    structured_tools = engine.get_tools_by_capability(ToolCapability.STRUCTURED_OUTPUT)
    print(f"  Structured output tools: {structured_tools}")

    state_tools = engine.get_state_tools()
    print(f"  State-aware tools: {state_tools}")

    interruptible = engine.get_interruptible_tools()
    print(f"  Interruptible tools: {interruptible}")

    print("\n")

    # 4. Store/Memory tools integration
    print("4. Creating memory management tools...")

    # Create store manager
    store_manager = StoreManager()

    # Create memory tools suite
    memory_tools = ToolEngine.create_store_tools_suite(
        store_manager=store_manager,
        namespace=("example", "session", "123"),
        include_tools=["store", "search", "retrieve"],
    )

    print(f"  Created {len(memory_tools)} memory tools:")
    for tool in memory_tools:
        print(f"    - {tool.name}: {tool.__tool_category__.value} category")

    print("\n")

    # 5. Retriever tool example
    print("5. Creating retriever tool...")

    # Mock retriever for example
    class ExampleRetriever:
        async def aget_relevant_documents(self, query: str):
            return [
                {
                    "page_content": f"Document about {query}",
                    "metadata": {"source": "example"},
                }
            ]

    retriever_tool = ToolEngine.create_retriever_tool(
        retriever=ExampleRetriever(),
        name="knowledge_retriever",
        description="Retrieve relevant knowledge",
    )

    print(f"  Created retriever: {retriever_tool.name}")
    print(f"  Capabilities: {retriever_tool.__tool_capabilities__}")

    print("\n")

    # 6. Tool augmentation example
    print("6. Augmenting existing tools...")

    # Augment calculator to be interruptible and state-aware
    enhanced_calc = ToolEngine.augment_tool(
        calculator,
        name="enhanced_calculator",
        make_interruptible=True,
        reads_state=True,
        state_keys=["calculation_history"],
        interrupt_message="Calculation will be performed",
    )

    print("  Enhanced calculator capabilities:")
    print(f"    - Interruptible: {enhanced_calc.is_interruptible}")
    print(f"    - State-aware: {enhanced_calc.reads_state}")
    print(f"    - Capabilities: {enhanced_calc.__tool_capabilities__}")

    print("\n")

    # 7. Create runnable for execution
    print("7. Creating runnable ToolNode...")

    runnable = engine.create_runnable()
    print(f"  Created ToolNode runnable: {type(runnable).__name__}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
