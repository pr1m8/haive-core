"""
Generic List Wrapper for Structured Output Models.

This module defines a reusable generic class `Listified[T]` that wraps a list of structured
Pydantic models while retaining rich metadata and semantic serialization.

It is useful for cases where an LLM or API should return:
- a typed list (e.g. summaries, answers, points)
- with a custom field name instead of "items"
- along with additional metadata such as the query, source, or context

Example use cases include:
- Wrapping a list of `Summary` models in a `SummaryList` output
- Returning extracted `ToolCall`s with a clear `tool_calls` field
- Returning `AgentStep`s with a `steps` key for LangGraph

Field names are automatically pluralized from the wrapped model’s name.

This model is compatible with Sphinx AutoAPI and supports custom serialization via `.model_dump()`.

Example:
    >>> class Summary(BaseModel):
    ...     topic: str
    ...     bullets: list[str]

    >>> class SummaryList(Listified[Summary]):
    ...     pass

    >>> summaries = SummaryList(items=[Summary(topic="AI", bullets=["..."])], source="user input")
    >>> summaries.model_dump()
    {'summaries': [{'topic': 'AI', 'bullets': ['...']}], 'source': 'user input'}
"""

from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class Listified(BaseModel, Generic[T]):
    """
    A generic wrapper for a list of structured Pydantic items.

    This class enables structured output for agents or APIs that return a list
    of typed objects (e.g. `Summary`, `Answer`, `Finding`), while preserving
    a meaningful outer schema with a custom field name (e.g., `summaries`).

    The name of the serialized list field is derived from the generic type's class name.

    Attributes:
        items (List[T]): The list of structured items (e.g. list of `Summary`).
        source (str): Descriptive string identifying the origin of this result
            (e.g. input query, filename, user prompt, etc.).

    Example:
        >>> class Summary(BaseModel):
        ...     topic: str
        ...     points: List[str]

        >>> class SummaryList(Listified[Summary]):
        ...     pass

        >>> result = SummaryList(items=[Summary(topic="AI", points=["point 1"])], source="user question")
        >>> result.model_dump()
        {'summaries': [{'topic': 'AI', 'points': ['point 1']}], 'source': 'user question'}
    """

    items: List[T] = Field(
        ...,
        description="List of structured items to wrap and serialize.",
    )
    source: str = Field(
        ...,
        description="What this list is based on (e.g., query, user input, file name).",
    )

    @classmethod
    def field_name(cls) -> str:
        """
        Returns the pluralized name of the generic type for output serialization.

        Returns:
            str: Name like 'summaries', 'answers', etc.
        """
        name = cls.__parameters__[0].__name__.lower()
        return name + "s" if not name.endswith("s") else name

    def model_dump(self, *args, **kwargs) -> dict:
        """
        Serializes the model with the list field renamed to match the generic type.

        Args:
            *args: Passed to BaseModel.model_dump.
            **kwargs: Passed to BaseModel.model_dump.

        Returns:
            dict: Custom dict with field name derived from generic and 'source'.
        """
        base = super().model_dump(*args, **kwargs)
        return {self.field_name(): base.pop("items"), "source": base.get("source")}

    class Config:
        arbitrary_types_allowed = True
