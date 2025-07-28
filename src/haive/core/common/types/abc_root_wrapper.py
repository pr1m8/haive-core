"""From typing import Any This module provides an abstract base class for root-wrapped
models that serialize with a named key.

This is useful for models that are used as the root of a response, but need to be serialized with a named key.

For example, if you have a model like this:
```python
class Query(ABCRootWrapper[str]):
    pass
```

It will serialize as `{"query": "Hello, world!"}` instead of `{"root": "Hello, world!"}`.
"""

from abc import ABC
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import RootModel

T = TypeVar("T")


class ABCRootWrapper(RootModel[T], Generic[T], ABC):
    """Abstract base class for root-wrapped models that serialize with a named key (like
    'query' instead of 'root').

    The key is inferred automatically from the class name (lowercased),
    unless explicitly overridden by setting `SERIALIZED_KEY`.

    Example:
        class Query(ABCRootWrapper[str]):
            # SERIALIZED_KEY = "query"  # Optional override
    """

    SERIALIZED_KEY: ClassVar[str | None] = None

    def model_dump(self, *args, **kwargs) -> Any:
        data = super().model_dump(*args, **kwargs)
        key = self._get_serialized_key()
        if "root" in data:
            data[key] = data.pop("root")
        return data

    def model_dump_json(self, *args, **kwargs) -> Any:
        key = self._get_serialized_key()
        return super().model_dump_json(*args, **kwargs).replace('"root":', f'"{key}":')

    @classmethod
    def _get_serialized_key(cls) -> str:
        if cls.SERIALIZED_KEY:
            return cls.SERIALIZED_KEY
        return cls.__name__.lower()
