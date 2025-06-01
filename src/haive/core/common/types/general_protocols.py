from typing import Protocol, Union, runtime_checkable


@runtime_checkable
class Nameable(Protocol):
    """Pure protocol for name attribute"""

    name: str


@runtime_checkable
class Identifiable(Protocol):
    """Pure protocol for id attribute"""

    id: Union[str, int]
