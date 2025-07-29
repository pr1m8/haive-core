from __future__ import annotations

"""Dynamic_Literal core module.

This module provides dynamic literal functionality for the Haive framework.

Classes:
    _DynLitMeta: _DynLitMeta implementation.
    attr: attr implementation.
    DynamicLiteral: DynamicLiteral implementation.

Functions:
    register: Register functionality.
    unregister: Unregister functionality.
"""


import contextlib
import inspect
from collections.abc import Iterable
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema

T = TypeVar("T", bound=str)


class _DynLitMeta(type):
    def __new__(mcls, name, bases, ns, **kw):
        cls = super().__new__(mcls, name, bases, ns, **kw)
        if not ns.get("__abstract__", False):
            if not hasattr(cls, "START_VALUES"):
                raise TypeError(f"{name} must define class attr START_VALUES")
            cls._values = set(cls.START_VALUES)
        return cls


class DynamicLiteral(str, Generic[T], metaclass=_DynLitMeta):
    """Dynamic "Literal-like" type with runtime-extensible allowed values.

    Meant to be used as a Pydantic field type.
    """

    __abstract__ = True
    _values: ClassVar[set[str]]
    START_VALUES: ClassVar[Iterable[str]]

    @classmethod
    def register(cls, *vals: str) -> None:
        cls._values.update(vals)

    @classmethod
    def unregister(cls, *vals: str) -> None:
        cls._values.difference_update(vals)

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._values))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source: type[Any], _handler: Any
    ) -> core_schema.CoreSchema:
        def _validate(v: Any) -> str:
            inspect.stack()[2].function
            if not isinstance(v, str):
                raise TypeError("string required")
            if v not in cls._values:
                raise ValueError(
                    f"invalid literal; allowed = {
                        sorted(
                            cls._values)!r}"
                )
            return v

        return core_schema.no_info_plain_validator_function(_validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, _core_schema: core_schema.CoreSchema, _handler: Any
    ) -> JsonSchemaValue:
        return {"type": "string", "enum": list(cls._values)}

    @classmethod
    def literal_type(cls) -> Any:
        return cast(Any, Literal[tuple(cls._values)])


def create_dynamic_literal(name: str, values: Iterable[str]) -> type[DynamicLiteral]:
    attrs = {"START_VALUES": tuple(values), "__qualname__": name}
    return type(name, (DynamicLiteral,), attrs)


# ──────────────────────────────── Demo Subclass ─────────────────────────


class Colour(DynamicLiteral):
    START_VALUES = ("red", "green", "blue")


class PaintJob(BaseModel):
    base: Colour
    accent: Colour

    class Config:
        extra = "forbid"


if __name__ == "__main__":
    pj1 = PaintJob(base="red", accent="green")

    Colour.register("purple")
    pj2 = PaintJob(base="purple", accent="blue")

    with contextlib.suppress(Exception):
        PaintJob(base="chartreuse", accent="green")
