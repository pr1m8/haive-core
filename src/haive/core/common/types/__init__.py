"""Module exports."""

from pathlib import Path
from typing import Any, Union

from haive.core.common.types.abc_root_wrapper import ABCRootWrapper

# Common type aliases
DictStrAny = dict[str, Any]
JsonType = Union[str, int, float, bool, None, dict[str, Any], list[Any]]
StrOrPath = Union[str, Path]

__all__ = ["ABCRootWrapper", "DictStrAny", "JsonType", "StrOrPath"]
