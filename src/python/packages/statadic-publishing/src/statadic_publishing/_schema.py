"""Strict decoding shared by the two file contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import Field, fields, is_dataclass
from pathlib import PurePosixPath
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
)

from statadic.api.reporting import FrozenMetadataValue, freeze_metadata


class _Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def relative_path(value: str, *, external: bool = False) -> str:
    """Return a portable relative path, allowing ``..`` only for external files."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or (not external and ".." in path.parts)
        or "\\" in value
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return value


def decode[T: _Dataclass](cls: type[T], value: object) -> T:
    """Decode exact dataclass keys and types, without coercion."""
    if not isinstance(value, dict):
        raise ValueError(f"{cls.__name__} must be an object")
    expected = {field.name for field in fields(cls)}
    if set(value) != expected:
        raise ValueError(
            f"{cls.__name__} keys: expected {sorted(expected)}, got {sorted(value)}"
        )
    hints = get_type_hints(cls)
    return cls(**{name: _value(hints[name], value[name]) for name in expected})


def _value(kind: object, value: object) -> object:
    if kind is FrozenMetadataValue:
        try:
            return freeze_metadata(value)
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
    origin = get_origin(kind)
    args = get_args(kind)
    if origin is Literal:
        if value not in args or not any(type(value) is type(a) for a in args):
            raise ValueError(f"expected one of {args}, got {value!r}")
    elif origin is UnionType:
        for option in args:
            try:
                return _value(option, value)
            except ValueError:
                pass
        raise ValueError(f"invalid value {value!r}")
    elif origin is tuple:
        if not isinstance(value, list):
            raise ValueError("expected an array")
        return tuple(_value(args[0], item) for item in value)
    elif origin in (dict, Mapping):
        if not isinstance(value, dict) or any(
            not isinstance(key, str) for key in value
        ):
            raise ValueError("expected an object with string keys")
        items = {key: item for key, item in value.items() if isinstance(key, str)}
        if args[1] is object:
            return items
        return {key: _value(args[1], item) for key, item in items.items()}
    elif isinstance(kind, type) and is_dataclass(kind):
        return decode(kind, value)
    elif type(value) is not kind:
        raise ValueError(f"expected {kind}, got {value!r}")
    return value
