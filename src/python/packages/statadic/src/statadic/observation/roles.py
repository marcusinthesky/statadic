"""Named input capabilities shared by designs and host model specifications."""

from collections.abc import Mapping  # noqa: TC003 -- runtime annotations
from typing import Literal, Protocol

__all__ = [
    "InputRole",
    "InputRoleProvider",
    "merge_grouping_roles",
    "merge_input_roles",
]

type InputRole = Literal["numeric", "categorical", "measure"]


class InputRoleProvider(Protocol):
    """A component declaring the named columns it consumes."""

    @property
    def input_columns(self) -> Mapping[str, InputRole]:
        """Return the required columns and their semantic input roles."""
        ...


def merge_input_roles(*groups: Mapping[str, InputRole]) -> dict[str, InputRole]:
    """Combine declarations, rejecting incompatible roles for the same name."""
    result: dict[str, InputRole] = {}
    for group in groups:
        for name, role in group.items():
            if name in result and result[name] != role:
                message = f"Column {name!r} has conflicting input roles."
                raise ValueError(message)
            result[name] = role
    return result


def merge_grouping_roles(
    roles: Mapping[str, InputRole], *names: str | None
) -> dict[str, InputRole]:
    """Add grouping keys, such as entities or units, to declared roles.

    A grouping key may be integer numeric or categorical labels. A key the
    other roles already read as numeric stays numeric; otherwise it is declared
    categorical. A key already declared as a measure is a conflict.
    """
    result = dict(roles)
    for name in names:
        if name is None or result.get(name) == "numeric":
            continue
        result = merge_input_roles(result, {name: "categorical"})
    return result
