# jaxtyping shape strings are runtime contracts.
"""Shared estimator schema checks; no fitted-state or numerical policy."""


def validate_parameter_names(names: tuple[str, ...] | None, width: int) -> None:
    """Check a static ordered schema without deriving an identity from data."""
    if names is None:
        return
    if (
        not isinstance(names, tuple)
        or len(names) != width
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        message = (
            "parameter_names must be a tuple of unique nonempty names matching columns."
        )
        raise ValueError(message)
