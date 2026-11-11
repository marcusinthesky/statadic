"""Scale per-test search budgets using the profile selected before collection."""

from __future__ import annotations

from typing import Any

from hypothesis import settings

# name: (per-test multiplier, optional cap, undecorated-test budget)
PROFILES: dict[str, tuple[int, int | None, int]] = {
    "statadic": (1, None, 100),
    "statadic-explore": (5, None, 500),
    "statadic-reference": (1, 20, 20),
}


def register_profiles() -> None:
    """Register independent profiles without inheriting Hypothesis's CI policy."""
    for name, (_, _, budget) in PROFILES.items():
        settings.register_profile(
            name,
            max_examples=budget,
            deadline=None,
            print_blob=True,
            derandomize=False,
            suppress_health_check=(),
        )


def examples(n: int, **overrides: Any) -> settings:  # noqa: ANN401 -- forwards Hypothesis settings
    """Preserve a test's cost tier while scaling it with the active profile.

    Additional Hypothesis settings, such as a justified health-check exception,
    are passed through. The budget itself is owned here.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        message = "An example budget must be a positive integer."
        raise ValueError(message)
    if "max_examples" in overrides:
        message = "Pass the base budget as the first argument."
        raise ValueError(message)
    profile = settings.get_current_profile_name()
    if profile not in PROFILES:
        message = f"Unknown STATADIC hypothesis profile: {profile!r}."
        raise ValueError(message)
    scale, cap, _ = PROFILES[profile]
    budget = n * scale
    return settings(
        max_examples=budget if cap is None else min(cap, budget), **overrides
    )
