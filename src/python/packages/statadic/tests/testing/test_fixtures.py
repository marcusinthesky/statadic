"""Explicitly loaded shared fixtures retain their deterministic behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import statadic.testing
import pytest
from statadic.testing import fixtures

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax


@pytest.mark.integration
def test_explicit_plugin_supplies_prng_key(prng_key: jax.Array) -> None:
    """The package conftest loads the fixture module explicitly.

    Args:
        prng_key: The fixture under test.

    """
    assert prng_key.shape == (2,)
    assert jnp.array_equal(prng_key, jnp.asarray(prng_key))


@pytest.mark.unit
def test_prng_key_is_the_declared_seed(prng_key: jax.Array) -> None:
    """It is ``PRNGKey(SEED)``, so a failure is reproducible from the constant.

    Args:
        prng_key: The fixture under test.

    """
    import jax  # noqa: PLC0415  # mirrors the fixture's own lazy import

    assert jnp.array_equal(prng_key, jax.random.PRNGKey(fixtures.SEED))


@pytest.mark.unit
@pytest.mark.property
def test_seeded_key_gives_independent_reproducible_keys(
    seeded_key: Callable[[int], jax.Array],
) -> None:
    """Different indices give different keys; the same index gives the same key.

    Args:
        seeded_key: The factory fixture under test.

    """
    assert jnp.array_equal(seeded_key(3), seeded_key(3))
    assert not jnp.array_equal(seeded_key(3), seeded_key(4))


@pytest.mark.unit
@pytest.mark.contract
def test_tolerances_are_the_shared_pair(tolerances: tuple[float, float]) -> None:
    """The suite compares float64 statistics against one documented pair.

    Args:
        tolerances: The ``(atol, rtol)`` fixture under test.

    """
    atol, rtol = tolerances
    assert (atol, rtol) == (fixtures.ATOL, fixtures.RTOL)
    assert 0.0 < atol < rtol


@pytest.mark.integration
@pytest.mark.contract
def test_the_fixtures_are_not_re_exported_from_statadic_testing() -> None:
    """``statadic.testing.__all__`` carries library helpers only, never fixtures."""
    exported = set(statadic.testing.__all__)
    assert exported.isdisjoint(set(fixtures.__all__))
    for name in fixtures.__all__:
        assert not hasattr(statadic.testing, name), (
            f"statadic.testing re-exports the fixture {name!r}; importing a fixture "
            f"by name yields a function pytest never sets up"
        )


@pytest.mark.integration
@pytest.mark.contract
def test_the_explicit_plugin_is_registered(
    pytestconfig: pytest.Config,
) -> None:
    """The fixture module is registered by name, so it can be looked up or blocked.

    Args:
        pytestconfig: The active config, whose plugin manager holds the entry.

    """
    assert pytestconfig.pluginmanager.hasplugin("statadic.testing.fixtures")


@pytest.mark.integration
@pytest.mark.contract
def test_fixtures_are_not_globally_registered() -> None:
    """Installing STATADIC does not inject a pytest plugin into other suites."""
    from importlib.metadata import entry_points  # noqa: PLC0415

    assert not any("statadic" in ep.name for ep in entry_points(group="pytest11"))
