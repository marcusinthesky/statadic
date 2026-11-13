# standalone package-maintenance entry point
"""t46.5 — the ground/discrepancy migration, pinned where prose cannot hold it.

Four things this suite exists to stop from drifting silently:

* **Every declaration module is discovered.** The battery cases read the
  shipped registry directly; this suite derives the modules that assign a
  ``DECLARED_AXIOMS`` table and checks their compatibility brands. Adding a
  declaring module cannot silently escape the sweep.
* **``energy_distance`` has no default ground metric.** ``"angular"`` cannot see
  scale (t46.1 Finding 2: ``E(X, 3X) == 0``), so callers must choose.
  ``test_energy_distance_metric_is_required_and_angular_stays_scale_blind`` pins
  both facts.
* **The shims are the only reason waves A-B stay green.** Identity, not just
  importability, is what keeps ``except NegativeEnergyFunctionalError`` and
  ``is`` checks working across the old and new paths.
* **``mixture_energy_distance`` gained the α guard in this migration** — it was
  the one energy callable that did not validate ``exponent`` before the move.

Runs under the t46.1 jaxtyping/beartype import hook (``tests/conftest.py``), so
the NumPy-vs-JAX annotation split in :mod:`statadic.discrepancy.metrize` is checked
here rather than merely asserted in a docstring.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.core.axioms import (
    MARKER_AXIOMS,
    Axioms,
    Metric,
    NegativeType,
    Premetric,
    Semimetric,
)
from statadic.discrepancy import metrize
from statadic.discrepancy.mmd import energy
from statadic.discrepancy.mmd.components import (
    candidate_energy_components,
)
from statadic.sample.packed import PackedClouds, pack

if TYPE_CHECKING:
    from jax import Array

if TYPE_CHECKING:
    from types import ModuleType

    import jax

_RESCALE = 3.0

# --- derived declaration-module coverage ---------------------------------


_PACKAGE_SRC = Path(__file__).parents[2] / "src" / "statadic"


def _declaring_modules() -> tuple[ModuleType, ...]:
    """Discover modules that assign ``DECLARED_AXIOMS`` in their own source."""
    module_names: list[str] = []
    for path in sorted(_PACKAGE_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        declared_here = any(
            (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "DECLARED_AXIOMS"
            )
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "DECLARED_AXIOMS"
                    for target in node.targets
                )
            )
            for node in tree.body
        )
        if declared_here:
            relative = path.relative_to(_PACKAGE_SRC.parent).with_suffix("")
            module_names.append(".".join(relative.parts))
    return tuple(importlib.import_module(name) for name in module_names)


_DECLARING_MODULES = _declaring_modules()


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("module", _DECLARING_MODULES, ids=lambda m: m.__name__)
def test_every_brand_is_the_strongest_marker_its_flags_imply(
    module: ModuleType,
) -> None:
    """A brand may never promise an axiom the declaration does not carry.

    Extends ``test_marker_brand_is_the_strongest_available`` to the shipped
    tables, including :mod:`statadic.discrepancy.metrize`, which the battery does
    not cover.

    Args:
        module: A migrated module carrying both declaration tables.

    """
    declared: dict[str, Axioms] = module.DECLARED_AXIOMS
    brands: dict[str, type[Premetric]] = module.DECLARED_BRANDS
    assert set(declared) == set(brands)
    chain = (Premetric, Semimetric, Metric)
    for name, flags in declared.items():
        brand = brands[name]
        assert MARKER_AXIOMS[brand] & ~flags == Axioms(0), (
            f"{name}: brand {brand.__name__} promises "
            f"{MARKER_AXIOMS[brand] & ~flags!r}, which is not declared"
        )
        # t46.1's rule, restated on the shipped tables: brand the marker
        # furthest down the linear chain whose flags the declaration implies.
        # `NegativeType` ties with `Metric` because it strengthens it with a
        # condition that is not one of the four flags.
        safe = [m for m in chain if MARKER_AXIOMS[m] & ~flags == Axioms(0)]
        narrowest = max(safe, key=lambda m: len(m.__mro__))
        permitted = {Metric, NegativeType} if narrowest is Metric else {narrowest}
        assert brand in permitted, (
            f"{name} brands {brand.__name__} but {narrowest.__name__} "
            f"is implied by {flags!r}"
        )


# --- the default nobody may silently change -------------------------------


@pytest.mark.component
@pytest.mark.regression
def test_energy_distance_metric_is_required_and_angular_stays_scale_blind() -> None:
    """Pin the required ground metric and the angular defect that motivates it.

    t46.1 measured that ``metric="angular"`` normalises before comparing, so a
    pure rescaling reads as "identical distributions".
    """
    # `inspect.signature` rather than `__defaults__`: the t46.1 import hook
    # wraps every statadic callable, and the wrapper does not carry the tuple.
    metric = inspect.signature(energy.energy_distance).parameters["metric"]
    assert metric.kind is inspect.Parameter.KEYWORD_ONLY
    assert metric.default is inspect.Parameter.empty

    x = jnp.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]])
    blind = float(
        energy.energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(_RESCALE * x)),
            metric="angular",
        )
    )
    seeing = float(
        energy.energy_distance(
            PackedClouds.from_array(jnp.asarray(x)),
            PackedClouds.from_array(jnp.asarray(_RESCALE * x)),
            metric="euclidean",
        )
    )

    # `angular_distance` is `arccos(cos_sim)`, and `arccos` is the ill-conditioned
    # direction at coincidence: a cosine that rounds to `1 - δ` rather than to
    # exactly 1 comes back as `√(2δ)`, so the achievable floor for a coincident
    # pair is `√eps`, not `eps` — ~2.1e-08 at float64 (the suite runs x64) and
    # ~3.5e-04 at float32. Whether `jnp.dot` lands on exactly 1.0 is a property
    # of the host BLAS, which is why the previous `abs=1e-12` passed here and
    # failed in CI at -6.6e-09. The claim under test survives intact: `blind` is
    # eight orders below `seeing`, so the scale-blindness is still unmistakable.
    coincident_floor: float = 8.0 * float(jnp.finfo(jnp.result_type(x)).eps) ** 0.5
    assert blind == pytest.approx(0.0, abs=coincident_floor)
    assert seeing > 1.0
    assert energy.DECLARED_AXIOMS["energy_distance[angular]"] == (
        Axioms.NONNEGATIVE | Axioms.ZERO_DIAGONAL | Axioms.SYMMETRY
    )


# --- the α ∈ (0, 2) guard --------------------------------------------------


class _ExponentCall(Protocol):
    """An adapter invoking one energy callable with a single sample."""

    def __call__(self, a: jax.Array, /, *, exponent: float) -> object:
        """Invoke the callable under test with ``exponent``."""
        ...


def _evaluate_energy_distance(a: jax.Array, *, exponent: float) -> Array:
    return energy.energy_distance(
        PackedClouds.from_array(jnp.asarray(a)),
        PackedClouds.from_array(jnp.asarray(a)),
        metric="angular",
        exponent=exponent,
    )


def _evaluate_energy_test_statistic(a: jax.Array, *, exponent: float) -> Array:
    return energy.energy_test_statistic(a, a, metric="angular", exponent=exponent)


def _evaluate_energy_distance_matrix(a: jax.Array, *, exponent: float) -> object:
    return energy.energy_distance_matrix(
        pack([a, a], require_positive_counts=True), metric="angular", exponent=exponent
    )


def _evaluate_mean_distance_matrix(a: jax.Array, *, exponent: float) -> Array:
    return energy.mean_distance_matrix(
        pack([a, a], require_positive_counts=True), metric="angular", exponent=exponent
    )


def _evaluate_mixture_energy_distance(a: jax.Array, *, exponent: float) -> Array:
    return energy.mixture_energy_distance(
        PackedClouds.from_array(jnp.asarray(a)),
        pack([a], require_positive_counts=True),
        jnp.array([1.0]),
        exponent=exponent,
    )


_GUARDED = (
    (
        "energy_distance",
        _evaluate_energy_distance,
    ),
    (
        "energy_test_statistic",
        _evaluate_energy_test_statistic,
    ),
    (
        "energy_distance_matrix",
        _evaluate_energy_distance_matrix,
    ),
    (
        "mean_distance_matrix",
        _evaluate_mean_distance_matrix,
    ),
    (
        "mixture_energy_distance",
        _evaluate_mixture_energy_distance,
    ),
)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(("name", "call"), _GUARDED, ids=[n for n, _ in _GUARDED])
@pytest.mark.parametrize("exponent", [0.0, 2.0, -0.5, 2.5])
def test_alpha_guard_rejects_the_closed_boundary(
    name: str,  # parametrize id only; the adapter carries the call  # noqa: ARG001
    call: _ExponentCall,
    exponent: float,
) -> None:
    """α must be strictly inside ``(0, 2)`` on every public energy callable.

    Both boundaries are rejected, not just the outside: at ``α = 2`` the
    statistic collapses to ``2||E[X] - E[Y]||²`` and negative type fails at
    ``α = 0``. ``mixture_energy_distance`` is included because it is the one
    callable that did **not** carry this guard before t46.5.

    Args:
        name: Callable under test (test id only).
        call: Adapter invoking it with a single sample.
        exponent: Rejected α.

    """
    a = jnp.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="exponent must be in the range"):
        call(a, exponent=exponent)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("exponent", [1e-6, 1.0, 2.0 - 1e-6])
def test_alpha_guard_admits_the_open_interior(exponent: float) -> None:
    """Euclidean exponents strictly inside ``(0, 2)`` are accepted.

    Args:
        exponent: Admitted α.

    """
    a = jnp.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert float(
        energy.energy_distance(
            PackedClouds.from_array(jnp.asarray(a)),
            PackedClouds.from_array(jnp.asarray(a)),
            exponent=exponent,
            metric="euclidean",
        )
    ) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.needs_import_hook
def test_sqrt_energy_functional_accepts_a_numpy_matrix_under_the_hook() -> None:
    """``ArrayLike`` admits both float64 NumPy and JAX arrays under the hook.

    The converted artifact door takes :data:`jax.typing.ArrayLike` and returns a
    JAX array; the typed legal metrization still lives separately in
    ``core.matrices``.
    """
    functional = np.array([[0.0, 4.0], [4.0, 0.0]], dtype=np.float64)
    expected = np.array([[0.0, 2.0], [2.0, 0.0]])
    np.testing.assert_array_equal(metrize.sqrt_energy_functional(functional), expected)
    np.testing.assert_array_equal(
        metrize.sqrt_energy_functional(jnp.asarray(functional)),
        expected,
    )


@pytest.mark.component
@pytest.mark.regression
def test_sqrt_energy_functional_is_bit_identical_to_the_pipeline_predecessor() -> None:
    """Substrate energy outputs must not move: same formula, same float64 path.

    Reproduces the body that lived at
    ``pipeline/stages/substrate/energy_metric.py::_energy_metric_from_functional``
    before the move and demands exact equality, not tolerance.
    """
    rng = np.random.default_rng(46_5)
    functional = np.abs(rng.normal(size=(16, 16))).astype(np.float64)
    functional = (functional + functional.T) / 2.0
    np.fill_diagonal(functional, 0.0)

    expected = np.sqrt(np.maximum(np.asarray(functional, dtype=np.float64), 0.0))
    np.testing.assert_array_equal(metrize.sqrt_energy_functional(functional), expected)


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize(
    "values",
    [np.array([[0.0, np.nan]]), np.array([[0.0, np.inf]])],
    ids=["nan", "inf"],
)
def test_sqrt_energy_functional_rejects_nonfinite(values: np.ndarray) -> None:
    """Non-finite stored functionals raise rather than propagate NaN.

    Args:
        values: A functional containing a non-finite entry.

    """
    with pytest.raises(metrize.NonfiniteEnergyFunctionalError):
        metrize.sqrt_energy_functional(values)


@pytest.mark.component
@pytest.mark.contract
def test_sqrt_energy_functional_rejects_a_material_sign_error() -> None:
    """A negative beyond the relative tolerance is a sign error, not rounding."""
    with pytest.raises(metrize.NegativeEnergyFunctionalError):
        metrize.sqrt_energy_functional(np.array([[0.0, -1.0]]))
    np.testing.assert_allclose(
        metrize.sqrt_energy_functional(np.array([[0.0, -1e-18]])),
        np.array([[0.0, 0.0]]),
    )


# --- the shims, inverted (t46.10) ---------------------------------------

#: Every legacy module path t46.5 kept alive as a re-export shim through waves
#: A-B. t46.10 deleted them and repointed the call sites, so the wave-B
#: identity assertions (``shim.name is target.name``) are no longer
#: expressible. The successor obligation is the *inverse*: the path must be
#: gone, so nothing can quietly re-acquire a dependency on it.
_RETIRED_MODULE_PATHS = [
    "statadic._energy_components",
    "statadic.metrics",
    "statadic.statistics",
    "statadic.wasserstein",
]


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.parametrize("dotted", _RETIRED_MODULE_PATHS)
def test_retired_shim_path_no_longer_imports(dotted: str) -> None:
    """The pre-t46 module paths are deleted, not merely deprecated.

    Args:
        dotted: A legacy ``statadic`` module path retired by t46.10.

    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(dotted)


@pytest.mark.integration
@pytest.mark.contract
def test_private_energy_component_names_live_on_the_migrated_module() -> None:
    """Resolve the packed helpers on ``discrepancy/_mmd/components.py`` alone.

    Wave B asserted this through two shims; the shims are gone, so the check is
    now that the migrated module is the single owner. The private accesses are
    the subject of the test, not incidental -- SLF001 is suppressed for exactly
    that reason.
    """
    from statadic.discrepancy.mmd import components as _components  # noqa: PLC0415

    assert _components.candidate_energy_components is candidate_energy_components
