# jaxtyping shape strings are runtime contracts.
"""The dissimilarity table three suites share — a plain module, not a conftest.

``CASES`` pairs every dissimilarity statadic ships with the contract it declares,
what it is measured to do on unstructured data, and the ``DMat`` brand a caller
may put on it. Three suites read it, each asking a different question:

* ``tests/testing/test_axioms.py`` — do the declared flags actually hold?
  (``statadic.testing.axioms`` driven by ``hypothesis`` and by fixed witnesses.)
* ``tests/core/test_axioms.py`` — is the brand implied by the flags, and is it
  the strongest one that is? (The ``MARKER_AXIOMS`` bridge.)
* ``tests/migration/test_ground_discrepancy_migration.py`` — does the shipped
  ``DECLARED_AXIOMS`` table still agree with what this one measures?

It is a module rather than a ``conftest.py`` on purpose (t49 C3): this is logic
and data, not lifecycle. Fixtures exist to set up and tear down state; a table
that three files import is an import. Putting it in a conftest would also make
it invisible — ``pytest --fixtures`` shows fixtures, nothing shows a conftest
constant — and would load it for every test in the tree rather than the three
that use it.

Why the strategies generate bounded, finite floats rather than ``st.floats()``
with defaults: NaN, ±inf and 1e308 are outside every one of these functions'
documented domain (``metrics.py`` says so explicitly for zero vectors), so
unbounded generation would measure the float model, not the axioms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    SEMIMETRIC,
    Axioms,
    Metric,
    NegativeType,
    Premetric,
    Semimetric,
)
from statadic.discrepancy.mmd import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    cosine_mean_mmd,
    distance_induced_kernel,
    energy,
    mmd,
    mmd_squared,
    rbf_kernel,
)

# Imported from the submodule, not the package: ``statadic.discrepancy.__init__``
# re-exports the ``mmd`` *function*, which shadows the module of the same name.
from statadic.discrepancy.mmd import DECLARED_AXIOMS as MMD_DECLARED_AXIOMS
from statadic.discrepancy.transport import sliced as transport
from statadic.geometry.ground import metrics
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from jaxtyping import Array
    from statadic.core.axioms import LawProperties
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )

# Fixed so JAX re-traces once per case rather than once per hypothesis draw;
# hypothesis varies the *values*, which is the axis that can break an axiom.
K_OBJECTS = 4
DIM = 3
SAMPLE_SIZE = 8
_SLICES = 256
_PROJECTION_KEY = jax.random.PRNGKey(7)

_ELEMENT_BOUND = 5.0
_BOUNDED = st.floats(
    min_value=-_ELEMENT_BOUND,
    max_value=_ELEMENT_BOUND,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
#: Strictly greater than ``2 * _ELEMENT_BOUND``, which is what makes the shifted
#: coordinate ranges of two objects provably disjoint. See :func:`objects`.
_SEPARATION = 11.0

# float64 is on (conftest), but energy/Wasserstein statistics on 8-point samples
# accumulate over O(n^2) terms; 1e-9 absolute would report summation noise as a
# violated axiom.
ATOL = 1e-9
RTOL = 1e-7
#: How negative the smallest Gram eigenvalue may be before "not PSD".
PSD_ATOL = 1e-8


def objects(case: Case) -> st.SearchStrategy[list[Array]]:
    """Draw ``K_OBJECTS`` arrays of ``shape``, non-degenerate **by construction**.

    Object ``i`` gets ``_SEPARATION * (i + 1)`` added along axis ``i % shape[-1]``.
    Because ``_SEPARATION > 2 * _ELEMENT_BOUND``:

    * every object owns a coordinate of magnitude at least
      ``_SEPARATION - _ELEMENT_BOUND``, so no row can have a near-zero norm — the
      documented NaN case for the direction-based metrics;
    * for any two objects the largest value differs by more than the element
      range, so they can coincide neither as arrays nor as multisets of values.

    Building the guarantee in rather than filtering it out matters: ``assume``
    rejections on a strategy that leans toward ``0.0`` are exactly how a
    property test starts failing ``filter_too_much`` intermittently, and this
    suite is the baseline six concurrent tasks run.

    Args:
        case: Dissimilarity case whose shape and object-space domain are used.

    Returns:
        A strategy producing a list of JAX arrays.

    """
    shape = case.shape
    last = shape[-1]

    def separate(raw: np.ndarray) -> list[Array]:
        drawn = jnp.asarray(raw)
        if case.requires_unit_norm:
            # Unit-sphere cases need distinct *directions*, not merely distinct
            # magnitudes. Put each draw in a disjoint slope interval before
            # normalization; this preserves variation without an ``assume``
            # filter and prevents the K=4/D=3 axis wrap from duplicating rays.
            directions: list[Array] = []
            for i in range(K_OBJECTS):
                value = 0.01 * drawn[i]
                value = value.at[0].add(1.0)
                value = value.at[1].add(4.0 * (i + 1))
                directions.append(value / jnp.linalg.norm(value))
            return directions
        return [
            drawn[i] + jnp.zeros(last).at[i % last].set(_SEPARATION * (i + 1))
            for i in range(K_OBJECTS)
        ]

    return hnp.arrays(
        dtype=np.float64, shape=(K_OBJECTS, *shape), elements=_BOUNDED
    ).map(separate)


def _as_point_metric(
    fn: Callable[[Array, Array], Array],
) -> Callable[[Array, Array], Array]:
    """Adapt an ``(n, d) x (m, d) -> (n, m)`` ground metric to two single points.

    Args:
        fn: A ``statadic.geometry.ground.metrics`` pairwise-distance function.

    Returns:
        ``g(a, b)`` returning the scalar distance between two ``(d,)`` vectors.

    """

    def scalar(a: Array, b: Array) -> Array:
        return fn(a[None, :], b[None, :])[0, 0]

    scalar.__name__ = getattr(fn, "__name__", "point_metric")
    return scalar


def _energy(metric: GroundDistanceName) -> Callable[[Array, Array], Array]:
    """Bind ``energy_distance`` to one ground metric.

    Args:
        metric: Ground metric name passed through to ``cdist``.

    Returns:
        ``g(a, b)`` returning the scalar energy distance between two samples.

    """

    def scalar(a: Array, b: Array) -> Array:
        return energy.energy_distance(
            PackedClouds.from_array(jnp.asarray(a)),
            PackedClouds.from_array(jnp.asarray(b)),
            exponent=1.0,
            metric=metric,
        )

    scalar.__name__ = f"energy_distance[{metric}]"
    return scalar


def _mmd(
    statistic: Callable[..., Array],
    key: str,
    *,
    univariate: bool = False,
    **bound: object,
) -> Callable[[Array, Array], Array]:
    """Bind an MMD statistic to the one declaration key it is measured under.

    ``mmd``/``mmd_squared`` key their contract on the ``(kernel, ground metric)``
    pair, so a case is a *bound* statistic, not the bare callable — the same
    shape as :func:`_energy` binding one ground metric.

    Args:
        statistic: ``mmd`` or ``mmd_squared``.
        key: The ``DECLARED_AXIOMS`` key this binding realizes; becomes the
            case name, so ``CASES_BY_NAME`` and the shipped table share an index.
        **bound: Keyword arguments pinning the kernel, ground metric and, for
            ``rbf``, the bandwidth.
        univariate: Add the feature axis to scalar-support contract fixtures.

    Returns:
        ``g(a, b)`` returning the scalar discrepancy between two samples.

    """

    def scalar(a: Array, b: Array) -> Array:
        return statistic(
            PackedClouds.from_array(a[..., None] if univariate else a),
            PackedClouds.from_array(b[..., None] if univariate else b),
            **bound,
        )

    scalar.__name__ = key
    return scalar


#: ``rbf`` ships no default bandwidth and no median heuristic, so the battery
#: closes one explicit factory value. The
#: value is arbitrary but fixed: the
#: axioms of an RBF MMD do not depend on it, and a fixed value keeps the battery
#: deterministic for the same reason ``_PROJECTION_KEY`` is fixed.
_RBF_BANDWIDTH = 1.0
_RBF_KERNEL = rbf_kernel(_RBF_BANDWIDTH)
_EUCLIDEAN_ENERGY_KERNEL = distance_induced_kernel(
    energy.energy_geometry(metrics.EUCLIDEAN, energy.UNIT_EXPONENT)
)
_ANGULAR_ENERGY_KERNEL = distance_induced_kernel(
    energy.energy_geometry(metrics.ANGULAR, energy.UNIT_EXPONENT)
)


def _sliced(a: Array, b: Array) -> Array:
    """Sliced 2-Wasserstein with a fixed key, so the battery is deterministic.

    Args:
        a: First sample, shape ``(n, d)``.
        b: Second sample, shape ``(n, d)``.

    Returns:
        Scalar sliced 2-Wasserstein distance.

    """
    return transport.sliced_wasserstein(
        _PROJECTION_KEY,
        PackedClouds.from_array(jnp.asarray(a)),
        PackedClouds.from_array(jnp.asarray(b)),
        n_projections=_SLICES,
        p=2,
    )


class AdversarialWitness(NamedTuple):
    """Named construction showing why a generic axiom is not universal."""

    name: str
    lost: Axioms
    build: Callable[[], list[Array]]
    fn: Callable[[Array, Array], Array] | None = None


class Case(NamedTuple):
    """One dissimilarity and the contract it declares.

    Attributes:
        name: Test id.
        fn: The uniform ``fn(a, b) -> scalar`` adapter.
        generic: Axioms observed on the fixed ``reference_objects`` — i.e. on
            data with no special structure. ``generic`` is a superset of
            ``declared``, and the gap is precisely the set of axioms that hold
            for typical data but have an adversarial counterexample. Pinning it
            turns a silent behaviour change into a diff.
        marker: The static brand a caller should put on a ``DMat`` built from
            this function. Must be the **strongest** marker — furthest down the
            chain, i.e. most axioms — whose ``MARKER_AXIOMS`` entry is implied by
            ``declared``. See ``test_marker_brand_is_implied_by_declared_axioms``
            and ``test_marker_brand_is_the_strongest_available``.
        properties: Orthogonal law properties exercised by separate witnesses.
        shape: Shape of a single object for the hypothesis strategy.
        requires_unit_norm: Whether the reference objects must be row-normalised
            before this dissimilarity is meaningful.
        witnesses: Named constructions for every generic/declaration gap.

    """

    name: str
    fn: Callable[[Array, Array], Array]
    generic: Axioms
    marker: type[Premetric]
    properties: LawProperties
    shape: tuple[int, ...]
    requires_unit_norm: bool = False
    witnesses: tuple[AdversarialWitness, ...] = ()

    @property
    def declared(self) -> Axioms:
        """Resolve the one shipped declaration for this dissimilarity.

        Documented here rather than in the class ``Attributes:`` block because
        it is a property, not a NamedTuple field, and pydoclint checks that
        block against the actual fields. A case never carries a test-side copy
        of the registry; ``generic`` above is a superset of what this returns.
        """
        return SHIPPED_DECLARED_AXIOMS[self.name]


SHIPPED_DECLARED_AXIOMS: dict[str, Axioms] = {
    **metrics.DECLARED_AXIOMS,
    **energy.DECLARED_AXIOMS,
    **transport.DECLARED_AXIOMS,
    **MMD_DECLARED_AXIOMS,
}


def _raw_scaling_collision() -> list[Array]:
    """Return distinct raw vectors occupying the same positive ray."""
    base = jnp.asarray([1.0, 2.0, 3.0])
    return [base, 4.0 * base, jnp.asarray([0.0, 1.0, 0.0])]


def _cloud_scaling_collision() -> list[Array]:
    """Return distinct empirical clouds with identical row directions."""
    base = jnp.tile(jnp.asarray([[1.0, 2.0, 3.0]]), (SAMPLE_SIZE, 1))
    other = jnp.tile(jnp.asarray([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))
    return [base, 4.0 * base, other]


def _linear_mean_collision() -> list[Array]:
    """Return distinct empirical clouds with the same arithmetic mean."""
    base = jnp.tile(jnp.asarray([[1.0, 2.0, 3.0]]), (SAMPLE_SIZE, 1))
    delta = jnp.tile(
        jnp.asarray([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]),
        (SAMPLE_SIZE // 2, 1),
    )
    other = jnp.tile(jnp.asarray([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))
    return [base, base + delta, other]


def _cosine_feature_collision() -> list[Array]:
    """Return distinct clouds with the same normalized-row feature mean."""
    base = jnp.tile(jnp.asarray([[1.0, 2.0, 3.0]]), (SAMPLE_SIZE, 1))
    scales = jnp.arange(1.0, SAMPLE_SIZE + 1.0)[:, None]
    other = jnp.tile(jnp.asarray([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))
    return [base, scales * base, other]


_ONE_PROJECTION_DESIGN = transport.ProjectionDesign(
    key=jax.random.PRNGKey(91),
    n_projections=1,
    p=2,
)


def _sliced_one_projection(a: Array, b: Array) -> Array:
    """Evaluate sliced Wasserstein under one fixed shared direction."""
    return transport.sliced_wasserstein_under_design(
        _ONE_PROJECTION_DESIGN,
        PackedClouds.from_array(jnp.asarray(a)),
        PackedClouds.from_array(jnp.asarray(b)),
    )


def _finite_projection_collision() -> list[Array]:
    """Return distinct clouds agreeing on the selected finite projection."""
    direction = _ONE_PROJECTION_DESIGN.directions(DIM, jnp.float64)[0]
    axis = jnp.argmin(jnp.abs(direction))
    basis = jax.nn.one_hot(axis, DIM, dtype=jnp.float64)
    orthogonal = basis - jnp.dot(basis, direction) * direction
    coordinates = jnp.arange(SAMPLE_SIZE, dtype=jnp.float64)[:, None]
    base = coordinates * direction
    return [base, base + orthogonal, base + direction]


_RAW_SCALING = AdversarialWitness(
    "positive-ray collision",
    Axioms.SEPARATION,
    _raw_scaling_collision,
)
_CLOUD_SCALING = AdversarialWitness(
    "row-direction collision",
    Axioms.SEPARATION,
    _cloud_scaling_collision,
)
_LINEAR_MEAN = AdversarialWitness(
    "equal-mean empirical clouds",
    Axioms.SEPARATION,
    _linear_mean_collision,
)
_COSINE_FEATURE = AdversarialWitness(
    "equal normalized-row feature means",
    Axioms.SEPARATION,
    _cosine_feature_collision,
)
_FINITE_PROJECTION = AdversarialWitness(
    "finite-projection collision",
    Axioms.SEPARATION,
    _finite_projection_collision,
    _sliced_one_projection,
)


CASES: tuple[Case, ...] = (
    # --- S2 ground: the object IS the vector in R^d, magnitude included ---
    Case(
        "euclidean_distance",
        _as_point_metric(metrics.euclidean_distance),
        generic=METRIC,
        marker=NegativeType,
        properties=NEGATIVE_TYPE,
        shape=(DIM,),
    ),
    Case(
        # Normalises first, so d(x, 2x) == 0: IDENTITY fails on R^d. PSEUDOMETRIC
        # is off the linear marker chain, so the brand drops all the way to
        # Premetric.
        "angular_distance",
        _as_point_metric(metrics.angular_distance),
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(DIM,),
        witnesses=(_RAW_SCALING,),
    ),
    Case(
        # Fails BOTH triangle (fixed witness) and identity (parallel vectors).
        "cosine_distance",
        _as_point_metric(metrics.cosine_distance),
        generic=SEMIMETRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(DIM,),
        witnesses=(_RAW_SCALING,),
    ),
    Case(
        "normalized_euclidean_distance",
        _as_point_metric(metrics.normalized_euclidean_distance),
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(DIM,),
        witnesses=(_RAW_SCALING,),
    ),
    # --- S3 discrepancy: the object is the empirical MEASURE, not the array ---
    Case(
        "energy_distance[euclidean]",
        _energy("euclidean"),
        generic=SEMIMETRIC,
        marker=Semimetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
    ),
    Case(
        "energy_distance[angular]",
        # The DEFAULT ground metric of `energy.energy_distance`, and it
        # inherits angular_distance's blindness to scale: d(X, 3X) == 0.
        _energy("angular"),
        generic=SEMIMETRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_CLOUD_SCALING,),
    ),
    Case(
        "w1_1d",
        _mmd(transport.w1_1d, "w1_1d", univariate=True),
        generic=METRIC,
        marker=Metric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE,),
    ),
    Case(
        "w2_1d",
        _mmd(transport.w2_1d, "w2_1d", univariate=True),
        generic=METRIC,
        marker=Metric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE,),
    ),
    Case(
        "sliced_wasserstein",
        _sliced,
        # A finite shared projection design can identify distinct
        # multivariate empirical measures. Typical draws separate, but the
        # universal law is only a pseudometric.
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_FINITE_PROJECTION,),
    ),
    # --- S3 discrepancy: MMD, keyed on (kernel, ground metric) (t50) ---
    # Every key in `statadic.discrepancy.mmd.DECLARED_AXIOMS` appears here, so
    # `tests/migration` can cross-check the shipped table against a measurement
    # rather than against itself. `generic` is what the fixed reference objects
    # actually produce under float64 — measured, not assumed; each `mmd` row
    # reaches METRIC on unstructured data, and each `mmd_squared` row SEMIMETRIC,
    # so every gap below is an adversarial-input gap, never an untested claim.
    Case(
        "mmd[rbf]",
        _mmd(mmd, "mmd[rbf]", kernel=_RBF_KERNEL),
        generic=METRIC,
        marker=NegativeType,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
    ),
    Case(
        "mmd[distance_induced[euclidean]]",
        _mmd(
            mmd,
            "mmd[distance_induced[euclidean]]",
            kernel=_EUCLIDEAN_ENERGY_KERNEL,
        ),
        generic=METRIC,
        marker=NegativeType,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
    ),
    Case(
        # Inherits angular_distance's blindness to scale, exactly as
        # `energy_distance[angular]` above does: the declaration drops to the
        # nonnegative/symmetric floor even though typical data looks like a metric.
        "mmd[distance_induced[angular]]",
        _mmd(
            mmd,
            "mmd[distance_induced[angular]]",
            kernel=_ANGULAR_ENERGY_KERNEL,
        ),
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_CLOUD_SCALING,),
    ),
    Case(
        # A linear-kernel MMD compares means only, so distinct samples sharing a
        # mean sit at distance zero: IDENTITY is unavailable by construction.
        "mmd[linear]",
        _mmd(mmd, "mmd[linear]", kernel=LINEAR_KERNEL),
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_LINEAR_MEAN,),
    ),
    Case(
        "mmd[cosine]",
        _mmd(mmd, "mmd[cosine]", kernel=COSINE_KERNEL),
        generic=METRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_COSINE_FEATURE,),
    ),
    Case(
        "mmd_squared[rbf]",
        _mmd(mmd_squared, "mmd_squared[rbf]", kernel=_RBF_KERNEL),
        generic=SEMIMETRIC,
        marker=Semimetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
    ),
    Case(
        "mmd_squared[distance_induced[euclidean]]",
        _mmd(
            mmd_squared,
            "mmd_squared[distance_induced[euclidean]]",
            kernel=_EUCLIDEAN_ENERGY_KERNEL,
        ),
        generic=SEMIMETRIC,
        marker=Semimetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
    ),
    Case(
        "mmd_squared[distance_induced[angular]]",
        _mmd(
            mmd_squared,
            "mmd_squared[distance_induced[angular]]",
            kernel=_ANGULAR_ENERGY_KERNEL,
        ),
        generic=SEMIMETRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_CLOUD_SCALING,),
    ),
    Case(
        "mmd_squared[linear]",
        _mmd(mmd_squared, "mmd_squared[linear]", kernel=LINEAR_KERNEL),
        generic=SEMIMETRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_LINEAR_MEAN,),
    ),
    Case(
        "mmd_squared[cosine]",
        _mmd(mmd_squared, "mmd_squared[cosine]", kernel=COSINE_KERNEL),
        generic=SEMIMETRIC,
        marker=Premetric,
        properties=NEGATIVE_TYPE,
        shape=(SAMPLE_SIZE, DIM),
        witnesses=(_COSINE_FEATURE,),
    ),
    Case(
        # The object is the already-normalized prototype supplied by pipeline.
        # STATADIC applies Euclidean chord distance on the unit sphere, so this is a
        # separating Hilbertian metric of negative type on its declared domain.
        "cosine_mean_mmd",
        cosine_mean_mmd,
        generic=METRIC,
        marker=NegativeType,
        properties=NEGATIVE_TYPE,
        shape=(DIM,),
        requires_unit_norm=True,
    ),
)

CASES_BY_NAME = {case.name: case for case in CASES}
IDS = [case.name for case in CASES]


def reference_objects(case: Case) -> list[Array]:
    """Build a fixed, non-degenerate object set for the deterministic checks.

    Args:
        case: The case under test.

    Returns:
        Six objects of the case's shape, offset so no two coincide and no row
        has near-zero norm.

    """
    key = jax.random.PRNGKey(20460001)
    n = 6
    raw = jax.random.normal(key, (n, *case.shape))
    offsets = jnp.arange(n, dtype=raw.dtype).reshape((n,) + (1,) * len(case.shape))
    values = list(raw + offsets + 1.0)
    if case.requires_unit_norm:
        return [value / jnp.linalg.norm(value) for value in values]
    return values
