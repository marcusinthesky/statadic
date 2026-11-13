# ruff: noqa: F722  # jaxtyping dimensions are runtime contracts.
"""Generic permutation testing with an explicit JAX/host result boundary.

``permutation_test_result`` is the primary numerical API. It accepts a PRNG key,
keeps every numerical value on device, and returns a registered
:class:`statadic.core.results.TestResult`. ``permutation_test_kernel`` is the smaller
statistic-plus-null assembly primitive used when callers already own resampling.

The eager host report of a generic test is
``to_host_test_result(permutation_test_result(...))``. Energy statistics use the
separate ``energy_permutation_test`` door, whose single ``EnergyGeometry``
argument makes the statistic and hoisted matrix construction inseparable.
"""

from __future__ import annotations

from collections.abc import (
    Callable,  # runtime type aliases
)
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Float, Int, Num  # noqa: TC002

from statadic.core.axioms import Law
from statadic.core.domains import (
    ConstructionOrigin,
)
from statadic.core.results import (
    TestResult,  # runtime annotations
)
from statadic.core.statistics import (
    CalibratedTest,
    CalibrationProvenance,
    EnergyTwoSampleStatistic,
    Homogeneity,
    RealizedStatistic,
    ResampledCalibration,
    ResampledNull,
    StatisticProvenance,
    UngroupedDesign,
)
from statadic.core.transforms import typed_vmap
from statadic.core.typing import (  # noqa: TC001
    Array,
    ArrayLike,
    PRNGKey,
    Static,
    static_jaxtyping,
)
from statadic.discrepancy.mmd.energy import (
    EnergyExponent,
    EnergyGeometry,  # noqa: TC001
    energy_test_statistic_from_blocks,
    energy_test_statistic_from_distances,
)
from statadic.geometry.ground.metrics import (
    cdist,
)
from statadic.inference.resampling.blocks import (
    grouped_block_reduction,
    map_replicates,
    replicate_counts,
)
from statadic.inference.resampling.permutation.kernel import (  # noqa: TC001
    _ALTERNATIVES,
    Alternative,
    _alternative_message,
    permutation_test_kernel,
)
from statadic.inference.resampling.results import (  # noqa: TC001
    HypothesisTest,
    to_host_test_result,
)

__all__ = [
    "EnergyLabelPermutation",
    "EnergyPermutationOrigin",
    "EnergyPermutationResult",
    "EnergyStatisticOrigin",
    "EnergyStatisticProvenance",
    "StatisticFn",
    "energy_permutation_test",
    "energy_permutation_test_result",
    "permutation_test_kernel",
    "permutation_test_result",
]

type StatisticFn = Callable[[Array, Array], Float[Array, ""]]  # jaxtyping scalar shape

#: Group count of the two-sample energy design: X and Y, in that pooled order.
_TWO_SAMPLE_GROUPS = 2


class EnergyStatisticOrigin[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](ConstructionOrigin):
    """Energy statistic whose geometry remains visible in its static type."""


class EnergyPermutationOrigin[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](ConstructionOrigin):
    """Label permutation of one typed energy-statistic construction."""


#: The statistic and the calibration this module produces, each named once.
#: Both stay generic in the three geometry parameters, so the ground law,
#: exponent and energy law still survive into the test result — the bundling
#: shortens the spelling and changes no implication.
type EnergyStatisticProvenance[
    GroundLawA: Law,
    ExponentA: EnergyExponent,
    EnergyLawA: Law,
] = StatisticProvenance[
    EnergyTwoSampleStatistic,
    EnergyStatisticOrigin[GroundLawA, ExponentA, EnergyLawA],
]
type EnergyLabelPermutation[
    GroundLawA: Law,
    ExponentA: EnergyExponent,
    EnergyLawA: Law,
] = CalibrationProvenance[
    UngroupedDesign,
    Homogeneity,
    ResampledCalibration,
    EnergyPermutationOrigin[GroundLawA, ExponentA, EnergyLawA],
]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EnergyPermutationResult[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
]:
    """Typed energy statistic, homogeneity test, null, and geometry."""

    test: CalibratedTest[
        Array,
        EnergyStatisticProvenance[GroundLawT, ExponentT, EnergyLawT],
        EnergyLabelPermutation[GroundLawT, ExponentT, EnergyLawT],
    ]
    null: ResampledNull[Array, ResampledCalibration]
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT] = field(
        metadata={"static": True}
    )

    @property
    def statistic(self) -> Array:
        """Observed energy statistic."""
        return self.test.statistic.value

    @property
    def pvalue(self) -> Array:
        """Monte-Carlo add-one p-value."""
        return self.test.pvalue

    @property
    def null_distribution(self) -> Array:
        """Resampled energy statistics."""
        return self.null.draws

    @property
    def n_resamples(self) -> int:
        """Static number of resamples."""
        return self.null.n_resamples


def energy_permutation_test[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    x: ArrayLike,
    y: ArrayLike,
    geometry: EnergyGeometry[GroundLawT, ExponentT, EnergyLawT],
    num_permutations: int = 999,
    seed: int | None = None,
    alternative: Alternative = "two-sided",
) -> HypothesisTest:
    """Perform an eager energy permutation test from one compatible geometry."""
    result = energy_permutation_test_result(
        jnp.asarray(x),
        jnp.asarray(y),
        geometry,
        random.PRNGKey(seed if seed is not None else 0),
        num_permutations=num_permutations,
        alternative=alternative,
    )
    return to_host_test_result(
        TestResult(
            statistic=result.statistic,
            pvalue=result.pvalue,
            null_distribution=result.null_distribution,
            n_resamples=result.n_resamples,
        )
    )


@static_jaxtyping("n m resamples")
def permutation_test_result(
    x: Num[Array, "n ..."],  # jaxtyping variadic shape
    y: Num[Array, "m ..."],  # jaxtyping variadic shape
    statistic_fn: Static[StatisticFn],
    key: PRNGKey,
    num_permutations: Static[int] = 999,
    alternative: Static[Alternative] = "two-sided",
) -> TestResult[Float[Array, ""]]:  # jaxtyping scalar shape
    """Return the transformable JAX result of a two-sample permutation test.

    ``statistic_fn``, ``num_permutations``, and ``alternative`` determine
    tracing or shapes and must be marked static when this
    function is wrapped in :func:`jax.jit`. ``x``, ``y``, and ``key`` remain
    dynamic and may carry a leading axis through :func:`jax.vmap`.

    Args:
        x: First on-device sample, shape ``(n, d)``.
        y: Second on-device sample, shape ``(m, d)``.
        statistic_fn: Traceable scalar statistic.
        key: JAX PRNG key. Numerical code never accepts or converts a seed.
        num_permutations: Static resample count.
        alternative: Static tail convention.

    Returns:
        A registered ``TestResult`` containing only JAX array leaves plus its
        static resample count.

    Raises:
        ValueError: If ``alternative`` is not recognized.

    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(_alternative_message(alternative))

    n = x.shape[0]
    m = y.shape[0]

    observed_stat = jnp.asarray(statistic_fn(x, y))
    combined = jnp.concatenate([x, y], axis=0)

    def permuted_statistic(
        resample_key: PRNGKey,
    ) -> Float[Array, ""]:  # jaxtyping scalar shape
        """Evaluate the statistic on one permuted label assignment.

        Args:
            resample_key: Key for this label permutation.

        Returns:
            The scalar statistic for the permuted split.

        """
        perm_idx = random.permutation(resample_key, n + m)
        perm_combined = combined[perm_idx]
        return jnp.asarray(statistic_fn(perm_combined[:n], perm_combined[n:]))

    keys = random.split(key, num_permutations)
    null_dist: Float[Array, " resamples"] = typed_vmap(permuted_statistic)(keys)
    return permutation_test_kernel(
        observed_stat, null_dist, num_permutations, alternative
    )


@static_jaxtyping("n m resamples")
def energy_permutation_test_result[
    GroundLawT: Law,
    ExponentT: EnergyExponent,
    EnergyLawT: Law,
](
    x: Num[Array, "n ..."],  # jaxtyping variadic shape
    y: Num[Array, "m ..."],  # jaxtyping variadic shape
    geometry: Static[EnergyGeometry[GroundLawT, ExponentT, EnergyLawT]],
    key: PRNGKey,
    num_permutations: Static[int] = 999,
    alternative: Static[Alternative] = "two-sided",
) -> EnergyPermutationResult[GroundLawT, ExponentT, EnergyLawT]:
    """Return a transformable distance-hoisted energy permutation result.

    Args:
        x: First on-device sample.
        y: Second on-device sample.
        geometry: Compatible ground, exponent regime, and result-law contract.
        key: Permutation PRNG key.
        num_permutations: Static number of null draws.
        alternative: Static tail convention.

    Returns:
        The observed statistic, p-value, and null draws as a ``TestResult``.

    """
    if alternative not in _ALTERNATIVES:
        raise ValueError(_alternative_message(alternative))

    n = x.shape[0]
    m = y.shape[0]
    combined = jnp.concatenate([x, y], axis=0)
    distances = cdist(combined, combined, metric=geometry.ground)
    distances = distances**geometry.exponent.value
    observed_stat = energy_test_statistic_from_distances(distances, n)

    keys = random.split(key, num_permutations)

    def draw_permutation(resample_key: Array) -> Int[Array, " _"]:
        return random.permutation(resample_key, n + m)

    perms: Int[Array, "resamples n+m"] = typed_vmap(  # jaxtyping shape
        draw_permutation
    )(keys)

    # Slot -> group labels are fixed by construction: a permutation sends the
    # first n pooled slots to X and the remaining m to Y, so only *which*
    # pooled rows fill those slots varies per replicate. That is exactly the
    # membership-matrix parameterisation `statadic.inference.resampling.blocks`
    # exploits, letting the replicate ride in C rather than in a reindexed D.
    slot_groups = (jnp.arange(n + m) >= n).astype(jnp.int32)

    def null_statistic(
        perm: Array,
    ) -> Float[Array, ""]:  # jaxtyping scalar shape
        """Recover one energy statistic without reindexing the pooled distances.

        Args:
            perm: Permutation indices into the pooled distance matrix.

        Returns:
            The scalar energy statistic for that permutation.

        """
        counts = replicate_counts(
            perm, slot_groups, _TWO_SAMPLE_GROUPS, n + m, distances.dtype
        )
        block_sums, _ = grouped_block_reduction(counts, distances)
        return energy_test_statistic_from_blocks(block_sums, n, m)

    # Chunked, not a bare vmap: `map_replicates` keeps the scan memory bound the
    # per-replicate gather used to provide, while still handing XLA a batched
    # dot_general to fuse (see the blocks module docstring for the measurement).
    null_dist: Float[Array, " resamples"] = map_replicates(null_statistic, perms)
    raw = permutation_test_kernel(
        observed_stat, null_dist, num_permutations, alternative
    )
    statistic = RealizedStatistic(
        value=raw.statistic,
        provenance=StatisticProvenance(),
    )
    test = CalibratedTest(
        statistic=statistic,
        pvalue=raw.pvalue,
        provenance=CalibrationProvenance(),
    )
    return EnergyPermutationResult(
        test=test,
        null=ResampledNull(
            draws=raw.null_distribution,
            n_resamples=raw.n_resamples,
        ),
        geometry=geometry,
    )
