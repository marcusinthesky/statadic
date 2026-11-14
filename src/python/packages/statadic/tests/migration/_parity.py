# jaxtyping shape strings are runtime contracts.
"""Provide shared comparators for the t05 parity harness.

This supports the energy/permutation vectorization campaign.

Later lanes (t05.2-t05.5, t05.8) import ``load_oracle`` /
``assert_decision_invariant`` and add their own
``-k``-selectable parity tests against the same frozen oracle — keep this module
generic (no lane-specific numerics) so it stays a stable contract across lanes.

Correctness-contract summary (see the t05 README "Correctness contract"):

- Bit-exact class: production block reductions are compared with independent
  gathered sums on exactly representable integer inputs in
  ``test_block_reduction_parity.py``.
- Tolerance + decision-invariant class: every optimization vs the oracle
  (``assert_decision_invariant``) — ``allclose`` at an empirically-set ``atol``
  plus an identical permutation p-value / equivalence verdict, because the
  baseline recomputes the matmul-induced metric per replicate and any
  hoisted/rebatched recompute drifts ~1e-8-1e-10 from XLA tiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    import jax

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "t05_energy_oracle.npz"

# Empirically measured (see test_parity_t05.py::test_measured_rebatch_drift):
# reslicing a fully-batched pairwise distance matrix vs recomputing the same
# block directly via a separately-batched (row-stacked) call drifts by
# max|delta| ~= 4.2e-8 in float64 (jax_enable_x64=True) on this host, from XLA
# matmul-tiling nondeterminism across batchings (see the repo's JAX-tiling
# note). DEFAULT_ATOL is set to ~10x that observed drift.
MEASURED_REBATCH_MAX_ABS_DIFF = 4.2146848510894035e-08
DEFAULT_ATOL = 1e-6  # ~10x MEASURED_REBATCH_MAX_ABS_DIFF, rounded up


class ParityResult(Protocol):
    """Numerical result surface consumed by the shared parity checks."""

    @property
    def energy_point_estimate(self) -> float: ...

    @property
    def studentized_statistic(self) -> float: ...

    @property
    def weights(self) -> jax.Array: ...

    @property
    def null_distribution(self) -> jax.Array: ...

    @property
    def bootstrap_distribution(self) -> jax.Array: ...

    @property
    def pvalue(self) -> float: ...

    @property
    def equivalence(self) -> bool | None: ...


def load_oracle() -> dict[str, np.ndarray]:
    """Load the frozen t05 golden oracle fixture.

    Returns:
        Dict with the fixture config (``seed``, ``n``, ``m``, ``d``, ``K``,
        ``num_permutations``, ``exponent``, ``metric``, ``solver``,
        ``selection_split``, ``weight_method``, ``reoptimize_in_permutation``),
        the frozen inputs (``x``, ``candidates``), the
        ``corrected_mixture_test`` outputs (``energy_point_estimate``,
        ``pvalue``, ``studentized_statistic``, ``weights``,
        ``null_distribution``, ``bootstrap_distribution``), and the reused
        scalar-energy oracle fields (``energy_distance_x_c0``,
        ``energy_terms_xx``, ``energy_terms_xy``, ``energy_terms_yy``,
        ``mixture_energy_distance``).

    """
    with np.load(_FIXTURE_PATH) as npz:
        return {key: npz[key] for key in npz.files}


# The two helpers below state their failure condition in prose rather than a
# `Raises:` section. Both raise `AssertionError` and only via `assert`, which
# pydoclint does not count as a raise (DOC502) unless
# `should-declare-assert-error-if-assert-statement-exists` is on — and turning
# that on would demand the section on every test function in the suite. Prose
# keeps the contract; the section would only keep the linter's model of it.
def assert_decision_invariant(
    result: ParityResult,
    oracle: dict[str, np.ndarray],
    *,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Assert tolerance-equality + identical test decision vs the oracle.

    This is the binding requirement for every non-bit-exact change in the t05
    campaign (t05.2 included): continuous fields must match within ``atol``
    AND the discrete test decision (permutation p-value, equivalence verdict)
    must be identical, since the baseline recomputes a matmul-induced metric
    per replicate and any hoisted/rebatched recompute is only ``allclose``,
    never bit-identical, vs the oracle.

    Args:
        result: A mixture test result (or any object
            exposing the same attribute names) produced by the code under
            test.
        oracle: The dict returned by :func:`load_oracle`.
        atol: Absolute tolerance for the continuous-field ``allclose`` checks.
            Defaults to :data:`DEFAULT_ATOL`, empirically set to ~10x the
            measured baseline rebatch drift.

    Fails with ``AssertionError`` if any continuous field falls outside ``atol``
    or the decision differs from the oracle.

    """
    assert jnp.allclose(
        result.energy_point_estimate, oracle["energy_point_estimate"], atol=atol
    ), "energy_point_estimate drifted beyond atol"
    assert jnp.allclose(
        result.studentized_statistic, oracle["studentized_statistic"], atol=atol
    ), "studentized_statistic drifted beyond atol"
    assert jnp.allclose(
        jnp.asarray(result.weights), jnp.asarray(oracle["weights"]), atol=atol
    ), "weights drifted beyond atol"
    assert jnp.allclose(
        jnp.asarray(result.null_distribution),
        jnp.asarray(oracle["null_distribution"]),
        atol=atol,
    ), "null_distribution drifted beyond atol"
    assert jnp.allclose(
        jnp.asarray(result.bootstrap_distribution),
        jnp.asarray(oracle["bootstrap_distribution"]),
        atol=atol,
    ), "bootstrap_distribution drifted beyond atol"

    # The binding requirement: the test DECISION must be identical, not just
    # the underlying floats. The permutation p-value uses a fixed (1 + count)
    # / (1 + num_permutations) formula (see permutation/_split_runner.py)
    # so an unchanged null-vs-observed ordering yields an *exact* p-value
    # match, not merely allclose.
    np.testing.assert_equal(
        float(result.pvalue),
        float(oracle["pvalue"]),
        err_msg="permutation p-value decision changed",
    )
    oracle_equivalence = oracle.get("equivalence")
    result_equivalence = getattr(result, "equivalence", None)
    assert result_equivalence == oracle_equivalence, (
        f"equivalence verdict changed: {result_equivalence} != {oracle_equivalence}"
    )
