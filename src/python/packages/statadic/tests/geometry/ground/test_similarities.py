# standalone package-maintenance entry point
"""``statadic.geometry.ground.similarities`` — the S2 Gram registry and its dispatcher.

The similarity-side sibling of ``test_metrics.py``: four kernel callables
(``linear``, ``cosine``, ``rbf``, ``distance_induced``) and the ``gram``
dispatcher that applies them.

What is **not** here: the energy/MMD parity identity, which is a property of the
statistic rather than of the Gram and lives in ``tests/discrepancy/test_mmd.py``.
PSD-ness is checked here, because :data:`statadic.geometry.ground.similarities.DECLARED_PSD`
is this module's declaration to keep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.core.domains import (
    FiniteValues,
)
from statadic.geometry.ground.similarities import (
    COSINE_KERNEL,
    LINEAR_KERNEL,
    PositiveSemidefiniteKernelProperties,
    cosine_gram,
    declare_similarity,
    distance_induced_gram,
    gram,
    linear_gram,
    rbf_gram,
    rbf_kernel,
)

from ..._budgets import examples
from ..._strategies import float_arrays, nonzero_rows
from ..._typing import ill_typed

if TYPE_CHECKING:
    from collections.abc import Callable

    from statadic.geometry.ground.similarities import (
        PositiveSemidefiniteKernelLaw,
        SimilarityStrategy,
    )

_PSD_TOLERANCE = -1e-5


def _min_eigenvalue(matrix: jnp.ndarray) -> float:
    """Return the smallest eigenvalue of a symmetrized square matrix."""
    symmetric = 0.5 * (matrix + matrix.T)
    return float(jnp.min(jnp.linalg.eigvalsh(symmetric)))


class TestLinearGram:
    """Tests for linear_gram."""

    @pytest.mark.unit
    @pytest.mark.numerical
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0], [3.0, 4.0]]), y=jnp.array([[5.0, 6.0]]))
    @given(x=float_arrays((2, 2)), y=float_arrays((1, 2)))
    def test_matches_inner_product(self, x: jax.Array, y: jax.Array) -> None:
        """Linear Gram is the matrix of pairwise inner products."""
        assert jnp.allclose(linear_gram(x, y), x @ y.T)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_output_shape(self) -> None:
        """Output is (n, m)."""
        x = jnp.zeros((4, 3))
        y = jnp.zeros((7, 3))
        assert linear_gram(x, y).shape == (4, 7)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    @given(x=float_arrays((3, 2)))
    def test_symmetry(self, x: jax.Array) -> None:
        """The self-Gram is symmetric."""
        k = linear_gram(x, x)
        assert jnp.allclose(k, k.T)


class TestCosineGram:
    """Tests for cosine_gram."""

    @pytest.mark.unit
    @pytest.mark.property
    def test_scale_invariance(self) -> None:
        """Cosine Gram ignores row magnitude."""
        x = jnp.array([[3.0, 4.0]])
        y = jnp.array([[0.6, 0.8]])
        assert jnp.allclose(cosine_gram(x, y), 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0], [-3.0, 4.0]]))
    @given(x=nonzero_rows(2, 2))
    def test_diagonal_is_one(self, x: jax.Array) -> None:
        """Self-similarity of a nonzero row is one."""
        assert jnp.allclose(jnp.diagonal(cosine_gram(x, x)), 1.0, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.property
    def test_coincides_with_linear_on_unit_rows(self) -> None:
        """On already-unit rows the cosine and linear Grams agree.

        This is the fact that makes ``mmd(kernel="linear")`` and the cosine-mean
        MMD *different operators* on Paper 1 data rather than one operator under
        two kernels: the normalization that distinguishes them happens after
        pooling, not per row.
        """
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(6, 5))
        unit = jnp.asarray(raw / np.linalg.norm(raw, axis=1, keepdims=True))
        assert jnp.allclose(cosine_gram(unit, unit), linear_gram(unit, unit), atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.contract
    def test_zero_row_is_nan(self) -> None:
        """A zero-norm row has no direction, so it yields NaN."""
        x = jnp.array([[0.0, 0.0]])
        y = jnp.array([[1.0, 0.0]])
        assert bool(jnp.all(jnp.isnan(cosine_gram(x, y))))


class TestRbfGram:
    """Tests for rbf_gram."""

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    @given(x=float_arrays((2, 2)))
    def test_diagonal_is_one(self, x: jax.Array) -> None:
        """Self-similarity is exactly one."""
        assert jnp.allclose(jnp.diagonal(rbf_gram(x, x, bandwidth=2.0)), 1.0)

    @pytest.mark.unit
    @pytest.mark.property
    def test_decays_with_distance(self) -> None:
        """Similarity falls as points separate."""
        x = jnp.array([[0.0]])
        near = jnp.array([[1.0]])
        far = jnp.array([[5.0]])
        k_near = float(rbf_gram(x, near, bandwidth=1.0)[0, 0])
        k_far = float(rbf_gram(x, far, bandwidth=1.0)[0, 0])
        assert 0.0 < k_far < k_near < 1.0

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(x=jnp.asarray(np.random.default_rng(1).normal(size=(8, 4))))
    @given(x=float_arrays((8, 4), bound=3.0))
    def test_bounded(self, x: jax.Array) -> None:
        """RBF values live in (0, 1], for inputs within ``±3`` so no underflow."""
        k = rbf_gram(x, x, bandwidth=1.5)
        assert bool(jnp.all(k > 0.0))
        assert bool(jnp.all(k <= 1.0 + 1e-6))


class TestDistanceInducedGram:
    """Tests for distance_induced_gram."""

    @pytest.mark.unit
    @pytest.mark.contract
    def test_metric_is_required(self) -> None:
        """Omitting the ground metric raises instead of returning NaN."""
        x = jnp.array([[3.0, 4.0], [0.0, 1.0]])
        with pytest.raises(TypeError, match="metric"):
            ill_typed(distance_induced_gram)(x, x)

    @pytest.mark.unit
    @pytest.mark.numerical
    def test_lean_kernel_form(self) -> None:
        """Matches ``d(x, z0) + d(y, z0) - d(x, y)`` at the origin.

        This is ``MeanEmbedding.lean:318`` ``inner_energyPointEmbed``:
        ``<Phi x, Phi y> = ||x|| + ||y|| - dist x y`` — unhalved.
        """
        x = jnp.array([[3.0, 4.0], [0.0, 1.0]])
        k = distance_induced_gram(x, x, metric="euclidean")
        norms = jnp.linalg.norm(x, axis=1)
        expected = (
            norms[:, None]
            + norms[None, :]
            - jnp.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        )
        assert jnp.allclose(k, expected, atol=1e-5)

    @pytest.mark.unit
    @pytest.mark.property
    def test_reference_shifts_the_gram(self) -> None:
        """The Gram depends on z0 — which is why it is a keyword argument."""
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        at_origin = distance_induced_gram(x, x, metric="euclidean")
        shifted = distance_induced_gram(
            x, x, metric="euclidean", reference=jnp.array([5.0, 5.0])
        )
        assert not bool(jnp.allclose(at_origin, shifted))

    @pytest.mark.unit
    @pytest.mark.property
    def test_psd_under_euclidean(self) -> None:
        """PSD holds on negative-type grounds (Lean theorem cited in the module)."""
        rng = np.random.default_rng(2)
        x = jnp.asarray(rng.normal(size=(12, 5)))
        assert _min_eigenvalue(distance_induced_gram(x, x, metric="euclidean")) >= (
            _PSD_TOLERANCE
        )

    @pytest.mark.unit
    @pytest.mark.property
    def test_exponent_is_applied(self) -> None:
        """A different exponent gives a different kernel."""
        x = jnp.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
        one = distance_induced_gram(x, x, metric="euclidean", exponent=1.0)
        half = distance_induced_gram(x, x, metric="euclidean", exponent=0.5)
        assert not bool(jnp.allclose(one, half))


class TestGramDispatcher:
    """Names never dispatch; declared strategy objects do."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("kernel", "direct"),
        [(LINEAR_KERNEL, linear_gram), (COSINE_KERNEL, cosine_gram)],
    )
    def test_declared_builtins_match_their_functions(
        self,
        kernel: SimilarityStrategy[PositiveSemidefiniteKernelLaw],
        direct: Callable[[jax.Array, jax.Array], jax.Array],
    ) -> None:
        """Evaluate built-ins only through their law-bearing strategies."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(
            gram(x, x, kernel=kernel),
            direct(x, x),
        )

    @pytest.mark.unit
    @pytest.mark.contract
    def test_angular_requires_an_explicit_reference(self) -> None:
        """The origin is not on the sphere, so require a valid reference.

        This is why :func:`statadic.discrepancy.mmd` resolves a shared reference
        from the data rather than letting the Gram default apply.
        """
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="reference is required"):
            distance_induced_gram(x, x, metric="angular")

    @pytest.mark.unit
    def test_rbf_with_bandwidth(self) -> None:
        """The factory closes bandwidth before numerical evaluation."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        assert jnp.allclose(
            gram(x, x, kernel=rbf_kernel(2.0)),
            rbf_gram(x, x, bandwidth=2.0),
        )

    @pytest.mark.component
    @pytest.mark.contract
    def test_third_party_kernel_requires_a_declaration(self) -> None:
        """Admit an extension after its kernel law is declared explicitly."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        declared = declare_similarity(
            linear_gram,
            law=PositiveSemidefiniteKernelProperties(),
            name="third-party-linear",
            required_evidence=FiniteValues,
        )
        assert jnp.allclose(gram(x, x, kernel=declared), linear_gram(x, x))

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("bare", ["linear", linear_gram, 42])
    def test_undeclared_inputs_raise(self, bare: object) -> None:
        """Reject name and callable dispatch without theoretical metadata."""
        x = jnp.array([[1.0, 2.0]])
        with pytest.raises(TypeError, match="SimilarityStrategy"):
            gram(x, x, kernel=ill_typed(bare))


def evaluate_unit_rbf(x: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate the RBF example at its fixed bandwidth."""
    return rbf_gram(x, y, bandwidth=1.0)


class TestDeclaredPsd:
    """Exercise the PSD promises carried by executable strategies."""

    @pytest.mark.unit
    @pytest.mark.property
    @examples(40)
    @example(
        builder=linear_gram,
        x=jnp.asarray(np.random.default_rng(3).normal(size=(10, 4))),
    )
    @example(
        builder=cosine_gram,
        x=jnp.asarray(np.random.default_rng(3).normal(size=(10, 4))),
    )
    @example(
        builder=evaluate_unit_rbf,
        x=jnp.asarray(np.random.default_rng(3).normal(size=(10, 4))),
    )
    @given(
        builder=st.sampled_from([linear_gram, cosine_gram, evaluate_unit_rbf]),
        x=nonzero_rows(10, 4),
    )
    def test_declared_strategies_are_psd(
        self, builder: Callable[[jax.Array, jax.Array], jax.Array], x: jax.Array
    ) -> None:
        """Built-in declared strategies are PSD on nonzero-row input."""
        assert _min_eigenvalue(builder(x, x)) >= _PSD_TOLERANCE


class TestJitCompatibility:
    """The registry has to survive the transforms the package runs on."""

    @pytest.mark.component
    @pytest.mark.contract
    def test_kernel_is_static_under_jit(self) -> None:
        """``gram`` traces with the declared strategy held static."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def evaluate_linear_self_gram(a: jax.Array) -> jax.Array:
            return gram(a, a, kernel=LINEAR_KERNEL)

        jitted = jax.jit(evaluate_linear_self_gram)
        assert jnp.allclose(jitted(x), linear_gram(x, x))

    @pytest.mark.component
    @pytest.mark.contract
    def test_distance_induced_traces(self) -> None:
        """Metric and exponent are static arguments, so a plain trace works."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        def evaluate_energy_self_gram(a: jax.Array) -> jax.Array:
            return distance_induced_gram(a, a, metric="euclidean", exponent=1.0)

        jitted = jax.jit(evaluate_energy_self_gram)
        assert jitted(x).shape == (2, 2)
