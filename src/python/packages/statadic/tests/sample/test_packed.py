# standalone package-maintenance entry point
"""Tests for the PackedClouds substrate (t13, hardened by t61)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import FrozenInstanceError, is_dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax import config
from statadic.sample import packed as packed_module
from statadic.sample.packed import (
    PackedClouds,
    checked_packed_clouds,
    masked_mean,
    masked_sum,
    pack,
    pair_chunks,
)

from .._budgets import examples
from .._strategies import float_arrays

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from hypothesis.strategies import SearchStrategy

config.update("jax_enable_x64", val=True)


class TestPack:
    """Check packing of unequal-size candidate clouds."""

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_shapes(self) -> None:
        """Verify pack shapes."""
        clouds = [jnp.ones((2, 3)), jnp.ones((5, 3)), jnp.ones((1, 3))]
        packed = pack(clouds)
        assert isinstance(packed, PackedClouds)
        assert packed.data.shape == (3, 5, 3)
        assert list(np.asarray(packed.counts)) == [2, 5, 1]

    @pytest.mark.unit
    @pytest.mark.property
    @example(
        c0=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        c1=jnp.array([[5.0, 6.0]]),
    )
    @given(c0=float_arrays((2, 2)), c1=float_arrays((1, 2)))
    @examples(40)
    def test_pack_preserves_values_in_leading_block(
        self, c0: jax.Array, c1: jax.Array
    ) -> None:
        """Verify pack preserves values in leading block."""
        packed = pack([c0, c1])
        assert jnp.array_equal(packed.data[0, :2], c0)
        assert jnp.array_equal(packed.data[1, :1], c1)
        # Padding is zero.
        assert jnp.array_equal(packed.data[1, 1:], jnp.zeros((1, 2)))

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_preserves_first_cloud_dtype(self) -> None:
        """Host-side padding keeps the dtype selected from the first cloud."""
        first = jnp.ones((2, 2), dtype=jnp.float64)
        second = jnp.ones((1, 2), dtype=jnp.float32)
        assert pack([first, second]).data.dtype == first.dtype

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_preserves_integer_dtype(self) -> None:
        """Shape hardening does not narrow the historical numeric carrier."""
        clouds = [jnp.ones((2, 2), dtype=jnp.int16), jnp.ones((1, 2), dtype=jnp.int16)]
        assert pack(clouds).data.dtype == jnp.int16

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_preserves_empty_input(self) -> None:
        """The historical empty carrier has zero extent on every data axis."""
        packed = pack([])
        assert packed.data.shape == (0, 0, 0)
        assert packed.data.dtype == jnp.float32
        assert packed.counts.shape == (0,)
        assert packed.counts.dtype == jnp.int32

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_rejects_inconsistent_feature_widths(self) -> None:
        """Feature width remains the one non-ragged cloud dimension."""
        with pytest.raises(ValueError, match="inconsistent cloud feature widths"):
            pack([jnp.ones((2, 2)), jnp.ones((3, 4))])

    @pytest.mark.unit
    @pytest.mark.contract
    @pytest.mark.parametrize("shape", [(), (3,), (2, 3, 4)])
    def test_pack_rejects_non_matrix_clouds(self, shape: tuple[int, ...]) -> None:
        """Rank errors fail at the eager boundary rather than during assignment."""
        with pytest.raises(ValueError, match="must be two-dimensional"):
            pack([jnp.ones(shape)])

    @pytest.mark.unit
    @pytest.mark.contract
    def test_pack_empty_cloud_policy_is_explicit(self) -> None:
        """The generic door admits zeros; the empirical door requires positivity."""
        generic = pack([jnp.ones((0, 2))])
        assert generic.data.shape == (1, 0, 2)
        np.testing.assert_array_equal(generic.counts, [0])

        with pytest.raises(ValueError, match="counts must lie in"):
            pack([jnp.ones((0, 2))], require_positive_counts=True)
        with pytest.raises(ValueError, match="at least one cloud"):
            pack([], require_positive_counts=True)

    @pytest.mark.integration
    @pytest.mark.contract
    @pytest.mark.parametrize("cloud_count", [0, 1, 8, 32])
    def test_pack_performs_no_host_round_trip(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cloud_count: int,
    ) -> None:
        """Packing stays on device: zero transfers, whatever the cloud count.

        This supersedes the t61 contract that `pack` make exactly *one* tree
        `device_get`/`device_put` pair. That bound was the best available while
        the padded grid was assembled in a NumPy host buffer -- one tree call
        rather than one physical copy per leaf. t65's fused scatter builds the
        grid on device, so the correct bound is now zero, and asserting it
        pins the host door shut rather than merely narrow.
        """
        original_device_get = packed_module.jax.device_get
        original_device_put = packed_module.jax.device_put
        device_get_payloads: list[object] = []
        transferred_payloads: list[object] = []

        def _counted_device_get(value: object) -> object:
            device_get_payloads.append(value)
            return original_device_get(value)

        def _counted_device_put(value: object) -> object:
            transferred_payloads.append(value)
            return original_device_put(value)

        monkeypatch.setattr(packed_module.jax, "device_get", _counted_device_get)
        monkeypatch.setattr(packed_module.jax, "device_put", _counted_device_put)
        packed = pack([jnp.ones((size, 2)) for size in range(1, cloud_count + 1)])
        assert device_get_payloads == []
        assert transferred_payloads == []
        # Not vacuous: the scatter still has to have produced the right grid.
        assert isinstance(packed.data, jax.Array)
        assert isinstance(packed.counts, jax.Array)
        m_max = cloud_count
        assert packed.data.shape == (cloud_count, m_max, 2 if cloud_count else 0)
        assert list(np.asarray(packed.counts)) == list(range(1, cloud_count + 1))

    @pytest.mark.component
    @pytest.mark.contract
    def test_pack_is_registered_frozen_pytree(self) -> None:
        """Registration keeps coordinates, counts and masses in field order."""
        packed = pack([jnp.ones((2, 2)), jnp.ones((3, 2))])
        leaves = jax.tree_util.tree_leaves(packed)
        treedef = jax.tree_util.tree_structure(packed)
        assert is_dataclass(packed)
        assert leaves[0] is packed.data
        assert leaves[1] is packed.counts
        assert leaves[2] is packed.masses
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, PackedClouds)
        assert vars(PackedClouds)["__dataclass_params__"].frozen
        with pytest.raises(FrozenInstanceError):
            packed.__setattr__("data", jnp.zeros_like(packed.data))

    @pytest.mark.integration
    @pytest.mark.contract
    def test_packed_clouds_survives_jax_transforms(self) -> None:
        """Checked leading batch axes admit JIT, vmap, and lax.map rebuilding."""
        first = pack([jnp.ones((2, 2)), jnp.ones((3, 2))])
        second = pack([2 * jnp.ones((2, 2)), 3 * jnp.ones((3, 2))])
        batched = PackedClouds(
            data=jnp.stack((first.data, second.data)),
            counts=jnp.stack((first.counts, second.counts)),
        )

        def rebuild(value: PackedClouds) -> PackedClouds:
            return PackedClouds(data=value.data, counts=value.counts)

        jitted = jax.jit(rebuild)(first)
        vmapped = jax.vmap(rebuild)(batched)
        lax_mapped = jax.lax.map(rebuild, batched)

        assert jitted.data.shape == (2, 3, 2)
        assert vmapped.data.shape == (2, 2, 3, 2)
        assert vmapped.counts.shape == (2, 2)
        assert lax_mapped.data.shape == vmapped.data.shape
        assert lax_mapped.counts.shape == vmapped.counts.shape

    @pytest.mark.component
    @pytest.mark.contract
    @pytest.mark.needs_import_hook
    @pytest.mark.parametrize(
        ("data_shape", "counts_shape", "counts_dtype"),
        [
            ((3, 4), (3,), jnp.int32),
            ((2, 3, 4), (), jnp.int32),
            ((2, 3, 4), (3,), jnp.int32),
            ((2, 2, 3, 4), (3, 2), jnp.int32),
            ((2, 3, 4), (2,), jnp.float32),
        ],
    )
    def test_packed_clouds_shape_is_enforced_at_runtime(
        self,
        data_shape: tuple[int, ...],
        counts_shape: tuple[int, ...],
        counts_dtype: jnp.dtype,
    ) -> None:
        """Wrong ranks, axes, or count dtype fail under the import hook."""
        with pytest.raises(TypeError if counts_dtype == jnp.float32 else ValueError):
            PackedClouds(
                data=jnp.ones(data_shape),
                counts=jnp.ones(counts_shape, dtype=counts_dtype),
            )

    @pytest.mark.integration
    @pytest.mark.contract
    def test_structural_checks_do_not_depend_on_the_import_hook(self) -> None:
        """A fresh non-statadic module copy still rejects malformed carrier structure."""
        module_name = "_statadic_packed_without_import_hook"
        module_path = packed_module.__file__
        assert module_path is not None
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            module_namespace = vars(module)
            raw_packed_clouds = cast(
                "Callable[..., object]", module_namespace["PackedClouds"]
            )
            raw_pack = cast("Callable[..., object]", module_namespace["pack"])
            raw_checked_packed_clouds = cast(
                "Callable[..., object]", module_namespace["checked_packed_clouds"]
            )

            with pytest.raises(ValueError, match="coordinate/count axes"):
                raw_packed_clouds(
                    data=jnp.ones((3, 4)),
                    counts=jnp.ones((3,), dtype=jnp.int32),
                )
            with pytest.raises(ValueError, match="coordinate/count axes"):
                raw_packed_clouds(
                    data=jnp.ones((2, 2, 3, 4)),
                    counts=jnp.ones((3, 2), dtype=jnp.int32),
                )
            with pytest.raises(TypeError, match="signed integers"):
                raw_packed_clouds(
                    data=jnp.ones((2, 3, 4)),
                    counts=jnp.ones((2,), dtype=jnp.float32),
                )
            with pytest.raises(TypeError, match="must be a JAX array"):
                raw_packed_clouds(
                    data=np.ones((2, 3, 4)),
                    counts=np.ones((2,), dtype=np.int32),
                )
            with pytest.raises(TypeError, match="masses must be a JAX array"):
                raw_packed_clouds(
                    data=jnp.ones((2, 3, 4)),
                    counts=jnp.full((2,), 3, dtype=jnp.int32),
                    masses=np.full((2, 3), 1 / 3),
                )
            with pytest.raises(
                TypeError, match="require_positive_counts must be a bool"
            ):
                raw_pack(
                    [jnp.ones((2, 1))],
                    require_positive_counts=1,
                )
            with pytest.raises(
                TypeError, match="require_positive_counts must be a bool"
            ):
                raw_checked_packed_clouds(
                    jnp.zeros((1, 2, 1)),
                    jnp.asarray([1], dtype=jnp.int32),
                    require_positive_counts="yes",
                )
        finally:
            sys.modules.pop(module_name, None)

    @pytest.mark.component
    @pytest.mark.contract
    @pytest.mark.parametrize(
        ("counts", "require_positive_counts"),
        [([-1, 2], False), ([1, 4], False), ([0, 2], True)],
    )
    def test_checked_factory_rejects_invalid_count_values(
        self,
        counts: list[int],
        *,
        require_positive_counts: bool,
    ) -> None:
        """The eager factory enforces bounds the tracer-safe carrier cannot."""
        data = jnp.zeros((2, 3, 1))
        device_counts = jnp.asarray(counts, dtype=jnp.int32)
        # The raw constructor is deliberately structural-only.
        PackedClouds(data=data, counts=device_counts)
        with pytest.raises(ValueError, match="counts must lie in"):
            checked_packed_clouds(
                data,
                device_counts,
                require_positive_counts=require_positive_counts,
            )

    @pytest.mark.component
    @pytest.mark.contract
    def test_checked_factory_accepts_generic_zero_and_positive_empirical_counts(
        self,
    ) -> None:
        """The two eager value-law regimes are explicit and stage-exported."""
        data = jnp.zeros((2, 3, 1))
        generic = checked_packed_clouds(
            data,
            jnp.asarray([0, 3], dtype=jnp.int32),
        )
        empirical = checked_packed_clouds(
            data,
            jnp.asarray([1, 3], dtype=jnp.int32),
            require_positive_counts=True,
        )
        assert generic.counts[0] == 0
        assert empirical.counts[0] == 1

    @pytest.mark.component
    @pytest.mark.contract
    def test_checked_factory_rejects_tracers_instead_of_faking_value_checks(
        self,
    ) -> None:
        """Concrete count validation is explicitly outside transformations."""
        data = jnp.zeros((2, 3, 1))
        counts = jnp.asarray([1, 3], dtype=jnp.int32)
        with pytest.raises(TypeError, match="eager value-law boundary"):
            jax.jit(checked_packed_clouds)(data, counts)


def _integer_valued_matrices() -> SearchStrategy[jax.Array]:
    """Integer-valued floats, so every summation order gives the same result."""
    return float_arrays((4, 3)).map(jnp.round)


class TestMaskedReductions:
    """Check masked reductions over packed candidate clouds."""

    @pytest.mark.unit
    @pytest.mark.reference
    @example(values=jnp.arange(12, dtype=jnp.float64).reshape(4, 3), rows=2, columns=2)
    @given(
        values=_integer_valued_matrices(),
        rows=st.integers(1, 4),
        columns=st.integers(1, 3),
    )
    @examples(40)
    def test_masked_sum_matches_direct_sum_on_leading_block(
        self, values: jax.Array, rows: int, columns: int
    ) -> None:
        """Verify masked sum matches direct sum on leading block."""
        got = masked_sum(values, jnp.asarray(rows), jnp.asarray(columns))
        expected = jnp.sum(values[:rows, :columns])
        assert jnp.array_equal(got, expected)

    @pytest.mark.unit
    @pytest.mark.reference
    @example(values=jnp.arange(12, dtype=jnp.float64).reshape(4, 3), rows=3, columns=2)
    @given(
        values=_integer_valued_matrices(),
        rows=st.integers(1, 4),
        columns=st.integers(1, 3),
    )
    @examples(40)
    def test_masked_mean_matches_direct_mean_on_leading_block(
        self, values: jax.Array, rows: int, columns: int
    ) -> None:
        """Verify masked mean matches direct mean on leading block."""
        got = masked_mean(values, jnp.asarray(rows), jnp.asarray(columns))
        expected = jnp.mean(values[:rows, :columns])
        assert jnp.allclose(got, expected)


class TestPairChunks:
    """Check padded upper-triangle pair chunking."""

    @pytest.mark.unit
    @pytest.mark.property
    @example(n=5, k=0, chunk=4)
    @given(n=st.integers(2, 7), k=st.sampled_from([0, 1]), chunk=st.integers(1, 6))
    @examples(40)
    def test_pair_chunks_covers_upper_triangle(
        self, n: int, k: int, chunk: int
    ) -> None:
        """Verify pair chunks covers upper triangle."""
        ii, jj, n_pairs, chunks = pair_chunks(n, k=k, chunk=chunk)
        exp_ii, exp_jj = np.triu_indices(n, k=k)
        assert n_pairs == exp_ii.shape[0]
        assert np.array_equal(ii, exp_ii)
        assert np.array_equal(jj, exp_jj)
        flat = chunks.reshape(-1, 2)[:n_pairs]
        assert np.array_equal(flat[:, 0], exp_ii)
        assert np.array_equal(flat[:, 1], exp_jj)
