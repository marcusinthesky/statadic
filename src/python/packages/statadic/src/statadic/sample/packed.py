"""Immutable empirical distributions with explicit ragged lengths and masses.

Coordinates, integer counts and atom probabilities are dynamic pytree leaves.
Checked factories establish probability-domain values; numerical construction
checks structure only. Pytree reconstruction bypasses the constructor so JAX
can rebuild shapes, placeholders and transformed leaves without validation.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Int  # noqa: TC002

from statadic.core.typing import Array, ArrayLike  # noqa: TC001

__all__ = [
    "PackedClouds",
    "checked_packed_clouds",
    "masked_mean",
    "masked_row_mask",
    "masked_sum",
    "pack",
    "pair_chunks",
]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_PACKED_DATA_AXES = 3
_CLOUD_DIMENSIONS = 2


def _require_jax_array(value: Array, *, field_name: str) -> None:
    """Require an immutable JAX array or a transform-time tracer leaf."""
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return
    message = (
        f"PackedClouds.{field_name} must be a JAX array; use pack or "
        "checked_packed_clouds to construct immutable leaves"
    )
    raise TypeError(message)


def _require_count_values(
    counts: Array,
    *,
    m_max: int,
    require_positive_counts: bool,
) -> None:
    """Validate concrete count bounds at an eager host boundary."""
    minimum = 1 if require_positive_counts else 0
    if require_positive_counts and counts.size == 0:
        message = "positive-count PackedClouds require at least one cloud"
        raise ValueError(message)
    if bool(jnp.any(counts < minimum)) or bool(jnp.any(counts > m_max)):
        interval = f"[1, {m_max}]" if require_positive_counts else f"[0, {m_max}]"
        message = f"PackedClouds counts must lie in {interval}, got {counts.tolist()}"
        raise ValueError(message)


def _require_boolean_flag(value: object, *, name: str) -> bool:
    """Validate an eager boolean option without relying on the import hook."""
    if not isinstance(value, bool):
        message = f"{name} must be a bool, got {type(value).__name__}"
        raise TypeError(message)
    return bool(value)


@jax.jit
def _uniform_support(data: Array) -> tuple[Array, Array]:
    """Generate fixed-support counts and uniform masses in a single dispatch."""
    n = data.shape[-2]
    counts = jnp.full(data.shape[:-2], n, dtype=jnp.int32)
    dtype = jnp.result_type(data, 1.0)
    atom_mass = jnp.asarray(1, dtype=dtype) / jnp.asarray(max(n, 1), dtype=dtype)
    masses = jnp.broadcast_to(atom_mass, data.shape[:-1])
    return counts, masses


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, init=False)
class PackedClouds:
    """Empirical distributions with coordinates, true counts and atom masses.

    All three arrays are dynamic pytree leaves. Ordinary construction checks
    structure; checked factories establish probability-domain values. Native
    pytree reconstruction bypasses the constructor, including for placeholders.

    Attributes:
        data: Padded clouds, shape ``(*batch, K, m_max, d)``. :func:`pack`
            writes zero padding; external inactive entries are ignored by the
            masked reducers and are not value-checked.
        counts: True (unpadded) size of each cloud, shape ``(*batch, K)``.
        masses: Atom probabilities, shape ``(*batch, K, m_max)``; padding is zero.
            A single distribution omits the K axis from all three fields.

    """

    data: Array
    counts: Array
    masses: Array

    def __init__(self, data: Array, counts: Array, masses: Array | None = None) -> None:
        """Enforce the carrier's tracer-safe structural contract in production."""
        _require_jax_array(data, field_name="data")
        _require_jax_array(counts, field_name="counts")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "counts", counts)
        if (
            self.data.ndim < _CLOUD_DIMENSIONS
            or self.counts.shape != self.data.shape[:-2]
        ):
            msg = "PackedClouds coordinate/count axes must agree"
            raise ValueError(msg)
        if not (
            jnp.issubdtype(self.data.dtype, jnp.floating)
            or jnp.issubdtype(self.data.dtype, jnp.integer)
        ):
            msg = "PackedClouds supports must be real numeric arrays"
            raise TypeError(msg)
        if not jnp.issubdtype(self.counts.dtype, jnp.signedinteger):
            msg = "PackedClouds counts must be signed integers"
            raise TypeError(msg)
        if masses is None:
            mask = jnp.arange(self.data.shape[-2]) < self.counts[..., None]
            dtype = jnp.result_type(self.data, 1.0)
            masses = mask.astype(dtype) / jnp.maximum(self.counts[..., None], 1).astype(
                dtype
            )
        _require_jax_array(masses, field_name="masses")
        object.__setattr__(self, "masses", masses)
        if self.masses.shape != self.data.shape[:-1] or not jnp.issubdtype(
            self.masses.dtype, jnp.floating
        ):
            msg = "PackedClouds masses must be floating and match atom axes"
            raise ValueError(msg)

    def tree_flatten(self) -> tuple[tuple[object, object, object], None]:
        """Expose coordinates, counts and masses as dynamic leaves."""
        return (self.data, self.counts, self.masses), None

    @classmethod
    def tree_unflatten(
        cls, metadata: object, leaves: tuple[object, ...]
    ) -> "PackedClouds":
        """Rebuild transformed leaves without invoking constructor validation."""
        del metadata
        result = object.__new__(cls)
        for name, value in zip(("data", "counts", "masses"), leaves, strict=True):
            object.__setattr__(result, name, value)
        return result

    @property
    def active_mask(self) -> Array:
        """True support positions, excluding inactive padding."""
        return jnp.arange(self.data.shape[-2]) < self.counts[..., None]

    @property
    def coordinates(self) -> Array:
        """Coordinates with inactive storage replaced before arithmetic."""
        return jnp.where(
            (self.active_mask & (self.masses > 0))[..., None], self.data, 0
        )

    @property
    def valid(self) -> Array:
        """Numerical probability-domain validity per distribution."""
        active = self.active_mask
        return (
            (self.counts > 0)
            & (self.data.shape[-1] > 0)
            & (self.counts <= self.data.shape[-2])
            & jnp.all(
                (~(active & (self.masses > 0))[..., None]) | jnp.isfinite(self.data),
                axis=(-2, -1),
            )
            & jnp.all(
                jnp.isfinite(self.masses)
                & (self.masses >= 0)
                & (active | (self.masses == 0)),
                axis=-1,
            )
            & (
                jnp.abs(jnp.sum(self.masses, axis=-1) - 1)
                <= 8 * jnp.finfo(self.masses.dtype).eps
            )
        )

    @property
    def is_uniform(self) -> Array:
        """Whether probabilities are uniform over the true support."""
        expected = self.active_mask.astype(self.masses.dtype) / jnp.maximum(
            self.counts[..., None], 1
        ).astype(self.masses.dtype)
        return jnp.all(self.masses == expected, axis=-1)

    @property
    def n_observations(self) -> int:
        """Number of distributions on the last cloud axis."""
        return self.data.shape[-3] if self.data.ndim > _CLOUD_DIMENSIONS else 1

    def mean(self) -> Array:
        """Mass-weighted mean of each empirical distribution."""
        return jnp.sum(self.coordinates * self.masses[..., None], axis=-2)

    def expectation(
        self, other: "PackedClouds", point_values: "Callable[[Array, Array], Array]"
    ) -> Array:
        """Integrate a point-pair matrix against the two empirical probabilities."""

        def pair(a: "PackedClouds", b: "PackedClouds") -> Array:
            active = (a.masses[:, None] > 0) & (b.masses[None, :] > 0)
            values = point_values(a.coordinates, b.coordinates)
            total = jnp.sum(
                jnp.where(active, values, 0) * a.masses[:, None] * b.masses[None, :]
            )
            return jnp.where(a.valid & b.valid, total, jnp.nan)

        return self.map_aligned(other, pair)

    def __getitem__(self, index: int | Array) -> "PackedClouds":
        """Index cloud axes without changing atom coordinates or probabilities."""
        if self.counts.ndim == 0:
            message = "Cloud indexing requires a collection, not a single distribution"
            raise ValueError(message)
        return PackedClouds(self.data[index], self.counts[index], self.masses[index])

    def with_coordinates(self, data: Array) -> "PackedClouds":
        """Replace coordinates, retaining support topology and masses."""
        if data.shape[:-1] != self.data.shape[:-1] or not data.shape[-1]:
            msg = "Replacement coordinates must retain cloud and atom axes"
            raise ValueError(msg)
        return PackedClouds(data, self.counts, self.masses)

    def affine(self, matrix: Array, offset: Array | None = None) -> "PackedClouds":
        """Apply a shared differentiable affine coordinate transformation."""
        values = self.coordinates @ matrix
        if offset is not None:
            values = values + offset
        return self.with_coordinates(jnp.where(self.active_mask[..., None], values, 0))

    def with_mass_logits(self, logits: Array) -> "PackedClouds":
        """Parameterize active probabilities, retaining zero mass on padding."""
        if logits.shape != self.data.shape[:-1]:
            msg = "Mass logits must match cloud and atom axes"
            raise ValueError(msg)
        masked = jnp.where(self.active_mask, logits, -jnp.inf)
        mass = jnp.where(self.active_mask, jax.nn.softmax(masked, axis=-1), 0)
        return PackedClouds(self.data, self.counts, mass)

    def take(self, indices: "ArrayLike | Sequence[int | float]") -> "PackedClouds":
        """Select whole distributions at an eager identity boundary."""
        if self.counts.ndim == 0:
            message = "Cloud selection requires a collection, not a single distribution"
            raise ValueError(message)
        raw: object = (
            list(indices)
            if isinstance(indices, (list, tuple))
            else getattr(indices, "tolist", lambda: None)()
        )
        if (
            not isinstance(raw, list)
            or not raw
            or any(type(i) is not int or i < 0 or i >= self.n_observations for i in raw)
        ):
            msg = "Measure selection requires nonempty in-bounds integer indices"
            raise ValueError(msg)
        selected = jnp.asarray(raw)
        return PackedClouds(
            jnp.take(self.data, selected, axis=-3),
            jnp.take(self.counts, selected, axis=-1),
            jnp.take(self.masses, selected, axis=-2),
        )

    @classmethod
    def from_points(
        cls,
        points: "ArrayLike | Sequence[Sequence[int | float]]",
        masses: "ArrayLike | Sequence[int | float] | None" = None,
        *,
        normalize: bool = False,
    ) -> "PackedClouds":
        """Validate one distribution with optional atom probabilities."""
        collection = cls.from_clouds(
            [points], None if masses is None else [masses], normalize=normalize
        )
        return cls(collection.data[0], collection.counts[0], collection.masses[0])

    @classmethod
    def from_array(cls, data: Array, masses: Array | None = None) -> "PackedClouds":
        """Build fixed-shape numerical supports; caller owns probability values.

        All atom positions are active. This structural door is transformable;
        use from_points/from_clouds to validate external values eagerly.
        """
        if masses is None:
            counts, uniform = _uniform_support(data)
            return cls(data, counts, uniform)
        if masses.shape[-1] not in (1, data.shape[-2]):
            msg = "Atom weights must match the support atom axis"
            raise ValueError(msg)
        axes = jnp.broadcast_shapes(data.shape[:-2], masses.shape[:-1])
        data = jnp.broadcast_to(data, (*axes, *data.shape[-2:]))
        masses = jnp.broadcast_to(masses, (*axes, data.shape[-2]))
        return cls(
            data, jnp.full(data.shape[:-2], data.shape[-2], dtype=jnp.int32), masses
        )

    def map_aligned(
        self,
        other: "PackedClouds",
        operation: "Callable[[PackedClouds, PackedClouds], Array]",
    ) -> Array:
        """Apply one pair operation over broadcast cloud axes using native vmap."""
        axes = jnp.broadcast_shapes(self.data.shape[:-2], other.data.shape[:-2])
        if not axes:
            return operation(self, other)

        def broadcast(value: "PackedClouds") -> "PackedClouds":
            capacity, dimensions = value.data.shape[-2:]
            return PackedClouds(
                jnp.broadcast_to(value.data, (*axes, capacity, dimensions)).reshape(
                    -1, capacity, dimensions
                ),
                jnp.broadcast_to(value.counts, axes).reshape(-1),
                jnp.broadcast_to(value.masses, (*axes, capacity)).reshape(-1, capacity),
            )

        result = jax.vmap(operation)(broadcast(self), broadcast(other))
        return result.reshape(axes + result.shape[1:])

    def map_cross(
        self,
        other: "PackedClouds",
        operation: "Callable[[PackedClouds, PackedClouds], Array]",
    ) -> Array:
        """Apply a pair operation to every pair of two unbatched collections."""
        if (
            self.data.ndim != _PACKED_DATA_AXES
            or other.data.ndim != _PACKED_DATA_AXES
            or not self.n_observations
            or not other.n_observations
        ):
            msg = (
                "Cross operations require nonempty cloud collections; "
                "use vmap for batches"
            )
            raise ValueError(msg)

        def row(a: "PackedClouds") -> Array:
            def entry(b: "PackedClouds") -> Array:
                return operation(a, b)

            return jax.vmap(entry)(other)

        return jax.vmap(row)(self)

    @classmethod
    def from_clouds(
        cls,
        clouds: "Sequence[ArrayLike | Sequence[Sequence[int | float]]]",
        masses: "Sequence[ArrayLike | Sequence[int | float]] | None" = None,
        *,
        normalize: bool = False,
    ) -> "PackedClouds":
        """Validate ragged supports and supplied or uniform probabilities."""
        if not isinstance(normalize, bool):
            msg = "normalize must be a bool"
            raise TypeError(msg)
        arrays = [jnp.asarray(cloud) for cloud in clouds]
        if not arrays or any(
            a.ndim != _CLOUD_DIMENSIONS or not all(a.shape) for a in arrays
        ):
            msg = "Measure supports must be nonempty matrices with nonempty features"
            raise ValueError(msg)
        supplied = None if masses is None else [jnp.asarray(m) for m in masses]
        if supplied is not None and len(supplied) != len(arrays):
            msg = "One mass vector is required for each support"
            raise ValueError(msg)
        dtype = jnp.result_type(*arrays, *(supplied or []), 1.0)
        if not jnp.issubdtype(dtype, jnp.floating):
            msg = "Empirical supports and masses must be real"
            raise TypeError(msg)
        packed = pack([a.astype(dtype) for a in arrays], require_positive_counts=True)
        rows = []
        for i, a in enumerate(arrays):
            mass = (
                jnp.full((a.shape[0],), 1 / a.shape[0], dtype=dtype)
                if supplied is None
                else supplied[i].astype(dtype)
            )
            if (
                mass.shape != (a.shape[0],)
                or not bool(jnp.all(jnp.isfinite(mass) & (mass >= 0)))
                or not bool(jnp.any(mass > 0))
            ):
                msg = (
                    "Atom masses must be finite nonnegative vectors with positive total"
                )
                raise ValueError(msg)
            if normalize:
                mantissa, exponent = jnp.frexp(mass)
                largest = jnp.max(
                    jnp.where(mass > 0, exponent, jnp.iinfo(exponent.dtype).min)
                )
                scaled = jnp.ldexp(mantissa, exponent - largest)
                mass = scaled / jnp.sum(scaled)
            elif not bool(jnp.abs(jnp.sum(mass) - 1) <= 8 * jnp.finfo(dtype).eps):
                msg = "Atom masses must sum to one; use normalize=True explicitly"
                raise ValueError(msg)
            rows.append(jnp.pad(mass, (0, packed.data.shape[-2] - mass.size)))
        return checked_packed_clouds(
            packed.data, packed.counts, jnp.stack(rows), require_positive_counts=True
        )


def checked_packed_clouds(
    data: Array,
    counts: Array,
    masses: Array | None = None,
    *,
    require_positive_counts: bool = False,
) -> PackedClouds:
    """Construct a packed carrier after eager validation of count values.

    This is the external-buffer door. It is intentionally host-only: count
    values cannot be tested with Python control flow while they are JAX
    tracers. It proves count bounds only; inactive padding values are ignored,
    not inspected or claimed to be zero. The raw :class:`PackedClouds`
    constructor remains available inside ``jit``/``vmap``/``lax.map`` and
    enforces structure only.

    Args:
        data: Padded JAX array of shape ``(*batch, K, m_max, d)``.
        counts: True row extents, shape ``(*batch, K)``.
        masses: Optional atom probabilities, with zero mass on padding.
        require_positive_counts: Require every extent to be at least one and
            reject an empty carrier. Statistical consumers such as T62 should
            enable this; the generic carrier retains zero-count compatibility.

    Returns:
        A structurally and value-checked :class:`PackedClouds`.

    Raises:
        TypeError: If called while ``counts`` is a JAX tracer, or if the boolean
            option or structural leaf/dtype contract is invalid.
        ValueError: If structural axes or a count value are outside their contracts.

    """
    positive_counts = _require_boolean_flag(
        require_positive_counts,
        name="require_positive_counts",
    )
    packed = PackedClouds(data=data, counts=counts, masses=masses)
    if isinstance(packed.counts, jax.core.Tracer):
        message = (
            "checked_packed_clouds is an eager value-law boundary; construct "
            "PackedClouds directly inside JAX transformations"
        )
        raise TypeError(message)
    host_counts = jnp.asarray(jax.device_get(packed.counts))
    _require_count_values(
        host_counts,
        m_max=int(packed.data.shape[-2]),
        require_positive_counts=positive_counts,
    )
    active = packed.active_mask
    if not bool(
        jnp.all(
            (~(active & (packed.masses > 0))[..., None]) | jnp.isfinite(packed.data)
        )
    ):
        msg = "Every active support coordinate must be finite"
        raise ValueError(msg)
    if not bool(
        jnp.all(jnp.isfinite(packed.masses) & (packed.masses >= 0))
    ) or not bool(jnp.all(active | (packed.masses == 0))):
        msg = "Atom masses must be finite, nonnegative and zero on padding"
        raise ValueError(msg)
    totals = jnp.sum(packed.masses, axis=-1)
    if not bool(
        jnp.all(
            jnp.abs(totals - jnp.where(packed.counts > 0, 1, 0))
            <= 8 * jnp.finfo(packed.masses.dtype).eps
        )
    ):
        msg = "Atom masses must sum to one"
        raise ValueError(msg)
    return packed


def pack(
    clouds: "Sequence[ArrayLike] | Array",
    *,
    require_positive_counts: bool = False,
) -> PackedClouds:
    """Zero-pad a ragged list of ``(m_k, d)`` clouds into a ``PackedClouds``.

    Args:
        clouds: List of ``K`` arrays, each of shape ``(m_k, d)``; ``m_k`` may
            vary across clouds, but ``d`` must be shared. Nonempty rows use
            their shared promoted dtype under the caller's JAX configuration.
        require_positive_counts: Require a nonempty list and ``m_k > 0`` for
            every cloud. Leave false for the historical ``pack([])`` result and
            for generic carriers that deliberately admit empty clouds.

    Returns:
        ``PackedClouds`` with ``data`` shape ``(K, m_max, d)`` (``m_max`` the
        largest ``m_k``) and ``counts`` shape ``(K,)``. The counts are derived
        from the original cloud shapes rather than supplied by the caller.

    Raises:
        TypeError: If ``require_positive_counts`` is not a boolean.
        ValueError: If a cloud is not rank two, feature widths differ, or the
            positive-count door receives an empty list/cloud.
        OverflowError: If a derived row count cannot fit the preserved int32 dtype.

    """
    positive_counts = _require_boolean_flag(
        require_positive_counts,
        name="require_positive_counts",
    )
    arrays = [jnp.asarray(cloud) for cloud in clouds]
    k = len(arrays)
    for index, cloud in enumerate(arrays):
        if not hasattr(cloud, "shape") or len(cloud.shape) != _CLOUD_DIMENSIONS:
            shape = getattr(cloud, "shape", None)
            message = f"cloud {index} must be two-dimensional, got shape {shape}"
            raise ValueError(message)
    dims = {int(cloud.shape[1]) for cloud in arrays}
    if len(dims) > 1:
        message = f"inconsistent cloud feature widths: {dims}"
        raise ValueError(message)
    d = dims.pop() if dims else 0
    sizes = [int(cloud.shape[0]) for cloud in arrays]
    if any(size > jnp.iinfo(jnp.int32).max for size in sizes):
        message = "cloud row counts must fit the preserved int32 count dtype"
        raise OverflowError(message)
    # These extents come from static array shapes, not external count values.
    # Validate in Python before array creation so pack remains usable under JIT.
    m_max = max(sizes) if k else 0
    if positive_counts and not sizes:
        message = "positive-count PackedClouds require at least one cloud"
        raise ValueError(message)
    if positive_counts and any(size == 0 for size in sizes):
        message = f"PackedClouds counts must lie in [1, {m_max}], got {sizes}"
        raise ValueError(message)
    counts = jnp.asarray(sizes, dtype=jnp.int32)
    dtype = arrays[0].dtype if k else jnp.float32
    # One fused scatter, not a per-cloud loop. `m_max` comes from Python
    # `.shape[0]` ints, so every index below is static; the rows are
    # concatenated once and written in a single `.at[].set`, which measured
    # ~9x faster than the NumPy buffer-and-slice-assign it replaces and
    # bit-identical to it. The obvious alternative — `jnp.pad` each cloud then
    # `stack` — measured ~12x *slower* than NumPy, because it emits one XLA op
    # per cloud instead of one for the batch.
    if k == 0 or m_max == 0:
        data = jnp.zeros((k, m_max, d), dtype=dtype)
    else:
        flat_rows = jnp.concatenate(arrays, axis=0)
        row_targets = jnp.asarray(
            [i * m_max + j for i, size in enumerate(sizes) for j in range(size)],
            dtype=jnp.int32,
        )
        # `flat_rows.dtype`, not `clouds[0].dtype`: the clouds may be NumPy, and
        # feeding a NumPy dtype to `jnp.zeros` requests a width JAX may not be
        # configured for (a float64 request warns and truncates under the
        # default config). Taking it from the already-converted rows asks for
        # the width the scatter will actually write.
        data = (
            jnp.zeros((k * m_max, d), dtype=flat_rows.dtype)
            .at[row_targets]
            .set(flat_rows)
            .reshape(k, m_max, d)
        )
    return PackedClouds(data=data, counts=counts)


def masked_row_mask(m_max: int, count: jnp.ndarray) -> jnp.ndarray:
    """Boolean/float validity mask ``ar < count`` for one padded row.

    Args:
        m_max: Padded row length.
        count: True (unpadded) size, scalar or shape ``(...,)``.

    Returns:
        Boolean array of shape ``(..., m_max)``, ``True`` where the row
        position is within the true (unpadded) extent.

    """
    ar = jnp.arange(m_max)
    return ar < count[..., None] if count.ndim else ar < count


def masked_sum(
    values: jnp.ndarray, row_count: jnp.ndarray, col_count: jnp.ndarray
) -> jnp.ndarray:
    """Sum of ``values`` over the valid ``(row_count, col_count)`` leading block.

    Zeroes the padded region before summing, matching the leading-block
    masked-reduction pattern used by the package's pairwise kernels.

    Args:
        values: Pairwise values, shape ``(m_max, n_max)``.
        row_count: Valid row extent (scalar).
        col_count: Valid column extent (scalar).

    Returns:
        Scalar sum over ``values[:row_count, :col_count]``.

    """
    m_max, n_max = values.shape
    row_mask = jnp.arange(m_max) < row_count
    col_mask = jnp.arange(n_max) < col_count
    valid = row_mask[:, None] & col_mask[None, :]
    return jnp.sum(jnp.where(valid, values, jnp.zeros((), dtype=values.dtype)))


def masked_mean(
    values: jnp.ndarray, row_count: jnp.ndarray, col_count: jnp.ndarray
) -> jnp.ndarray:
    """Mean of ``values`` over the valid ``(row_count, col_count)`` leading block.

    Args:
        values: Pairwise values, shape ``(m_max, n_max)``.
        row_count: Valid row extent (scalar).
        col_count: Valid column extent (scalar).

    Returns:
        Scalar mean over ``values[:row_count, :col_count]``.

    """
    total = masked_sum(values, row_count, col_count)
    denom = row_count.astype(values.dtype) * col_count.astype(values.dtype)
    return total / denom


# Pairs per lax.map step for chunked upper-triangle pair kernels. Callers may
# override the shared default via `chunk`.
DEFAULT_PAIR_CHUNK = 16


def pair_chunks(
    n: int, k: int = 0, chunk: int = DEFAULT_PAIR_CHUNK
) -> tuple[
    Int[Array, " p"],  # noqa: F722  # jaxtyping shape
    Int[Array, " p"],  # noqa: F722  # jaxtyping shape
    int,
    Int[Array, "c q 2"],  # noqa: F722  # jaxtyping shape
]:
    """Upper-triangle ``(i, j)`` index pairs, padded and chunked for ``lax.map``.

    Produces the ``i <= j`` (``k=0``) or ``i < j`` (``k=1``) entries needed by
    upper-triangle cloud reductions, since the lower triangle is redundant
    matmul work in exact arithmetic (author-approved contract relaxation
    2026-07-18; mirrored entries are equal only up to
    summation-order/XLA-tiling tolerance).

    Args:
        n: Matrix side length (number of clouds).
        k: Diagonal offset for :func:`numpy.triu_indices` (0 keeps the
            diagonal, 1 drops it).
        chunk: Pairs per ``lax.map`` step.

    Returns:
        Tuple ``(ii, jj, n_pairs, chunks)`` where ``ii``/``jj`` are the
        unpadded row/column indices, ``n_pairs`` their count, and ``chunks``
        the zero-padded ``(n_chunks, chunk, 2)`` pair grid.

    """
    # `jnp.triu_indices` returns int64 under x64, and the padded grid is int32
    # by contract (it indexes clouds, and `PackedClouds` pins the count dtype).
    # NumPy downcast that silently on assignment; a JAX scatter warns today and
    # is documented to become an error, so the narrowing is explicit here.
    ii, jj = (index.astype(jnp.int32) for index in jnp.triu_indices(n, k=k))
    n_pairs = int(ii.shape[0])
    n_pad = (-n_pairs) % chunk
    pairs = jnp.zeros((n_pairs + n_pad, 2), dtype=jnp.int32)
    pairs = pairs.at[:n_pairs, 0].set(ii)
    pairs = pairs.at[:n_pairs, 1].set(jj)
    return ii, jj, n_pairs, pairs.reshape(-1, chunk, 2)
