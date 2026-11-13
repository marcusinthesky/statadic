"""Permutation plans: random draws or complete enumeration over a design."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from typing import Literal

import jax.numpy as jnp

from statadic.api.comparison._shared.design import (
    DesignLabels,  # noqa: TC001 -- runtime annotations
    ExchangeabilityDesign,
    PlannedDraws,
    _design_labels,
    _role_name,
)
from statadic.api.comparison._shared.draws import (
    _multinomial,
    _multiset_orderings,
    _placed,
    _within_blocks,
)
from statadic.api.data import Observations  # noqa: TC001 -- runtime annotations
from statadic.api.design import level_codes
from statadic.api.embedding import RowIds  # noqa: TC001 -- runtime annotations
from statadic.core.random import permutation_batch
from statadic.core.typing import Array, ArrayLike  # noqa: TC001 -- runtime annotations
from statadic.inference.disco import disco_permutation_key
from statadic.inference.dispersion import dispersion_permutation_key


@dataclass(frozen=True)
class PermutationPlan:
    """Specify random draws or complete enumeration, never interchange their tails.

    ``unit`` names the unit of assignment: every row of a unit carries one
    label and moves with it. ``blocks`` names exchangeability blocks: labels
    never move between blocks. Each is a column of the fitted grouping
    ``Observations`` or labels aligned with ``row_ids``. Leaving both unset
    relabels rows freely. Declare the design that actually assigned the labels:
    permuting repeated visits of randomized subjects as free rows is a
    different, usually invalid, test.

    Supplied ``indices`` are row-index schedules and require exact row IDs;
    with a declared design each one must respect it. Complete enumeration
    visits each distinct admissible label arrangement once (every arrangement
    of the full group has the same multiplicity, so the p-value is unchanged)
    within ``max_exact_arrangements``.
    """

    n: int = 999
    seed: int = 0
    method: Literal["random", "exact"] = "random"
    indices: ArrayLike | None = field(default=None, repr=False)
    row_ids: RowIds | None = None
    unit: DesignLabels | None = None
    blocks: DesignLabels | None = None
    max_exact_arrangements: int = 100_000

    def __post_init__(self) -> None:
        """Snapshot caller-owned buffers before retaining the plan."""
        if self.indices is not None:
            object.__setattr__(self, "indices", jnp.array(self.indices, copy=True))
        if self.row_ids is not None:
            object.__setattr__(self, "row_ids", tuple(self.row_ids))
        for role in ("unit", "blocks"):
            value = getattr(self, role)
            if value is not None and not isinstance(value, str):
                tolist = getattr(value, "tolist", None)
                items = tolist() if callable(tolist) else value
                if not isinstance(items, Sequence):
                    message = f"PermutationPlan {role} labels must be a sequence."
                    raise ValueError(message)
                object.__setattr__(self, role, tuple(items))

    @property
    def supplied(self) -> bool:
        """Whether the caller, rather than a seeded draw, supplied the schedules."""
        return self.indices is not None

    @property
    def key(self) -> Array:
        """Key of the native label-permutation stream (DISCO and PERMANOVA)."""
        return disco_permutation_key(self.seed)

    @property
    def residual_key(self) -> Array:
        """Key of the native PERMDISP residual-permutation stream."""
        return dispersion_permutation_key(self.seed)

    def design(
        self, row_ids: RowIds, observations: Observations | None = None
    ) -> ExchangeabilityDesign:
        """Resolve units and blocks, checking that units nest within blocks."""
        self._check_common(row_ids)
        count = len(row_ids)
        units = (
            tuple(range(count))
            if self.unit is None
            else _design_labels("unit", self.unit, row_ids, self.row_ids, observations)
        )
        blocks = (
            (0,) * count
            if self.blocks is None
            else _design_labels(
                "blocks", self.blocks, row_ids, self.row_ids, observations
            )
        )
        unit_ids, unit_codes = level_codes(units)
        block_ids, block_codes = level_codes(blocks)
        row_units = tuple(int(code) for code in unit_codes.tolist())
        row_blocks = tuple(int(code) for code in block_codes.tolist())
        unit_blocks: list[int] = [-1] * len(unit_ids)
        for unit, block in zip(row_units, row_blocks, strict=True):
            if unit_blocks[unit] not in (-1, block):
                message = (
                    f"Unit {unit_ids[unit]!r} spans exchangeability blocks; "
                    "assignment units must nest within blocks."
                )
                raise ValueError(message)
            unit_blocks[unit] = block
        return ExchangeabilityDesign(
            _role_name(self.unit),
            _role_name(self.blocks),
            row_units,
            tuple(unit_blocks),
            unit_ids,
            block_ids,
        )

    def arrangements(
        self,
        codes: Array,
        row_ids: RowIds,
        observations: Observations | None = None,
    ) -> PlannedDraws:
        """Return the label arrangements to evaluate, the design and its size.

        Each row of the result holds the group code of every observation under
        one draw. The count is the number of distinct admissible arrangements.
        """
        design = self.design(row_ids, observations)
        labels = tuple(int(code) for code in codes.tolist())
        unit_labels: list[int] = [-1] * len(design.unit_ids)
        for row, unit in enumerate(design.row_units):
            if unit_labels[unit] not in (-1, labels[row]):
                message = (
                    f"Group labels vary within unit {design.unit_ids[unit]!r}; "
                    "labels assigned per unit cannot vary inside it."
                )
                raise ValueError(message)
            unit_labels[unit] = labels[row]
        per_block = [
            [unit_labels[u] for u in design.units_in(block)]
            for block in range(len(design.block_ids))
        ]
        total = 1
        for block_labels in per_block:
            total *= _multinomial(block_labels)
        self._check_size(total, design)
        row_units = jnp.asarray(design.row_units, dtype=jnp.int32)
        if self.method == "exact":
            self._check_budget(total)
            unit_draws = jnp.asarray(
                _placed(
                    [design.units_in(block) for block in range(len(per_block))],
                    [_multiset_orderings(block) for block in per_block],
                    len(unit_labels),
                ),
                dtype=codes.dtype,
            )
            return PlannedDraws(unit_draws[:, row_units], design, total)
        if self.indices is not None:
            arrangements = codes[self._supplied(len(row_ids))]
            if not design.free:
                self._check_orbit(arrangements, design, unit_labels)
            return PlannedDraws(arrangements, design, total)
        self._check_random()
        if design.free:
            schedules = permutation_batch(
                self.key, jnp.arange(len(row_ids), dtype=jnp.int32), self.n
            )
            return PlannedDraws(codes[schedules], design, total)
        moved = _within_blocks(
            self.key, jnp.asarray(unit_labels, dtype=codes.dtype), design, self.n
        )
        return PlannedDraws(moved[:, row_units], design, total)

    def schedules(
        self,
        row_ids: RowIds,
        *,
        residual: bool = False,
        key: Array | None = None,
        observations: Observations | None = None,
    ) -> PlannedDraws:
        """Return row-index permutations, the design and the group size.

        Row permutations move observations (Mantel objects, PERMDISP
        residuals), so only ``blocks`` applies; whole-unit moves would need
        equal-size units and are rejected. ``key`` selects a native stream.
        """
        if self.unit is not None:
            message = (
                "Whole-unit moves are not supported for row permutations "
                "(Mantel objects, PERMDISP residuals); declare blocks instead."
            )
            raise ValueError(message)
        design = self.design(row_ids, observations)
        rows = [design.units_in(block) for block in range(len(design.block_ids))]
        total = 1
        for members in rows:
            total *= factorial(len(members))
        self._check_size(total, design)
        count = len(row_ids)
        if self.method == "exact":
            self._check_budget(total)
            enumerated = _placed(
                rows, [list(permutations(members)) for members in rows], count
            )
            return PlannedDraws(jnp.asarray(enumerated, dtype=jnp.int32), design, total)
        if self.indices is not None:
            values = self._supplied(count)
            blocks = jnp.asarray(design.unit_blocks, dtype=jnp.int32)
            if not bool(jnp.all(blocks[values] == blocks)):
                message = (
                    "A supplied schedule moves rows between exchangeability blocks."
                )
                raise ValueError(message)
            return PlannedDraws(values, design, total)
        self._check_random()
        stream = key
        if stream is None:
            stream = self.residual_key if residual else self.key
        identity = jnp.arange(count, dtype=jnp.int32)
        if design.free:
            return PlannedDraws(
                permutation_batch(stream, identity, self.n), design, total
            )
        return PlannedDraws(
            _within_blocks(stream, identity, design, self.n), design, total
        )

    def _check_common(self, row_ids: RowIds) -> None:
        if self.method not in {"random", "exact"}:
            message = "Permutation method must be 'random' or 'exact'."
            raise ValueError(message)
        if self.row_ids is not None and tuple(self.row_ids) != row_ids:
            message = (
                "Permutation row IDs must exactly match the retained observation order."
            )
            raise ValueError(message)
        if self.method == "exact" and self.indices is not None:
            message = "Exact enumeration cannot accept a supplied schedule subset."
            raise ValueError(message)

    def _check_random(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**32
        ):
            message = "Permutation seed must be an integer in [0, 2**32)."
            raise ValueError(message)

    def _check_size(self, total: int, design: ExchangeabilityDesign) -> None:
        # A free design keeps failing closed through its statistics instead.
        if total <= 1 and not design.free:
            message = (
                "The declared design admits only the observed arrangement; "
                "no permutation test is possible."
            )
            raise ValueError(message)
        if self.method == "random" and (
            isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 1
        ):
            message = "Random permutation count must be a positive integer."
            raise ValueError(message)

    def _check_budget(self, total: int) -> None:
        if (
            isinstance(self.max_exact_arrangements, bool)
            or not isinstance(self.max_exact_arrangements, int)
            or total > self.max_exact_arrangements
        ):
            message = (
                f"Complete enumeration of {total} arrangements exceeds "
                "max_exact_arrangements."
            )
            raise ValueError(message)

    def _supplied(self, count: int) -> Array:
        if self.row_ids is None:
            message = "Supplied permutation schedules require explicit row IDs."
            raise ValueError(message)
        values = jnp.asarray(self.indices)
        if (
            values.shape != (self.n, count)
            or not jnp.issubdtype(values.dtype, jnp.integer)
            or not bool(jnp.all(jnp.sort(values, axis=1) == jnp.arange(count)))
        ):
            message = "Each supplied row must be a complete integer index permutation."
            raise ValueError(message)
        return values

    @staticmethod
    def _check_orbit(
        arrangements: Array, design: ExchangeabilityDesign, unit_labels: list[int]
    ) -> None:
        """Require supplied relabellings to lie in the declared design's orbit."""
        row_units = jnp.asarray(design.row_units, dtype=jnp.int32)
        first = jnp.asarray(
            [design.row_units.index(unit) for unit in range(len(unit_labels))],
            dtype=jnp.int32,
        )
        per_unit = arrangements[:, first]
        constant = bool(jnp.all(per_unit[:, row_units] == arrangements))
        observed = jnp.asarray(unit_labels, dtype=arrangements.dtype)
        preserved = all(
            bool(
                jnp.all(
                    jnp.sort(per_unit[:, members], axis=1)
                    == jnp.sort(observed[members])
                )
            )
            for members in (
                jnp.asarray(design.units_in(block), dtype=jnp.int32)
                for block in range(len(design.block_ids))
            )
        )
        if not (constant and preserved):
            message = (
                "A supplied schedule is not an arrangement the declared design "
                "admits (labels must stay constant within units and in block)."
            )
            raise ValueError(message)
