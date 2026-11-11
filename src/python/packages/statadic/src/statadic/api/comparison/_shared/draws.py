"""Counting, enumerating and drawing admissible arrangements."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence  # noqa: TC003 -- runtime annotations
from functools import partial
from itertools import combinations, product
from math import factorial

import jax
import jax.numpy as jnp

from statadic.api.comparison._shared.design import (
    ExchangeabilityDesign,  # noqa: TC001 -- runtime annotations
)
from statadic.association.dispersion import disco_decomposition
from statadic.core.random import permutation_batch
from statadic.core.typing import Array  # noqa: TC001 -- runtime annotations


@partial(jax.jit, static_argnames=("n_groups",))
def _disco_draws(values: Array, labels: Array, n_groups: int) -> Array:
    """Batch the existing statistic under one stable compiled callable."""
    return jax.vmap(disco_decomposition, in_axes=(None, 0, None))(
        values, labels, n_groups
    )[-1]


def _multinomial(labels: Sequence[int]) -> int:
    """Count the distinct orderings of a label multiset."""
    total = factorial(len(labels))
    for count in Counter(labels).values():
        total //= factorial(count)
    return total


def _multiset_orderings(labels: Sequence[int]) -> list[tuple[int, ...]]:
    """Enumerate distinct orderings of a label multiset, observed order first."""
    size = len(labels)
    counts = Counter(labels)
    levels = sorted(counts)
    first = {level: labels.index(level) for level in levels}
    levels.sort(key=first.__getitem__)
    found: list[tuple[int, ...]] = []
    current = [0] * size

    def assign(open_positions: tuple[int, ...], position: int) -> None:
        if position == len(levels):
            found.append(tuple(current))
            return
        level = levels[position]
        for chosen in combinations(open_positions, counts[level]):
            for index in chosen:
                current[index] = level
            assign(
                tuple(index for index in open_positions if index not in chosen),
                position + 1,
            )

    assign(tuple(range(size)), 0)
    observed = tuple(labels)
    found.sort(key=lambda item: item != observed)
    return found


def _placed(
    members: Sequence[Sequence[int]],
    choices: Sequence[Sequence[Sequence[int]]],
    size: int,
) -> list[list[int]]:
    """Combine per-block choices into full vectors (Cartesian product).

    ``choices[b]`` lists the values to place at ``members[b]``, in order.
    """
    placed: list[list[int]] = []
    for combination in product(*choices):
        vector = [0] * size
        for positions, values in zip(members, combination, strict=True):
            for position, value in zip(positions, values, strict=True):
                vector[position] = value
        placed.append(vector)
    return placed


def _within_blocks(
    key: Array, values: Array, design: ExchangeabilityDesign, draws: int
) -> Array:
    """Permute unit-level values uniformly within each exchangeability block.

    A uniform permutation of all units, stably regrouped by block, orders each
    block's units uniformly; matching it to the block-sorted identity moves
    every value only inside its block.
    """
    blocks = jnp.asarray(design.unit_blocks, dtype=jnp.int32)
    units = jnp.arange(len(design.unit_blocks), dtype=jnp.int32)
    base = jnp.argsort(blocks, stable=True)

    def regroup(order: Array) -> Array:
        grouped = order[jnp.argsort(blocks[order], stable=True)]
        return jnp.zeros_like(values).at[base].set(values[grouped])

    return jax.vmap(regroup)(permutation_batch(key, units, draws))
