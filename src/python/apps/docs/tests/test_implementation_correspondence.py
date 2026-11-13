"""Conformance witnesses for actual packed operations, not Python proofs."""

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.sample.packed import masked_sum, pair_chunks

pytestmark = pytest.mark.native


def test_masked_sum_matches_exact_rational_leading_blocks():
    # Integer values and these tiny sums are exactly representable in both
    # supported dtypes. NaNs deliberately poison only the discarded padding.
    for rows in range(4):
        for cols in range(5):
            values = np.arange(12, dtype=float).reshape(3, 4) - 6
            expected = sum(
                (Fraction(int(values[i, j])) for i in range(rows) for j in range(cols)),
                Fraction(0),
            )
            values[rows:, :] = np.nan
            values[:, cols:] = np.nan
            actual = masked_sum(
                jnp.asarray(values), jnp.asarray(rows), jnp.asarray(cols)
            )
            assert float(actual) == float(expected)


def test_pair_chunks_retains_exact_upper_triangle_before_padding():
    for n in range(7):
        for k in (0, 1):
            for chunk in (1, 3, 16):
                ii, jj, count, chunks = pair_chunks(n, k=k, chunk=chunk)
                expected = [(i, j) for i in range(n) for j in range(i + k, n)]
                assert list(zip(ii.tolist(), jj.tolist(), strict=True)) == expected
                assert count == len(expected)
                flat = np.asarray(chunks).reshape(-1, 2)
                np.testing.assert_array_equal(
                    flat[:count], np.asarray(expected).reshape(-1, 2)
                )
                assert np.all(flat[count:] == 0)
