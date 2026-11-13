# jaxtyping shape strings are runtime contracts.
"""Property batteries for declared contracts, loaded explicitly by tests.

These helpers take a callable and check its claimed laws against observed
values. They depend on shared core contracts without choosing a statistical
implementation or proving a universal theorem. Shipping them in the wheel lets
downstream consumers apply the same batteries to their own dissimilarities.

``fixtures`` is loaded explicitly by a consuming suite's initial conftest;
fixture functions are not re-exported here. The curated names below are ordinary
library helpers.
"""

from __future__ import annotations

__all__ = [
    "assert_axioms",
    "assert_negative_type",
    "dissimilarity_matrix",
    "measure_axioms",
    "negative_type_eigmin",
]

from statadic.testing.axioms import (
    assert_axioms,
    assert_negative_type,
    dissimilarity_matrix,
    measure_axioms,
    negative_type_eigmin,
)
