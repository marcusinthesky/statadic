"""Specify undirected dyadic samples, regressors and symmetric node effects.

DyadicSample aligns an outcome with endpoint labels; DyadicModel selects named
regressors and optional node effects. Estimation belongs to
statadic.estimation.least_squares.dyadic, and multinomial node-bootstrap inference
belongs to
statadic.inference.dyadic. Validation contains eager specification checks shared by
those consumers.
"""

from statadic.law.dyadic.specification import (
    DyadicInputError,
    DyadicModel,
    DyadicSample,
)

__all__ = ["DyadicInputError", "DyadicModel", "DyadicSample"]
