"""GEL overidentification tests from retained fits.

Fit fixed or affine moments with :mod:`statadic.estimation.gel`, then pass the fit
to :func:`gel_test`. Fixed fits support LR, LM and Wald; affine fits currently
support LR only. P-values require explicit iid/uncorrelated contributions,
solver convergence and the retained numerical-rank checks. This package owns
calibration, while fitting owns the criterion and its numerical solution.
"""

from statadic.inference.gel.hypothesis import AffineGelResult, GelResult, gel_test

__all__ = ["AffineGelResult", "GelResult", "gel_test"]
