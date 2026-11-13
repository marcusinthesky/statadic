---
type: troubleshooting
description: Recognize common installation, fitting, distance and notebook problems.
---

# Troubleshooting

(installation-or-import-fails)=

## Installation or import fails

**Cause** The interpreter can belong to a different environment from the one
where STATADIC was installed, or the Python version can be unsupported.

**Fix** Follow [installation](installation.md) with Python 3.13, then run the
example with `.venv/bin/python` from the checkout. Check the interpreter used
by your editor against the environment where the package was installed.

(a-fit-or-interval-is-unavailable)=

## A fit or interval is unavailable

**Cause** Identification, convergence, residual degrees of freedom and the
reference calculation each have their own availability checks.

**Fix** Inspect the fitted result and inference status separately, since each
status describes a different stage of the calculation. Read the method's
assumptions and [result validity](../concepts/result-validity.md) before
changing the model or its inference request.

(an-angular-distance-is-nan)=

## An angular distance is NaN

**Cause** Angular geometry compares directions, and a zero vector has no
direction to compare with the other observations.

**Fix** Inspect the inputs and choose a scientifically appropriate treatment
for zero vectors, then read the [geometry explanation](../concepts/geometry.md)
before selecting a ground if magnitude matters. The choice changes the comparison.

(a-notebook-does-not-start)=

## A notebook does not start

**Cause** The website displays a static execution, while a downloaded notebook
needs a Python kernel and the dependencies declared in its source.

**Fix** Open it with `uvx marimo edit --sandbox notebook.py` and allow the first
dependency installation to finish. Use Python 3.13. Inspect import errors in that environment.
