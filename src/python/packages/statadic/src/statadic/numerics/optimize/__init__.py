"""Numerical algorithms and checked solver adapters.

Use ``backends`` for maintained QR, iterative-linear and scalar/BFGS adapters;
``simplex`` for constrained quadratic and affine optimization; ``psd`` for
matrix feasibility repairs; ``glm`` for canonical-response solvers; and
``integration`` for Gaussian-tail integration. Import the required child to
load its numerical machinery.

Solver status and residuals describe numerical evidence. They do not choose a
statistical reference distribution or establish identification.
"""

__all__: list[str] = []
