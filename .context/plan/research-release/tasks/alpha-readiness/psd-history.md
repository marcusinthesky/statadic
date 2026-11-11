---
type: document
title: Historical PSD solver observations
description: Dated retained implementation observations moved from the numerical module; these are not current alpha qualification.
links:
  - .context/plan/research-release/tasks/alpha-readiness/final-sweep.md
  - .context/plan/research-release/tasks/alpha-readiness/c-record.md
---

# Historical PSD solver observations

Retained verbatim from `optimize/psd.py:26–115` at `5694676` during the final
0.5.0a1 sweep. The source records a 2026-08-08 repair and bounded production
observations. Its historical phrases, counts and claims retain that original
scope; they are neither fresh measurements nor alpha qualification. Current
operation and derivative contracts remain in the numerical module.

```text
distortion so callers can quantify the trade-off.

Why Newton rather than Dykstra alternating projections
------------------------------------------------------
Measured, not assumed.  The previous implementation ran Higham's Dykstra
alternating projections under a fixed 200-iteration budget with a ``1e-10``
relative-change tolerance.  On the production ``Σ_dist`` shape (52×52, Paper 3)
that tolerance was **never** reached: all 20 sampled matrices exhausted the
budget at ``max(dx, dy, dxy) ≈ 3e-5``, and the returned iterate sat ``7e-4``
(max entry) away from the same iteration run to convergence.  Dykstra converges
linearly, so the budget could not be tightened cheaply.

Qi-Sun Newton reaches the KKT point of the dual in 8-11 Newton steps on the
same 20 matrices — against 200 alternating projections — and lands ``4.7e-12``
from a Dykstra oracle run to 5000 iterations rather than ``7e-4``.

**Those 20 seeds are not the whole story, and this module does not state an
unconditional nearest-matrix contract.** ``converged`` is returned as a leaf
precisely because a real production matrix can fall outside what the sampled
seeds represent; callers that need *nearest* rather than *feasible* must read
it. The energy-robustness ``robust-higham`` arm does not read it, and describes
itself as a heuristic accordingly.

**The Paper 3 "boundary plateau" was a solver defect, fixed 2026-08-08 — it was
never a property of the input.** Two earlier notes here read the same stall as a
degenerate generalized Jacobian at the cone boundary, then as a state that "no
longer exists"; both were wrong, and the second was written from a single run
that happened to land on the good side of a coin flip.

Two independent defects, both required for the stall:

1. **The CG ridge was floored.** It read ``√eps · max(‖g‖, 1)``, pinning the
   regularisation at ``√eps`` = 1.49e-08 for *every* iterate with ``‖g‖ < 1`` —
   24 800x the ``n²·eps`` = 6.00e-13 stopping tolerance — so the Newton
   direction could not resolve the last ~1e-08 of gradient at any budget. It is
   now ``√eps · ‖g‖``, which tracks the accuracy still being demanded.
2. **The Armijo test ran below its own objective's resolution.** At
   ``‖g‖ ≈ 3.6e-08`` the decrement ``1e-4·t·slope`` is 1e-18 to 1e-28 against a
   dual objective of magnitude 251.76 whose float64 granularity is 5.59e-14: the
   line search was comparing bit-identical numbers, accepting on rounding noise
   for three steps and then exhausting all 30 halvings, forcing ``step_size = 0``
   and freezing the iterate. That is why the residual was bit-identical at 25,
   50, 100, 200 and 400 Newton steps and read as a plateau. Backtracking is now
   skipped once the demanded decrement falls below ``|value|·eps``.

Because the verdict turned on rounding, it was not reproducible. Before the fix,
a ``1e-16`` relative (one ulp) symmetric perturbation of the production
``Σ_dist`` flipped it: 6 of 8 draws converged at ~7.2e-14, 2 of 8 stalled — the
same 2-in-8 rate at 1e-15, 1e-14 and 1e-12. That is the whole history of the
published ``psd-converged`` flag: ``False@4.0977e-08``, then
``True@7.329280e-14``, then ``False@3.612131e-08``, across runs whose inputs
differed only in the last bits. Fixing the ridge alone moved the production
matrix off the knife edge but left the rate at 2-3 in 8 on perturbed draws; both
fixes together give **33 of 33** convergences at ~6.6e-14 over the same sweep.

Measured at the ``panel.py`` call site, one process, per dtype:

===========  ==========  ===================  ==========  ==========
input dtype  tolerance   residual             converged   iterations
===========  ==========  ===================  ==========  ==========
float64      6.0041e-13  **6.741998e-14**     True        7
float32      3.2234e-04  2.183340e-04         False       5
===========  ==========  ===================  ==========  ==========

The float32 row reports ``False`` by :data:`_NEAREST_CLAIM_CEILING`, not by its
stopping tolerance, which it does meet. Stopping at the dtype's roundoff floor
is not the same as being nearest, and the dual gradient ``diag(X) - e`` is
dimensionless, so the claim is gated on an absolute bar as well as a scaled one.

Precision policy
----------------
The working dtype is the input's; nothing here promotes to float64 or mutates
the caller's ``jax_enable_x64``.  The dual convergence tolerance, the Cholesky
eigenvalue floor, and the Newton system's Tikhonov level are all derived from
``n`` and the input dtype (:func:`_dual_tolerance`, :func:`_relative_floor`),
rather than pinning float64 constants into a float32 computation.

**The nearest-matrix contract is stated at float64.**  Over those 20 seeds,
float64 converges on 20/20; float32 converges on 11/20 and lands ``4.2e-03``
from the oracle.  That gap is the problem, not the solver: the nearest
correlation matrix is boundary-singular, so its smallest eigenvalues sit at
float32 epsilon and no amount of iteration resolves them.  In float32 the
return is a feasible unit-diagonal PSD repair with ``converged=False`` saying
so — the honest answer rather than a tolerance widened until it reads True.
Every production caller enables x64.

Convergence is reported as a leaf, never raised, so a capped iteration is
visible to a traced caller instead of aborting it.

References:
```
