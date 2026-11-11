---
type: document
title: Projection and refit resource follow-up
description: Source-based optimization candidates after the adverse final-wheel pilot, without a claim of measured improvement.
links:
  - .context/plan/research-release/tasks/alpha-readiness/final-sweep.md
  - .context/plan/research-release/tasks/alpha-readiness/c-record.md
---

# Projection and refit resource follow-up

Read-only source inspection on 2026-11-09 followed the 26-case pilot of wheel
SHA-256 `9fb4f29b318d4a7b2e1848daa9622c95c6774cb78500cda1787c1b441a44ec33`.
The inspected pilot is `build/final-sweep/reference-pilot/manifest.json`: one
session, one warmup and three samples. Parity passed all 26 cases; every case
lost whole-process peak RSS. This is pilot evidence, not the required three
sessions of five warmups and 30 samples. The gate remains unchanged.
The coordinator later repeated all 26 cases at the final lock in
`reference-pilot-locked/`; parity again passed and every RSS gate failed.
The [final sweep](final-sweep.md) owns those current ratios. Timings below refer
to the original inspected pilot and are not pooled with the repeat.

## Highest-priority projection investigation

The energy and kernel public projections share an eager, undecorated
`geometry/barycentre/mmd/solve.py:_exact_barycentre_qp_core` at line 225. Each
call with a positive refinement budget creates fresh `keep` and `refine`
functions at lines 254 and 257, closing over numerical arrays, then invokes
`lax.cond` at line 279. The inner canonical QP already has a stable JIT entry
in `optimize/simplex.py:1081`. The outer conditional is outside that entry.

Repeated tracing or compilation of this outer conditional is a strong,
unmeasured explanation for the common projection overhead. Both candidate
warm medians were approximately 0.213 seconds, while their independent peers
were approximately 0.000294 and 0.002902 seconds. Those timings alone do not
prove a compiler-cache cause.

The smallest proposed repair is a stable module-level JIT of the existing
core, with tolerance and iteration cap static and Gram, cross and target
self-term explicit dynamic arguments. Keep the original objective, initial
caller tolerance, conditional refinement, total iteration budget, canonical
KKT and original-energy Frank-Wolfe certificates. First establish the cause
with compile logs for repeated equal-shape calls and changed numerical
inputs. Retain boundary, duplicate-candidate, float32/64 and cap controls.
Under `vmap`, a conditional may execute both solver branches; a scalar-path
improvement would not establish a batched cost improvement.

## Remove unnecessary candidate materialization

`solve.py:_candidate_list`, lines 188–222, slices every cloud into a Python
list and evaluates `bool(candidate.valid)` separately for each member.
`sample/packed.py:168` calculates a complete numerical probability check;
`__getitem__` at line 223 slices all three carrier leaves. This creates
avoidable dispatch and host synchronization.

There are exactly two repository callers. Kernel components at
`geometry/barycentre/mmd/weights.py:115` ignore the returned list. Energy
components at `energy/weights.py:77` return it only so the public solver can
take its length at line 355. Replace this helper with static shape/count
checks, one target-validity check and one batched candidate-validity
reduction, returning the count. Preserve error behavior, masses and dynamic
validity; do not cache a previous validity verdict.

After that bounded change, consider one stable numerical boundary covering
component reductions, tangent geometry, QP and array-valued diagnostics.
The current eager path performs a QR and eigenspectrum at `solve.py:104–107`,
host scalar conversion at line 129 and multiple diagnostic conversions at
lines 149–160 and 173–184. Keep Python request validation and diagnostic
presentation outside the compiled calculation. Preserve genuine precomputed
component callers and both numerical and scientific checks. Moving component
work outside the timed raw-cloud workload would require a separately matched
contract; it is not a performance repair of that workload.

## Dyadic refits: host work rather than an established cache defect

`inference/dyadic/bootstrap.py:325` runs a Python loop over draws. Each draw
does host rank selection, dynamic retained-column slicing, solve and several
scalar synchronizations (`:253–305`). `operators/design.py:354` calculates a
spectral norm before the declared-order rank profile. `_effective_rank` in
`bootstrap.py:75–103` separately calculates a masked full-design SVD. Effects
also perform the within-R-squared fit. The rank-profile function itself is
already a stable module-level JIT at `operators/design.py:233`; the fixed
warm schedule reuses its shapes and retained widths.

A narrow next step is a stable numerical kernel after host column selection,
returning coefficient, status and summary arrays together, followed by one
host transfer. The existing QR owner can retain factors and derive rank from
R (`optimize/backends.py:321–351`), potentially replacing the second large
SVD. That requires explicit near-threshold agreement with the present sliced
design cutoff, including original dimensions. Do not replace declared-order
selection with pivoting, solve a zero-padded rank-deficient design, change the
node-count bootstrap law or omit invalid draws. No source evidence establishes
fresh per-warm-call closure compilation as the dominant dyadic cause.

## Row-pairs and RSS limits

`statadic_performance/reference_refits.py:73–85` constructs and JITs the row-pairs
callable once during preparation. Its nested refit functions execute within
that stable outer trace. The approximately 1.029 warm ratio from three pilot
samples does not establish a cache defect. Inspect optimized HLO and the full
declared workload before removing fit-state calculations: unused arrays may
already be eliminated, while finite-state and rank checks remain observable.
Retain the explicit-Q solve and original inference validity.

Package-root, estimation and inference namespaces are already navigation
namespaces. A remaining bounded import-coupling candidate is
`optimize/backends.py:15`, which imports Optimistix for QR-only consumers;
only its minimizer functions use that dependency. A local import or narrower
owner could avoid that unrelated import, but no RSS saving is measured and
transitive dependencies must be checked. JAX runtime, compilation and retained
executables can still dominate whole-process peak RSS.

No native or harness changes, tests, compile probes or measurements were run
for this inspection. None of these proposals establishes a time or RSS win.
The coordinator separately owns clean-launch RSS provenance, the full-size
QP feasibility probe and the final resource summary. Rejected inherited-RSS
launches must remain excluded; subtracting a runtime baseline, changing scope
or relaxing the per-session 0.90 gate is not proposed.
