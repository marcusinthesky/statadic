# Corrective-candidate verification contract

This is the execution specification for retained alpha operations after review
dispositions and corrective-source freeze. It does not authorize a long campaign
now. Use existing studies/performance/evidence owners; add only missing workload
contracts and adapters. Keep failed, absent, stale and inconclusive results visible.

## Per-operation evidence

The [scientific CSV](scientific-evidence.csv) retains selectors, source status,
exact study methods/scenarios and declared method-specific Monte Carlo metrics.
[Frozen study definitions](study-designs.json) make the joins reproducible.
Declarations currently link 165 variants to bias/MSE, 123 to size, 55 to power
and 101 to coverage. These overlap and are not counts of qualified variants.
Semantic omissions (F006/F014) must be repaired before freezing that denominator.

For each retained substantive variant, record:

- Estimand, units, bias correction, rooting, normalization, weight semantics,
  constraints, covariance and calibration/reference choices.
- Exact primary-source passage and implementation correspondence, explicit
  theorem hypotheses and human review status. Missing proofs are not numerical
  failures; passing Lean is not implementation verification.
- Independent formula/software controls, tolerances justified by conditioning
  and precision, invalid/singular/boundary cases, and fitted-state consistency.
- Relevant statistical dimensions: null size, alternatives/power, bias/MSE,
  interval coverage/width and procedure failures. Mark genuinely inapplicable
  dimensions with a reason; deterministic interfaces cannot hide stochastic
  procedures as wrappers.
- Execution modes and differentiation target, precise shape/static controls,
  compile reuse, output materialization and resource bounds.

Registration, collected tests and same-owner parity do not count as independent
executed evidence. Bind imported source or exact wheel, lock, definitions and
retained results before and after execution. Use independent trial identities
and failure-inclusive qualification accounting. Never condition coverage/size
only on successful fits or stop after a favorable result.

## JAX analysis matrix

| Target | Required bounded checks | Limits to preserve |
| --- | --- | --- |
| Smooth scalar objective | Independent value/gradient; JVP/VJP duality; analytic or independently differenced Hessian/HVP; symmetry where justified. | State interior, tie/sort/rank boundaries and precision. |
| Fixed finite numerical iterate | Check the explicitly chosen unrolled/implicit/custom derivative against that target. | It is not automatically the converged solution derivative. |
| Optimal/profile value | Check envelope assumptions, inner stationarity and higher derivatives separately. | Stopping an argmax can give a first gradient and an incorrect nested Hessian. |
| Fitted solution | Require a supported implicit/unrolled rule and nonsingular identification conditions. | A pytree or successful JIT does not establish differentiability. |
| Batch/resample kernel | Eager/JIT/vmap/chunked equality, changing values, remainder chunks, invalid lanes and fixed draw IDs. | Static shape changes can legitimately recompile; document them. |
| Iterative control | Check fixed carry shapes, masks, termination and intended derivative mode. | Dynamic `lax.while_loop` is not a general reverse-mode solution. |
| Host preparation/report | Snapshot state, instance ownership, identities, row alignment, dependencies and precision. | Eager host conversions are intentional here. |

Inspect lowered/optimized work before asserting duplicate execution. Check
closure/static-argument capture, unused-output elimination, stable compiled
boundaries, dense Jacobian/Hessian/tensor storage and lax loop choices. Do not
replace an efficient sufficient-statistic reduction with per-row AD or a dense
tensor simply to make syntax uniform.

## Performance and memory gates

Run retained workloads on this available CPU with matched reference methods and
outputs. Use n/width/conditioning, support imbalance, instrument count, positive
fraction, cluster/time structure and solver difficulty relevant to each method.
Preserve exact objectives, precision, tolerances, failure policy and requested
evidence. Coefficients-only and complete-fit-plus-inference are separate products.

The primary gate is repeated numerical work: fixed-shape refits, batches,
resampling and supported derivative objectives. Record preparation/transfer,
lowering/compilation, fresh combined calls, synchronized warm distributions and
host materialization separately. Native repeated-call compilation is part of
the public workload until a stable compiled boundary is explicitly used.

Use at least three independent workers. Require every session's median
candidate/reference time ratio below one for a warm win; mixed ordering is
inconclusive. Report raw samples, failures and adverse regimes. A practical
effect threshold should be prespecified per workload, rather than selected after
measurement. Historical CPU44 results describe earlier revisions only.

Compare the same memory quantity: isolated total-process peak RSS against RSS;
retained output bytes against output bytes; compiler estimates against compatible
estimates. Total RSS includes imports/compilation and remains visible even when
kernel workspace improves. Missing allocator evidence is unavailable, not zero.
Never equate a compiler temporary estimate with a reference process's memory.
Joint superiority requires both the declared speed and memory gates. No ARM/GPU
claim follows from this machine. A losing or unmeasured regime gets a narrowed
claim, corrective implementation task or explicit deferral—not a blanket win.
