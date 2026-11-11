# CPU performance evidence

For the 44-case comparison with spreg, lifelines, linearmodels, statsmodels,
POT and other libraries, see the [CPU timings and peak memory report](cpu-reference-performance.md).
It includes both measured revisions, all individual timings and replay instructions.

This internal workspace consumer measures existing STATADIC implementations. It adds
no performance claim or process precision change to the runtime library. Use the
pinned environment and locked commands from the repository root:

```bash
just python::performance::check
just python::performance::smoke --output /tmp/statadic-performance-smoke
just python::performance::bench /tmp/statadic-performance-baseline
just python::performance::report /tmp/statadic-performance-baseline
just python::performance::compare /tmp/statadic-performance-baseline /tmp/statadic-performance-candidate
just python::performance::profile /tmp/statadic-performance-profile
```

Every output directory is new and immutable after completion; retain its sibling
`.tar` archive. Smoke executes all workload families at small sizes in both
precisions. Bench uses three independent processes per workload/precision, five
synchronized warmups and thirty retained warm samples per variant. Each worker
sets CPU precision before importing JAX. Persistent compilation caching is disabled;
each variant clears in-memory caches before lowering and compilation. Preparation,
transfer, lowering, compilation, first compiled call and warm elapsed samples are
separate fields. Warm calls synchronize every pytree leaf.

Workers discard inherited `JAX_*` settings before selecting precision and a fixed
compiler policy: JIT enabled, most optimizations enabled, and highest matmul
precision. Records retain effective settings; comparisons reject absent or
different compiler identities. Comparison and report destinations must be outside
retained input directories; report writes replace the destination inode.

Each variant also runs once per session in its own fresh cold worker. The
combined cold JIT-call latency includes first tracing, lowering, compilation and
execution after inputs are transferred; it excludes imports and fixture setup.
It is separate from first execution of an already-compiled executable. Host
output materialization is measured after execution and may use zero-copy views
on CPU. Eager timing, general batching/derivative modes, accelerators and kernel
allocator peaks are explicitly unsupported in these records.

The workloads are unequal-support RBF MMD value/gradient and Euclidean energy;
complete OLS fit, parameter projection and fit-plus-inference; and supplied-row
refits with one frozen schedule under several chunk sizes. Parameter projection
allows compiler dead-code elimination of unused full-fit outputs; it is not a new
least-squares implementation. Chunk variants retain identical draw identities.
The graph-profile workload projects equal-mass article clouds onto one line,
constructs squared Wasserstein costs and RBF weights, then evaluates pooled
SAR log likelihood with and without its projection-angle derivative.

Before recording a successful measurement, independent NumPy float64 witnesses
check RBF MMD and its analytic input gradient, raw Euclidean V-energy, SVD least
squares and classical covariance, every supplied-row refit, and the projected
spatial profile and its active-sort analytic derivative. Reference work
runs outside timed execution. Records retain maximum discrepancies, fixed dtype
tolerances and their bounded-fixture rationale; finite but incorrect results
fail the campaign. These checks do not establish general implementation proofs.

Separate memory workers record process peak RSS including imports, compilation
and retained buffers. Compiler argument/output/temporary estimates are separate;
missing analysis is unavailable, never zero. Neither metric claims kernel/device
allocator peak. Profile retains JAX trace files when its profiler is available.

Compare refuses mismatched hardware, backend, precision, dependency lock,
workload design, threads, versions, cache policy or sampling protocol. It reports
per-session median ratios and retains raw samples; it does not certify speedups.
Code revisions may differ deliberately for optimization comparisons. Keep full
matched campaigns over time and investigate distributions before optimizing.

`history/` preserves the former exploratory benchmark notebook, helper and t05
JSON through file renames. They are historical context, not measurements produced
by this application. No historical DVC campaign is reconstructed here.

## Current wheel against independent references

The reference pilot uses an already installed candidate wheel in a separate
environment and a JAX-free peer process. It never modifies either environment.
Both environments must use versions from `src/python/uv.lock`; the candidate
process checks every installed STATADIC payload file against the supplied wheel.
Use a frozen resolved scope written by the release scope resolver, rather than
the authored selection alone. From the pinned repository environment:

```bash
just python::performance::reference pilot \
  --wheel /absolute/path/statadic.whl \
  --scope /absolute/path/resolved-scope.json --lock /absolute/path/statadic/src/python/uv.lock \
  --python /absolute/path/wheel-env/bin/python \
  --reference-python /absolute/path/reference-env/bin/python \
  --output /absolute/path/new-reference-pilot
just python::performance::reference verify \
  --campaign /absolute/path/new-reference-pilot --allow-pilot \
  --wheel /absolute/path/statadic.whl \
  --scope /absolute/path/resolved-scope.json --lock /absolute/path/statadic/src/python/uv.lock
```

The original bounded pilot measures OLS plus HC3 and fixed-graph SLX with classical
covariance against statsmodels, unequal weighted RBF squared-MMD plus an analytic
coordinate gradient against scikit-learn kernel values, a capped-simplex QP
against SciPy SLSQP, fixed-graph Gaussian SAR fit and expected-information
covariance against spreg, and censored ordinary lognormal AFT fit and observed
information against SciPy BFGS with an independent analytic score/Hessian.
It defaults to one session,
one warmup and three retained calls; `--sessions 3` requests three independent
sessions. `--cases ols-hc3` restricts a feasibility pilot. Inputs are prepared
before timing; both sides return matching synchronized numerical consumer
outputs. First-call latency and whole-process peak RSS include compilation where
applicable. These small deterministic cases do not cover an entire statistical
family, production data, float32, accelerators or general solution derivatives.
The weighted RBF adapter validates its empirical carriers during preparation;
their coordinates and masses remain dynamic inputs to the compiled value and
coordinate-gradient calculation.

All raw outputs, timing distributions, process identities, logs and exact input
identities remain in a new retained directory and sibling archive. The verifier
checks output agreement before deriving ratios. A timing or RSS loss remains
visible. A pilot is bounded measurement evidence, not statistical qualification.

The frozen scope selects its performance policy. The current alpha uses `report`:
at least one completed matched session for every requested workload, with the
actual executed case list and all warm-time and total-process RSS ratios retained.
Exact wheel, lock, harness, numerical parity and valid measurements still apply;
losses do not fail the performance policy. An absent workload still fails coverage.

The legacy/default `superiority` policy requires every workload and reviewed subcase in the frozen scope,
three independent sessions, five warmups and thirty samples. In **each** session,
both the warm median ratio and whole-process peak RSS ratio must be at most
`0.90`. A median over sessions cannot hide an adverse session.

Kernel PCA and Isomap fit-plus-held-out-transform cases compare against
scikit-learn; adjusted geometric Freedman–Lane compares observed and supplied-null
statistics against independent NumPy projections. These are measured operations,
not claims of population consistency, calibrated inference or full-family coverage.

The current registry enumerates exact input/output subcases for fourteen workload
IDs. It adds relative-precision WLS/HC3; grouped logit and exposure/offset Poisson;
joint estimated-dispersion Gamma/NB2; sample and constant-correlation Ledoit-Wolf;
supplied dyadic node refits with/without effects; fixed actual-W and three-update
centered-HAC affine GMM; weighted Euclidean energy/gradient; weighted unequal-support
real-line W1/W2; energy and RBF target-simplex projections; supplied row refits;
time-varying-W unit-effect panels and changing-roster observed pooled panels;
and ordinary/delayed-entry Weibull/lognormal AFT. `reference_workloads.CASES`, the
deterministic fixtures and `reference.QUALIFICATION_CASES` own their precise
operations, sizes, conventions and required subcases. They do not establish
coverage of other family variants.

The fifteenth workload, joint censored Gaussian SAR-AFT, still has no matched
full 500-by-20 fit/identified-information reference adapter. It cannot inherit a
small integration or objective-only pilot. Complete-scope `reference qualify`
and release verification therefore fail closed. Adapter registration is not
passing parity, statistical or performance evidence: those require actual
execution. Retained native and historical campaigns keep their descriptive
interpretation, including adverse RSS observations.
