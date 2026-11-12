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

## CPU measurement provenance

The measurements below were collected in the original downstream repository
using its `statadic_qualification` DVC stage and synthetic fixture. That stage and its
`data/shared/statadic_qualification/` artifacts are not part of this standalone checkout;
its `dvc repro` commands cannot be run here. These records describe their measured
configurations and do not establish results for another machine or data set.

For standalone implementation checks, run `just python::statadic::test-fresh` from the
repository root. The [release guide](../../../../infra/release/README.md) describes additional optional
reference and qualification lanes. These tests do not reproduce the historical
benchmark or substitute for the original retained artifacts.

Benchmarks record preparation, host-to-device transfer, cold lowering/compilation
and synchronized warm repetitions separately, along with shape, dtype, actual
device, JAX/backend versions and native thread settings. Compiler-reported argument,
output and temporary allocation bytes are distinct from process resident memory
or measured device peak memory. A missing allocator report is unavailable evidence,
never a zero-memory claim. Timing distributions are evidence, not brittle
unit-test thresholds.

The measured operations include OLS, row-pairs refits, dense/sparse spatial
products and checked solves, projected MMD objectives, unequal-support transport,
checked capped QP and compiled Hotelling calculations. Synthetic numerical witnesses separately cover
SLX, fixed effects and a noncommuting-W spatial-panel profile. These operator
measurements do not qualify scalable sparse log-determinants, full likelihood
throughput or unmeasured general transport, packing and shrinkage paths. See the
[spatial/panel guide](../../packages/statadic/docs/reference/spatial-panels.md) for family-specific limits.

Additional bounded fitting measurements compare full and parameter-only linear
fits, fixed-schedule refits, fixed/iterated GMM and a full 64-unit SAR likelihood.
GMM records include failed and mixed valid/failed lanes. Isolated comparison
processes own their float32/float64 choice; the governed qualification stage
retains its caller-owned float64 contract. A single small full-likelihood
fixture does not establish scalable likelihood throughput. Recorded compiler
allocations remain distinct from allocator peaks and retained statistical state.

Bootstrap schedules and retained outputs occupy memory proportional to requested
draws. The shared chunk executor bounds concurrent per-refit workspaces; it does
not make total retained storage independent of the number of draws. The benchmark
varies chunk size with one fixed schedule, preserving paired draw identity.

Larger fixed-shape CPU measurements also exercise unequal supports and objective
gradients. Dense packed projected-MMD gradients at 128 versus 511 points in eight
dimensions required approximately 314 MB of compiler temporaries in float64
(157 MB in float32). This is a workspace limitation of that measured path;
unequal-support correctness and transformation support do not establish efficient
memory use at arbitrary cloud sizes. Compiler estimates are not allocator peaks.

Hotelling is an application consumer, not another STATADIC model family. Its
economic equations, certificates and equilibrium algorithms stay app-local;
the benchmark exercises their compiled numerical state and composition with
STATADIC. Certified application entrypoints own float64. The new Threefry innovation
schedule is a separate scientific change, not a numerical equivalence claim.

No GPU measurements, general resampling validity, all-family certification or
release qualification are implied. For nonlinear fitted-solution derivatives,
an optimizer adjoint additionally needs a valid stationary condition and regularity;
[Optimistix documents that distinction](https://docs.kidger.site/optimistix/api/adjoints/).

## Current-wheel matched references

The repository performance application compares an installed
candidate wheel with independent peer processes. It prepares identical inputs
before timing, then measures the complete numerical consumer outputs below.
Prepared empirical carriers keep coordinates and masses dynamic. The public target-projection and ordered dyadic-rank calculations use eager
execution, preserving the procedure being measured.

The executable definitions are
[`reference_workloads.CASES`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference_workloads.py),
[`reference_fixtures`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference_fixtures.py)
and [`reference.QUALIFICATION_CASES`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference.py).
They freeze exact inputs, sizes, conventions, output vectors and required
subcases. These workload contracts do not cover every statistical variant in
their named families.

| Required workload | Enumerated consumers and independent references |
| --- | --- |
| `linear-fit-and-inference` | OLS and relative-precision WLS coefficients plus HC3 covariance; statsmodels |
| `canonical-glm` | Grouped-binomial logit with offset and Poisson with exposure/offset: fit, model covariance, likelihood and means; statsmodels |
| `dispersion-likelihood` | Gamma and NB2 joint estimated dispersion: coefficients, physical dispersion, likelihood and physical-coordinate observed information; independent SciPy likelihood/analytic score and information differences |
| `covariance-shrinkage` | Sample covariance and constant-correlation Ledoit–Wolf target, weight and estimate; independent NumPy moments. This is not sklearn's identity target. |
| `dyadic-node-refits` | Supplied multinomial node counts with and without node effects: ordered rank selection, focal coefficient, SSE, overall/within R² and active counts; independent weighted SVD refits |
| `affine-gmm` | Fixed non-efficient actual W and three centered Bartlett-HAC updates: fit, retained actual W, final HAC covariance and criterion; independent SciPy SVD/Bartlett calculations |
| `weighted-mmd-energy` | Unequal weighted RBF V-MMD and Euclidean exponent-one V-energy values plus coordinate gradients; sklearn kernels and independent analytic gradients |
| `empirical-transport` | Unequal weighted real-line W1 and squared W2; SciPy W1 and independent monotone coupling. Assignment, sliced and Sinkhorn performance are outside this contract. |
| `measure-projections` | Energy and RBF target-simplex mixtures: weights and original objective; independent SciPy SLSQP over separately calculated distance/kernel means |
| `qp-gmv` | Capped-simplex quadratic weights and objective; SciPy SLSQP |
| `supplied-row-refits` | Initial OLS and all supplied aligned row-pairs coefficient refits; independent NumPy SVD solves |
| `spatial-operators` | Fixed-graph SLX fit/classical covariance/fitted values and Gaussian SAR fit/expected covariance/fitted/reduced means; statsmodels and spreg |
| `spatial-panel` | Time-varying-W unit-effect SAR fit and expected information, plus changing-roster/full-graph-missing pooled SAR fit and observed information; independent period-block Gaussian calculations |
| `ordinary-aft` | Right-censored Weibull/lognormal, ordinary and delayed entry: fit, likelihood and observed information; independent SciPy likelihoods/analytic derivatives |
| `joint-censored-sar-aft` | **Unsupported matched reference:** full 500-unit ×20-period fit and identified observed information. Small integration or objective-only pilots do not replace it. |
| `ordination-fit-transform` | Kernel PCA and Isomap training and held-out coordinates; scikit-learn with matched centering, bandwidth or union-neighbor paths and frozen queries |
| `geometric-regression-permutations` | Observed and supplied Freedman–Lane partial statistics and p-value; independent NumPy projections |

A workload definition specifies what to measure. Numerical agreement,
statistical behavior and resource use each need their own results. The current alpha selects `report`: completed matched sessions
for every requested workload, numerical agreement and the actual executed case
list, with all adverse ratios visible. A bounded case does not establish
full-family coverage. The legacy/default `superiority` policy additionally
requires every registered subcase, three independent sessions, five warmups,
thirty samples and warm-time and total-process RSS ratios at most `0.90` in each
session. Process RSS includes imports, compilation and buffers that remain allocated.
It measures the whole process rather than incremental workload memory. Cold/first-call costs remain separate.
The unsupported joint workload leaves the complete set of measurements
incomplete under both policies.
