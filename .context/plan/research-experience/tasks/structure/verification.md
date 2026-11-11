# Structure audit verification

The audit covers 220 runtime Python files and 186 standalone pytest modules,
plus documentation, studies, performance, shared evidence and Lean tooling.
Independent runtime and application reviews checked the proposed boundaries,
test paths, fixture risks and migration policy. Direct imports establish candidate
ownership; they do not establish executed coverage.

No runtime module or test was relocated by this audit. The separately authorized
studies package reorganization is implemented. The audit's recursive certification
freshness prerequisite is also implemented: new or changed nested Lean tools and
docs source modules invalidate old certificates. The focused Lean registry gate
passed 33 tests, lint, formatting and source-correspondence checks.

## Repository checks

The final uninterrupted frozen `just check` passed. The following notes preserve
earlier failures and their corrections rather than describing current failures.

- `just fmt` and `just sync` ran explicitly.
- `just check` passed runtime lint, strict typing, 2,679 tests with 161 skips,
  distribution builds and 15 shared-evidence tests. It stopped in the studies
  lane when float32 GMS projections raised numerical-unavailability errors.
- The narrow correction gives those existing native failures a typed exception
  and structured counts; arbitrary runtime errors still propagate. No numerical
  algorithm, random schedule or precision setting changed. A new native-result
  witness passes strict typing and native projection/GMS/Wolak tests in float64
  (18 passed). The dedicated numerical projection regressions passed 10 tests.
- The complete frozen studies gate subsequently passed 261 tests with 3 skips.
  Performance passed 24 tests. Documentation passed 83 tests plus 16 native tests;
  strict Sphinx, Lean, LaTeX and agent checks passed.
- Additional float32 GMS/Wolak reference diagnostics found seven failures:
  numerical projection availability and float64-level comparison tolerances.
  This limitation is recorded in the user documentation. It is not hidden by
  the successful float64 reference lane or by retaining failed study attempts.

The later uninterrupted frozen `just check` passed: 2,680 runtime tests, 287
studies tests, 24 performance tests, 83 documentation tests, 16 native tests and
the evidence, Lean, LaTeX and agent gates. The earlier failed aggregate and
component checks above remain historical evidence. Broader runtime moves remain
proposals; all-method scientific evaluation remains an independent open task.
