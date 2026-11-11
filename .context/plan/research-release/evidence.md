# Research-release evidence

The 2026-11-06 assessment checked a release-priorities proposal against commit `0d68571`. Paths use the short forms from the [research-workflow handoff](../api-ergonomics/tasks/research-workflow/handoff.md). Line numbers are hints from that commit.

## Verdicts

| # | Proposal | Verdict | Record |
| --- | --- | --- | --- |
| 1 | End-to-end OLS and Poisson walkthroughs on the installed wheel | Accept, P0 | [walkthroughs](tasks/walkthroughs/README.md) |
| 2 | Minimal `statadic-publishing` OLS/HC3 slice | Accept, P1; owner chose to plan it now | [publishing companion](../publishing-companion/README.md) |
| 3a | `fit.plot` accessor | Defer until walkthrough evidence exists | [plot-accessor](tasks/plot-accessor/README.md) |
| 3b | HoloViews as the primary stack | Agree; stays closed | `holoviews-stack` |
| 3c | Confidence bands on predicted curves | Agree; stays blocked as a statistical-method project | `prediction-bands` |
| 4 | Multi-model statistical tables | Agree; stays blocked until a real manuscript needs one | `statistical-table` |
| 5 | Freeze the serialization contract; no Orbax | Accept, P0; Orbax stays closed | [artifact-contract](tasks/artifact-contract/README.md), `orbax-checkpoints` |
| 6 | No ExtensionArray or xarray yet | Agree; stays blocked | `labelled-arrays` |
| 7 | Scientific release gate | Accept as additions to the existing hold | `publication-hold` |

## Findings

### Walkthroughs

- `examples/first_regression.py` still builds `Observations.from_arrays` and checks `fit.valid` and each row's validity by hand. It produces no figure and does not save or reload a predictor.
- `examples/` has no Poisson example. Only the manuscript listing `src/latex/projects/01_statadic/src/listings/poisson.py` shows exposure, mapping input, `prediction_line` and `save_predictor` together.
- `apps/docs/scripts/check_examples.py` runs the examples in both precision lanes, but against the editable workspace install.
- `scripts/check_distribution.py:22-154` installs the wheel outside the checkout, but runs a smoke script that asserts Polars and PyArrow are absent. It saves and reloads one OLS predictor and runs no example.
- `statadic.api.plots` exports only `prediction_line`. A residual diagnostic needs hand-built `PlotSpec` coordinates from `fit.core.fitted_values`.

### Publishing companion

- The [accepted design](../release-experience/publishing-design.md) specifies `build`, `verify` and `archive`, evidence bundle v1 and an OLS/HC3 vertical slice. The five `publishing-companion` tasks each held one line of acceptance.
- The design says STATADIC renders reports only as text, HTML and LaTeX. That premise is stale: `save_analysis` now writes the full report as schema-3 JSON, including None values, warnings, notes, metadata and plots (`api/artifacts/_snapshot.py:38-85`), and `load_analysis` restores it.
- `statadic-evidence` already provides `write_json`, `fingerprint`, `sha256`, `inventory`, `verify_inventory`, `write_archive` and `extract_archive`.
- `src/latex` pins Tectonic to a 6.2 MB vendored bundle, `bundle/tlextras-2022.0r0-statadic.zip`. `just latex::bundle` already audits an upload by recompiling it in an empty directory, and `tools/check-log` rejects unresolved references and overfull boxes.
- The `01_statadic` analysis already records `provenance.json` with a base revision, an analysis hash and a package-source hash. It prototypes, without schema or verification, what the companion generalizes.

### Artifacts

- Already present: an exact schema check (`api/artifacts/_store.py:238-243`), and tests for manifest corruption, schema-1 rejection, caller precision and object payloads (`tests/api/test_artifacts.py:288-365`).
- The manifest does not record which `statadic` version wrote it (`_store.py:181-194`).
- The rejection message "Unsupported artifact schema version." names neither the version found nor the version supported.
- The compatibility policy is one sentence: "Earlier schemas are rejected without migration" (`docs/reference/composition.md:359`). No page separates predictors, analysis snapshots and reproduction.
- The installed-wheel lane checks OLS predictor parity only.

### Release gate

- `publication-hold` already requires a human review of all eight Lean claims, a frozen release, retained qualification campaigns, a cold hosted gate and regenerated paper claims.
- `src/lean/registry/claims.json` already separates Lean mathematics from implementation. For example, `STATADIC-MMD-002` lists "IEEE rounding, overflow, JAX execution and XLA semantics are not verified".
- The `studies` and `performance` applications keep calibration and performance evidence apart.
- The `01_statadic` draft still records `af1314d` with a modified source state, because it was generated before commit `0d68571`. Rerun `just latex::generate 01_statadic` when a clean record is wanted.
