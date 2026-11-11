# Publishing-companion handoff

Read this before starting the `bundle`, `paper`, `operations` or `replication` task. The [accepted design](../release-experience/publishing-design.md) and its 2026-11-06 amendment are authoritative. Where this guide is more specific, follow this guide. The [research-release plan](../research-release/README.md) sets the order and the [research-release handoff](../research-release/handoff.md) the settled decisions. Environment, source rules and finishing steps come from the [research-workflow handoff](../api-ergonomics/tasks/research-workflow/handoff.md).

## Settled decisions

- Publication is on hold. The companion builds and archives locally; it never uploads, submits or publishes.
- The package is `src/python/packages/statadic-publishing`, importable as `statadic_publishing`. It depends on `statadic` and `statadic-evidence`. STATADIC never imports it, and STATADIC's core CLI gains no commands.
- The first slice is one synthetic OLS/HC3 analysis, one table, one diagnostic figure and a verified archive that rebuilds outside the checkout. Reviewer revision, paper-wide claim tracking, generic scheduling and journal integrations come later.
- The companion never infers assumptions from an estimator name, never rewrites conclusions, and never treats a missing status as approval.

## Reuse; do not reimplement

| Need | Use |
| --- | --- |
| Report JSON with None values, warnings, notes and plots | `statadic.api.artifacts.save_analysis` and `load_analysis`; these replace the design's "report adapter" |
| Tables and figures | `Report.to_latex`, `PlotSpec.to_pgfplots`, and `statadic.api.plots.residual_plot` from [walkthroughs](../research-release/tasks/walkthroughs/README.md) |
| Canonical JSON, hashes, inventories, archives | `statadic_evidence.write_json`, `fingerprint`, `sha256`, `inventory`, `verify_inventory`, `write_archive`, `extract_archive` |
| TeX engine and packages | `tectonic -X compile --untrusted --only-cached --bundle <zip>`, with `src/latex/bundle/tlextras-2022.0r0-statadic.zip`, 6.2 MB in Git LFS |
| Log checks | `src/latex/tools/check-log` |
| Citations | `src/latex/refs.bib`, which `just sync` derives from `.context/reference/` records |

The synthetic paper lives in the package at `examples/ols-hc3/`, not under `src/latex/projects/`. `src/latex` tracks generated outputs so a checkout compiles as it stands, whereas a companion bundle is built into a fresh directory.

## Package wiring

Follow the `statadic-evidence` pattern:

- `src/python/packages/statadic-publishing/pyproject.toml` depends on `statadic[data]`, because the example starts from a frame, and on `statadic-evidence`, and has a `dev` group.
- Add `"packages/statadic-publishing/src"` to `source_roots` in `src/python/tach.toml`, and `"statadic_publishing"` to the first `[[modules]] paths` list.
- Add `statadic_publishing = ["statadic", "statadic_evidence"]` under `[tool.statadic_architecture.workspace_dependencies]` in `src/python/pyproject.toml`. The architecture script requires source roots and this table to agree.
- Add `mod publishing 'packages/statadic-publishing/justfile'` to `src/python/justfile`, and call `just publishing::check` and `just publishing::fmt` beside the `evidence` calls in its `check` and `fmt` recipes.
- Run `uv lock` in `src/python`; the lock change belongs with the package. The workspace already includes `packages/*`.
- Add `README.md` and `ARCHITECTURE.md` to the package, as `statadic-evidence` does.

## Testing

- Tests that compile TeX need `tectonic` and the vendored bundle. Both exist in the pinned devenv after `just latex::setup`.
- Witness comparisons with statsmodels use the `reference` group and `requires_optional`, following STATADIC's reference tests.
- A test that rebuilds outside the checkout installs freshly built wheels into a new venv, following `scripts/check_distribution.py`.

## Finishing a task

Follow the inherited finishing steps, and also run `just python::publishing::check`. Do not commit unless the owner asks.
