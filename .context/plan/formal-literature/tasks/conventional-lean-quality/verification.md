# Verification

Verified locally on 2026-11-06 against the prepared Lean 4.35.0 environment.
No hosted workflows were run or changed, and no dependencies were added or upgraded.

## Changes

- Batteries is Lake's conventional lint driver. Default library roots are covered by
  complete Mathlib-generated public import indexes.
- The quality gate checks indexes, builds current targets with warnings as errors,
  then runs Batteries and Mathlib text style. Lean checks include the gate.
- Mathlib's text-lint API supplies both checks and available fixes through a small
  interpreted path adapter. The adapter is included in the compiled integrity census.
- Eight missing documentation strings were added. No lint suppressions were needed.
  Theorem statements, hypotheses and proof bodies are unchanged.
- Source-bound statement pins were explicitly regenerated after reviewing the
  documentation/index/configuration changes. All 17 claim mappings are retained.
- A manual-only prek hook avoids unprovisioned builds in ordinary pre-commit checks.
  Native diagnostics, import-graph, shake and sandboxed kernel replay have local recipes.
- Three persistent subprocess regression tests cover build freshness and fail-fast
  behavior after index, build and lint failures.

## Passed

- `just lean::lint`: complete indexes, warning-as-error build, both Batteries roots,
  and Mathlib text style. No adoption exception file was needed.
- `just lean::registry-test`: Ruff checks and formatting, all 39 tool regression
  tests, and scientific correspondence checks.
- `just lean::statement-pin`: compiled trust and claim validation, then reviewed
  source-pin refresh; 17 literature mappings and 240 compiled declarations.
- `just lean::certify`, rerun within the repository check after the pin refresh.
- `just lean::lint-builtin` and `just lean::quality-report`: no native diagnostic
  records were emitted. Native environment linting reported no registered linters;
  Batteries supplies the separate environment lint gate.
- `just lean::imports`: no unused-transitive-import candidates for either root.
- All ordinary pre-commit hooks and the workflow metadata/view checks passed within
  `just check`. Changed Markdown, shell syntax and justfile formatting also passed.

## Limitations

The required repository-wide `just check` was attempted once. It passed source
hooks, plan checks, Lean lint, certification and blueprint HTML generation, then
failed in existing `blueprint-pdf`: Tectonic could not locate the physical font
`cmbx10.pfb`. Python/Testmon suites and the later LaTeX/agent checks were therefore
not reached. This font provisioning issue was not changed by the tooling rollout.

A clean-cache rebuild, hosted CI, native `lake shake`, and sandboxed `lake check`
were not run. Native report mode and import-graph findings remain advisory under
upstream semantics; the Batteries and Mathlib text-style gate fails on findings.
