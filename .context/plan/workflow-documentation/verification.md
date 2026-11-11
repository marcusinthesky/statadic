# Workflow documentation verification — 31 October 2026

## Delivered

Four task-first sections: Start here, Everyday work, Compose models, Reference.
The homepage exposes fit, uncertainty, LaTeX export and model composition
shortcuts. A short workflow and a six-piece component map introduce the design
before generic protocols. Existing methods, deep guides and notebook paths remain.

The first regression retains one fit through diagnostics, inference, prediction
and LaTeX export; the regression lab downloads the inference report matching the
selected covariance. A shared booktabs template and pinned compiler helper serve
both TeX preparation and offline checks. Python setup prepares their resources.

The new capstone composes lognormal survival, period effects and fixed spatial
coefficients of 0 and 0.2. A predeclared administrative limit right-censors one
spell. The zero-coefficient comparison checks agreement with the independent-spell
panel likelihood. Fixed coefficients are not estimates of spatial dependence.

## Checks executed

- **Full `devenv shell -- just check` passed.** This includes hooks, workflow
  validation, STATADIC lint/types, the complete no-Testmon fast suite, distribution
  build, documentation checks, Lean/blueprint, LaTeX and agent projections.
- Scientific suite: **2,677 passed, 161 skipped, five warnings**, 482.36 seconds.
- Documentation app: **55 passed** with JAX x64 disabled; **14 passed** in a
  separate x64 native process. Tests compare OLS, HC3 standard errors and normal
  intervals with independent NumPy arithmetic; check report values, escaping,
  unavailable fields and warnings; compile the actual report; and test the
  censored spatial composition in both precision modes.
- Strict Sphinx build, six strict marimo source checks, three existing tutorial
  commands in both modes, and all six executed snapshots passed.
- The revised regression and new spatial example also ran in both precision
  modes in a clean installed-package environment containing only STATADIC runtime
  dependencies (JAX 0.11.2, NumPy 2.5.4). This verifies tutorial execution, not
  general scientific qualification of that dependency combination.
- Real Chrome checks verified the four navigation groups, all local targets
  across 58 generated HTML pages, component-map mobile rendering, source
  downloads, and covariance-specific LaTeX downloads. Changing covariance
  retained the displayed regression slope. No JavaScript errors were seen.
- The generated booktabs PDF was rendered and visually inspected: table columns
  and retained notes fit the tested landscape/footnotesize template. Compiler
  checks reject overfull warnings.
- SHA-256 verification confirmed all **421 scientific source/test files**
  unchanged, and the package pyproject remains byte-identical to the extracted
  source. No public scientific API, numerical default or production tolerance
  changed. The workspace lock and runtime dependencies were not changed.

## Network and scientific limits

`just python::docs::setup` succeeded with the existing local cache. Two isolated
fresh-cache attempts were blocked by upstream HTTP **429 Too Many Requests**
for the pinned Tectonic bundle index. Cold-cache download preparation therefore
remains unverified; cached setup and pinned offline compilation passed. Checks
remain offline and do not hide failed downloads or substitute another bundle.

Slow, GPU, optional external-reference and statistical qualification suites were
not rerun. The fast-suite skips are not passes. No student usability study or
ARM execution was performed. The capstone's small integration budget and finite
reported integration diagnostic do not establish integration accuracy or
statistical calibration. Single-model LaTeX rendering is not an automatic
multi-model publication-table engine. No deployment, commit or push was made.
