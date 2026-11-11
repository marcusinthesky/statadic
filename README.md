---
description: "Composable statistics, inspectable mathematics, reproducible evidence."
---

# STATADIC

![STATADIC logo: two curves resolve into one fit through three observations](src/python/apps/docs/static/brand/statadic-curve-fit-animated.svg)

[![Status: alpha](https://img.shields.io/badge/status-alpha-orange)](src/python/packages/statadic/CHANGELOG.md)
[![Python 3.13–3.14](https://img.shields.io/badge/python-3.13%E2%80%933.14-3776AB?logo=python&logoColor=white)](src/python/packages/statadic/docs/start/installation.md)
[![JAX 0.11](https://img.shields.io/badge/JAX-0.11-2F6FB3)](https://github.com/jax-ml/jax)
[![Proofs: Lean 4 and Mathlib](https://img.shields.io/badge/proofs-Lean%204%20%2B%20Mathlib-2F6FB3)](src/lean/README.md)
[![Manuscripts: LaTeX and PGFplots](https://img.shields.io/badge/manuscripts-LaTeX%20%2B%20PGFplots-008080?logo=latex&logoColor=white)](src/latex/README.md)
[![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![Types: Pyrefly strict](https://img.shields.io/badge/types-Pyrefly%20strict-2F6FB3)](https://pyrefly.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**Composable statistics. Inspectable mathematics. Reproducible evidence.**

Fit familiar statistical models on JAX. Change supported components while keeping
the fitted state, assumptions and evidence inspectable.

[Regression](src/python/packages/statadic/docs/reference/regression.md),
[Ordination](src/python/packages/statadic/docs/reference/ordination.md),
[Estimand](src/python/packages/statadic/docs/reference/estimand.md),
[Hypothesis](src/python/packages/statadic/docs/reference/hypotheses.md) and
[Report](src/python/packages/statadic/docs/reference/reporting.md) share one workflow.

**Align → Specify → Estimate → Inspect → Diagnose → Predict or infer → Report.**

[Start](src/python/packages/statadic/docs/start/index.md) ·
[Methods](src/python/packages/statadic/docs/methods.md) ·
[Evidence](src/python/packages/statadic/docs/validation/index.md)

Lean-checked results, numerical comparisons and statistical studies are distinct
evidence; none verifies every floating-point implementation.

Declared Python range: 3.13–3.14; repository environments: Linux x86-64 and ARM64.
See [installation](src/python/packages/statadic/docs/start/installation.md).

For development, see [contributor commands](CONTRIBUTING.md); after `just setup`,
run `just python::docs::build` and `just python::docs::serve`.
