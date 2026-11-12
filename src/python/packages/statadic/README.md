# STATADIC

**Composable statistics. Inspectable mathematics. Reproducible evidence.**

Fit a familiar model on JAX, then change supported design, estimation or inference
components. Retained state connects your result to its assumptions and evidence.
Use [Regression](docs/reference/regression.md) for outcomes,
[Ordination](docs/reference/ordination.md) for coordinates and
[Estimand](docs/reference/estimand.md) for covariance or supplied moments.
State a [Hypothesis](docs/reference/hypotheses.md); inspect a
[Report](docs/reference/reporting.md). Ordination can also supply predictor terms.

**Align → Specify → Estimate → Inspect → Diagnose → Predict or infer → Report.**

[Install](docs/start/installation.md) · [First regression](docs/start/first-regression.py) ·
[Choose a method](docs/methods.md) · [Concepts](docs/concepts/index.md)

A retained result holds training preparation, diagnostics and availability
information. A numerical fit alone establishes neither identification nor valid inference.

Caller precision governs computation; transformation support is
[method-specific](docs/reference/transformation-support.md). Energy distance needs an explicit metric;
angular geometry ignores scale.

Inspect [validation](docs/validation/index.md), [limitations](docs/validation/known-limitations.md)
and [selected mathematical foundations](docs/formal/index.md) separately.
Human source-fidelity review remains pending; Lean results do not verify JAX implementations.

Python 3.13 and 3.14; table and plotting extras are optional.

## Development

Install the package into a dedicated environment, then use its `just` recipes:

```bash
uv sync --project src/python --package statadic
cd src/python/packages/statadic
just lint
just typecheck
just test-fresh tests/core/test_domains.py
```

[CONTRIBUTING.md](../../../../CONTRIBUTING.md#package-development) covers typing,
test lanes, compatibility checks and module ownership. The
[release guide](../../../../infra/release/README.md) covers versions and publishing.
