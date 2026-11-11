# STATADIC notebooks

The notebooks in this directory explore STATADIC methods and computational performance. They are exploratory examples, not statistical qualification studies. The [interactive lab catalogue](../docs/labs/index.md) provides a guided learning sequence and its own launch instructions.

<!-- insitu:begin gittree
id = "gittree"
path = "src/python/packages/statadic/notebooks"
depth = 1
-->

```text
.
└── __marimo__
```

<!-- insitu:end -->

- The former homogeneity Monte Carlo notebook is retired. Use the
  [permutation](../docs/labs/permutation-evidence.md) and
  [calibration](../docs/labs/calibration.md) labs for bounded teaching examples.
  They do not reproduce its full distribution, dimension and effect-size grid.
  Retained research experiments belong to the studies application.
- The former kernel-mixing notebook is retired. Its mixture optimization and
  unequal-support examples are covered by the current mixture studies and
  focused independent numerical tests. Use packed probability measures with
  their true atom counts and masses; repeated-row equalization changes them.
- The historical benchmark notebook, environment helper and retained JSON now live in `../../../apps/performance/history/`. Use the internal performance application for new retained CPU measurements; historical records do not establish general hardware performance.

Run the guided replacements from the repository root:

```bash
just python::docs::lab permutation-evidence
just python::docs::lab calibration
```

STATADIC's declared runtime uses CPU JAX; a GPU setup requires separate caller configuration. Do not infer GPU timing or inferential calibration from an exploratory notebook. Consult [capabilities](../docs/capabilities.md) and [validation](../docs/validation.md) for the evidence and limits of supported configurations.
