# Paper verification

The analysis produced its declared outputs, retained HC3 values agreed with
statsmodels, and the snapshot included the residual plot. The initial combined
publishing check reported three passed and one failed: the offline manuscript
compile needed resources absent from the prescribed vendored TeX subset.

Paper, operations and replication paused at that failed premise. On 2026-11-06
the owner explicitly authorized refreshing the vendored bundle. The coordinator
added an ordinary article/bibliography/PGFplots warmup fixture to the existing
refresh recipe. The pinned upstream returned HTTP 429 on preparation attempts;
the original bundle pin and offline compilation requirement remain unchanged.

The refresh subsequently succeeded. The publishing check passed four tests in
both the default and x64-enabled environments, including the analysis, statsmodels
witness, offline compilation and unchanged log checks. Cross-lane review found
no blocking findings. A final check also linted and formatted the shipped example
scripts and passed four tests. The coordinator established acceptance and marked
this task done from passing checkpoint 1. Publication remains on hold.

Checkpoint 1 passed and establishes acceptance; see the [integrated results](../../../research-release/verification.md).
