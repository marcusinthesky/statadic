---
type: evidence
description: Configuration-specific numerical and statistical limits, with actions for an analysis.
---

# Known limitations

The tables collect observed limits for particular methods, configurations and
outputs, together with the action each observation calls for in an analysis.

<!-- statadic:known-limitations -->

## How to read this

Begin with the procedure and configuration that match your analysis, then read
the method's reference page and its entry in the [study catalogue](research.md).
A numerical stress test measures a different property from an interval-coverage
study, even when both calculations use the same estimator.

The ordinary AFT interval study requested 10,000 trials per scenario and required
a one-sided lower coverage bound of at least 0.94 for nominal 95% intervals.
Unavailable intervals counted as uncovered. These criteria describe the
particular study, while the optimization and precision observations describe
numerical examples with their own inputs and tolerances.

Changing precision, integration budget or the statistical procedure calls for
fresh evidence about the resulting calculation. A nearby configuration can
provide a useful comparison, but its results alone cannot establish the behavior
of the procedure you intend to report.
