# Rubric v1

This is a new Python research-workflow adaptation, **not** StatLLM's original
expert rubric. Source: https://github.com/yili-hong/StatLLM, pinned in `core.py`.
The selected original problem IDs and dataset paths are in `tasks.json`; original
prompts are `ProblemDescription/PDxxxx.txt`, SAS programs are
`Humanverified_SAScode/SCxxxx.txt` under that pinned source.

| Component | Points | Criterion |
| --- | ---: | --- |
| Reproducible contract | 10 | Two fresh offline replays; valid schema and correct identity fields |
| Estimates | 40 | Equal weight per declared scalar/array; all elements satisfy tolerances |
| Inference | 25 | Model-based covariance, reference laws, intervals, marginal contrasts and adjustments |
| LaTeX | 10 | Fresh offline untrusted Tectonic compile produces a PDF |
| Interpretation | 10 | Human: 10 correct; 5 minor omission; 0 missing or materially misleading |
| Assigned package | 5 | Human: 5 genuine assigned-package estimation/inference; 0 substitution or unverifiable |

Automatic score is **/85**. Final score stays `null` until review; it is never
silently rescaled to /100. Review includes JSON/TeX numerical agreement at stated
rounding, plot correctness, statistical meaning, assumptions and implementation
inspection. Imported module names and matching outputs alone cannot prove use of
a package. Fabrication, leaked gold answers or undisclosed replacement fitting
are integrity violations and force the final score to zero.

A reviewer supplies an external JSON file, never mounted in the agent workspace:

```json
{
  "submission_sha256": "source digest from run.json",
  "reviewer": "reviewer-id",
  "interpretation": 10,
  "package_policy": 5,
  "integrity_violation": false,
  "notes": "Reviewed replayed results, rendered report, figures and source."
}
```

Invoke `run.py review RUN_DIRECTORY REVIEW_FILE`. Existing reviews are not
overwritten. For blinded prose assessment, show randomized report IDs without
package labels; package-policy review separately sees source. No LLM judge or
paid evaluation calls are enabled by default.

## Scientific boundaries

The independent witness uses NumPy QR/least squares and SciPy likelihood
optimization/distribution functions, not STATADIC as its own oracle. Tests compare
synthetic witnesses with statsmodels. These tests do **not** establish agreement
on all 20 source datasets or qualification of every native STATADIC route.
Calibrate both packages on each task before interpreting a measured study.

Adaptations are explicit in each task. In particular PD0035 encodes a string
factor omitted by the SAS declaration; PD0128 models binary Pain=Yes, not Duration;
PD0096 fixes equal-weight LS means; PD0142 distinguishes raw from studentized
residuals. Dataset cleanup is fixed before agent execution. SAS code is a
methodological reference, not an infallible numerical oracle. No Cox, Kaplan–Meier,
penalized GAM or spatial tasks have been silently substituted.

The suite overrepresents classical regression. Repeated tasks share datasets.
Summary means weight tasks equally after averaging repetitions. Descriptive
bootstrap intervals resample dataset clusters, not individual attempts. A frontier
point needs the same full task set and known usage for every run. Incomplete token
accounting remains unavailable, never zero; prices are user-supplied USD per million
with a date and source. Provider nondeterminism, cached inputs and subscription
billing limit claims of strict replayability or actual marginal dollar spending.
