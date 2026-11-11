# Fit contract specification

Normative for `fit-contract`, and read by the wave 3 family migrations.

- Files:
  - New: `src/statadic/api/fit_statistics.py` and `tests/api/test_fit_statistics.py`.
  - Edited: `src/statadic/api/base.py`, `src/statadic/api/capabilities.py`,
    `tests/api/regression/test__linear.py` (moving tests out),
    `tests/api/test_reporting.py`, the capabilities test (create one if the layout
    expects it), and `src/python/tach.toml`.
  - In `src/statadic/api/reporting.py`: the name constants and label typing only.
- `LikelihoodSummary`, a frozen dataclass:

  ```python
  log_likelihood: float          # NaN when the fit is invalid
  n_parameters: int              # k under the specification's counting rule
  n_observations: int            # n entering the likelihood
  unavailable_reason: str        # used when log_likelihood is not finite
  ```

  Validate non-bool integers, `n_parameters >= 0`, `n_observations >= 1` and a
  nonempty reason string.
- `FitStatistics` moves unchanged. Add
  `@classmethod from_likelihood(cls, summary) -> FitStatistics`, which returns
  `information_metrics(...)` from the summary plus the information note.
- `information_metrics(log_likelihood, n_parameters, n_observations, *, reason)`
  moves with its signature and behaviour unchanged, as does `_MIN_HQIC_OBSERVATIONS`.
- `gaussian_least_squares_summary(core: LinearFit) -> LikelihoodSummary`: the
  Gaussian maximum-likelihood log likelihood at the profiled scale, with the
  precision-weight term, `k = rank + 1` and `n`. It is NaN when unusable, with the
  existing reason.
- `linear_fit_statistics(core, *, weighted) -> FitStatistics` is the current
  `_linear_fit_statistics` body: `exact-fit-r-squared` has already fixed it. Build
  its likelihood metrics through `from_likelihood(gaussian_least_squares_summary(core))`,
  keeping metric order and notes byte-for-byte.
- `_INFORMATION_NOTE` and `_r_metric` move and stay private.
- `api/base.py`:
  - Add `Result.likelihood_summary(self) -> LikelihoodSummary | None: return None`.
  - `Result.fit_statistics()` returns `FitStatistics()` when the summary is `None`,
    else `FitStatistics.from_likelihood(summary)`.
  - Re-import the moved names for transition (`FitStatistics`,
    `information_metrics`, `_INFORMATION_NOTE`, and `_linear_fit_statistics` bound
    to `linear_fit_statistics`), each marked `# Transitional: removed by fit-cleanup`.
  - Remove the moved names from base's `__all__`, and remove the definitions and
    imports that base no longer needs.
- `api/reporting.py`:
  - `type FitStatisticName = Literal["n_observations", "n_parameters", "r_squared", "adjusted_r_squared", "uncentred_r_squared", "log_likelihood", "aic", "bic", "hqic"]`.
  - `FIT_STATISTIC_NAMES: tuple[FitStatisticName, ...]` in that order. Export both.
  - `_METRIC_LABELS` becomes `Mapping[FitStatisticName, str]`.
  - A test asserts the label keys equal `FIT_STATISTIC_NAMES`.
- `api/capabilities.py`: add both Protocols to `__all__`.

  ```python
  class ReportsFitStatistics(Protocol):
      def fit_statistics(self) -> FitStatistics: ...

  class SummarizesLikelihood(Protocol):
      def likelihood_summary(self) -> LikelihoodSummary | None: ...
  ```

- `tach.toml`:
  - Add a `[[modules]]` entry for `statadic.api.fit_statistics`, with
    `cannot_depend_on_external = ["numpy", "scipy"]` and `depends_on` of
    `statadic.api.reporting`, `statadic.core` and `statadic.estimation.linear`.
  - Add `statadic.api.fit_statistics` to the `depends_on` of `statadic.api.base`, the
    capabilities module if declared, `statadic.api.inference`, `statadic.api.regression`,
    `statadic.api.spatial` and `statadic.api.survival`.
- Tests:
  - Move `test_information_metrics_formulas_and_order`,
    `test_information_metrics_unavailability` and the `FitStatistics` default test
    from `test__linear.py` to `test_fit_statistics.py`.
  - Add `LikelihoodSummary` validation, `from_likelihood` equality with
    `information_metrics` plus the note, a minimal `Result` subclass whose
    `fit_statistics()` derives from `likelihood_summary()`, and static
    `ReportsFitStatistics` conformance for every family result, including
    `HurdleResult`.
