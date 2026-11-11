# Verification

Focused regression input, artifact and precision tests passed together (63 tests):
`devenv shell -- just python::statadic::test-fresh tests/api/test_inputs.py tests/api/test_regression_artifacts.py tests/api/test_artifacts.py tests/api/test_data.py`.

Focused regression input, regression and precision tests passed together (25 tests):
`devenv shell -- just python::statadic::test-fresh tests/api/test_inputs.py tests/api/test_regression.py tests/api/test_data.py`.

Final repository-wide checks and synchronization remain the integration owner's responsibility.

## Integration review

An independent review on 2026-11-05 probed each acceptance criterion directly against
the integrated source and then passed the full `just check`. See the
[integration verification](../../verification.md).

The review changed the `InvalidFitError` message from `...diagnostics.; solver_status=0`
to `OLS: <warnings verbatim> (solver_status=0, identified=False)`. The existing test,
which requires each warning verbatim, passes.
