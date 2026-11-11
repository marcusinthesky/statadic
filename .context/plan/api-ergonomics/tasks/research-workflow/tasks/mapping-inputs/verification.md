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

The review fixed two Ruff findings in `tests/api/test_inputs.py`. One was a magic cluster
count, now derived from the data. The other was a positional boolean literal, now
`np.bool_(1)`. The tests' assertions are unchanged.
