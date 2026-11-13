"""Test failure-only subprocess diagnostics without executing numerical code."""

import subprocess
from unittest.mock import Mock

import _spatial_diagnostic as diagnostic
import pytest


def test_diagnostic_child_is_valid_python():
    compile(diagnostic.DIAGNOSTIC, "panel-diagnostic", "exec")


def test_success_does_not_run_diagnostics(monkeypatch):
    run = Mock()
    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    environment = {"JAX_ENABLE_X64": "0", "XLA_FLAGS": "flags"}
    diagnostic.run_with_diagnostic("primary", environment)
    run.assert_called_once_with(
        [diagnostic.sys.executable, "-c", "primary"],
        env=environment,
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("primary_timeout", [False, True])
@pytest.mark.parametrize(
    "diagnostic_outcome", ["success", "failure", "timeout", "oserror"]
)
def test_diagnostics_preserve_original_failure_and_environment(
    monkeypatch, primary_timeout, diagnostic_outcome
):
    primary = (
        subprocess.TimeoutExpired("primary", 60)
        if primary_timeout
        else subprocess.CalledProcessError(3, "primary")
    )
    result = {
        "success": None,
        "failure": subprocess.CalledProcessError(4, "diagnostic"),
        "timeout": subprocess.TimeoutExpired("diagnostic", 45),
        "oserror": OSError("cannot start diagnostic"),
    }[diagnostic_outcome]
    run = Mock(side_effect=[primary, result])
    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    environment = {"JAX_ENABLE_X64": "1", "STATADIC_CI_SEED": "exact"}
    with pytest.raises(type(primary)) as caught:
        diagnostic.run_with_diagnostic("primary", environment)
    assert caught.value is primary
    assert run.call_count == 2
    args, kwargs = run.call_args
    assert args[0] == [diagnostic.sys.executable, "-c", diagnostic.DIAGNOSTIC]
    assert kwargs == {"env": environment, "check": True, "timeout": 45}
