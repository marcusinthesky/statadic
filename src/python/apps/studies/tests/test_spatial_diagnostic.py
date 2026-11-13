"""Test bounded failure diagnostics and replay input preservation."""

import subprocess
from unittest.mock import Mock

import _panel_replay as replay
import _spatial_diagnostic as diagnostic
import numpy as np
import pytest


def test_saved_inputs_preserve_rounded_values(tmp_path):
    arrays = tuple(np.array([value], dtype=np.float32) for value in (1.0, 2.0, 3.0))
    path = tmp_path / "inputs.npz"
    replay.save_inputs(path, *arrays)
    for actual, expected in zip(replay.load_inputs(path), arrays, strict=True):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_replay_rejects_pickled_arrays(tmp_path):
    path = tmp_path / "inputs.npz"
    np.savez(path, x=np.array([object()]), y=[1], graphs=[1])
    with pytest.raises(ValueError, match="Object arrays"):
        replay.load_inputs(path)


@pytest.mark.parametrize("x64", [False, True])
def test_replay_converts_saved_values_to_requested_precision(x64):
    original = np.array([np.nextafter(np.float32(1), np.float32(2))])
    converted = replay.precision_inputs((original,) * 3, x64)
    for array in converted:
        assert array.dtype == (np.float64 if x64 else np.float32)
        np.testing.assert_array_equal(array, original)


def test_success_does_not_run_diagnostics(monkeypatch, tmp_path):
    run = Mock()
    monkeypatch.setattr(diagnostic.subprocess, "run", run)
    monkeypatch.setattr(diagnostic, "REPOSITORY", tmp_path)
    environment = {"JAX_ENABLE_X64": "0", "XLA_FLAGS": "flags"}
    diagnostic.run_with_diagnostic("primary", environment)
    run.assert_called_once_with(
        [diagnostic.sys.executable, "-c", "primary"],
        env=environment,
        check=True,
        timeout=60,
    )
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("x64", ["0", "1"])
@pytest.mark.parametrize("primary_timeout", [False, True])
@pytest.mark.parametrize(
    "diagnostic_outcome", ["success", "failure", "timeout", "oserror"]
)
def test_diagnostics_preserve_original_failure_and_environment(
    monkeypatch, tmp_path, x64, primary_timeout, diagnostic_outcome
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
    monkeypatch.setattr(diagnostic, "REPOSITORY", tmp_path)
    environment = {"JAX_ENABLE_X64": x64, "STATADIC_CI_SEED": "exact"}
    with pytest.raises(type(primary)) as caught:
        diagnostic.run_with_diagnostic("primary", environment)
    assert caught.value is primary
    assert run.call_count == 2
    args, kwargs = run.call_args
    path = tmp_path / "build/ci/panel-diagnostic" / f"x64-{x64}" / "inputs.npz"
    assert args[0] == [
        diagnostic.sys.executable,
        str(diagnostic.REPLAY),
        "capture",
        str(path),
        x64,
    ]
    assert kwargs == {
        "env": environment
        | {"PANEL_PRIMARY_FAILURE": f"{type(primary).__name__}: {primary}"},
        "check": True,
        "timeout": 45,
    }
