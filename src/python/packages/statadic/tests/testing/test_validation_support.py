"""Adversarial checks for numerical assertions and mandatory reference execution."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from .._budgets import examples
from .._typing import ill_typed
from .._validation.numeric import Tolerance, assert_finite_close

_EXACT = Tolerance(rtol=0, atol=0, rationale="Literal arithmetic witness.")
_PACKAGE = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_comparison_accepts_explicit_absolute_and_relative_budgets() -> None:
    """Small nonzero errors are permitted only by the supplied output budget."""
    assert_finite_close(
        [0.0005, 100.05],
        [0.0, 100.0],
        tolerance=Tolerance(rtol=0.001, atol=0.001, rationale="Known rounding budget."),
        label="estimates",
    )
    assert_finite_close(2, 2, tolerance=_EXACT, label="count")


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize(
    ("actual", "expected", "reason"),
    [
        ([[1.0], [2.0]], [1.0, 2.0], "shape"),
        ([np.nan], [np.nan], "nonfinite independent reference"),
        ([np.inf], [np.inf], "nonfinite independent reference"),
        ([np.nan], [1.0], "nonfinite valid-result output"),
        ([-1.0], [1.0], "estimates"),
        ([1.0000001], [1.0], "estimates"),
    ],
)
def test_comparison_rejects_malformed_or_incorrect_outputs(
    actual: np.typing.ArrayLike, expected: np.typing.ArrayLike, reason: str
) -> None:
    """Broadcasting, matching NaNs and plausible wrong values cannot pass."""
    with pytest.raises(AssertionError, match=reason):
        assert_finite_close(actual, expected, tolerance=_EXACT, label="estimates")


@pytest.mark.unit
@pytest.mark.contract
@examples(100)
@example(value=-1.0, field="rtol")
@example(value=np.nan, field="rtol")
@example(value=np.inf, field="atol")
@example(value=-np.inf, field="atol")
@given(
    value=st.one_of(
        st.floats(max_value=0.0, exclude_max=True, allow_nan=False),
        st.just(float("nan")),
        st.just(float("inf")),
    ),
    field=st.sampled_from(["rtol", "atol"]),
)
def test_tolerance_rejects_invalid_budgets(value: float, field: str) -> None:
    """A nonfinite or negative comparison budget is never meaningful."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        Tolerance(
            rtol=value if field == "rtol" else 0,
            atol=value if field == "atol" else 0,
            rationale="Invalid budget witness.",
        )


@pytest.mark.unit
@pytest.mark.contract
def test_tolerance_requires_rationale_and_is_immutable() -> None:
    """Comparison choices must be justified before observing the result."""
    with pytest.raises(ValueError, match="rationale"):
        Tolerance(rtol=0, atol=0, rationale="  ")
    with pytest.raises(FrozenInstanceError):
        ill_typed(_EXACT).atol = 1.0


def _run_policy(
    directory: Path, source: str, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Exercise the actual package conftest in a fresh isolated pytest process."""
    shutil.copyfile(_PACKAGE / "tests" / "conftest.py", directory / "conftest.py")
    shutil.copyfile(_PACKAGE / "tests" / "_budgets.py", directory / "_budgets.py")
    (directory / "__init__.py").touch()
    (directory / "test_policy.py").write_text(source, encoding="utf-8")
    environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    environment.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(  # isolated native pytest invocation  # noqa: S603
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(_PACKAGE / "pyproject.toml"),
            "--confcutdir",
            str(directory),
            "-p",
            "no:monitor",
            "-p",
            "no:tach",
            "-n",
            "0",
            "--no-testmon",
            "-m",
            "",
            "-q",
            "test_policy.py",
            *arguments,
        ],
        cwd=directory,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_compilation_cache_cleanup_is_opt_in_and_module_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: str
) -> None:
    """Each module releases caches after its tests only in bounded CI mode."""
    monkeypatch.setenv("STATADIC_CI_MEMORY_BOUNDED", enabled)
    source = """
import gc
import inspect
import jax
import pytest
from pathlib import Path

events = Path("events")
def record(event):
    with events.open("a") as stream:
        stream.write(event + "\\n")
jax.clear_caches = lambda: record("clear")
def collect():
    if inspect.currentframe().f_back.f_code.co_name == "_release_compiled_memory":
        record("collect")
gc.collect = collect
@pytest.mark.unit
def test_one():
    record("one")
@pytest.mark.unit
def test_two():
    record("two")
"""
    result = _run_policy(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "events").read_text().splitlines() == [
        "one",
        "two",
        *(["clear", "collect"] if enabled == "1" else []),
    ]


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("scenario", "body", "expected_code", "diagnostic"),
    [
        ("complete", "assert True", 0, "1 passed"),
        ("skip", "pytest.skip('oracle unavailable')", 1, "skip/xfail is forbidden"),
        ("xfail", "pytest.xfail('known mismatch')", 1, "skip/xfail is forbidden"),
        ("xpass", "assert True", 1, "skip/xfail is forbidden"),
        ("missing", "assert True", 4, "cannot import"),
        ("no_requirement", "assert True", 4, "requires requires_optional"),
        ("broken", "assert True", 4, "cannot import"),
        ("empty", "assert True", 4, "No external_reference tests were selected"),
    ],
)
def test_required_reference_policy(
    tmp_path: Path, scenario: str, body: str, expected_code: int, diagnostic: str
) -> None:
    """Required nodes must exist, import their comparator and execute assertions."""
    dependency = "numpy"
    if scenario == "missing":
        dependency = "statadic_missing_comparator"
    elif scenario == "broken":
        dependency = "statadic_broken_comparator"
        (tmp_path / f"{dependency}.py").write_text(
            "raise ImportError('installed comparator is broken')\n", encoding="utf-8"
        )
    marker = (
        "@pytest.mark.unit\n"
        if scenario == "empty"
        else "@pytest.mark.unit\n"
        "@pytest.mark.reference\n"
        "@pytest.mark.external_reference\n"
    )
    xfail = (
        "@pytest.mark.xfail(reason='unexpected success')\n"
        if scenario == "xpass"
        else ""
    )
    requirement = (
        ""
        if scenario == "no_requirement"
        else f"@pytest.mark.requires_optional('{dependency}')\n"
    )
    if scenario == "complete":
        body += (
            "; properties = dict(request.node.user_properties)"
            "; assert properties['reference_nodeid'] == request.node.nodeid"
            "; assert properties['reference_nodeid'].endswith('::test_comparison')"
            "; assert properties['reference_dependencies'] == '[\"numpy\"]'"
        )
    source = (
        "import pytest\n"
        f"{marker}{xfail}{requirement}"
        f"def test_comparison(request):\n    {body}\n"
    )
    result = _run_policy(tmp_path, source, "--require-external-references")
    output = result.stdout + result.stderr
    assert result.returncode == expected_code, output
    assert diagnostic in output, output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("source", "selection", "diagnostic"),
    [
        (
            "@pytest.mark.slow\ndef test_unscoped():\n    assert True\n",
            ["-m", "not slow"],
            "expected exactly one scope marker",
        ),
        (
            (
                "@pytest.mark.unit\n@pytest.mark.component\n"
                "def test_conflicting_scopes():\n    assert True\n"
            ),
            [],
            "expected exactly one scope marker",
        ),
        (
            (
                "pytestmark = pytest.mark.unit\n@pytest.mark.unit\n"
                "def test_repeated_inherited_scope():\n    assert True\n"
            ),
            ["-k", "does_not_match"],
            "expected exactly one scope marker",
        ),
    ],
)
def test_scope_policy_checks_all_nodes_before_selection(
    tmp_path: Path, source: str, selection: list[str], diagnostic: str
) -> None:
    """Invalid declarations fail even when pytest would deselect the node."""
    result = _run_policy(tmp_path, "import pytest\n" + source, *selection)
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert diagnostic in output, output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    "source",
    [
        "pytestmark = pytest.mark.unit\ndef test_module_scope():\n    assert True\n",
        (
            "class TestScoped:\n"
            "    pytestmark = pytest.mark.component\n"
            "    def test_class_scope(self):\n        assert True\n"
        ),
        (
            '@pytest.mark.parametrize("value", [\n'
            "    pytest.param(1, marks=pytest.mark.integration),\n"
            "    pytest.param(2, marks=pytest.mark.integration),\n"
            "])\n"
            "def test_parameter_scope(value):\n    assert value in (1, 2)\n"
        ),
    ],
)
def test_scope_policy_accepts_one_inherited_declaration(
    tmp_path: Path, source: str
) -> None:
    """Module, class and parametrized declarations each define one scope."""
    result = _run_policy(tmp_path, "import pytest\n" + source)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "passed" in output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("declaration", "diagnostic"),
    [
        ("@pytest.mark.requires_optional()", "requires_optional requires exactly one"),
        (
            "@pytest.mark.requires_optional('a', 'b')",
            "requires_optional requires exactly one",
        ),
        (
            "@pytest.mark.requires_optional(name='a')",
            "requires_optional requires exactly one",
        ),
        (
            "@pytest.mark.requires_optional('   ')",
            "requires_optional requires exactly one",
        ),
        (
            "@pytest.mark.requires_optional(None)",
            "requires_optional requires exactly one",
        ),
        ("@pytest.mark.requires_program()", "requires_program requires exactly one"),
        (
            "@pytest.mark.requires_program('a', 'b')",
            "requires_program requires exactly one",
        ),
        (
            "@pytest.mark.requires_program(name='a')",
            "requires_program requires exactly one",
        ),
        (
            "@pytest.mark.requires_program('  ')",
            "requires_program requires exactly one",
        ),
        (
            "@pytest.mark.requires_program(None)",
            "requires_program requires exactly one",
        ),
    ],
)
def test_requirement_marker_arguments_are_validated(
    tmp_path: Path, declaration: str, diagnostic: str
) -> None:
    """Requirement markers carry one nonblank positional string only."""
    source = (
        "import pytest\n"
        "@pytest.mark.unit\n"
        f"{declaration}\n"
        "def test_requirement():\n    assert True\n"
    )
    result = _run_policy(tmp_path, source)
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert diagnostic in output, output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("marks", "diagnostic"),
    [
        (
            (
                "@pytest.mark.external_reference\n"
                "@pytest.mark.requires_optional('numpy')\n"
            ),
            "external_reference requires reference",
        ),
        (
            "@pytest.mark.external_reference\n@pytest.mark.reference\n",
            "external_reference requires requires_optional",
        ),
        (
            "@pytest.mark.qualification\n",
            "qualification requires statistical",
        ),
    ],
)
def test_marker_implications_are_validated(
    tmp_path: Path, marks: str, diagnostic: str
) -> None:
    """External comparisons and qualification declare their parent purpose."""
    result = _run_policy(
        tmp_path,
        "import pytest\n@pytest.mark.unit\n"
        + marks
        + "def test_marked():\n    assert True\n",
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, output
    assert diagnostic in output, output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
def test_external_preflight_imports_only_selected_nodes(tmp_path: Path) -> None:
    """A deselected external comparison does not import its comparator."""
    (tmp_path / "statadic_deselected_comparator.py").write_text(
        "raise ImportError('deselected comparator must not import')\n",
        encoding="utf-8",
    )
    source = (
        "import pytest\n"
        "@pytest.mark.unit\n@pytest.mark.reference\n"
        "@pytest.mark.external_reference\n"
        "@pytest.mark.requires_optional('numpy')\n"
        "def test_selected_comparison():\n    assert True\n\n"
        "@pytest.mark.unit\n@pytest.mark.reference\n"
        "@pytest.mark.external_reference\n"
        "@pytest.mark.requires_optional('statadic_deselected_comparator')\n"
        "def test_deselected_comparison():\n    assert True\n"
    )
    result = _run_policy(
        tmp_path,
        source,
        "--require-external-references",
        "-k",
        "test_selected_comparison",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
    assert "deselected comparator must not import" not in output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
def test_unknown_marker_still_fails_under_strict_markers(tmp_path: Path) -> None:
    """The standardized scope policy leaves pytest strict-marker checks active."""
    result = _run_policy(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.unit\n@pytest.mark.unknown_policy_marker\n"
        "def test_unknown_marker():\n    assert True\n",
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "unknown_policy_marker" in output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("module", "expected_code", "diagnostic"),
    [
        ("statadic_missing_comparator", 0, "1 skipped"),
        ("statadic_broken_comparator", 1, "missing_transitive_comparator_dependency"),
        ("numpy.statadic_unsupported_interface", 1, "statadic_unsupported_interface"),
    ],
)
def test_optional_reference_distinguishes_absent_from_broken(
    tmp_path: Path, module: str, expected_code: int, diagnostic: str
) -> None:
    """Optional absence skips; broken packages and unsupported APIs fail."""
    (tmp_path / "statadic_broken_comparator.py").write_text(
        "import missing_transitive_comparator_dependency\n", encoding="utf-8"
    )
    result = _run_policy(
        tmp_path,
        "import pytest\n"
        "@pytest.mark.unit\n"
        "def test_comparison(optional_reference):\n"
        f"    optional_reference('{module}')\n",
    )
    output = result.stdout + result.stderr
    assert result.returncode == expected_code, output
    assert diagnostic in output, output


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("profile", "budget", "decorated_budget"),
    [
        ("statadic", 100, 40),
        ("statadic-reference", 20, 20),
        ("statadic-explore", 500, 200),
    ],
)
@pytest.mark.parametrize("ci", [False, True], ids=["ordinary", "ci"])
def test_native_hypothesis_profile_is_effective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    budget: int,
    decorated_budget: int,
    *,
    ci: bool,
) -> None:
    """CLI-selected budgets reach collection without weakened health checks."""
    if ci:
        monkeypatch.setenv("CI", "1")
    else:
        monkeypatch.delenv("CI", raising=False)
    result = _run_policy(
        tmp_path,
        "from hypothesis import settings\n"
        "from ._budgets import examples\n"
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "@examples(40)\n"
        "def decorated(): pass\n"
        "def test_profile():\n"
        f"    assert settings.get_current_profile_name() == '{profile}'\n"
        f"    assert settings.default.max_examples == {budget}\n"
        "    assert settings.default.deadline is None\n"
        "    assert not settings.default.suppress_health_check\n"
        "    assert settings.default.print_blob\n"
        "    assert not settings.default.derandomize\n"
        "    effective = decorated._hypothesis_internal_use_settings.max_examples\n"
        f"    assert effective == {decorated_budget}\n",
        f"--hypothesis-profile={profile}",
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
@pytest.mark.contract
@pytest.mark.subprocess
@pytest.mark.parametrize(
    ("configured", "expected"), [(None, True), ("false", False), ("true", True)]
)
def test_test_process_respects_explicit_jax_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected: bool,  # noqa: FBT001
) -> None:
    """Legacy x64 default applies only without a caller environment choice."""
    if configured is None:
        monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    else:
        monkeypatch.setenv("JAX_ENABLE_X64", configured)
    result = _run_policy(
        tmp_path,
        "from jax import config\n"
        "import pytest\n"
        "pytestmark = pytest.mark.unit\n"
        "def test_precision():\n"
        f"    assert config.x64_enabled is {expected!r}\n",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "1 passed" in output
