"""Mark canonical documentation notebook tests as end-to-end cases."""

from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[4] / "packages/statadic/docs"
SECTIONS = tuple(DOCS / name for name in ("start", "tutorials", "guides"))
MISSING_TESTS = pytest.StashKey[Path]()


def pytest_addoption(parser):
    parser.addoption(
        "--require-notebook-tests",
        metavar="PATH",
        help="Require an ordinary test collected from this notebook",
    )


def pytest_collection_modifyitems(items):
    """Apply only to tests collected from the canonical notebook directories."""
    for item in items:
        path = Path(str(item.path)).resolve()
        if path.suffix == ".py" and any(
            path.is_relative_to(section) for section in SECTIONS
        ):
            item.add_marker(pytest.mark.e2e)


def pytest_collection_finish(session):
    target = session.config.getoption("--require-notebook-tests")
    if target is None:
        return
    target = Path(target).resolve()
    if not any(
        Path(item.path).resolve() == target and not isinstance(item, pytest.DoctestItem)
        for item in session.items
    ):
        session.config.stash[MISSING_TESTS] = target


def pytest_sessionfinish(session):
    if session.config.stash.get(
        MISSING_TESTS, None
    ) is not None and session.exitstatus in (
        pytest.ExitCode.OK,
        pytest.ExitCode.NO_TESTS_COLLECTED,
    ):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter):
    target = terminalreporter.config.stash.get(MISSING_TESTS, None)
    if target is not None:
        terminalreporter.write_line(
            f"No ordinary tests collected from notebook: {target}"
        )
