"""Record canonical pytest collection and execution for CI shard verification."""

import json
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config._ci_universe = []
    config._ci_selected = []
    config._ci_reports = {}
    config._ci_worker_universes = []
    config._ci_durations = {}


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config, items):
    yield
    config._ci_universe = [item.nodeid for item in items]
    shard = os.environ.get("STATADIC_CI_SHARD")
    if shard is not None:
        assignment = json.loads(Path(os.environ["STATADIC_CI_PLAN"]).read_text())[
            "assignment"
        ]
        selected = [
            item
            for item in items
            if assignment[item.nodeid.split("::", 1)[0]] == int(shard)
        ]
        deselected = [
            item
            for item in items
            if assignment[item.nodeid.split("::", 1)[0]] != int(shard)
        ]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
    config._ci_selected = [item.nodeid for item in items]


def pytest_runtest_logreport(report):
    # The controller receives reports from xdist workers; setup skips count.
    config = _controller_config
    if config is None:
        return
    outcomes = config._ci_reports.setdefault(report.nodeid, {})
    outcomes.setdefault(report.when, []).append(report.outcome)
    filename = report.nodeid.split("::", 1)[0]
    config._ci_durations[filename] = (
        config._ci_durations.get(filename, 0.0) + report.duration
    )
    if report.failed:
        print(
            f"\nCI failure: {report.nodeid} ({report.when})\n{report.longreprtext}",
            flush=True,
        )


_controller_config = None


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    global _controller_config
    _controller_config = node.config


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error):
    result = node.workeroutput.get("ci_shard", {})
    node.config._ci_worker_universes.append(result)
    if error:
        node.config._ci_worker_universes.append({"worker_error": str(error)})


def pytest_sessionfinish(session, exitstatus):
    config = session.config
    result = {"universe": config._ci_universe, "selected": config._ci_selected}
    if hasattr(config, "workeroutput"):
        config.workeroutput["ci_shard"] = result
        return
    workers = config._ci_worker_universes
    if workers:
        result = {
            "workers": workers,
            "reports": config._ci_reports,
            "durations": config._ci_durations,
        }
    result["exitstatus"] = int(exitstatus)
    target = os.environ.get("STATADIC_CI_PLUGIN_OUTPUT")
    if target:
        Path(target).write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
