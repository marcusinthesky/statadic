"""Record canonical pytest collection and execution for CI shard verification."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest


def pytest_configure(config):
    global _controller_config
    if not hasattr(config, "workerinput"):
        _controller_config = config
    config._ci_universe = []
    config._ci_selected = []
    config._ci_reports = {}
    config._ci_worker_universes = []
    config._ci_durations = {}
    config._ci_node_durations = {}
    config._ci_active_nodes = {}
    config._ci_last_started = None
    config._ci_checkpoint_at = 0.0
    config._ci_timing_target = None
    target = os.environ.get("STATADIC_CI_PLUGIN_OUTPUT")
    plan = os.environ.get("STATADIC_CI_PLAN")
    shard = os.environ.get("STATADIC_CI_SHARD")
    if target and plan and shard is not None and not hasattr(config, "workerinput"):
        config._ci_timing_target = Path(target).with_name("timings.json")
        payload = Path(plan).read_bytes()
        config._ci_plan_digest = hashlib.sha256(payload).hexdigest()
        config._ci_shard = int(shard)
        planned = json.loads(payload)
        config._ci_pending_files = {}
        for node in planned.get("universe", []):
            filename = node.split("::", 1)[0]
            if planned["assignment"][filename] == config._ci_shard:
                config._ci_pending_files.setdefault(filename, set()).add(node)


def _atomic_json(target, value):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint(config, *, force=False):
    """Retain completed nodes every fifteen seconds, independently of success."""
    target = config._ci_timing_target
    now = time.monotonic()
    if target is None or (not force and now - config._ci_checkpoint_at < 15.0):
        return
    nodes = {
        node: {"reports": phases, "durations": config._ci_node_durations[node]}
        for node, phases in config._ci_reports.items()
        if "teardown" in phases
    }
    _atomic_json(
        target,
        {
            "schema_version": 1,
            "kind": "test-timings",
            "plan_sha256": config._ci_plan_digest,
            "shard": config._ci_shard,
            "execution": {
                "workers": config.getoption("numprocesses", default=0) or 0,
                "distribution": config.getoption("dist", default="no"),
            },
            "nodes": nodes,
        },
    )
    config._ci_checkpoint_at = now


def pytest_runtest_logstart(nodeid, location):
    config = _controller_config
    if config is None or config._ci_timing_target is None:
        return
    print(f"\nCI starting: {nodeid}", flush=True)
    config._ci_last_started = nodeid
    config._ci_active_nodes[nodeid] = {"started_at": time.time()}
    _active_snapshot(config)


def _active_snapshot(config):
    if config._ci_timing_target is None:
        return
    _atomic_json(
        config._ci_timing_target.with_name("last-started.json"),
        {"last_started": config._ci_last_started, "active": config._ci_active_nodes},
    )


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
    durations = config._ci_node_durations.setdefault(report.nodeid, {})
    durations.setdefault(report.when, []).append(report.duration)
    filename = report.nodeid.split("::", 1)[0]
    config._ci_durations[filename] = (
        config._ci_durations.get(filename, 0.0) + report.duration
    )
    if report.failed:
        print(
            f"\nCI failure: {report.nodeid} ({report.when})\n{report.longreprtext}",
            flush=True,
        )
    if report.when == "teardown":
        config._ci_active_nodes.pop(report.nodeid, None)
        _active_snapshot(config)
        pending = getattr(config, "_ci_pending_files", {}).get(filename)
        if pending is not None:
            pending.discard(report.nodeid)
        _checkpoint(config, force=pending == set())
    elif report.when == "setup" and report.nodeid in config._ci_active_nodes:
        worker = getattr(report, "worker_id", "controller")
        config._ci_active_nodes[report.nodeid]["worker"] = worker
        _active_snapshot(config)


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
    _checkpoint(config, force=True)
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
        _atomic_json(Path(target), result)
