"""Explicit synchronization can shrink generated indexes before budget checks."""

import runpy
import sys
from pathlib import Path


def test_sync_checks_budget_after_refreshing_generated_views(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    main = runpy.run_path(str(root / "src/python/scripts/workflow/script.py"))["main"]
    events = []
    monkeypatch.setattr(sys, "argv", ["workflow", "sync"])
    for name, replacement in {
        "read_records": lambda: [],
        "schema_check": lambda *args, budget=True: events.append(("schema", budget)),
        "validate_records": lambda *args: [],
        "bibliography": lambda *args, **kwargs: events.append("bibliography"),
        "insitu": lambda *args: events.append("sync"),
    }.items():
        monkeypatch.setitem(main.__globals__, name, replacement)
    assert main() == 0
    assert events == [("schema", False), "bibliography", "sync", ("schema", True)]
