"""Timing contracts exercise ordering instead of brittle wall-clock thresholds."""

import pytest
from statadic_performance.measurement import compiler_memory, repetitions, timed


def test_timing_and_warmup_synchronize_every_call():
    calls = []
    result = {"nested": [object()]}

    def call():
        calls.append("call")
        return result

    def sync(value):
        assert value is result
        calls.append("sync")

    clock = iter([1.0, 1.2, 2.0, 2.5])
    samples = repetitions(call, sync, warmups=2, samples=2, clock=lambda: next(clock))
    assert samples == pytest.approx([0.2, 0.5])
    assert calls == ["call", "sync"] * 4


def test_timing_includes_synchronization():
    order = []

    def clock():
        order.append("clock")
        return float(len(order))

    timed(lambda: order.append("call"), lambda _: order.append("sync"), clock)
    assert order == ["clock", "call", "sync", "clock"]


def test_missing_memory_is_unavailable():
    class Missing:
        def memory_analysis(self):
            return None

    assert compiler_memory(Missing())["status"] == "unavailable"
    assert "temp_size_in_bytes" not in compiler_memory(object())


def test_partial_memory_is_unavailable():
    class Partial:
        def memory_analysis(self):
            return object()

    assert compiler_memory(Partial())["status"] == "unavailable"
