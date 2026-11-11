"""Report generators and availability witnesses independent of rendering."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st
from statadic.api.reporting import Metric, ParameterRow, Report, TestRow


def report_numbers() -> st.SearchStrategy[int | float]:
    """Finite report scalars, including exact counts and extreme magnitudes."""
    return st.one_of(
        st.integers(-(2**63), 2**63),
        st.floats(allow_nan=False, allow_infinity=False),
        st.sampled_from([-0.0, 5e-324, -1e-4, 1e-3, 1e100]),
    )


@dataclass(frozen=True)
class ReportCase:
    """Primitive inputs; construct hooked API records outside strategy draws."""

    available: bool
    interval: bool
    value: int | float
    count: int

    def build(self) -> Report:
        """Construct a report without consuming annotation-hook randomness in draws."""
        return _build_report(self)


@st.composite
def reports(draw: st.DrawFn) -> ReportCase:
    """Draw valid point, interval, metric and test availability combinations."""
    return ReportCase(
        draw(st.booleans()),
        draw(st.booleans()),
        draw(report_numbers()),
        draw(st.integers(1, 10**12)),
    )


def _build_report(case: ReportCase) -> Report:
    """Assemble typed records from a valid primitive case."""
    available, interval, value, count = (
        case.available,
        case.interval,
        case.value,
        case.count,
    )
    parameter = ParameterRow(
        "coefficient",
        value if available else None,
        valid=available,
        unavailable_reason=None if available else "coefficient withheld",
        ci_lower=-1.0 if available and interval else None,
        ci_upper=1.0 if available and interval else None,
        interval_status="available" if available and interval else "unavailable",
        interval_unavailable_reason=None
        if available and interval
        else "interval withheld",
    )
    metric = Metric(
        "criterion",
        value if available else None,
        available=available,
        unavailable_reason=None if available else "criterion withheld",
    )
    test = TestRow(
        "test",
        1.0 if available else None,
        pvalue=0.5 if available else None,
        law="normal" if available else None,
        law_parameters=(0.0, 1.0) if available else (),
        available=available,
        unavailable_reason=None if available else "test withheld",
    )
    return Report(
        "generated",
        parameters=(parameter,),
        metrics=(metric, Metric("n_observations", count)),
        tests=(test,),
    )


def expected_metric_state(
    *, available: bool, has_value: bool, has_reason: bool
) -> bool:
    """Require a value when available and a reason when withheld."""
    return (available and has_value and not has_reason) or (
        not available and not has_value and has_reason
    )


def expected_interval_state(*, available: bool, endpoints: bool, reason: bool) -> bool:
    """Require two endpoints or a reason for withholding the requested interval."""
    return (available and endpoints and not reason) or (
        not available and not endpoints and reason
    )
