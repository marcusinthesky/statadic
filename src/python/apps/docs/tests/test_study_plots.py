"""The visualization preserves retained failure-inclusive evidence and limits."""

from statadic_docs_labs.study_plots import operating_plot, operating_points


def report():
    return {
        "profile": "evaluation",
        "scenarios": {
            "adverse": {
                "requested": 100,
                "methods": {
                    "fit": {
                        "valid": 90,
                        "failed": 10,
                        "size_failure_inclusive": {"mean": 0.2},
                        "conditional_rejection": {"mean": 1 / 9},
                        "failure_rate": {"mean": 0.1},
                        "qualification": {
                            "size_lower": 0.13,
                            "size_upper": 0.28,
                            "size_status": "excess-error",
                            "failure_lower": 0.06,
                            "failure_upper": 0.16,
                        },
                    }
                },
            }
        },
    }


def test_plot_uses_retained_bounds_and_unconditional_rates(tmp_path):
    points = operating_points(report())
    assert set(points) == {"size", "failure"}
    assert points["size"][0]["mean"] == 0.2
    assert points["size"][0]["lower"] == 0.13
    assert points["size"][0]["status"] == "excess-error"
    target = tmp_path / "evidence.svg"
    assert operating_plot(
        report(),
        {"id": "example", "alpha": 0.05, "margin": 0.01, "failure_budget": 0.005},
        target,
    )
    rendered = target.read_text()
    assert "excess-error" in rendered
    assert "90/100 valid, 10 failed" in rendered
    assert "0.06" in rendered and "0.005" in rendered


def test_missing_bounds_do_not_create_a_chart(tmp_path):
    target = tmp_path / "absent.svg"
    assert not operating_plot({"scenarios": {}}, {}, target)
    assert not target.exists()
