"""Independent NumPy witnesses for the bounded benchmark fixtures.

These witnesses are checks of measured numerical outputs, not proofs of Python
or JAX implementation correctness or guarantees outside the retained fixtures.
No STATADIC, JAX or benchmark implementation is called to construct a reference.
"""

from __future__ import annotations

import numpy as np


def rbf_reference(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """RBF bandwidth-one V-MMD and analytic derivative with respect to x."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xx = x[:, None, :] - x[None, :, :]
    xy = x[:, None, :] - y[None, :, :]
    yy = y[:, None, :] - y[None, :, :]
    kxx = np.exp(-np.sum(xx * xx, axis=-1) / 2)
    kxy = np.exp(-np.sum(xy * xy, axis=-1) / 2)
    kyy = np.exp(-np.sum(yy * yy, axis=-1) / 2)
    n, m = len(x), len(y)
    gradient = -2 * np.sum(kxx[:, :, None] * xx, axis=1) / n**2 + 2 * np.sum(
        kxy[:, :, None] * xy, axis=1
    ) / (n * m)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean()), gradient


def energy_reference(x: np.ndarray, y: np.ndarray) -> float:
    """Raw Euclidean empirical V-energy, retaining self-pairs."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def distance(a, b):
        return np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)).mean()

    return float(2 * distance(x, y) - distance(x, x) - distance(y, y))


def linear_reference(x: np.ndarray, y: np.ndarray) -> dict:
    """SVD least squares and classical SSE/(n-p) covariance on fixed full-rank data."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    condition = float(np.linalg.cond(x))
    if rank != x.shape[1] or condition > 100:
        raise ValueError("Reference fixture outside full-rank, condition <= 100 domain")
    prediction = x @ beta
    residual = y - prediction
    sse = float(residual @ residual)
    df = len(y) - x.shape[1]
    if df <= 0:
        raise ValueError("Reference requires positive residual degrees of freedom")
    coefficient_map = np.linalg.pinv(x)
    normalized = coefficient_map @ coefficient_map.T
    scale = sse / df
    return {
        "parameters": beta,
        "fitted_values": prediction,
        "residuals": residual,
        "sse": sse,
        "common_scale": scale,
        "normalized_covariance": normalized,
        "covariance": scale * normalized,
        "rank": rank,
        "df_resid": df,
        "condition_number": condition,
    }


def graph_profile_reference(angle, points, design, outcome):
    """Independent active-sort derivative, profiled OLS and determinant Jacobian."""
    angle, points, design, outcome = (
        np.asarray(v, dtype=np.float64) for v in (angle, points, design, outcome)
    )
    direction = np.array([np.cos(angle), np.sin(angle)])
    projection = points @ direction
    order = np.argsort(projection, axis=1)
    z = np.take_along_axis(projection, order, axis=1)
    dz = np.take_along_axis(
        points @ np.array([-direction[1], direction[0]]), order, axis=1
    )
    delta = z[:, None] - z[None, :]
    logits = -np.mean(delta**2, axis=-1) / 0.17
    dlogits = -2 * np.mean(delta * (dz[:, None] - dz[None, :]), axis=-1) / 0.17
    np.fill_diagonal(logits, -np.inf)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    w = exp / exp.sum(axis=1, keepdims=True)
    a = np.eye(len(w)) - 0.35 * w
    da = -0.35 * w * (dlogits - (w * dlogits).sum(axis=1, keepdims=True))
    x = design.reshape(-1, design.shape[-1])
    y = (outcome @ a.T).ravel()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    rss = residual @ residual
    periods, n = outcome.shape
    value = periods * np.linalg.slogdet(a)[1] - y.size / 2 * (
        1 + np.log(2 * np.pi * rss / y.size)
    )
    derivative = (
        periods * np.trace(np.linalg.solve(a, da))
        - periods * n * (residual @ (outcome @ da.T).ravel()) / rss
    )
    return value, derivative


def witness(
    workload: str, variant: str, inputs: tuple, result: object, dtype: str
) -> dict:
    """Reject wrong-but-finite results against independent prespecified references."""
    if dtype not in {"float32", "float64"}:
        raise ValueError("Unsupported witness precision")
    rtol, atol = (5e-5, 5e-7) if dtype == "float32" else (2e-10, 2e-12)
    checks = []
    conditions = []

    def compare(name, actual, expected, *, exact=False):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape:
            raise ValueError(f"Numerical witness shape mismatch: {name}")
        if not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise ValueError(f"Nonfinite numerical witness: {name}")
        discrepancy = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        limit = np.zeros_like(discrepancy) if exact else atol + rtol * np.abs(expected)
        passed = bool(np.all(discrepancy <= limit))
        checks.append(
            {
                "quantity": name,
                "max_absolute_error": float(np.max(discrepancy)),
                "max_allowed_error": float(np.max(limit)),
                "passed": passed,
                "exact": exact,
            }
        )
        if not passed:
            raise ValueError(f"Numerical witness disagreement: {name}: {checks[-1]}")

    if workload == "discrepancy":
        if variant == "mmd-value-gradient":
            expected_value, expected_gradient = rbf_reference(*inputs)
            value, gradient = result
            compare("MMD squared", value, expected_value)
            compare("analytic x gradient", gradient, expected_gradient)
        elif variant == "energy-euclidean":
            compare("raw Euclidean V-energy", result, energy_reference(*inputs))
        else:
            raise ValueError("Unknown discrepancy witness variant")
    elif workload == "graph-profile":
        expected_value, expected_derivative = graph_profile_reference(*inputs)
        if variant == "profile-value":
            compare("profile log likelihood", result, expected_value)
        elif variant == "profile-value-angle-gradient":
            compare("profile log likelihood", result[0], expected_value)
            compare(
                "analytic projection-angle derivative", result[1], expected_derivative
            )
        else:
            raise ValueError("Unknown graph-profile witness variant")
    elif workload == "ols":
        reference = linear_reference(*inputs)
        conditions.append(reference["condition_number"])
        if variant == "parameters-projection":
            compare("parameters", result, reference["parameters"])
        elif variant == "full-fit":
            for name in (
                "parameters",
                "fitted_values",
                "residuals",
                "sse",
                "common_scale",
                "normalized_covariance",
            ):
                compare(name, getattr(result, name), reference[name])
            compare("rank", result.rank, reference["rank"], exact=True)
            compare(
                "residual degrees of freedom",
                result.df_resid,
                reference["df_resid"],
                exact=True,
            )
            compare("valid fit", result.valid, True, exact=True)
        elif variant == "fit-and-inference":
            parameters, uncertainty = result
            compare("parameters", parameters, reference["parameters"])
            compare(
                "classical parameter covariance",
                uncertainty.covariance.values,
                reference["covariance"],
            )
            compare("available uncertainty", uncertainty.valid, True, exact=True)
        else:
            raise ValueError("Unknown OLS witness variant")
    elif workload == "row-refits":
        x, y, rows = inputs
        references = [linear_reference(x[index], y[index]) for index in rows]
        conditions.extend(reference["condition_number"] for reference in references)
        compare(
            "per-draw refit parameters",
            result.values,
            np.stack([r["parameters"] for r in references]),
        )
        compare("retained draw IDs", result.draw_ids, np.arange(len(rows)), exact=True)
        compare(
            "per-draw validity",
            result.valid,
            np.ones(len(rows), dtype=bool),
            exact=True,
        )
        compare(
            "per-draw status", result.status, np.zeros(len(rows), dtype=int), exact=True
        )
        compare("requested count", result.requested_count, len(rows), exact=True)
        compare("valid count", result.valid_count, len(rows), exact=True)
        compare("failed count", result.failed_count, 0, exact=True)
    else:
        raise ValueError("Unknown witness workload")
    return {
        "status": "passed",
        "reference": "independent NumPy float64 direct formulas/SVD",
        "rtol": rtol,
        "atol": atol,
        "tolerance_rationale": (
            "Fixed dtype budgets for these bounded bandwidth-one reductions and "
            "full-rank condition<=100 least-squares fixtures; accommodates float32 "
            "reduction/QR rounding against NumPy64, not a universal error bound. "
            "Identity/status/count checks are exact."
        ),
        "maximum_reference_condition_number": max(conditions) if conditions else None,
        "checks": checks,
    }
