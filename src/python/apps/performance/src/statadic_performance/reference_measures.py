"""Independent weighted measure and target-projection reference consumers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def monotone_transport(left, right, left_mass, right_mass):
    """Exact one-dimensional coupling by moving remaining atom mass."""
    left_order, right_order = np.argsort(left), np.argsort(right)
    x, y = left[left_order], right[right_order]
    a, b = left_mass[left_order].copy(), right_mass[right_order].copy()
    i = j = 0
    result = np.zeros(2)
    while i < len(x) and j < len(y):
        mass = min(a[i], b[j])
        difference = abs(x[i] - y[j])
        result += mass * np.asarray([difference, difference**2])
        a[i] -= mass
        b[j] -= mass
        if a[i] <= 0:
            i += 1
        if b[j] <= 0:
            j += 1
    return result


def energy_value_gradient(x, y, a, b):
    """Euclidean V-energy and its coordinate gradient, including self pairs."""
    from scipy.spatial.distance import cdist

    xx, yy, xy = cdist(x, x), cdist(y, y), cdist(x, y)
    value = 2 * a @ xy @ b - a @ xx @ a - b @ yy @ b
    difference_xx, difference_xy = x[:, None] - x[None, :], x[:, None] - y[None, :]
    # A coincident pair's selected norm derivative is zero, as declared by the
    # candidate ground strategy. It is not a smoothness claim at collisions.
    self_gradient = np.sum(
        difference_xx / np.where(xx > 0, xx, 1)[:, :, None] * a[None, :, None], axis=1
    )
    cross_gradient = np.sum(
        difference_xy / np.where(xy > 0, xy, 1)[:, :, None] * b[None, :, None], axis=1
    )
    return value, 2 * a[:, None] * (cross_gradient - self_gradient)


def projection_peer(
    target,
    candidates,
    *,
    kernel,
    objective_tolerance=1e-12,
    gap_tolerance=1e-6,
    max_steps=10_000,
):
    """Construct independent mean distances/kernels and solve the actual QP."""
    from scipy.optimize import minimize
    from scipy.spatial.distance import cdist

    if kernel:
        from sklearn.metrics.pairwise import rbf_kernel

        def pair(x, y):
            return rbf_kernel(x, y, gamma=0.5).mean()

    else:

        def pair(x, y):
            return cdist(x, y).mean()

    gram = np.asarray([[pair(x, y) for y in candidates] for x in candidates])
    cross = np.asarray([pair(target, y) for y in candidates])
    target_self = pair(target, target)
    sign = 1 if kernel else -1

    def objective(weights):
        return sign * (target_self - 2 * cross @ weights + weights @ gram @ weights)

    def gradient(weights):
        return sign * 2 * (gram @ weights - cross)

    k = len(candidates)
    fit = minimize(
        objective,
        np.full(k, 1 / k),
        jac=gradient,
        method="SLSQP",
        bounds=[(0, 1)] * k,
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones(k),
        },
        options={"ftol": objective_tolerance, "maxiter": max_steps},
    )
    derivative = gradient(fit.x)
    gap = fit.x @ derivative - derivative.min()
    valid = (
        fit.success
        and abs(fit.x.sum() - 1) <= 1e-8
        and fit.x.min() >= -1e-8
        and gap <= gap_tolerance * (1 + abs(fit.fun))
    )
    return {"values": np.r_[fit.x, fit.fun], "valid": valid}


def prepare_measure(
    case: str, side: str, inputs: tuple, *, configuration=None
) -> tuple[Callable, tuple]:
    """Validate carriers outside timing; retain actual coordinates/masses inside."""
    if side == "statadic":
        import jax
        import jax.numpy as jnp
        from statadic.sample.packed import PackedClouds

        if case in {"weighted-energy-value-gradient", "quantile-transport-unequal"}:
            left = PackedClouds.from_points(inputs[0], inputs[2])
            right = PackedClouds.from_points(inputs[1], inputs[3])
            if case == "weighted-energy-value-gradient":
                from statadic.discrepancy.mmd.energy import energy_distance

                def objective(coordinates, left, right):
                    return energy_distance(
                        left.with_coordinates(coordinates), right, metric="euclidean"
                    )

                differentiated = jax.value_and_grad(objective)

                def execute(left, right):
                    value, gradient = differentiated(left.data, left, right)
                    return {
                        "values": jnp.r_[value, gradient.ravel()],
                        "valid": left.valid & right.valid,
                    }

            else:
                from statadic.discrepancy.transport.univariate import (
                    wasserstein_1d_power,
                )

                def execute(left, right):
                    return {
                        "values": jnp.stack(
                            [
                                wasserstein_1d_power(left, right, p=1),
                                wasserstein_1d_power(left, right, p=2),
                            ]
                        ),
                        "valid": left.valid & right.valid,
                    }

            return jax.jit(execute), (left, right)
        if configuration is None:
            raise ValueError("Projection workload requires frozen solver settings")
        left = PackedClouds.from_array(jax.device_put(inputs[0]))
        candidates = PackedClouds.from_array(jax.device_put(inputs[1]))
        if case == "simplex-energy-projection":
            from statadic.geometry.barycentre.mmd.energy import (
                energy_barycentre_weights,
            )

            def execute(target, candidates):
                weights, diagnostics = energy_barycentre_weights(
                    target,
                    candidates,
                    metric="euclidean",
                    return_diagnostics=True,
                    tol=configuration["candidate_fw_tolerance"],
                    maxiter=configuration["max_steps"],
                )
                return {
                    "values": jnp.r_[weights, diagnostics.energy],
                    "valid": diagnostics.converged,
                }

        elif case == "kernel-target-projection":
            from statadic.geometry.barycentre.mmd import kernel_barycentre_weights
            from statadic.geometry.ground.similarities import RBFKernel

            def execute(target, candidates):
                weights, diagnostics = kernel_barycentre_weights(
                    target,
                    candidates,
                    kernel=RBFKernel(bandwidth=1),
                    return_diagnostics=True,
                    tol=configuration["candidate_fw_tolerance"],
                    maxiter=configuration["max_steps"],
                )
                return {
                    "values": jnp.r_[weights, diagnostics.mmd_squared],
                    "valid": diagnostics.converged,
                }

        else:
            raise ValueError(f"Unknown measure case: {case}")
        # The public projection doors deliberately retain eager numerical
        # validation. Do not put their host diagnostics inside a new JIT.
        return execute, (left, candidates)
    if side != "reference":
        raise ValueError(f"Unknown reference side: {side}")
    if case == "weighted-energy-value-gradient":

        def execute(x, y, a, b):
            value, gradient = energy_value_gradient(x, y, a, b)
            return {"values": np.r_[value, gradient.ravel()], "valid": True}

    elif case == "quantile-transport-unequal":
        from scipy.stats import wasserstein_distance

        def execute(x, y, a, b):
            values = monotone_transport(x[:, 0], y[:, 0], a, b)
            # The public SciPy W1 supplies a separate check on the independent
            # monotone-coupling implementation used for W2.
            w1 = wasserstein_distance(x[:, 0], y[:, 0], a, b)
            return {
                "values": np.r_[w1, values[1]],
                "valid": abs(w1 - values[0]) <= 1e-10,
            }

    elif case in {"simplex-energy-projection", "kernel-target-projection"}:
        if configuration is None:
            raise ValueError("Projection workload requires frozen solver settings")

        def execute(target, candidates):
            return projection_peer(
                target,
                candidates,
                kernel=case == "kernel-target-projection",
                objective_tolerance=configuration["reference_objective_tolerance"],
                gap_tolerance=configuration["reference_gap_tolerance"],
                max_steps=configuration["max_steps"],
            )

    else:
        raise ValueError(f"Unknown measure case: {case}")
    return execute, inputs
