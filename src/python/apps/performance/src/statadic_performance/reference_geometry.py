"""Matched ordination consumers and a fixed-geometry partial permutation test.

Ordination calls time the full host estimate and held-out transformation. The
geometric case times numerical response centering, full/reduced QR projections,
every supplied Freedman--Lane draw and its add-one tail. Host schema checking is
preparation on both sides. None of these cases establishes statistical validity.
Only candidate preparation imports JAX or STATADIC; peer workers remain independent.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _oriented(train, query, eigenvalues):
    """Fix each simple eigenaxis by its largest training coordinate, on both sides."""
    pivots = np.argmax(np.abs(train), axis=0)
    signs = np.where(train[pivots, np.arange(train.shape[1])] < 0, -1, 1)
    return np.r_[eigenvalues, (train * signs).ravel(), (query * signs).ravel()]


def geometric_permutation_peer(functional, design, schedules):
    """Use independent NumPy pulled-back bases instead of transformed response Grams.

    For A=H_reduced+P R_reduced, projection SS is tr((A'Q)' B (A'Q)).
    Its total is tr(B)+2 tr(H B R P'). This evaluates each exact supplied
    schedule with thin products, avoiding an artificially cubic peer refit.
    """
    gram = -0.5 * (
        functional
        - functional.mean(axis=0, keepdims=True)
        - functional.mean(axis=1, keepdims=True)
        + functional.mean()
    )
    full = np.linalg.qr(design, mode="reduced")[0]
    reduced = np.linalg.qr(design[:, :2], mode="reduced")[0]
    total = np.trace(gram)
    df_test, df_residual = full.shape[1] - reduced.shape[1], len(design) - full.shape[1]

    def components(full_ss, reduced_ss, moved_total):
        within, between = moved_total - full_ss, full_ss - reduced_ss
        return np.asarray(
            [
                within,
                between,
                moved_total,
                between / (moved_total - reduced_ss),
                (between / df_test) / (within / df_residual),
                df_test,
                df_residual,
            ]
        )

    observed = components(
        np.sum(full * (gram @ full)), np.sum(reduced * (gram @ reduced)), total
    )
    fitted_left = reduced @ (reduced.T @ gram)
    cross = fitted_left - (fitted_left @ reduced) @ reduced.T

    def pullback(basis, inverse):
        permuted = basis[inverse]
        return (
            reduced @ (reduced.T @ basis) + permuted - reduced @ (reduced.T @ permuted)
        )

    draws = []
    for order in schedules:
        if np.array_equal(order, np.arange(len(design))):
            draws.append(observed[4])
            continue
        inverse = np.argsort(order)
        pulled_full, pulled_reduced = (
            pullback(full, inverse),
            pullback(reduced, inverse),
        )
        moved_total = total + 2 * cross[np.arange(len(design)), order].sum()
        draws.append(
            components(
                np.sum(pulled_full * (gram @ pulled_full)),
                np.sum(pulled_reduced * (gram @ pulled_reduced)),
                moved_total,
            )[4]
        )
    draws = np.asarray(draws)
    pvalue = (1 + np.count_nonzero(draws >= observed[4])) / (1 + len(draws))
    values = np.r_[observed, draws, pvalue]
    return {
        "values": values,
        "valid": bool(np.isfinite(values).all() and observed[0] > 0),
    }


def prepare_geometry(
    case: str, side: str, inputs: tuple, *, configuration: dict
) -> tuple[Callable, tuple]:
    """Bind static method options; retain all fit and transform work in the call."""
    if case == "geometric-freedman-lane":
        return _prepare_geometric(side, inputs)
    if case not in {"kernel-pca-fit-transform", "isomap-fit-transform"}:
        raise ValueError(f"Unknown geometric consumer case: {case}")
    dim = configuration["dimensions"]
    if side == "reference":
        from sklearn.decomposition import KernelPCA
        from sklearn.manifold import Isomap

        def execute(training, query):
            specification = (
                KernelPCA(
                    n_components=dim,
                    kernel="rbf",
                    gamma=1 / (2 * configuration["bandwidth"] ** 2),
                    eigen_solver="dense",
                )
                if case == "kernel-pca-fit-transform"
                else Isomap(
                    n_neighbors=configuration["neighbors"],
                    n_components=dim,
                    eigen_solver="dense",
                    path_method="FW",
                    neighbors_algorithm="brute",
                    n_jobs=1,
                )
            )
            coordinates = specification.fit_transform(training)
            projected = specification.transform(query)
            spectrum = (
                specification.eigenvalues_
                if case == "kernel-pca-fit-transform"
                else specification.kernel_pca_.eigenvalues_
            )
            values = _oriented(coordinates, projected, spectrum[:dim])
            return {"values": values, "valid": np.isfinite(values).all()}

        return execute, inputs
    if side != "statadic":
        raise ValueError(f"Unknown benchmark side: {side}")
    import jax
    import jax.numpy as jnp
    from statadic.api.ordination import (
        DistanceGram,
        KernelGram,
        Ordination,
        PrincipalCoordinates,
        ShortestPathDistances,
    )
    from statadic.geometry.ground.similarities import RBFKernel

    names = tuple(f"feature_{index}" for index in range(inputs[0].shape[1]))
    specification = (
        Ordination(
            KernelGram(RBFKernel(configuration["bandwidth"])), PrincipalCoordinates(dim)
        )
        if case == "kernel-pca-fit-transform"
        else Ordination(
            DistanceGram(
                ShortestPathDistances(configuration["neighbors"], features=names)
            ),
            PrincipalCoordinates(dim, negative_inertia="positive-subspace"),
        )
    )

    def execute(training, query):
        fitted = specification.estimate(training)
        projected = fitted.transform(query)
        train = fitted.coordinates
        pivots = jnp.argmax(jnp.abs(train), axis=0)
        signs = jnp.where(train[pivots, jnp.arange(dim)] < 0, -1, 1)
        values = jnp.concatenate(
            [
                fitted.spectrum.eigenvalues[:dim],
                (train * signs).ravel(),
                (projected * signs).ravel(),
            ]
        )
        return {"values": values, "valid": jnp.all(jnp.isfinite(values))}

    # Both implementations perform a full fit and held-out transform, including
    # their own host preparation. Their validation is deliberately not traced.
    return execute, tuple(map(jax.device_put, inputs))


def _prepare_geometric(side, inputs):
    functional, design, schedules = inputs
    if design.shape[1] != 3 or np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The benchmark requires the identified full [1,z,x] design")
    if not np.array_equal(design[:, 0], np.ones(len(design))):
        raise ValueError(
            "The reduced benchmark design must retain its explicit intercept"
        )
    if side == "reference":
        return geometric_permutation_peer, inputs
    if side != "statadic":
        raise ValueError(f"Unknown benchmark side: {side}")
    import jax
    import jax.numpy as jnp
    from statadic.association.geometric import centered_response_gram, geometric_projection
    from statadic.inference.calibration import add_one_p_kernel
    from statadic.inference.geometric import freedman_lane_null

    def execute(functional, design, schedules):
        gram = centered_response_gram(functional)
        full = jnp.linalg.qr(design, mode="reduced")[0]
        reduced = jnp.linalg.qr(design[:, :2], mode="reduced")[0]
        observed = geometric_projection(gram, full, reduced)
        draws = freedman_lane_null(gram, full, reduced, schedules)
        pvalue = add_one_p_kernel(observed.statistic, draws.values)
        values = jnp.concatenate(
            [
                jnp.stack(
                    (
                        observed.within,
                        observed.between,
                        observed.total,
                        observed.r_squared,
                        observed.statistic,
                        observed.df_test,
                        observed.df_residual,
                    )
                ),
                draws.values,
                pvalue[None],
            ]
        )
        return {"values": values, "valid": observed.valid & jnp.all(draws.valid)}

    return jax.jit(execute), tuple(map(jax.device_put, (functional, design, schedules)))
