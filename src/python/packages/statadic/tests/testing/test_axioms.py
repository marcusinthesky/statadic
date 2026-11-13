# standalone package-maintenance entry point
"""The axiom battery — every dissimilarity statadic ships, against what it declares.

This is :mod:`statadic.testing.axioms` exercised on real inputs: ``assert_axioms``,
``measure_axioms``, ``assert_negative_type``, ``dissimilarity_matrix`` and
``negative_type_eigmin`` are the code under test, and the ``CASES`` table in
``tests/_cases.py`` supplies the dissimilarities they are pointed at.

Two planes, deliberately:

* **property** — ``hypothesis`` drives random object sets through
  :func:`statadic.testing.axioms.assert_axioms`; the declared flags must hold on
  every draw.
* **witness** — fixed adversarial inputs prove what random draws cannot reach:
  ``cosine_distance`` violates the triangle inequality, and every normalising
  dissimilarity (including ``energy_distance`` on its *default* ground metric)
  violates the identity of indiscernibles. Without them the declarations would
  be untested over-claims.

The third plane t49 split out — **static**, where the marker lattice is a
pyrefly error rather than a convention — lives in ``tests/core/test_typing.py``
with the ``DMat`` brand it checks; the ``MARKER_AXIOMS`` bridge from brand to
flags lives in ``tests/core/test_axioms.py``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given
from hypothesis import strategies as st
from statadic.core.axioms import (
    METRIC,
    NEGATIVE_TYPE,
    Axioms,
)
from statadic.testing.axioms import (
    assert_axioms,
    assert_negative_type,
    dissimilarity_matrix,
    measure_axioms,
    negative_type_eigmin,
)

from .._budgets import examples
from .._cases import (
    ATOL,
    CASES_BY_NAME,
    DIM,
    IDS,
    PSD_ATOL,
    RTOL,
    SAMPLE_SIZE,
    SHIPPED_DECLARED_AXIOMS,
    objects,
    reference_objects,
)


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.parametrize("name", IDS)
@examples(
    20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(data=st.data())
def test_declared_axioms_hold(name: str, data: st.DataObject) -> None:
    """Every declared axiom holds on randomly drawn objects."""
    case = CASES_BY_NAME[name]
    drawn = data.draw(objects(case))
    assert_axioms(case.fn, case.declared, drawn, atol=ATOL, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.parametrize("name", IDS)
def test_negative_type_on_reference_objects(name: str) -> None:
    """The double-centred Gram matrix ``-½ J D J`` is PSD where claimed."""
    case = CASES_BY_NAME[name]
    if NEGATIVE_TYPE not in case.properties:
        pytest.skip("negative type not declared")
    assert_negative_type(case.fn, reference_objects(case), atol=PSD_ATOL)


@pytest.mark.component
@pytest.mark.property
@pytest.mark.parametrize("name", IDS)
def test_generic_behaviour_is_pinned(name: str) -> None:
    """Pin what each dissimilarity does on unstructured data.

    Two assertions, and the second is the interesting one. ``declared`` must hold
    (it holds everywhere, so it holds here too), and ``generic`` must be exactly
    what a fixed, structure-free object set produces. The gap between them is the
    set of axioms that survive typical data and die on an adversarial input —
    which is the reason a *declaration* has to be the worst case, not the
    observed case.
    """
    case = CASES_BY_NAME[name]
    measured = measure_axioms(case.fn, reference_objects(case), atol=ATOL, rtol=RTOL)
    assert case.declared & ~measured == Axioms(0), (
        f"{case.name} declares {case.declared} but measured {measured}"
    )
    assert measured == case.generic, (
        f"{case.name} measured {measured} on reference objects, pinned {case.generic}"
    )
    assert case.declared & case.generic == case.declared, (
        f"{case.name} declares {case.declared}, which is not implied by its "
        f"generic behaviour {case.generic}"
    )


@pytest.mark.component
@pytest.mark.contract
@pytest.mark.parametrize("name", IDS)
def test_every_generic_declaration_gap_has_a_named_witness(name: str) -> None:
    """Each apparent generic axiom loss is exhibited by case-owned data."""
    case = CASES_BY_NAME[name]
    gap = case.generic & ~SHIPPED_DECLARED_AXIOMS[case.name]
    covered = Axioms(0)
    for witness in case.witnesses:
        measured = measure_axioms(
            witness.fn or case.fn,
            witness.build(),
            atol=ATOL,
            rtol=RTOL,
        )
        assert witness.lost & measured == Axioms(0), (
            f"{case.name}: witness {witness.name!r} did not refute "
            f"{witness.lost!r}; measured {measured!r}"
        )
        covered |= witness.lost
    assert covered == gap, (
        f"{case.name}: generic/declaration gap {gap!r}, witness coverage {covered!r}"
    )


@pytest.mark.component
@pytest.mark.property
def test_zero_diagonal_and_separation_are_independent_measurements() -> None:
    """Reflexivity and point separation no longer rise or fall together."""
    three_objects = [jnp.asarray(i) for i in range(3)]

    def separated_nonreflexive(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(a == b, 1.0, -2.0)

    separated = measure_axioms(separated_nonreflexive, three_objects)
    assert Axioms.SEPARATION in separated
    assert Axioms.ZERO_DIAGONAL not in separated
    assert Axioms.NONNEGATIVE not in separated
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in separated

    def nonseparating_reflexive(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        del a, b
        return jnp.asarray(0.0)

    reflexive = measure_axioms(nonseparating_reflexive, three_objects)
    assert Axioms.ZERO_DIAGONAL in reflexive
    assert Axioms.SEPARATION not in reflexive
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in reflexive


@pytest.mark.component
@pytest.mark.property
def test_cosine_distance_is_not_a_metric() -> None:
    """A fixed witness that ``cosine_distance`` violates the triangle inequality.

    ``x = (1, 0)``, ``y = (1, 1)``, ``z = (0, 1)``:
    ``d(x, z) = 1`` but ``d(x, y) + d(y, z) = 2(1 - 1/sqrt(2)) ~ 0.586``.
    Together with ``test_normalising_metrics_fail_identity_of_indiscernibles``
    this is why cosine's declaration is only ``NONNEGATIVE | SYMMETRY`` and its
    brand only ``Premetric``.
    """
    cosine = CASES_BY_NAME["cosine_distance"].fn
    witness = [
        jnp.array([1.0, 0.0]),
        jnp.array([1.0, 1.0]),
        jnp.array([0.0, 1.0]),
    ]
    d = dissimilarity_matrix(cosine, witness)
    assert float(d[0, 2]) > float(d[0, 1] + d[1, 2])
    assert Axioms.TRIANGLE not in measure_axioms(cosine, witness)
    with pytest.raises(AssertionError, match="TRIANGLE"):
        assert_axioms(cosine, METRIC, witness)


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.parametrize(
    "name",
    ["angular_distance", "cosine_distance", "normalized_euclidean_distance"],
)
def test_normalising_metrics_fail_identity_of_indiscernibles(name: str) -> None:
    """``d(x, 2x) == 0``: the three normalising metrics are pseudometrics on R^d.

    Every one of them divides by ``||x||`` before comparing, so a vector and any
    positive multiple of it are indistinguishable. Their reference behaviour
    (``Case.generic``) reads METRIC only because random vectors are never
    parallel; the guarantee is PSEUDOMETRIC (and less, for cosine). This is the
    correction t46.1 makes to the declarations.
    """
    fn = CASES_BY_NAME[name].fn
    x = jnp.array([1.0, 2.0, 3.0])
    witness = [x, 2.0 * x, jnp.array([0.0, 1.0, 0.0])]
    assert float(dissimilarity_matrix(fn, witness)[0, 1]) == pytest.approx(0.0)
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in measure_axioms(
        fn, witness, atol=ATOL, rtol=RTOL
    )
    with pytest.raises(AssertionError, match="SEPARATION"):
        assert_axioms(fn, METRIC, witness, atol=ATOL, rtol=RTOL)
    # The declared guarantee, by contrast, survives the adversarial witness.
    assert_axioms(fn, CASES_BY_NAME[name].declared, witness, atol=ATOL, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.property
def test_angular_energy_distance_cannot_see_rescaling() -> None:
    """``energy_distance``'s DEFAULT ground metric fails the identity of indiscernibles.

    ``energy.energy_distance`` defaults to ``metric="angular"``, and angular
    distance is direction-only. So for **any** sample ``X`` with all-positive
    rows, ``E(X, 3X) == 0`` even though ``X`` and ``3X`` are different empirical
    measures — the statistic reports "identical distributions" for a pure
    rescaling. With ``metric="euclidean"`` the same pair is far apart.

    This is the whole reason the battery declares worst-case axioms: the
    reference measurement (``Case.generic``) reads SEMIMETRIC because generic
    samples are not rescalings of one another.
    """
    x = jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (SAMPLE_SIZE, 1))
    rescaled = 4.0 * x
    assert not bool(jnp.array_equal(x, rescaled))
    angular = CASES_BY_NAME["energy_distance[angular]"].fn
    euclidean = CASES_BY_NAME["energy_distance[euclidean]"].fn
    assert float(angular(x, rescaled)) == pytest.approx(0.0, abs=1e-8)
    assert float(euclidean(x, rescaled)) > 1.0
    witness = [x, rescaled, jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))]
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in measure_axioms(
        angular, witness, atol=ATOL, rtol=RTOL
    )
    assert_axioms(
        angular,
        CASES_BY_NAME["energy_distance[angular]"].declared,
        witness,
        atol=ATOL,
        rtol=RTOL,
    )


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.parametrize(
    "name",
    [
        "mmd[distance_induced[angular]]",
        "mmd_squared[distance_induced[angular]]",
    ],
)
def test_angular_mmd_cannot_see_rescaling(name: str) -> None:
    """Distance-induced angular MMD inherits the ground metric's scale blindness."""
    x = jnp.tile(jnp.array([[1.0, 0.0, 0.0]]), (SAMPLE_SIZE, 1))
    rescaled = 4.0 * x
    other = jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))
    fn = CASES_BY_NAME[name].fn
    witness = [x, rescaled, other]
    assert float(dissimilarity_matrix(fn, witness)[0, 1]) == pytest.approx(
        0.0, abs=ATOL
    )
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in measure_axioms(
        fn, witness, atol=ATOL, rtol=RTOL
    )
    assert_axioms(fn, CASES_BY_NAME[name].declared, witness, atol=ATOL, rtol=RTOL)


@pytest.mark.integration
@pytest.mark.property
@pytest.mark.parametrize("prefix", ["mmd", "mmd_squared"])
@pytest.mark.parametrize("kernel", ["linear", "cosine"])
def test_noncharacteristic_mmd_kernels_fail_identity(prefix: str, kernel: str) -> None:
    """Distinct empirical measures can share a linear/cosine mean embedding."""
    base = jnp.tile(jnp.array([[1.0, 2.0, 3.0]]), (SAMPLE_SIZE, 1))
    if kernel == "linear":
        delta = jnp.tile(jnp.array([[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]), (4, 1))
        indistinguishable = base + delta
    else:
        scales = jnp.arange(1.0, SAMPLE_SIZE + 1.0)[:, None]
        indistinguishable = scales * base
    other = jnp.tile(jnp.array([[0.0, 1.0, 0.0]]), (SAMPLE_SIZE, 1))
    name = f"{prefix}[{kernel}]"
    fn = CASES_BY_NAME[name].fn
    witness = [base, indistinguishable, other]
    assert float(dissimilarity_matrix(fn, witness)[0, 1]) == pytest.approx(
        0.0, abs=ATOL
    )
    assert (Axioms.ZERO_DIAGONAL | Axioms.SEPARATION) not in measure_axioms(
        fn, witness, atol=ATOL, rtol=RTOL
    )


@pytest.mark.component
@pytest.mark.property
def test_squared_mmd_need_not_satisfy_triangle() -> None:
    """Squared Hilbert distance is a semimetric, not the rooted MMD metric."""
    fn = CASES_BY_NAME["mmd_squared[linear]"].fn
    points = [jnp.full((SAMPLE_SIZE, DIM), value) for value in (0.0, 1.0, 2.0)]
    distances = dissimilarity_matrix(fn, points)
    assert float(distances[0, 2]) > float(distances[0, 1] + distances[1, 2])
    assert Axioms.TRIANGLE not in measure_axioms(fn, points, atol=ATOL, rtol=RTOL)


@pytest.mark.component
@pytest.mark.property
def test_euclidean_distance_survives_the_same_witness() -> None:
    """The control: ``euclidean_distance`` keeps IDENTITY where the others lose it."""
    fn = CASES_BY_NAME["euclidean_distance"].fn
    x = jnp.array([1.0, 2.0, 3.0])
    witness = [x, 2.0 * x, jnp.array([0.0, 1.0, 0.0])]
    assert_axioms(fn, METRIC, witness, atol=ATOL, rtol=RTOL)


@pytest.mark.component
@pytest.mark.property
def test_cosine_distance_is_of_negative_type_but_not_a_metric() -> None:
    """The case the linear marker chain cannot express, made explicit.

    ``cosine_distance`` is ``½||x_hat - y_hat||^2`` — of negative type, yet it
    fails both TRIANGLE and IDENTITY. ``NegativeType`` derives from ``Metric``
    and ``Semimetric`` would promise IDENTITY, so the brand is ``Premetric`` and
    the negative type travels as a checked property, not a static marker.
    """
    case = CASES_BY_NAME["cosine_distance"]
    drawn = reference_objects(case)
    assert negative_type_eigmin(case.fn, drawn) >= -PSD_ATOL
    assert Axioms.TRIANGLE not in measure_axioms(case.fn, drawn, atol=ATOL, rtol=RTOL)


@pytest.mark.component
@pytest.mark.property
def test_negative_type_witness_rejects_an_indefinite_dissimilarity() -> None:
    """The negative-type battery has a fixed failing witness, not only passes."""
    values = jnp.asarray(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 10.0],
            [1.0, 10.0, 0.0],
        ]
    )
    three_objects = [jnp.asarray(i) for i in range(3)]

    def indefinite(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return values[a, b]

    assert negative_type_eigmin(indefinite, three_objects) < -PSD_ATOL
    with pytest.raises(AssertionError, match="not of negative type"):
        assert_negative_type(indefinite, three_objects, atol=PSD_ATOL)


@pytest.mark.component
@pytest.mark.contract
def test_measure_axioms_rejects_too_few_objects() -> None:
    """Two objects cannot exhibit a triangle violation, so the battery refuses."""
    with pytest.raises(ValueError, match="at least 3 objects"):
        measure_axioms(
            CASES_BY_NAME["euclidean_distance"].fn,
            [jnp.zeros(DIM), jnp.ones(DIM)],
        )
