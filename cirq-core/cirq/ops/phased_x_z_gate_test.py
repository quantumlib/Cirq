# pylint: disable=wrong-or-nonexistent-copyright-notice
import random

import numpy as np
import pytest
import sympy

import cirq


def test_init_properties():
    g = cirq.PhasedXZGate(x_exponent=0.125, z_exponent=0.25, axis_phase_exponent=0.375)
    assert g.x_exponent == 0.125
    assert g.z_exponent == 0.25
    assert g.axis_phase_exponent == 0.375


def test_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.PhasedXZGate(x_exponent=0.25, z_exponent=0.5, axis_phase_exponent=0.75)
    )

    # Sensitive to each parameter.
    eq.add_equality_group(cirq.PhasedXZGate(x_exponent=0, z_exponent=0.5, axis_phase_exponent=0.75))
    eq.add_equality_group(
        cirq.PhasedXZGate(x_exponent=0.25, z_exponent=0, axis_phase_exponent=0.75)
    )
    eq.add_equality_group(cirq.PhasedXZGate(x_exponent=0.25, z_exponent=0.5, axis_phase_exponent=0))

    # Different from other gates.
    eq.add_equality_group(cirq.PhasedXPowGate(exponent=0.25, phase_exponent=0.75))
    eq.add_equality_group(cirq.X)
    eq.add_equality_group(cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0))


def test_canonicalization():
    def f(x, z, a):
        return cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)

    # Canonicalizations are equivalent.
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(f(-1, 0, 0), f(-3, 0, 0), f(1, 1, 0.5))
    """
    # Canonicalize X exponent (-1, +1].
    if isinstance(x, numbers.Real):
        x %= 2
        if x > 1:
            x -= 2
    # Axis phase exponent is irrelevant if there is no X exponent.
    # Canonicalize Z exponent (-1, +1].
    if isinstance(z, numbers.Real):
        z %= 2
        if z > 1:
            z -= 2

    # Canonicalize axis phase exponent into (-0.5, +0.5].
    if isinstance(a, numbers.Real):
        a %= 2
        if a > 1:
            a -= 2
        if a <= -0.5:
            a += 1
            x = -x
        elif a > 0.5:
            a -= 1
            x = -x
    """

    # X rotation gets canonicalized.
    t = f(3, 0, 0)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0
    t = f(1.5, 0, 0)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0

    # Z rotation gets canonicalized.
    t = f(0, 3, 0)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == 1
    assert t.axis_phase_exponent == 0
    t = f(0, 1.5, 0)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == -0.5
    assert t.axis_phase_exponent == 0

    # Axis phase gets canonicalized.
    t = f(0.5, 0, 2.25)._canonical()
    assert t.x_exponent == 0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0.25
    t = f(0.5, 0, 1.25)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0.25
    t = f(0.5, 0, 0.75)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == -0.25

    # 180 degree rotations don't need a virtual Z.
    t = f(1, 1, 0.5)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0
    t = f(1, 0.25, 0.5)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == -0.375
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(t), cirq.unitary(f(1, 0.25, 0.5)), atol=1e-8
    )

    # Axis phase is irrelevant when not rotating.
    t = f(0, 0.25, 0.5)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == 0.25
    assert t.axis_phase_exponent == 0


def test_from_matrix():
    # Axis rotations.
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X ** 0.1)),
        cirq.PhasedXZGate(x_exponent=0.1, z_exponent=0, axis_phase_exponent=0),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X ** -0.1)),
        cirq.PhasedXZGate(x_exponent=-0.1, z_exponent=0, axis_phase_exponent=0),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Y ** 0.1)),
        cirq.PhasedXZGate(x_exponent=0.1, z_exponent=0, axis_phase_exponent=0.5),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Y ** -0.1)),
        cirq.PhasedXZGate(x_exponent=-0.1, z_exponent=0, axis_phase_exponent=0.5),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Z ** -0.1)),
        cirq.PhasedXZGate(x_exponent=0, z_exponent=-0.1, axis_phase_exponent=0),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Z ** 0.1)),
        cirq.PhasedXZGate(x_exponent=0, z_exponent=0.1, axis_phase_exponent=0),
        atol=1e-8,
    )

    # Pauli matrices.
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(np.eye(2)),
        cirq.PhasedXZGate(x_exponent=0, z_exponent=0, axis_phase_exponent=0),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.X)),
        cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Y)),
        cirq.PhasedXZGate(x_exponent=1, z_exponent=0, axis_phase_exponent=0.5),
        atol=1e-8,
    )
    assert cirq.approx_eq(
        cirq.PhasedXZGate.from_matrix(cirq.unitary(cirq.Z)),
        cirq.PhasedXZGate(x_exponent=0, z_exponent=1, axis_phase_exponent=0),
        atol=1e-8,
    )

    # Round trips.
    a = random.random()
    b = random.random()
    c = random.random()
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=c)
    assert cirq.approx_eq(cirq.PhasedXZGate.from_matrix(cirq.unitary(g)), g, atol=1e-8)


@pytest.mark.parametrize(
    'unitary',
    [
        cirq.testing.random_unitary(2),
        cirq.testing.random_unitary(2),
        cirq.testing.random_unitary(2),
        np.array([[0, 1], [1j, 0]]),
    ],
)
def test_from_matrix_close_unitary(unitary: np.ndarray):
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.PhasedXZGate.from_matrix(unitary)), unitary, atol=1e-8
    )


@pytest.mark.parametrize(
    'unitary',
    [
        cirq.testing.random_unitary(2),
        cirq.testing.random_unitary(2),
        cirq.testing.random_unitary(2),
        np.array([[0, 1], [1j, 0]]),
    ],
)
def test_from_matrix_close_kraus(unitary: np.ndarray):
    gate = cirq.PhasedXZGate.from_matrix(unitary)
    kraus = cirq.kraus(gate)
    assert len(kraus) == 1
    cirq.testing.assert_allclose_up_to_global_phase(kraus[0], unitary, atol=1e-8)


def test_protocols():
    a = random.random()
    b = random.random()
    c = random.random()
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=c)
    cirq.testing.assert_implements_consistent_protocols(g)

    # Symbolic.
    t = sympy.Symbol('t')
    g = cirq.PhasedXZGate(x_exponent=t, z_exponent=b, axis_phase_exponent=c)
    cirq.testing.assert_implements_consistent_protocols(g)
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=t, axis_phase_exponent=c)
    cirq.testing.assert_implements_consistent_protocols(g)
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=t)
    cirq.testing.assert_implements_consistent_protocols(g)


def test_inverse():
    a = random.random()
    b = random.random()
    c = random.random()
    q = cirq.LineQubit(0)
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=c).on(q)

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(g ** -1), np.transpose(np.conjugate(cirq.unitary(g))), atol=1e-8
    )


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterized(resolve_fn):
    a = random.random()
    b = random.random()
    c = random.random()
    g = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=c)
    assert not cirq.is_parameterized(g)

    t = sympy.Symbol('t')
    gt = cirq.PhasedXZGate(x_exponent=t, z_exponent=b, axis_phase_exponent=c)
    assert cirq.is_parameterized(gt)
    assert resolve_fn(gt, {'t': a}) == g
    gt = cirq.PhasedXZGate(x_exponent=a, z_exponent=t, axis_phase_exponent=c)
    assert cirq.is_parameterized(gt)
    assert resolve_fn(gt, {'t': b}) == g
    gt = cirq.PhasedXZGate(x_exponent=a, z_exponent=b, axis_phase_exponent=t)
    assert cirq.is_parameterized(gt)
    assert resolve_fn(gt, {'t': c}) == g


def test_str_diagram():
    g = cirq.PhasedXZGate(x_exponent=0.5, z_exponent=0.25, axis_phase_exponent=0.125)

    assert str(g) == "PhXZ(a=0.125,x=0.5,z=0.25)"

    cirq.testing.assert_has_diagram(
        cirq.Circuit(g.on(cirq.LineQubit(0))),
        """
0: ───PhXZ(a=0.125,x=0.5,z=0.25)───
    """,
    )
