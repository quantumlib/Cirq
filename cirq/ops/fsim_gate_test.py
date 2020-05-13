# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import sympy

import cirq


def test_fsim_init():
    f = cirq.FSimGate(1, 2)
    assert f.theta == 1
    assert f.phi == 2

    f2 = cirq.FSimGate(theta=1, phi=2)
    assert f2.theta == 1
    assert f2.phi == 2


def test_fsim_eq():
    eq = cirq.testing.EqualsTester()
    a, b = cirq.LineQubit.range(2)

    eq.add_equality_group(cirq.FSimGate(1, 2), cirq.FSimGate(1, 2))
    eq.add_equality_group(cirq.FSimGate(2, 1))
    eq.add_equality_group(cirq.FSimGate(0, 0))
    eq.add_equality_group(cirq.FSimGate(1, 1))
    eq.add_equality_group(
        cirq.FSimGate(1, 2).on(a, b),
        cirq.FSimGate(1, 2).on(b, a))


def test_fsim_approx_eq():
    assert cirq.approx_eq(cirq.FSimGate(1, 2),
                          cirq.FSimGate(1.00001, 2.00001),
                          atol=0.01)


@pytest.mark.parametrize('theta, phi', [
    (0, 0),
    (np.pi / 3, np.pi / 5),
    (-np.pi / 3, np.pi / 5),
    (np.pi / 3, -np.pi / 5),
    (-np.pi / 3, -np.pi / 5),
    (np.pi / 2, 0.5),
])
def test_fsim_consistent(theta, phi):
    gate = cirq.FSimGate(theta=theta, phi=phi)
    cirq.testing.assert_implements_consistent_protocols(gate)


def test_fsim_circuit():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.FSimGate(np.pi / 2, np.pi).on(a, b),
        cirq.FSimGate(-np.pi, np.pi / 2).on(a, b),
    )
    cirq.testing.assert_has_diagram(
        c, """
0: ───FSim(0.5π, π)───FSim(-π, 0.5π)───
      │               │
1: ───FSim(0.5π, π)───FSim(-π, 0.5π)───
    """)
    cirq.testing.assert_has_diagram(c,
                                    """
0: ---FSim(0.5pi, pi)---FSim(-pi, 0.5pi)---
      |                 |
1: ---FSim(0.5pi, pi)---FSim(-pi, 0.5pi)---
        """,
                                    use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c,
                                    """
0: ---FSim(1.5707963267948966, pi)---FSim(-pi, 1.5707963267948966)---
      |                              |
1: ---FSim(1.5707963267948966, pi)---FSim(-pi, 1.5707963267948966)---
""",
                                    use_unicode_characters=False,
                                    precision=None)
    c = cirq.Circuit(
        cirq.FSimGate(sympy.Symbol('a') + sympy.Symbol('b'), 0).on(a, b))
    cirq.testing.assert_has_diagram(
        c, """
0: ───FSim(a + b, 0)───
      │
1: ───FSim(a + b, 0)───
    """)


def test_resolve():
    f = cirq.FSimGate(sympy.Symbol('a'), sympy.Symbol('b'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'a': 2})
    assert f == cirq.FSimGate(2, sympy.Symbol('b'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'b': 1})
    assert f == cirq.FSimGate(2, 1)
    assert not cirq.is_parameterized(f)


def test_fsim_unitary():
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )

    # Theta
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=np.pi / 2, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=-np.pi / 2, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=np.pi, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=2*np.pi, phi=0)),
        cirq.unitary(cirq.FSimGate(theta=0, phi=0)),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=-np.pi/2, phi=0)),
        cirq.unitary(cirq.FSimGate(theta=3/2*np.pi, phi=0)),
        atol=1e-8)

    # Phi
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=np.pi / 2)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1j],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=-np.pi / 2)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1j],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=np.pi)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=0)),
        cirq.unitary(cirq.FSimGate(theta=0, phi=2*np.pi)),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=0, phi=-np.pi/2)),
        cirq.unitary(cirq.FSimGate(theta=0, phi=3/2*np.pi)),
        atol=1e-8)

    # Both.
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.FSimGate(theta=np.pi / 4, phi=np.pi / 3)),
        np.array([
            [1, 0, 0, 0],
            [0, s, -1j * s, 0],
            [0, -1j * s, s, 0],
            [0, 0, 0, 0.5 - 1j * np.sqrt(0.75)],
        ]),
        atol=1e-8,
    )


@pytest.mark.parametrize('theta, phi', (
    (0, 0),
    (0.1, 0.1),
    (-0.1, 0.1),
    (0.1, -0.1),
    (-0.1, -0.1),
    (np.pi / 2, np.pi / 6),
    (np.pi, np.pi),
    (3.5 * np.pi, 4 * np.pi),
))
def test_fsim_iswap_cphase(theta, phi):
    q0, q1 = cirq.NamedQubit('q0'), cirq.NamedQubit('q1')
    iswap = cirq.ISWAP**(-theta * 2 / np.pi)
    cphase = cirq.CZPowGate(exponent=-phi / np.pi)
    iswap_cphase = cirq.Circuit((iswap.on(q0, q1), cphase.on(q0, q1)))
    fsim = cirq.FSimGate(theta=theta, phi=phi)
    assert np.allclose(cirq.unitary(iswap_cphase), cirq.unitary(fsim))


def test_repr():
    f = cirq.FSimGate(sympy.Symbol('a'), sympy.Symbol('b'))
    cirq.testing.assert_equivalent_repr(f)


def test_fsim_json_dict():
    assert cirq.FSimGate(theta=0.123, phi=0.456)._json_dict_() == {
        'cirq_type': 'FSimGate',
        'theta': 0.123,
        'phi': 0.456,
    }
