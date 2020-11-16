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


def test_fsim_resolve():
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


def test_fsim_repr():
    f = cirq.FSimGate(sympy.Symbol('a'), sympy.Symbol('b'))
    cirq.testing.assert_equivalent_repr(f)


def test_fsim_json_dict():
    assert cirq.FSimGate(theta=0.123, phi=0.456)._json_dict_() == {
        'cirq_type': 'FSimGate',
        'theta': 0.123,
        'phi': 0.456,
    }


def test_phased_fsim_init():
    f = cirq.PhasedFSimGate(1, 2, 3, 4, 5)
    assert f.theta == 1
    assert f.phi == 2
    assert f.delta_plus == 3
    assert f.delta_minus == 4 - 2 * np.pi
    assert f.delta_minus_off_diagonal == 5 - 2 * np.pi

    f2 = cirq.PhasedFSimGate(theta=1,
                             phi=2,
                             delta_plus=3,
                             delta_minus=4,
                             delta_minus_off_diagonal=5)
    assert f2.theta == 1
    assert f2.phi == 2
    assert f.delta_plus == 3
    assert f.delta_minus == 4 - 2 * np.pi
    assert f.delta_minus_off_diagonal == 5 - 2 * np.pi


@pytest.mark.parametrize('theta, phi, phase_angles_before, phase_angles_after',
                         (
                             (0, 0, (0, 0), (0, 0)),
                             (1, 2, (3, 4), (5, 7)),
                             (np.pi / 5, np.pi / 6, (0.1, 0.2), (0.3, 0.5)),
                         ))
def test_phased_fsim_from_phase_angles_and_fsim(theta, phi, phase_angles_before,
                                                phase_angles_after):
    f = cirq.PhasedFSimGate.from_phase_angles_and_fsim(theta, phi,
                                                       phase_angles_before,
                                                       phase_angles_after)
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.rz(phase_angles_before[0]).on(q0),
        cirq.rz(phase_angles_before[1]).on(q1),
        cirq.FSimGate(theta, phi).on(q0, q1),
        cirq.rz(phase_angles_after[0]).on(q0),
        cirq.rz(phase_angles_after[1]).on(q1))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(f),
                                                    cirq.unitary(c),
                                                    atol=1e-8)


@pytest.mark.parametrize('phase_angles_before, phase_angles_after', (
    ((0, 0), (0, 0)),
    ((0.1, 0.2), (0.3, 0.7)),
    ((0.1, 0.2), (0.3, -0.8)),
    ((-0.1, -0.2), (-0.3, -0.9)),
    ((np.pi / 5, -np.pi / 3), (0, np.pi / 2)),
))
def test_phased_fsim_recreate_from_phase_angles(phase_angles_before,
                                                phase_angles_after):
    f = cirq.PhasedFSimGate.from_phase_angles_and_fsim(np.pi / 3, np.pi / 5,
                                                       phase_angles_before,
                                                       phase_angles_after)
    f2 = cirq.PhasedFSimGate.from_phase_angles_and_fsim(f.theta, f.phi,
                                                        f.phase_angles_before,
                                                        f.phase_angles_after)
    assert cirq.approx_eq(f, f2)


@pytest.mark.parametrize('phase_angles_before, phase_angles_after', (
    ((0, 0), (0, 0)),
    ((0.1, 0.2), (0.3, 0.7)),
    ((-0.1, 0.2), (0.3, 0.8)),
    ((-0.1, -0.2), (0.3, -0.9)),
    ((np.pi, np.pi / 6), (-np.pi / 2, 0)),
))
def test_phased_fsim_phase_angle_symmetry(phase_angles_before,
                                          phase_angles_after):
    f = cirq.PhasedFSimGate.from_phase_angles_and_fsim(np.pi / 3, np.pi / 5,
                                                       phase_angles_before,
                                                       phase_angles_after)
    for d in (-10, -7, -2 * np.pi, -0.2, 0, 0.1, 0.2, np.pi, 8, 20):
        phase_angles_before2 = (phase_angles_before[0] + d,
                                phase_angles_before[1] + d)
        phase_angles_after2 = (phase_angles_after[0] - d,
                               phase_angles_after[1] - d)
        f2 = cirq.PhasedFSimGate.from_phase_angles_and_fsim(
            np.pi / 3, np.pi / 5, phase_angles_before2, phase_angles_after2)
        assert cirq.approx_eq(f, f2)


def test_phased_fsim_eq():
    eq = cirq.testing.EqualsTester()
    a, b = cirq.LineQubit.range(2)

    eq.add_equality_group(cirq.PhasedFSimGate(1, 2, 3, 4, 5),
                          cirq.PhasedFSimGate(1, 2, 3, 4, 5))
    eq.add_equality_group(cirq.PhasedFSimGate(2, 1, 3, 4, 5))
    eq.add_equality_group(cirq.PhasedFSimGate(0, 0, 0, 0, 0))
    eq.add_equality_group(cirq.PhasedFSimGate(1, 0, 0, 0, 0))
    eq.add_equality_group(cirq.PhasedFSimGate(0, 1, 0, 0, 0))
    eq.add_equality_group(cirq.PhasedFSimGate(0, 0, 1, 0, 0))
    eq.add_equality_group(cirq.PhasedFSimGate(0, 0, 0, 1, 0))
    eq.add_equality_group(cirq.PhasedFSimGate(0, 0, 0, 0, 1))
    eq.add_equality_group(cirq.PhasedFSimGate(1, 1, 0, 0, 0))
    eq.add_equality_group(
        cirq.PhasedFSimGate(1, 2, 3, 4, 5).on(a, b),
        cirq.PhasedFSimGate(1, 2, 3, 4, 5).on(b, a))


def test_phased_fsim_approx_eq():
    assert cirq.approx_eq(cirq.PhasedFSimGate(1, 2, 3, 4, 5),
                          cirq.PhasedFSimGate(1.00001, 2.00001, 3.00001,
                                              4.00004, 5.00005),
                          atol=0.01)


@pytest.mark.parametrize(
    'theta, phi, delta_plus, delta_minus, delta_minus_off_diagonal', [
        (0, 0, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
        (np.pi / 3, np.pi / 5, 0, 0, 0),
        (np.pi / 3, np.pi / 5, 1, 0, 0),
        (np.pi / 3, np.pi / 5, 0, 1, 0),
        (np.pi / 3, np.pi / 5, 0, 0, 1),
        (-np.pi / 3, np.pi / 5, 1, 0, 0),
        (np.pi / 3, -np.pi / 5, 0, 1, 0),
        (-np.pi / 3, -np.pi / 5, 0, 0, 1),
        (np.pi, 0, 0, 0, sympy.Symbol('a')),
    ])
def test_phased_fsim_consistent(theta, phi, delta_plus, delta_minus,
                                delta_minus_off_diagonal):
    gate = cirq.PhasedFSimGate(
        theta=theta,
        phi=phi,
        delta_plus=delta_plus,
        delta_minus=delta_minus,
        delta_minus_off_diagonal=delta_minus_off_diagonal)
    cirq.testing.assert_implements_consistent_protocols(gate)


def test_phased_fsim_circuit():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.PhasedFSimGate(np.pi / 2, np.pi, np.pi / 2, 0,
                            -np.pi / 4).on(a, b),
        cirq.PhasedFSimGate(-np.pi, np.pi / 2, np.pi / 10, np.pi / 5,
                            3 * np.pi / 10).on(a, b),
    )
    cirq.testing.assert_has_diagram(
        c, """
0: ───PhFSim(0.5π, π, 0.5π, 0, -0.25π)───PhFSim(-π, 0.5π, 0.1π, 0.2π, 0.3π)───
      │                                  │
1: ───PhFSim(0.5π, π, 0.5π, 0, -0.25π)───PhFSim(-π, 0.5π, 0.1π, 0.2π, 0.3π)───
    """)
    # pylint: disable=line-too-long
    cirq.testing.assert_has_diagram(c,
                                    """
0: ---PhFSim(0.5pi, pi, 0.5pi, 0, -0.25pi)---PhFSim(-pi, 0.5pi, 0.1pi, 0.2pi, 0.3pi)---
      |                                      |
1: ---PhFSim(0.5pi, pi, 0.5pi, 0, -0.25pi)---PhFSim(-pi, 0.5pi, 0.1pi, 0.2pi, 0.3pi)---
        """,
                                    use_unicode_characters=False)
    cirq.testing.assert_has_diagram(c,
                                    """
0: ---PhFSim(1.5707963267948966, pi, 1.5707963267948966, 0, -0.7853981633974483)---PhFSim(-pi, 1.5707963267948966, 0.3141592653589793, 0.6283185307179586, 0.9424777960769379)---
      |                                                                            |
1: ---PhFSim(1.5707963267948966, pi, 1.5707963267948966, 0, -0.7853981633974483)---PhFSim(-pi, 1.5707963267948966, 0.3141592653589793, 0.6283185307179586, 0.9424777960769379)---
""",
                                    use_unicode_characters=False,
                                    precision=None)
    # pylint: enable=line-too-long
    c = cirq.Circuit(
        cirq.PhasedFSimGate(
            sympy.Symbol('a') + sympy.Symbol('b'), 0, sympy.Symbol('c'),
            sympy.Symbol('d'),
            sympy.Symbol('a') - sympy.Symbol('b')).on(a, b))
    cirq.testing.assert_has_diagram(
        c, """
0: ───PhFSim(a + b, 0, c, d, a - b)───
      │
1: ───PhFSim(a + b, 0, c, d, a - b)───
    """)


def test_phased_fsim_resolve():
    f = cirq.PhasedFSimGate(sympy.Symbol('a'), sympy.Symbol('b'),
                            sympy.Symbol('c'), sympy.Symbol('d'),
                            sympy.Symbol('e'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'a': 1})
    assert f == cirq.PhasedFSimGate(1, sympy.Symbol('b'), sympy.Symbol('c'),
                                    sympy.Symbol('d'), sympy.Symbol('e'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'b': 2})
    assert f == cirq.PhasedFSimGate(1, 2, sympy.Symbol('c'), sympy.Symbol('d'),
                                    sympy.Symbol('e'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'c': 3})
    assert f == cirq.PhasedFSimGate(1, 2, 3, sympy.Symbol('d'),
                                    sympy.Symbol('e'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'d': 4})
    assert f == cirq.PhasedFSimGate(1, 2, 3, 4, sympy.Symbol('e'))
    assert cirq.is_parameterized(f)

    f = cirq.resolve_parameters(f, {'e': 5})
    assert f == cirq.PhasedFSimGate(1, 2, 3, 4, 5)
    assert not cirq.is_parameterized(f)


def test_phased_fsim_unitary():
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=0)),
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
        cirq.unitary(cirq.PhasedFSimGate(theta=np.pi / 2, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, 0, -1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=-np.pi / 2, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=np.pi, phi=0)),
        np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(
        cirq.PhasedFSimGate(theta=2 * np.pi, phi=0)),
                               cirq.unitary(cirq.PhasedFSimGate(theta=0,
                                                                phi=0)),
                               atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=-np.pi / 2, phi=0)),
        cirq.unitary(cirq.PhasedFSimGate(theta=3 / 2 * np.pi, phi=0)),
        atol=1e-8)

    # Phi
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=np.pi / 2)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1j],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=-np.pi / 2)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1j],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=np.pi)),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(cirq.PhasedFSimGate(theta=0,
                                                                phi=0)),
                               cirq.unitary(
                                   cirq.PhasedFSimGate(theta=0, phi=2 * np.pi)),
                               atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=-np.pi / 2)),
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=3 / 2 * np.pi)),
        atol=1e-8)

    # Theta and phi
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=np.pi / 4, phi=np.pi / 3)),
        np.array([
            [1, 0, 0, 0],
            [0, s, -1j * s, 0],
            [0, -1j * s, s, 0],
            [0, 0, 0, 0.5 - 1j * np.sqrt(0.75)],
        ]),
        atol=1e-8,
    )

    # Deltas
    w6 = np.exp(1j * np.pi / 6)
    np.testing.assert_allclose(
        cirq.unitary(
            cirq.PhasedFSimGate(theta=0,
                                phi=0,
                                delta_plus=np.pi / 2,
                                delta_minus=np.pi / 3)),
        np.array([
            [1, 0, 0, 0],
            [0, -w6.conjugate(), 0, 0],
            [0, 0, w6, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=0, phi=0)),
        cirq.unitary(
            cirq.PhasedFSimGate(theta=0, phi=0, delta_minus_off_diagonal=0.2)),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(cirq.PhasedFSimGate(theta=np.pi, phi=0)),
        cirq.unitary(
            cirq.PhasedFSimGate(theta=np.pi,
                                phi=0,
                                delta_minus_off_diagonal=0.2)),
        atol=1e-8)
    np.testing.assert_allclose(
        cirq.unitary(
            cirq.PhasedFSimGate(theta=-np.pi / 2,
                                phi=0,
                                delta_plus=np.pi / 2,
                                delta_minus_off_diagonal=np.pi / 3)),
        np.array([
            [1, 0, 0, 0],
            [0, 0, 1j * w6, 0],
            [0, 1j * -w6.conjugate(), 0, 0],
            [0, 0, 0, -1],
        ]),
        atol=1e-8,
    )
    np.testing.assert_allclose(cirq.unitary(
        cirq.PhasedFSimGate(theta=np.pi / 2, phi=0)),
                               cirq.unitary(
                                   cirq.PhasedFSimGate(theta=np.pi / 2,
                                                       phi=0,
                                                       delta_minus=0.2)),
                               atol=1e-8)
    np.testing.assert_allclose(cirq.unitary(
        cirq.PhasedFSimGate(theta=-np.pi / 2, phi=0)),
                               cirq.unitary(
                                   cirq.PhasedFSimGate(theta=-np.pi / 2,
                                                       phi=0,
                                                       delta_minus=0.2)),
                               atol=1e-8)


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
def test_phased_fsim_vs_fsim(theta, phi):
    g1 = cirq.FSimGate(theta, phi)
    g2 = cirq.PhasedFSimGate(theta, phi)
    assert np.allclose(cirq.unitary(g1), cirq.unitary(g2))


def test_phased_fsim_repr():
    f = cirq.PhasedFSimGate(sympy.Symbol('a'), sympy.Symbol('b'),
                            sympy.Symbol('c'), sympy.Symbol('d'),
                            sympy.Symbol('e'))
    cirq.testing.assert_equivalent_repr(f)


def test_phased_fsim_json_dict():
    assert cirq.PhasedFSimGate(theta=0.12,
                               phi=0.34,
                               delta_plus=0.56,
                               delta_minus=0.78,
                               delta_minus_off_diagonal=0.9)._json_dict_() == {
                                   'cirq_type': 'PhasedFSimGate',
                                   'theta': 0.12,
                                   'phi': 0.34,
                                   'delta_plus': 0.56,
                                   'delta_minus': 0.78,
                                   'delta_minus_off_diagonal': 0.9,
                               }
