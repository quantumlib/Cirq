# Copyright 2018 The Cirq Developers
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

from cirq.ops.two_qubit_noise_channels import *

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.eye(2)
XI = np.kron(X, I)
XX = np.kron(X, X)
XY = np.kron(X, Y)
XZ = np.kron(X, Z)
YI = np.kron(Y, I)
YX = np.kron(Y, X)
YY = np.kron(Y, Y)
YZ = np.kron(Y, Z)
ZI = np.kron(Z, I)
ZX = np.kron(Z, X)
ZY = np.kron(Z, Y)
ZZ = np.kron(Z, Z)
IX = np.kron(I, X)
IY = np.kron(I, Y)
IZ = np.kron(I, Z)

round_to_6_prec = cirq.CircuitDiagramInfoArgs(known_qubits=None,
                                              known_qubit_count=None,
                                              use_unicode_characters=True,
                                              precision=6,
                                              qubit_map=None)

round_to_3_prec = cirq.CircuitDiagramInfoArgs(known_qubits=None,
                                              known_qubit_count=None,
                                              use_unicode_characters=True,
                                              precision=3,
                                              qubit_map=None)


def assert_mixtures_equal(actual, expected):
    """Assert equal for tuple of mixed scalar and array types."""
    for a, e in zip(actual, expected):
        np.testing.assert_almost_equal(a[0], e[0])
        np.testing.assert_almost_equal(a[1], e[1])


def test_two_qubit_asymmetric_depolarizing_channel():
    d = two_qubit_asymmetric_depolarize(0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                        0.07, 0.08, 0.09, 0.10, 0.001, 0.002,
                                        0.003, 0.004, 0.005)
    np.testing.assert_almost_equal(
        cirq.channel(d),
        (np.sqrt(0.435) * np.eye(4), np.sqrt(0.01) * XI, np.sqrt(0.02) * XX,
         np.sqrt(0.03) * XY, np.sqrt(0.04) * XZ, np.sqrt(0.05) * YI,
         np.sqrt(0.06) * YX, np.sqrt(0.07) * YY, np.sqrt(0.08) * YZ,
         np.sqrt(0.09) * ZI, np.sqrt(0.10) * ZX, np.sqrt(0.001) * ZY,
         np.sqrt(0.002) * ZZ, np.sqrt(0.003) * IX, np.sqrt(0.004) * IY,
         np.sqrt(0.005) * IZ))
    assert cirq.has_channel(d)


def test_two_qubit_asymmetric_depolarizing_mixture():
    d = two_qubit_asymmetric_depolarize(0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                        0.07, 0.08, 0.09, 0.10, 0.001, 0.002,
                                        0.003, 0.004, 0.005)
    assert_mixtures_equal(
        cirq.mixture(d),
        ((0.435 * np.eye(4), 0.01 * XI, 0.02 * XX, 0.03 * XY, 0.04 * XZ,
          0.05 * YI, 0.06 * YX, 0.07 * YY, 0.08 * YZ, 0.09 * ZI, 0.10 * ZX,
          0.001 * ZY, 0.002 * ZZ, 0.003 * IX, 0.004 * IY, 0.005 * IZ)))
    assert cirq.has_mixture_channel(d)


def test_two_qubit_asymmetric_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(
        TwoQubitAsymmetricDepolarizingChannel(0.01, 0.02, 0.03, 0.04, 0.05,
                                              0.06, 0.07, 0.08, 0.09, 0.10,
                                              0.001, 0.002, 0.003, 0.004,
                                              0.005))


def test_two_qubit_asymmetric_depolarizing_channel_str():
    assert (
        str(
            two_qubit_asymmetric_depolarize(0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                            0.07, 0.08, 0.09, 0.10, 0.001,
                                            0.002, 0.003, 0.004, 0.005)) ==
        'two_qubit_asymmetric_depolarize(p_xi=0.01,p_xx=0.02,p_xy=0.03, p_xz=0.04'
        'p_yi=0.05,p_yx=0.06,p_yy=0.07, p_yz=0.08'
        'p_zi=0.09,p_xx=0.10,p_zy=0.001, p_zz=0.002'
        'p_ix=0.003,p_iy=0.004,p_iz=0.005)')


permutations = [
    (0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
]


def test_two_qubit_asymmetric_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    c = two_qubit_asymmetric_depolarize(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    et.make_equality_group(lambda: c)
    [
        et.add_equality_group(
            two_qubit_asymmetric_depolarize(*p) for p in permutations)
    ]
    et.add_equality_group(
        two_qubit_asymmetric_depolarize(0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                        0.07, 0.08, 0.09, 0.10, 0.001, 0.002,
                                        0.003, 0.004, 0.005))
    et.add_equality_group(
        two_qubit_asymmetric_depolarize(0.005, 0.004, 0.003, 0.002, 0.001, 0.1,
                                        0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
                                        0.03, 0.02, 0.01))


neg_perm = []
for tup in permutations:
    neg_perm.append(tuple(-x for x in tup))


@pytest.mark.parametrize('p_xi,p_xx,p_xy,p_xz,'
                         'p_yi,p_yx,p_yy,p_yz,'
                         'p_zi,p_zx,p_zy,p_zz,'
                         'p_ix,p_iy,p_iz',
                         (*neg_perm,
                          (0.01, -0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                           0.09, 0.10, 0.001, 0.002, 0.003, 0.004, 0.005)))
def test_two_qubit_asymmetric_depolarizing_channel_negative_probability(
    p_xi, p_xx, p_xy, p_xz, p_yi, p_yx, p_yy, p_yz, p_zi, p_zx, p_zy, p_zz,
    p_ix, p_iy, p_iz):
    with pytest.raises(ValueError, match='was less than 0'):
        two_qubit_asymmetric_depolarize(p_xi, p_xx, p_xy, p_xz, p_yi, p_yx,
                                        p_yy, p_yz, p_zi, p_zx, p_zy, p_zz,
                                        p_ix, p_iy, p_iz)


gr_1_perm = []
for tup in permutations:
    gr_1_perm.append(tuple(x + 1 if x > 0 else x for x in tup))


@pytest.mark.parametrize('p_xi,p_xx,p_xy,p_xz,'
                         'p_yi,p_yx,p_yy,p_yz,'
                         'p_zi,p_zx,p_zy,p_zz,'
                         'p_ix,p_iy,p_iz',
                         (*gr_1_perm,
                          (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                           0.10, 0.10, 0.20, 0.30, 0.40, 0.50)))
def test_two_qubit_asymmetric_depolarizing_channel_bigly_probability(
    p_xi, p_xx, p_xy, p_xz, p_yi, p_yx, p_yy, p_yz, p_zi, p_zx, p_zy, p_zz,
    p_ix, p_iy, p_iz):
    with pytest.raises(ValueError, match='was greater than 1'):
        two_qubit_asymmetric_depolarize(p_xi, p_xx, p_xy, p_xz, p_yi, p_yx,
                                        p_yy, p_yz, p_zi, p_zx, p_zy, p_zz,
                                        p_ix, p_iy, p_iz)


def test_two_qubit_asymmetric_depolarizing_channel_text_diagram():
    a = two_qubit_asymmetric_depolarize(0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
                                        0.07, 0.08, 0.09, 0.10, 0.001, 0.002,
                                        0.003, 0.0, 0.005)
    assert (cirq.circuit_diagram_info(
        a, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('A(0.00111111,0.00222222,0.00333333'
                          '0.00444444,0.00555555,0.00666666'
                          '0.00777777,0.00888888,0.00999999'
                          '0.01111111,0.00011111,0.00022222'
                          '0.00033333,0.00044444,0.00055555)',)))
    assert (cirq.circuit_diagram_info(
        a, args=round_to_3_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('A(0.001,0.002,0.003'
                          '0.004,0.005,0.006'
                          '0.007,0.008,0.009'
                          '0.011,0.000,0.000'
                          '0.000,0.000,0.000)',)))


def test_two_qubit_depolarizing_channel():
    d = cirq.depolarize(0.015)
    np.testing.assert_almost_equal(
        cirq.channel(d),
        (np.sqrt(0.85) * np.eye(4), np.sqrt(0.01) * XI, np.sqrt(0.01) * XX,
         np.sqrt(0.01) * XY, np.sqrt(0.01) * XZ, np.sqrt(0.01) * YI,
         np.sqrt(0.01) * YX, np.sqrt(0.01) * YY, np.sqrt(0.01) * YZ,
         np.sqrt(0.01) * ZI, np.sqrt(0.01) * ZX, np.sqrt(0.01) * ZY,
         np.sqrt(0.01) * ZZ, np.sqrt(0.01) * IX, np.sqrt(0.01) * IY,
         np.sqrt(0.01) * IZ))
    assert cirq.has_channel(d)


def test_two_qubit_depolarizing_mixture():
    d = cirq.depolarize(0.3)
    assert_mixtures_equal(
        cirq.mixture(d),
        (0.85 * np.eye(4), 0.01 * XI, 0.01 * XX, 0.01 * XY, 0.01 * XZ,
         0.01 * YI, 0.01 * YX, 0.01 * YY, 0.01 * YZ, 0.01 * ZI, 0.01 * ZX,
         0.01 * ZY, 0.01 * ZZ, 0.01 * IX, 0.01 * IY, 0.01 * IZ))
    assert cirq.has_mixture_channel(d)


def test_two_qubit_depolarizing_channel_repr():
    cirq.testing.assert_equivalent_repr(TwoQubitDepolarizingChannel(0.3))


def test_two_qubit_depolarizing_channel_str():
    assert str(cirq.depolarize(0.3)) == 'depolarize(p=0.3)'


def test_two_qubit_depolarizing_channel_eq():
    et = cirq.testing.EqualsTester()
    c = cirq.depolarize(0.0)
    et.make_equality_group(lambda: c)
    et.add_equality_group(two_qubit_depolarize(0.1))
    et.add_equality_group(two_qubit_depolarize(0.9))
    et.add_equality_group(two_qubit_depolarize(1.0))


def test_two_qubit_depolarizing_channel_invalid_probability():
    with pytest.raises(ValueError, match='was less than 0'):
        two_qubit_depolarize(-0.1)
    with pytest.raises(ValueError, match='was greater than 1'):
        two_qubit_depolarize(1.1)


def test_two_qubit_depolarizing_channel_text_diagram():
    d = two_qubit_depolarize(0.1234567)
    assert (cirq.circuit_diagram_info(
        d, args=round_to_6_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('D(0.123457)',)))
    assert (cirq.circuit_diagram_info(
        d, args=round_to_3_prec) == cirq.CircuitDiagramInfo(
            wire_symbols=('D(0.12)',)))


@pytest.mark.parametrize('eigen_gate_type', [
    XXGate,
    XYGate,
    XZGate,
    YXGate,
    YYGate,
    YZGate,
    ZXGate,
    ZYGate,
    ZZGate,
    IXGate,
    IYGate,
    IZGate,
])
def test_phase_insensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
        eigen_gate_type)


def test_xi_init():
    assert XIGate(exponent=0.5).exponent == 0.5
    assert XIGate(exponent=5).exponent == 5
    assert (XIGate()**0.5).exponent == 0.5


def test_xi_str():
    assert str(XIGate()) == 'XI'
    assert str(XIGate()**0.5) == 'XI**0.5'
    assert str(XIGate()**-0.25) == 'XI**-0.25'


def test_xi_repr():
    assert repr(XIGate()) == 'cirq.XI'
    assert repr(XIGate()**0.5) == '(cirq.XI**0.5)'
    assert repr(XIGate()**-0.25) == '(cirq.XI**-0.25)'


def test_xi_unitary():
    assert np.allclose(
        cirq.unitary(XIGate()),
        np.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [1., 0., 0., 0.],
                  [0., 1., 0., 0.]]))

    assert np.allclose(
        cirq.unitary(XIGate()**0.5),
        np.array([[0.5 + 0.5j, 0, 0.5 - 0.5j,
                   0], [0, 0.5 + 0.5j, 0, 0.5 - 0.5j],
                  [0.5 - 0.5j, 0, 0.5 + 0.5j, 0],
                  [0, 0.5 - 0.5j, 0, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XIGate()**-0.5),
        np.array([[0.5 - 0.5j, 0, 0.5 + 0.5j,
                   0], [0, 0.5 - 0.5j, 0, 0.5 + 0.5j],
                  [0.5 + 0.5j, 0, 0.5 - 0.5j, 0],
                  [0, 0.5 + 0.5j, 0, 0.5 - 0.5j]]))


def test_xx_init():
    assert XXGate(exponent=0.5).exponent == 0.5
    assert XXGate(exponent=5).exponent == 5
    assert (XXGate()**0.5).exponent == 0.5


def test_xx_str():
    assert str(XXGate()) == 'XX'
    assert str(XXGate()**0.5) == 'XX**0.5'
    assert str(XXGate()**-0.25) == 'XX**-0.25'


def test_xx_repr():
    assert repr(XXGate()) == 'cirq.XX'
    assert repr(XXGate()**0.5) == '(cirq.XX**0.5)'
    assert repr(XXGate()**-0.25) == '(cirq.XX**-0.25)'


def test_xx_unitary():
    assert np.allclose(
        cirq.unitary(XXGate()),
        np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]))

    assert np.allclose(
        cirq.unitary(XXGate()**0.5),
        np.array([[0.5 + 1j * 0.5, 0, 0, 0.5 + 1j * -0.5],
                  [0, 0.5 + 1j * 0.5, 0.5 + 1j * -0.5, 0],
                  [0, 0.5 + 1j * -0.5, 0.5 + 1j * 0.5, 0],
                  [0.5 + 1j * -0.5, 0, 0, 0.5 + 1j * 0.5]]))

    assert np.allclose(
        cirq.unitary(XXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XXGate()**-0.5),
        np.array([[0.5 + 1j * -0.5, 0, 0, 0.5 + 1j * 0.5],
                  [0, 0.5 + 1j * -0.5, 0.5 + 1j * 0.5, 0],
                  [0, 0.5 + 1j * 0.5, 0.5 + 1j * -0.5, 0],
                  [0.5 + 1j * 0.5, 0, 0, 0.5 + 1j * -0.5]]))


def test_xy_init():
    assert XYGate(exponent=0.5).exponent == 0.5
    assert XYGate(exponent=5).exponent == 5
    assert (XYGate()**0.5).exponent == 0.5


def test_xy_str():
    assert str(XYGate()) == 'XY'
    assert str(XYGate()**0.5) == 'XY**0.5'
    assert str(XYGate()**-0.25) == 'XY**-0.25'


def test_xy_repr():
    assert repr(XYGate()) == 'cirq.XY'
    assert repr(XYGate()**0.5) == '(cirq.XY**0.5)'
    assert repr(XYGate()**-0.25) == '(cirq.XY**-0.25)'


def test_xy_unitary():
    assert np.allclose(
        cirq.unitary(XYGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(XYGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, -0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_xz_init():
    assert XZGate(exponent=0.5).exponent == 0.5
    assert XZGate(exponent=5).exponent == 5
    assert (XZGate()**0.5).exponent == 0.5


def test_xz_str():
    assert str(XZGate()) == 'XZ'
    assert str(XZGate()**0.5) == 'XZ**0.5'
    assert str(XZGate()**-0.25) == 'XZ**-0.25'


def test_xz_repr():
    assert repr(XZGate()) == 'cirq.XZ'
    assert repr(XZGate()**0.5) == '(cirq.XZ**0.5)'
    assert repr(XZGate()**-0.25) == '(cirq.XZ**-0.25)'


def test_xz_unitary():
    assert np.allclose(
        cirq.unitary(XZGate()),
        np.array([[0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(XZGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, -0.5 + 0.5j],
                  [0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(XZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(XZGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, -0.5 - 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yi_init():
    assert YIGate(exponent=0.5).exponent == 0.5
    assert YIGate(exponent=5).exponent == 5
    assert (YIGate()**0.5).exponent == 0.5


def test_yi_str():
    assert str(YIGate()) == 'YI'
    assert str(YIGate()**0.5) == 'YI**0.5'
    assert str(YIGate()**-0.25) == 'YI**-0.25'


def test_yi_repr():
    assert repr(YIGate()) == 'cirq.YI'
    assert repr(YIGate()**0.5) == '(cirq.YI**0.5)'
    assert repr(YIGate()**-0.25) == '(cirq.YI**-0.25)'


def test_yi_unitary():
    assert np.allclose(
        cirq.unitary(YIGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YIGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YIGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j],
                  [-0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yx_init():
    assert YXGate(exponent=0.5).exponent == 0.5
    assert YXGate(exponent=5).exponent == 5
    assert (YXGate()**0.5).exponent == 0.5


def test_yx_str():
    assert str(YXGate()) == 'YX'
    assert str(YXGate()**0.5) == 'YX**0.5'
    assert str(YXGate()**-0.25) == 'YX**-0.25'


def test_yx_repr():
    assert repr(YXGate()) == 'cirq.YX'
    assert repr(YXGate()**0.5) == '(cirq.YX**0.5)'
    assert repr(YXGate()**-0.25) == '(cirq.YX**-0.25)'


def test_yx_unitary():
    assert np.allclose(
        cirq.unitary(YXGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YXGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yy_init():
    assert YYGate(exponent=0.5).exponent == 0.5
    assert YYGate(exponent=5).exponent == 5
    assert (YYGate()**0.5).exponent == 0.5


def test_yy_str():
    assert str(YYGate()) == 'YY'
    assert str(YYGate()**0.5) == 'YY**0.5'
    assert str(YYGate()**-0.25) == 'YY**-0.25'


def test_yy_repr():
    assert repr(YYGate()) == 'cirq.YY'
    assert repr(YYGate()**0.5) == '(cirq.YY**0.5)'
    assert repr(YYGate()**-0.25) == '(cirq.YY**-0.25)'


def test_yy_unitary():
    assert np.allclose(
        cirq.unitary(YYGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [-1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YYGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, 0. + 0.j, -0.5 + 0.5j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [-0.5 + 0.5j, 0. + 0.j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0. + 0.j, -0.5 - 0.5j],
                  [0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j],
                  [-0.5 - 0.5j, 0. + 0.j, 0. + 0.j, 0.5 - 0.5j]]))


def test_yz_init():
    assert YZGate(exponent=0.5).exponent == 0.5
    assert YZGate(exponent=5).exponent == 5
    assert (YZGate()**0.5).exponent == 0.5


def test_yz_str():
    assert str(YZGate()) == 'YZ'
    assert str(YZGate()**0.5) == 'YZ**0.5'
    assert str(YZGate()**-0.25) == 'YZ**-0.25'


def test_yz_repr():
    assert repr(YZGate()) == 'cirq.YZ'
    assert repr(YZGate()**0.5) == '(cirq.YZ**0.5)'
    assert repr(YZGate()**-0.25) == '(cirq.YZ**-0.25)'


def test_yz_unitary():
    assert np.allclose(
        cirq.unitary(YZGate()),
        np.array([[0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(YZGate()**0.5),
        np.array([[0.5 + 0.5j, 0. + 0.j, -0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j],
                  [0.5 + 0.5j, 0. + 0.j, 0.5 + 0.5j, 0. + 0.j],
                  [0. + 0.j, -0.5 - 0.5j, 0. + 0.j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(YZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(YZGate()**-0.5),
        np.array([[0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, -0.5 + 0.5j],
                  [-0.5 + 0.5j, 0. + 0.j, 0.5 - 0.5j, 0. + 0.j],
                  [0. + 0.j, 0.5 - 0.5j, 0. + 0.j, 0.5 - 0.5j]]))


def test_zi_init():
    assert ZIGate(exponent=0.5).exponent == 0.5
    assert ZIGate(exponent=5).exponent == 5
    assert (ZIGate()**0.5).exponent == 0.5


def test_zi_str():
    assert str(ZIGate()) == 'ZI'
    assert str(ZIGate()**0.5) == 'ZI**0.5'
    assert str(ZIGate()**-0.25) == 'ZI**-0.25'


def test_zi_repr():
    assert repr(ZIGate()) == 'cirq.ZI'
    assert repr(ZIGate()**0.5) == '(cirq.ZI**0.5)'
    assert repr(ZIGate()**-0.25) == '(cirq.ZI**-0.25)'


def test_zi_unitary():
    assert np.allclose(
        cirq.unitary(ZIGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZIGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]]))


def test_zx_init():
    assert ZXGate(exponent=0.5).exponent == 0.5
    assert ZXGate(exponent=5).exponent == 5
    assert (ZXGate()**0.5).exponent == 0.5


def test_zx_str():
    assert str(ZXGate()) == 'ZX'
    assert str(ZXGate()**0.5) == 'ZX**0.5'
    assert str(ZXGate()**-0.25) == 'ZX**-0.25'


def test_zx_repr():
    assert repr(ZXGate()) == 'cirq.ZX'
    assert repr(ZXGate()**0.5) == '(cirq.ZX**0.5)'
    assert repr(ZXGate()**-0.25) == '(cirq.ZX**-0.25)'


def test_zx_unitary():
    assert np.allclose(
        cirq.unitary(ZXGate()),
        np.array([[0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**0.5),
        np.array([[0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, -0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 + 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, -0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 - 0.5j, 0.5 - 0.5j]]))


def test_zy_init():
    assert ZYGate(exponent=0.5).exponent == 0.5
    assert ZYGate(exponent=5).exponent == 5
    assert (ZYGate()**0.5).exponent == 0.5


def test_zy_str():
    assert str(ZYGate()) == 'ZY'
    assert str(ZYGate()**0.5) == 'ZY**0.5'
    assert str(ZYGate()**-0.25) == 'ZY**-0.25'


def test_zy_repr():
    assert repr(ZYGate()) == 'cirq.ZY'
    assert repr(ZYGate()**0.5) == '(cirq.ZY**0.5)'
    assert repr(ZYGate()**-0.25) == '(cirq.ZY**-0.25)'


def test_zy_unitary():
    assert np.allclose(
        cirq.unitary(ZYGate()),
        np.array([[0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**0.5),
        np.array([[0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 - 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [-0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, -0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j]]))


def test_zz_init():
    assert ZZGate(exponent=0.5).exponent == 0.5
    assert ZZGate(exponent=5).exponent == 5
    assert (ZZGate()**0.5).exponent == 0.5


def test_zz_str():
    assert str(ZZGate()) == 'ZZ'
    assert str(ZZGate()**0.5) == 'ZZ**0.5'
    assert str(ZZGate()**-0.25) == 'ZZ**-0.25'


def test_zz_repr():
    assert repr(ZZGate()) == 'cirq.ZZ'
    assert repr(ZZGate()**0.5) == '(cirq.ZZ**0.5)'
    assert repr(ZZGate()**-0.25) == '(cirq.ZZ**-0.25)'


def test_zz_unitary():
    assert np.allclose(
        cirq.unitary(ZZGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, -1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(ZZGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. - 1.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]]))


def test_ix_init():
    assert IXGate(exponent=0.5).exponent == 0.5
    assert IXGate(exponent=5).exponent == 5
    assert (IXGate()**0.5).exponent == 0.5


def test_ix_str():
    assert str(IXGate()) == 'IX'
    assert str(IXGate()**0.5) == 'IX**0.5'
    assert str(IXGate()**-0.25) == 'IX**-0.25'


def test_ix_repr():
    assert repr(IXGate()) == 'cirq.IX'
    assert repr(IXGate()**0.5) == '(cirq.IX**0.5)'
    assert repr(IXGate()**-0.25) == '(cirq.IX**-0.25)'


def test_ix_unitary():
    assert np.allclose(
        cirq.unitary(IXGate()),
        np.array([[0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IXGate()**0.5),
        np.array([[0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(IXGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IXGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 + 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 - 0.5j]]))


def test_iy_init():
    assert IYGate(exponent=0.5).exponent == 0.5
    assert IYGate(exponent=5).exponent == 5
    assert (IYGate()**0.5).exponent == 0.5


def test_iy_str():
    assert str(IYGate()) == 'IY'
    assert str(IYGate()**0.5) == 'IY**0.5'
    assert str(IYGate()**-0.25) == 'IY**-0.25'


def test_iy_repr():
    assert repr(IYGate()) == 'cirq.IY'
    assert repr(IYGate()**0.5) == '(cirq.IY**0.5)'
    assert repr(IYGate()**-0.25) == '(cirq.IY**-0.25)'


def test_iy_unitary():
    assert np.allclose(
        cirq.unitary(IYGate()),
        np.array([[0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 1.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j],
                  [0. + 0.j, 0. + 0.j, 0. + 1.j, 0. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IYGate()**0.5),
        np.array([[0.5 + 0.5j, -0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0.5 + 0.5j, 0.5 + 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, -0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, 0.5 + 0.5j, 0.5 + 0.5j]]))

    assert np.allclose(
        cirq.unitary(IYGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IYGate()**-0.5),
        np.array([[0.5 - 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [-0.5 + 0.5j, 0.5 - 0.5j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0.5 - 0.5j, 0.5 - 0.5j],
                  [0. + 0.j, 0. + 0.j, -0.5 + 0.5j, 0.5 - 0.5j]]))


def test_iz_init():
    assert IZGate(exponent=0.5).exponent == 0.5
    assert IZGate(exponent=5).exponent == 5
    assert (IZGate()**0.5).exponent == 0.5


def test_iz_str():
    assert str(IZGate()) == 'IZ'
    assert str(IZGate()**0.5) == 'IZ**0.5'
    assert str(IZGate()**-0.25) == 'IZ**-0.25'


def test_iz_repr():
    assert repr(IZGate()) == 'cirq.IZ'
    assert repr(IZGate()**0.5) == '(cirq.IZ**0.5)'
    assert repr(IZGate()**-0.25) == '(cirq.IZ**-0.25)'


def test_iz_unitary():
    assert np.allclose(
        cirq.unitary(IZGate()),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, -1. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, -1. + 0.j]]))

    assert np.allclose(
        cirq.unitary(IZGate()**0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 1.j]]))

    assert np.allclose(
        cirq.unitary(IZGate()**0),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))

    assert np.allclose(
        cirq.unitary(IZGate()**-0.5),
        np.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. - 1.j, 0. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                  [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. - 1.j]]))
