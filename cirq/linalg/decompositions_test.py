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

import random

import numpy as np
import pytest

import cirq

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
SQRT_X = np.array([[1, 1j], [1j, 1]])
c = np.exp(1j * np.pi / 4)
SQRT_SQRT_X = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
SWAP = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])


def assert_kronecker_factorization_within_tolerance(matrix, g, f1, f2):
    restored = g * cirq.linalg.combinators.kron(f1, f2)
    assert not np.any(np.isnan(restored)), "NaN in kronecker product."
    assert np.allclose(restored, matrix), "Can't factor kronecker product."


def assert_kronecker_factorization_not_within_tolerance(matrix, g, f1, f2):
    restored = g * cirq.linalg.combinators.kron(f1, f2)
    assert (np.any(np.isnan(restored) or
                   not np.allclose(restored, matrix)))

def assert_magic_su2_within_tolerance(mat, a, b):
    M = cirq.linalg.decompositions.MAGIC
    MT = cirq.linalg.decompositions.MAGIC_CONJ_T
    recon = cirq.linalg.combinators.dot(
        MT,
        cirq.linalg.combinators.kron(a, b),
        M)
    assert np.allclose(recon, mat), "Failed to decompose within tolerance."

@pytest.mark.parametrize('matrix', [
    X,
    cirq.kron(X, X),
    cirq.kron(X, Y),
    cirq.kron(X, np.eye(2))
])
def test_map_eigenvalues_identity(matrix):
    identity_mapped = cirq.map_eigenvalues(matrix, lambda e: e)
    assert np.allclose(matrix, identity_mapped)


@pytest.mark.parametrize('matrix,exponent,desired', [
    [X, 2, np.eye(2)],
    [X, 3, X],
    [Z, 2, np.eye(2)],
    [H, 2, np.eye(2)],
    [Z, 0.5, np.diag([1, 1j])],
    [X, 0.5, np.array([[1j, 1], [1, 1j]]) * (1 - 1j) / 2],
])
def test_map_eigenvalues_raise(matrix, exponent, desired):
    exp_mapped = cirq.map_eigenvalues(matrix, lambda e: complex(e)**exponent)
    assert np.allclose(desired, exp_mapped)


@pytest.mark.parametrize('f1,f2', [
    (H, X),
    (H * 1j, X),
    (H, SQRT_X),
    (H, SQRT_SQRT_X),
    (H, H),
    (SQRT_SQRT_X, H),
    (X, np.eye(2)),
    (1j * X, np.eye(2)),
    (X, 1j * np.eye(2)),
    (-X, 1j * np.eye(2)),
    (X, X),
] + [
    (cirq.testing.random_unitary(2), cirq.testing.random_unitary(2))
    for _ in range(10)
])
def test_kron_factor(f1, f2):
    p = cirq.kron(f1, f2)
    g, g1, g2 = cirq.kron_factor_4x4_to_2x2s(p)
    assert abs(np.linalg.det(g1) - 1) < 0.00001
    assert abs(np.linalg.det(g2) - 1) < 0.00001
    assert np.allclose(g * cirq.kron(g1, g2), p)
    assert_kronecker_factorization_within_tolerance(
        p, g, g1, g2)


@pytest.mark.parametrize('f1,f2', [
    (cirq.testing.random_special_unitary(2),
     cirq.testing.random_special_unitary(2))
    for _ in range(10)
])
def test_kron_factor_special_unitaries(f1, f2):
    p = cirq.kron(f1, f2)
    g, g1, g2 = cirq.kron_factor_4x4_to_2x2s(p)
    assert np.allclose(cirq.kron(g1, g2), p)
    assert abs(g - 1) < 0.000001
    assert cirq.is_special_unitary(g1)
    assert cirq.is_special_unitary(g2)
    assert_kronecker_factorization_within_tolerance(
        p, g, g1, g2)


def test_kron_factor_fail():
    mat = cirq.kron_with_controls(cirq.CONTROL_TAG, X)
    g, f1, f2 = cirq.kron_factor_4x4_to_2x2s(mat)
    with pytest.raises(ValueError):
        assert_kronecker_factorization_not_within_tolerance(
            mat, g, f1, f2)
    mat = cirq.kron_factor_4x4_to_2x2s(np.diag([1, 1, 1, 1j]))
    with pytest.raises(ValueError):
        assert_kronecker_factorization_not_within_tolerance(
            mat, g, f1, f2)


def recompose_so4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert cirq.is_special_unitary(a)
    assert cirq.is_special_unitary(b)

    magic = np.array([[1, 0, 0, 1j],
                      [0, 1j, 1, 0],
                      [0, 1j, -1, 0],
                      [1, 0, 0, -1j]]) * np.sqrt(0.5)
    result = np.real(cirq.dot(np.conj(magic.T),
                              cirq.kron(a, b),
                              magic))
    assert cirq.is_orthogonal(result)
    return result


@pytest.mark.parametrize('m', [
    cirq.testing.random_special_orthogonal(4)
    for _ in range(10)
])
def test_so4_to_magic_su2s(m):
    a, b = cirq.so4_to_magic_su2s(m)
    m2 = recompose_so4(a, b)
    assert_magic_su2_within_tolerance(m2, a, b)
    assert np.allclose(m, m2)


@pytest.mark.parametrize('a,b', [
    (cirq.testing.random_special_unitary(2),
     cirq.testing.random_special_unitary(2))
    for _ in range(10)
])
def test_so4_to_magic_su2s_known_factors(a, b):
    m = recompose_so4(a, b)
    a2, b2 = cirq.so4_to_magic_su2s(m)
    m2 = recompose_so4(a2, b2)

    assert np.allclose(m2, m)

    # Account for kron(A, B) = kron(-A, -B).
    if np.linalg.norm(a + a2) > np.linalg.norm(a - a2):
        assert np.allclose(a2, a)
        assert np.allclose(b2, b)
    else:
        assert np.allclose(a2, -a)
        assert np.allclose(b2, -b)


@pytest.mark.parametrize('mat', [
    np.diag([0, 1, 1, 1]),
    np.diag([0.5, 2, 1, 1]),
    np.diag([1, 1j, 1, 1]),
    np.diag([1, 1, 1, -1]),
])
def test_so4_to_magic_su2s_fail(mat):
    with pytest.raises(ValueError):
        _ = cirq.so4_to_magic_su2s(mat)


@pytest.mark.parametrize('x,y,z', [
    [(random.random() * 2 - 1) * np.pi * 2 for _ in range(3)]
    for _ in range(10)
])
def test_kak_canonicalize_vector(x, y, z):
    i = np.eye(2)
    m = cirq.unitary(cirq.KakDecomposition(
        global_phase=1,
        single_qubit_operations_after=(i, i),
        interaction_coefficients=(x, y, z),
        single_qubit_operations_before=(i, i)))

    kak = cirq.kak_canonicalize_vector(x, y, z, atol=1e-10)
    a1, a0 = kak.single_qubit_operations_after
    x2, y2, z2 = kak.interaction_coefficients
    b1, b0 = kak.single_qubit_operations_before
    m2 = cirq.unitary(kak)

    assert 0.0 <= x2 <= np.pi / 4
    assert 0.0 <= y2 <= np.pi / 4
    assert -np.pi / 4 < z2 <= np.pi / 4
    assert abs(x2) >= abs(y2) >= abs(z2)
    assert x2 < np.pi / 4 - 1e-10 or z2 >= 0
    assert cirq.is_special_unitary(a1)
    assert cirq.is_special_unitary(a0)
    assert cirq.is_special_unitary(b1)
    assert cirq.is_special_unitary(b0)
    assert np.allclose(m, m2)


@pytest.mark.parametrize('target', [
    np.eye(4),
    SWAP,
    SWAP * 1j,
    CZ,
    CNOT,
    SWAP.dot(CZ),
] + [
    cirq.testing.random_unitary(4)
    for _ in range(10)
])
def test_kak_decomposition(target):
    kak = cirq.kak_decomposition(target)
    np.testing.assert_allclose(cirq.unitary(kak), target, atol=1e-8)


def test_kak_decomposition_eq():
    eq = cirq.testing.EqualsTester()

    eq.make_equality_group(lambda: cirq.KakDecomposition(
        global_phase=1,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.Y)),
        interaction_coefficients=(0.3, 0.2, 0.1),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    ))

    eq.add_equality_group(cirq.KakDecomposition(
        global_phase=-1,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.Y)),
        interaction_coefficients=(0.3, 0.2, 0.1),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    ))

    eq.make_equality_group(lambda: cirq.KakDecomposition(
        global_phase=1,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.H)),
        interaction_coefficients=(0.3, 0.2, 0.1),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    ))

    eq.make_equality_group(lambda: cirq.KakDecomposition(
        global_phase=1,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.Y)),
        interaction_coefficients=(0.5, 0.2, 0.1),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    ))


def test_kak_repr():
    cirq.testing.assert_equivalent_repr(cirq.KakDecomposition(
        global_phase=1j,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.Y)),
        interaction_coefficients=(0.3, 0.2, 0.1),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    ))

    assert repr(cirq.KakDecomposition(
        global_phase=1,
        single_qubit_operations_before=(cirq.unitary(cirq.X),
                                        cirq.unitary(cirq.Y)),
        interaction_coefficients=(0.5, 0.25, 0),
        single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
    )) == """
cirq.KakDecomposition(
    interaction_coefficients=(0.5, 0.25, 0),
    single_qubit_operations_before=(
        np.array([[0j, (1+0j)], [(1+0j), 0j]]),
        np.array([[0j, -1j], [1j, 0j]]),
    ),
    single_qubit_operations_after=(
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[(1+0j), 0j], [0j, (-1+0j)]]),
    ),
    global_phase=1)
""".strip()


def test_kak_str():
    v = cirq.KakDecomposition(
        interaction_coefficients=(0.3 * np.pi / 4, 0.2 * np.pi / 4,
                                  0.1 * np.pi / 4),
        single_qubit_operations_before=(cirq.unitary(cirq.I),
                                        cirq.unitary(cirq.X)),
        single_qubit_operations_after=(cirq.unitary(cirq.Y),
                                       cirq.unitary(cirq.Z)),
        global_phase=1j)
    print(v)
    assert str(v) == """KAK {
    xyz*(4/π): 0.3, 0.2, 0.1
    before: (0*π around X) ⊗ (1*π around X)
    after: (1*π around Y) ⊗ (1*π around Z)
}"""


def test_axis_angle_decomposition_eq():
    eq = cirq.testing.EqualsTester()

    eq.make_equality_group(lambda: cirq.AxisAngleDecomposition(
        angle=1, axis=(0.8, 0.6, 0), global_phase=-1))
    eq.add_equality_group(
        cirq.AxisAngleDecomposition(angle=5,
                                    axis=(0.8, 0.6, 0),
                                    global_phase=-1))
    eq.add_equality_group(
        cirq.AxisAngleDecomposition(angle=1,
                                    axis=(0.8, 0, 0.6),
                                    global_phase=-1))
    eq.add_equality_group(
        cirq.AxisAngleDecomposition(angle=1, axis=(0.8, 0.6, 0),
                                    global_phase=1))


def test_axis_angle_decomposition_repr():
    cirq.testing.assert_equivalent_repr(
        cirq.AxisAngleDecomposition(angle=1,
                                    axis=(0, 0.6, 0.8),
                                    global_phase=-1))


def test_axis_angle_decomposition_str():
    assert str(cirq.axis_angle(cirq.unitary(cirq.X))) == '1*π around X'
    assert str(cirq.axis_angle(cirq.unitary(cirq.Y))) == '1*π around Y'
    assert str(cirq.axis_angle(cirq.unitary(cirq.Z))) == '1*π around Z'
    assert str(cirq.axis_angle(cirq.unitary(
        cirq.H))) == '1*π around 0.707*X+0.707*Z'
    assert str(cirq.axis_angle(cirq.unitary(
        cirq.H**0.5))) == '0.5*π around 0.707*X+0.707*Z'
    assert str(
        cirq.axis_angle(
            cirq.unitary(cirq.X**0.25) @ cirq.unitary(cirq.Y**0.25)
            @ cirq.unitary(cirq.Z**
                           0.25))) == '0.477*π around 0.679*X+0.281*Y+0.679*Z'


def test_axis_angle_decomposition_unitary():
    u = cirq.testing.random_unitary(2)
    u = cirq.unitary(cirq.T)
    a = cirq.axis_angle(u)
    np.testing.assert_allclose(u, cirq.unitary(a), atol=1e-8)


def test_axis_angle():
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Ry(1e-10))),
                          cirq.AxisAngleDecomposition(angle=0,
                                                      axis=(1, 0, 0),
                                                      global_phase=1),
                          atol=1e-8)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Rx(np.pi))),
                          cirq.AxisAngleDecomposition(angle=np.pi,
                                                      axis=(1, 0, 0),
                                                      global_phase=1),
                          atol=1e-8)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.X)),
                          cirq.AxisAngleDecomposition(angle=np.pi,
                                                      axis=(1, 0, 0),
                                                      global_phase=1j),
                          atol=1e-8)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.X**0.5)),
                          cirq.AxisAngleDecomposition(angle=np.pi / 2,
                                                      axis=(1, 0, 0),
                                                      global_phase=np.exp(
                                                          1j * np.pi / 4)),
                          atol=1e-8)
    assert cirq.approx_eq(
        cirq.axis_angle(cirq.unitary(cirq.X**-0.5)),
        cirq.AxisAngleDecomposition(angle=-np.pi / 2,
                                    axis=(1, 0, 0),
                                    global_phase=np.exp(-1j * np.pi / 4)))

    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Y)),
                          cirq.AxisAngleDecomposition(angle=np.pi,
                                                      axis=(0, 1, 0),
                                                      global_phase=1j),
                          atol=1e-8)

    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.Z)),
                          cirq.AxisAngleDecomposition(angle=np.pi,
                                                      axis=(0, 0, 1),
                                                      global_phase=1j),
                          atol=1e-8)

    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.H)),
                          cirq.AxisAngleDecomposition(angle=np.pi,
                                                      axis=(np.sqrt(0.5), 0,
                                                            np.sqrt(0.5)),
                                                      global_phase=1j),
                          atol=1e-8)

    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.H**0.5)),
                          cirq.AxisAngleDecomposition(
                              angle=np.pi / 2,
                              axis=(np.sqrt(0.5), 0, np.sqrt(0.5)),
                              global_phase=np.exp(1j * np.pi / 4)),
                          atol=1e-8)


def test_axis_angle_canonicalize():
    a = cirq.AxisAngleDecomposition(angle=np.pi * 2.3,
                                    axis=(1, 0, 0),
                                    global_phase=1j).canonicalize()
    assert a.global_phase == -1j
    assert a.axis == (1, 0, 0)
    np.testing.assert_allclose(a.angle, np.pi * 0.3, atol=1e-8)

    a = cirq.AxisAngleDecomposition(angle=np.pi / 2,
                                    axis=(-1, 0, 0),
                                    global_phase=1j).canonicalize()
    assert a.global_phase == 1j
    assert a.axis == (1, 0, 0)
    assert a.angle == -np.pi / 2

    a = cirq.AxisAngleDecomposition(angle=np.pi + 0.01,
                                    axis=(1, 0, 0),
                                    global_phase=1j).canonicalize(atol=0.1)
    assert a.global_phase == 1j
    assert a.axis == (1, 0, 0)
    assert a.angle == np.pi + 0.01

    a = cirq.AxisAngleDecomposition(angle=np.pi + 0.01,
                                    axis=(1, 0, 0),
                                    global_phase=1j).canonicalize(atol=0.001)
    assert a.global_phase == -1j
    assert a.axis == (1, 0, 0)
    assert np.isclose(a.angle, -np.pi + 0.01)


def test_axis_angle_canonicalize_approx_equal():
    a1 = cirq.AxisAngleDecomposition(angle=np.pi,
                                     axis=(1, 0, 0),
                                     global_phase=1)
    a2 = cirq.AxisAngleDecomposition(angle=-np.pi,
                                     axis=(1, 0, 0),
                                     global_phase=-1)
    b1 = cirq.AxisAngleDecomposition(angle=np.pi,
                                     axis=(1, 0, 0),
                                     global_phase=-1)
    assert cirq.approx_eq(a1, a2, atol=1e-8)
    assert not cirq.approx_eq(a1, b1, atol=1e-8)


def test_axis_angle_init():
    a = cirq.AxisAngleDecomposition(angle=1, axis=(0, 1, 0), global_phase=1j)
    assert a.angle == 1
    assert a.axis == (0, 1, 0)
    assert a.global_phase == 1j

    with pytest.raises(ValueError, match='normalize'):
        cirq.AxisAngleDecomposition(angle=1, axis=(0, 0.5, 0), global_phase=1)
