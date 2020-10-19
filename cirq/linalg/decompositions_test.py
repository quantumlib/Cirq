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
from cirq import value
from cirq import unitary_eig

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


def _random_unitary_with_close_eigenvalues():
    U = cirq.testing.random_unitary(4)
    d = np.diag(np.exp([-0.2312j, -0.2312j, -0.2332j, -0.2322j]))
    return U @ d @ U.conj().T


@pytest.mark.parametrize(
    'matrix',
    [
        X,
        np.eye(4),
        np.diag(
            np.exp([-1j * np.pi * 1.23, -1j * np.pi * 1.23, -1j * np.pi * 1.23
                   ])),

        # a global phase with a tiny perturbation
        np.diag(np.exp([-0.2312j, -0.2312j, -0.2312j, -0.2312j])) +
        np.random.random((4, 4)) * 1e-100,

        # also after a similarity transformation, demonstrating
        # that the effect is due to close eigenvalues, not diagonality
        _random_unitary_with_close_eigenvalues(),
    ])
def test_unitary_eig(matrix):
    # np.linalg.eig(matrix) won't work for the perturbed matrix
    d, vecs = unitary_eig(matrix)

    # test both unitarity and correctness of decomposition
    np.testing.assert_allclose(matrix,
                               vecs @ np.diag(d) @ vecs.conj().T,
                               atol=1e-14)


def test_non_unitary_eig():
    with pytest.raises(Exception):
        unitary_eig(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 0, 1, 2], [3, 4, 5, 6]]))


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
] + [(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2))
     for _ in range(10)])
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


def test_kak_vector_empty():
    assert len(cirq.kak_vector([])) == 0


def test_kak_plot_empty():
    cirq.scatter_plot_normalized_kak_interaction_coefficients([])


@pytest.mark.parametrize('target', [
    np.eye(4),
    SWAP,
    SWAP * 1j,
    CZ,
    CNOT,
    SWAP @ CZ,
] + [cirq.testing.random_unitary(4) for _ in range(10)])
def test_kak_decomposition(target):
    kak = cirq.kak_decomposition(target)
    np.testing.assert_allclose(cirq.unitary(kak), target, atol=1e-8)


def test_kak_decomposition_unitary_object():
    op = cirq.ISWAP(*cirq.LineQubit.range(2))**0.5
    kak = cirq.kak_decomposition(op)
    np.testing.assert_allclose(cirq.unitary(kak), cirq.unitary(op), atol=1e-8)
    assert cirq.kak_decomposition(kak) is kak


def test_kak_decomposition_invalid_object():
    with pytest.raises(TypeError, match='unitary effect'):
        _ = cirq.kak_decomposition('test')

    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.eye(3))

    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.eye(8))

    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.ones((4, 4)))

    with pytest.raises(ValueError, match='4x4 unitary matrix'):
        _ = cirq.kak_decomposition(np.zeros((4, 4)))

    nil = cirq.kak_decomposition(np.zeros((4, 4)), check_preconditions=False)
    np.testing.assert_allclose(cirq.unitary(nil), np.eye(4), atol=1e-8)


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

    eq.add_equality_group(
        cirq.KakDecomposition(
            global_phase=1,
            single_qubit_operations_before=(np.eye(2), np.eye(2)),
            interaction_coefficients=(0.3, 0.2, 0.1),
            single_qubit_operations_after=(np.eye(2), np.eye(2)),
        ),
        cirq.KakDecomposition(interaction_coefficients=(0.3, 0.2, 0.1)),
    )

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

    assert repr(
        cirq.KakDecomposition(
            global_phase=1,
            single_qubit_operations_before=(cirq.unitary(cirq.X),
                                            cirq.unitary(cirq.Y)),
            interaction_coefficients=(0.5, 0.25, 0),
            single_qubit_operations_after=(np.eye(2), cirq.unitary(cirq.Z)),
        )) == """
cirq.KakDecomposition(
    interaction_coefficients=(0.5, 0.25, 0),
    single_qubit_operations_before=(
        np.array([[0j, (1+0j)], [(1+0j), 0j]], dtype=np.complex128),
        np.array([[0j, -1j], [1j, 0j]], dtype=np.complex128),
    ),
    single_qubit_operations_after=(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[(1+0j), 0j], [0j, (-1+0j)]], dtype=np.complex128),
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
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.ry(1e-10))),
                          cirq.AxisAngleDecomposition(angle=0,
                                                      axis=(1, 0, 0),
                                                      global_phase=1),
                          atol=1e-8)
    assert cirq.approx_eq(cirq.axis_angle(cirq.unitary(cirq.rx(np.pi))),
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


def test_scatter_plot_normalized_kak_interaction_coefficients():
    a, b = cirq.LineQubit.range(2)
    data = [
        cirq.kak_decomposition(cirq.unitary(cirq.CZ)),
        cirq.unitary(cirq.CZ),
        cirq.CZ,
        cirq.Circuit(cirq.H(a), cirq.CNOT(a, b)),
    ]
    ax = cirq.scatter_plot_normalized_kak_interaction_coefficients(data)
    assert ax is not None
    ax2 = cirq.scatter_plot_normalized_kak_interaction_coefficients(
        data, s=1, c='blue', ax=ax, include_frame=False, label=f'test')
    assert ax2 is ax


def _vector_kron(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Vectorized implementation of kron for square matrices."""
    s_0, s_1 = first.shape[-2:], second.shape[-2:]
    assert s_0[0] == s_0[1]
    assert s_1[0] == s_1[1]
    out = np.einsum('...ab,...cd->...acbd', first, second)
    s_v = out.shape[:-4]
    return out.reshape(s_v + (s_0[0] * s_1[0],) * 2)


def _local_two_qubit_unitaries(samples, random_state):
    kl_0 = np.array([
        cirq.testing.random_unitary(2, random_state=random_state)
        for _ in range(samples)
    ])
    kl_1 = np.array([
        cirq.testing.random_unitary(2, random_state=random_state)
        for _ in range(samples)
    ])

    return _vector_kron(kl_0, kl_1)


_kak_gens = np.array([np.kron(X, X), np.kron(Y, Y), np.kron(Z, Z)])


def _random_two_qubit_unitaries(num_samples: int,
                                random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE'):
    # Randomly generated two-qubit unitaries and the KAK vectors (not canonical)
    kl = _local_two_qubit_unitaries(num_samples, random_state)

    kr = _local_two_qubit_unitaries(num_samples, random_state)

    prng = value.parse_random_state(random_state)
    # Generate the non-local part by explict matrix exponentiation.
    kak_vecs = prng.rand(num_samples, 3) * np.pi
    gens = np.einsum('...a,abc->...bc', kak_vecs, _kak_gens)
    evals, evecs = np.linalg.eigh(gens)
    A = np.einsum('...ab,...b,...cb', evecs, np.exp(1j * evals), evecs.conj())

    return np.einsum('...ab,...bc,...cd', kl, A, kr), kak_vecs


def _local_invariants_from_kak(vector: np.ndarray) -> np.ndarray:
    r"""Local invariants of a two-qubit unitary from its KAK vector.

    Any 2 qubit unitary may be expressed as

    $U = k_l A k_r$
    where $k_l, k_r$ are single qubit (local) unitaries and

    $$
    A = \exp( i * \sum_{j=x,y,z} k_j \sigma_{(j,0)}\sigma_{(j,1)})
    $$

    Here $(k_x,k_y,k_z)$ is the KAK vector.

    Args:
        vector: Shape (...,3) tensor representing different KAK vectors.

    Returns:
        The local invariants associated with the given KAK vector. Shape
        (..., 3), where first two elements are the real and imaginary parts
        of G1 and the third is G2.

    References:
        "A geometric theory of non-local two-qubit operations"
        https://arxiv.org/abs/quant-ph/0209120
    """
    vector = np.asarray(vector)
    # See equation 30 in the above reference. Compared to their notation, the k
    # vector equals c/2.
    kx = vector[..., 0]
    ky = vector[..., 1]
    kz = vector[..., 2]
    cos, sin = np.cos, np.sin
    G1R = (cos(2 * kx) * cos(2 * ky) * cos(2 * kz))**2
    G1R -= (sin(2 * kx) * sin(2 * ky) * sin(2 * kz))**2

    G1I = 0.25 * sin(4 * kx) * sin(4 * ky) * sin(4 * kz)

    G2 = cos(4 * kx) + cos(4 * ky) + cos(4 * kz)
    return np.moveaxis(np.array([G1R, G1I, G2]), 0, -1)


_random_unitaries, _kak_vecs = _random_two_qubit_unitaries(100, random_state=11)


def test_kak_vector_matches_vectorized():
    actual = cirq.kak_vector(_random_unitaries)
    expected = np.array([cirq.kak_vector(u) for u in _random_unitaries])
    np.testing.assert_almost_equal(actual, expected)


def test_KAK_vector_local_invariants_random_input():
    actual = _local_invariants_from_kak(cirq.kak_vector(_random_unitaries))
    expected = _local_invariants_from_kak(_kak_vecs)

    np.testing.assert_almost_equal(actual, expected)


def test_kak_vector_on_weyl_chamber_face():
    # unitaries with KAK vectors from I to ISWAP
    theta_swap = np.linspace(0, np.pi / 4, 10)
    k_vecs = np.zeros((10, 3))
    k_vecs[:, (0, 1)] = theta_swap[:, np.newaxis]

    kwargs = dict(global_phase=1j,
                  single_qubit_operations_before=(X, Y),
                  single_qubit_operations_after=(Z, 1j * X))
    unitaries = np.array([
        cirq.unitary(
            cirq.KakDecomposition(interaction_coefficients=(t, t, 0), **kwargs))
        for t in theta_swap
    ])

    actual = cirq.kak_vector(unitaries)
    np.testing.assert_almost_equal(actual, k_vecs)


@pytest.mark.parametrize('unitary,expected',
                         ((np.eye(4), (0, 0, 0)), (SWAP, [np.pi / 4] * 3),
                          (SWAP * 1j, [np.pi / 4] * 3),
                          (CNOT, [np.pi / 4, 0, 0]), (CZ, [np.pi / 4, 0, 0]),
                          (CZ @ SWAP, [np.pi / 4, np.pi / 4, 0]),
                          (np.kron(X, X), (0, 0, 0))))
def test_KAK_vector_weyl_chamber_vertices(unitary, expected):
    actual = cirq.kak_vector(unitary)
    np.testing.assert_almost_equal(actual, expected)


cases = [np.eye(3), SWAP.reshape((2, 8)), SWAP.ravel()]


@pytest.mark.parametrize('bad_input', cases)
def test_kak_vector_wrong_matrix_shape(bad_input):
    with pytest.raises(ValueError, match='to have shape'):
        cirq.kak_vector(bad_input)


def test_kak_vector_negative_atol():
    with pytest.raises(ValueError, match='must be positive'):
        cirq.kak_vector(np.eye(4), atol=-1.0)


def test_kak_vector_input_not_unitary():
    with pytest.raises(ValueError, match='must correspond to'):
        cirq.kak_vector(np.zeros((4, 4)))


@pytest.mark.parametrize('unitary', [
    cirq.testing.random_unitary(4),
    cirq.unitary(cirq.IdentityGate(2)),
    cirq.unitary(cirq.SWAP),
    cirq.unitary(cirq.SWAP**0.25),
    cirq.unitary(cirq.ISWAP),
    cirq.unitary(cirq.CZ**0.5),
    cirq.unitary(cirq.CZ),
])
def test_kak_decompose(unitary: np.ndarray):
    kak = cirq.kak_decomposition(unitary)
    circuit = cirq.Circuit(kak._decompose_(cirq.LineQubit.range(2)))
    np.testing.assert_allclose(cirq.unitary(circuit), unitary, atol=1e-8)
    assert len(circuit) == 5
    assert len(list(circuit.all_operations())) == 8


def test_num_two_qubit_gates_required():
    for i in range(4):
        assert cirq.num_cnots_required(
            _two_qubit_circuit_with_cnots(i).unitary()) == i

    assert cirq.num_cnots_required(np.eye(4)) == 0


def test_num_two_qubit_gates_required_invalid():
    with pytest.raises(ValueError, match="(4,4)"):
        cirq.num_cnots_required(np.array([[1]]))


def _two_qubit_circuit_with_cnots(num_cnots=3, a=None, b=None):
    random.seed(32123)
    if a is None or b is None:
        a, b = cirq.LineQubit.range(2)

    def random_one_qubit_gate():
        return cirq.PhasedXPowGate(phase_exponent=random.random(),
                                   exponent=random.random())

    def one_cz():
        return [
            cirq.CZ.on(a, b),
            random_one_qubit_gate().on(a),
            random_one_qubit_gate().on(b),
        ]

    return cirq.Circuit([
        random_one_qubit_gate().on(a),
        random_one_qubit_gate().on(b), [one_cz() for _ in range(num_cnots)]
    ])
