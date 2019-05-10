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

import cirq


def test_reflection_matrix_pow_consistent_results():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = cirq.reflection_matrix_pow(x, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_x, sqrt_x), x, atol=1e-10)

    ix = x * np.sqrt(1j)
    sqrt_ix = cirq.reflection_matrix_pow(ix, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_ix, sqrt_ix), ix, atol=1e-10)

    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    cube_root_h = cirq.reflection_matrix_pow(h, 1/3)
    np.testing.assert_allclose(
        np.dot(np.dot(cube_root_h, cube_root_h), cube_root_h),
        h,
        atol=1e-8)

    y = np.array([[0, -1j], [1j, 0]])
    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5j)
    yh = np.kron(y, h)
    sqrt_yh = cirq.reflection_matrix_pow(yh, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_yh, sqrt_yh), yh, atol=1e-10)


def test_reflection_matrix_sign_preference_under_perturbation():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = np.array([[1, -1j], [-1j, 1]]) * (1 + 1j) / 2
    np.testing.assert_allclose(cirq.reflection_matrix_pow(x, 0.5),
                               sqrt_x,
                               atol=1e-8)

    # Sqrt should behave well when phased by less than 90 degrees.
    # (When rotating by more it's ambiguous. For example, 181 = 91*2 = -89*2.)
    for perturbation in [0, 0.1, -0.1, 0.3, -0.3, 0.49, -0.49]:
        px = x * complex(-1)**perturbation
        expected_sqrt_px = sqrt_x * complex(-1)**(perturbation / 2)
        sqrt_px = cirq.reflection_matrix_pow(px, 0.5)
        np.testing.assert_allclose(np.dot(sqrt_px, sqrt_px), px, atol=1e-10)
        np.testing.assert_allclose(sqrt_px, expected_sqrt_px, atol=1e-10)


def test_match_global_phase():
    a = np.array([[5, 4], [3, -2]])
    b = np.array([[0.000001, 0], [0, 1j]])
    c, d = cirq.match_global_phase(a, b)
    np.testing.assert_allclose(c, -a, atol=1e-10)
    np.testing.assert_allclose(d, b * -1j, atol=1e-10)


def test_match_global_phase_zeros():
    z = np.array([[0, 0], [0, 0]])
    b = np.array([[1, 1], [1, 1]])

    z1, b1 = cirq.match_global_phase(z, b)
    np.testing.assert_allclose(z, z1, atol=1e-10)
    np.testing.assert_allclose(b, b1, atol=1e-10)

    z2, b2 = cirq.match_global_phase(z, b)
    np.testing.assert_allclose(z, z2, atol=1e-10)
    np.testing.assert_allclose(b, b2, atol=1e-10)

    z3, z4 = cirq.match_global_phase(z, z)
    np.testing.assert_allclose(z, z3, atol=1e-10)
    np.testing.assert_allclose(z, z4, atol=1e-10)


def test_match_global_no_float_error_when_axis_aligned():
    a = np.array([[1, 1.1], [-1.3, np.pi]])
    a2, _ = cirq.match_global_phase(a, a)
    a3, _ = cirq.match_global_phase(a * 1j, a * 1j)
    a4, _ = cirq.match_global_phase(-a, -a)
    a5, _ = cirq.match_global_phase(a * -1j, a * -1j)

    assert np.all(a2 == a)
    assert np.all(a3 == a)
    assert np.all(a4 == a)
    assert np.all(a5 == a)


def test_targeted_left_multiply_matches_kron_then_dot():
    t = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    m = np.array([[2, 3], [5, 7]])
    i = np.eye(2)

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m,
                                    right_target=t.reshape((2, 2, 2)),
                                    target_axes=[0]),
        np.dot(cirq.kron(m, i, i), t).reshape((2, 2, 2)),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m,
                                    right_target=t.reshape((2, 2, 2)),
                                    target_axes=[1]),
        np.dot(cirq.kron(i, m, i), t).reshape((2, 2, 2)),
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m,
                                    right_target=t.reshape((2, 2, 2)),
                                    target_axes=[2]),
        np.dot(cirq.kron(i, i, m), t).reshape((2, 2, 2)),
        atol=1e-8)


def test_targeted_left_multiply_reorders_matrices():
    t = np.eye(4).reshape((2, 2, 2, 2))
    m = np.array([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 0, 1,
        0, 0, 1, 0,
    ]).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m,
                                    right_target=t,
                                    target_axes=[0, 1]),
        m,
        atol=1e-8)

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m,
                                    right_target=t,
                                    target_axes=[1, 0]),
        np.array([
            1, 0, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0,
            0, 1, 0, 0,
        ]).reshape((2, 2, 2, 2)),
        atol=1e-8)


def test_targeted_left_multiply_out():
    left = np.array([[2, 3], [5, 7]])
    right = np.array([1, -1])
    out = np.zeros(2)

    result = cirq.targeted_left_multiply(left_matrix=left,
                                         right_target=right,
                                         target_axes=[0],
                                         out=out)
    assert result is out
    np.testing.assert_allclose(
        result,
        np.array([-1, -2]),
        atol=1e-8)


def test_targeted_conjugate_simple():
    a = np.array([[0, 1j], [0, 0]])
    # yapf: disable
    b = np.reshape(
        np.array([1, 2, 3, 4,
                  5, 6, 7, 8,
                  9, 10, 11, 12,
                  13, 14, 15, 16]),
        (2,) * 4
    )
    expected = np.reshape(
        np.array([11, 12, 0, 0,
                  15, 16, 0, 0,
                  0, 0, 0, 0,
                  0, 0, 0, 0]),
        (2,) * 4
    )
    # yapf: enable
    result = cirq.targeted_conjugate_about(a, b, [0])
    # Should move lower right block to upper right.
    # Conjugation should result in multiplication by 1 (since 1j * -1j == 1).
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate():
    a = np.reshape([0, 1, 2j, 3j], (2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0])
    expected = np.einsum('ij,jklm,ln->iknm', a, b,
                         np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)

    result = cirq.targeted_conjugate_about(a, b, [1])
    expected = np.einsum('ij,kjlm,mn->kiln', a, b,
                         np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_multiple_indices():
    a = np.reshape(np.arange(16) + 1j, (2, 2, 2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0, 1])
    expected = np.einsum('ijkl,klmn,mnop->ijop', a, b,
                         np.transpose(np.conjugate(a), (2, 3, 0, 1)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_multiple_indices_flip_order():
    a = np.reshape(np.arange(16) + 1j, (2, 2, 2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [1, 0], [3, 2])
    expected = np.einsum('ijkl,lknm,mnop->jipo', a, b,
                         np.transpose(np.conjugate(a), (2, 3, 0, 1)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_out():
    a = np.reshape([0, 1, 2j, 3j], (2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    buffer = np.empty((2,) * 4, dtype=a.dtype)
    out = np.empty((2,) * 4, dtype=a.dtype)
    result = cirq.targeted_conjugate_about(a, b, [0], buffer=buffer, out=out)
    assert result is out
    expected = np.einsum('ij,jklm,ln->iknm', a, b,
                         np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)


def test_apply_matrix_to_slices():
    # Output is input.
    with pytest.raises(ValueError, match='out'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(
            target=target,
            matrix=np.eye(2),
            out=target,
            slices=[0, 1])

    # Wrong matrix size.
    with pytest.raises(ValueError, match='shape'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(
            target=target,
            matrix=np.eye(3),
            slices=[0, 1])

    # Empty case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.array(range(5)),
            matrix=np.eye(0),
            slices=[]),
        np.array(range(5)))

    # Middle 2x2 of 4x4 case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.eye(4),
            matrix=np.array([[2, 3], [5, 7]]),
            slices=[1, 2]),
        np.array([
            [1, 0, 0, 0],
            [0, 2, 3, 0],
            [0, 5, 7, 0],
            [0, 0, 0, 1]
        ]))

    # Middle 2x2 of 4x4 with opposite order case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.eye(4),
            matrix=np.array([[2, 3], [5, 7]]),
            slices=[2, 1]),
        np.array([
            [1, 0, 0, 0],
            [0, 7, 5, 0],
            [0, 3, 2, 0],
            [0, 0, 0, 1]
        ]))

    # Complicated slices of tensor case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.array(range(8)).reshape((2, 2, 2)),
            matrix=np.array([[0, 1], [1, 0]]),
            slices=[
                (0, slice(None), 0),
                (1, slice(None), 0)
            ]
        ).reshape((8,)),
        [4, 1, 6, 3, 0, 5, 2, 7])

    # Specified output case.
    out = np.zeros(shape=(4,))
    actual = cirq.apply_matrix_to_slices(
        target=np.array([1, 2, 3, 4]),
        matrix=np.array([[2, 3], [5, 7]]),
        slices=[1, 2],
        out=out)
    assert actual is out
    np.testing.assert_allclose(
        actual,
        np.array([1, 13, 31, 4]))


def test_partial_trace():
    a = np.reshape(np.arange(4), (2, 2))
    b = np.reshape(np.arange(9) + 4, (3, 3))
    c = np.reshape(np.arange(16) + 13, (4, 4))
    tr_a = np.trace(a)
    tr_b = np.trace(b)
    tr_c = np.trace(c)
    tensor = np.reshape(np.kron(a, np.kron(b, c)), (2, 3, 4, 2, 3, 4))

    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, []),
        tr_a * tr_b * tr_c)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0]),
        a * tr_b * tr_c)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [1]),
        b * tr_a * tr_c)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [2]),
        c * tr_a * tr_b)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0, 1]),
        np.reshape(np.kron(a, b), (2, 3, 2, 3)) * tr_c)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [1, 2]),
        np.reshape(np.kron(b, c), (3, 4, 3, 4)) * tr_a)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0, 2]),
        np.reshape(np.kron(a, c), (2, 4, 2, 4)) * tr_b)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0, 1, 2]),
        tensor)

    # permutes indices
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [1, 0]),
        np.reshape(np.kron(b, a), (3, 2, 3, 2)) * tr_c)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [2, 0, 1]),
        np.reshape(np.kron(c, np.kron(a, b)), (4, 2, 3, 4, 2, 3)))

def test_partial_trace_non_kron():
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 0] = 1
    tensor[1, 1, 1, 1] = 4
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0]),
        np.array([[1, 0], [0, 4]]))


def test_partial_trace_invalid_inputs():
    with pytest.raises(ValueError, match='2, 3, 2, 2'):
        cirq.partial_trace(
            np.reshape(np.arange(2 * 3 * 2 * 2), (2, 3, 2, 2)), [1])
    with pytest.raises(ValueError, match='2'):
        cirq.partial_trace(
            np.reshape(np.arange(2 * 2 * 2 * 2), (2,) * 4), [2])
