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
import cirq.testing
from cirq import linalg


def test_reflection_matrix_pow_consistent_results():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = cirq.reflection_matrix_pow(x, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_x, sqrt_x), x, atol=1e-10)

    ix = x * np.sqrt(1j)
    sqrt_ix = cirq.reflection_matrix_pow(ix, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_ix, sqrt_ix), ix, atol=1e-10)

    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    cube_root_h = cirq.reflection_matrix_pow(h, 1 / 3)
    np.testing.assert_allclose(np.dot(np.dot(cube_root_h, cube_root_h), cube_root_h), h, atol=1e-8)

    y = np.array([[0, -1j], [1j, 0]])
    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5j)
    yh = np.kron(y, h)
    sqrt_yh = cirq.reflection_matrix_pow(yh, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_yh, sqrt_yh), yh, atol=1e-10)


def test_reflection_matrix_sign_preference_under_perturbation():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = np.array([[1, -1j], [-1j, 1]]) * (1 + 1j) / 2
    np.testing.assert_allclose(cirq.reflection_matrix_pow(x, 0.5), sqrt_x, atol=1e-8)

    # Sqrt should behave well when phased by less than 90 degrees.
    # (When rotating by more it's ambiguous. For example, 181 = 91*2 = -89*2.)
    for perturbation in [0, 0.1, -0.1, 0.3, -0.3, 0.49, -0.49]:
        px = x * complex(-1) ** perturbation
        expected_sqrt_px = sqrt_x * complex(-1) ** (perturbation / 2)
        sqrt_px = cirq.reflection_matrix_pow(px, 0.5)
        np.testing.assert_allclose(np.dot(sqrt_px, sqrt_px), px, atol=1e-10)
        np.testing.assert_allclose(sqrt_px, expected_sqrt_px, atol=1e-10)


def test_match_global_phase():
    a = np.array([[5, 4], [3, -2]])
    b = np.array([[0.000001, 0], [0, 1j]])
    c, d = cirq.match_global_phase(a, b)
    np.testing.assert_allclose(c, -a, atol=1e-10)
    np.testing.assert_allclose(d, b * -1j, atol=1e-10)


def test_match_global_phase_incompatible_shape():
    a = np.array([1])
    b = np.array([1, 2])
    c, d = cirq.match_global_phase(a, b)
    assert c.shape == a.shape
    assert d.shape == b.shape
    assert c is not a
    assert d is not b
    assert np.allclose(c, a)
    assert np.allclose(d, b)

    a = np.array([])
    b = np.array([])
    c, d = cirq.match_global_phase(a, b)
    assert c.shape == a.shape
    assert d.shape == b.shape
    assert c is not a
    assert d is not b
    assert np.allclose(c, a)
    assert np.allclose(d, b)


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
        cirq.targeted_left_multiply(
            left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[0]
        ),
        np.dot(cirq.kron(m, i, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(
            left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[1]
        ),
        np.dot(cirq.kron(i, m, i), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(
            left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[2]
        ),
        np.dot(cirq.kron(i, i, m), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(
            left_matrix=m, right_target=t.reshape((2, 2, 2)), target_axes=[2]
        ),
        np.dot(cirq.kron(i, i, m), t).reshape((2, 2, 2)),
        atol=1e-8,
    )

    common = t.reshape((2, 2, 2))
    with pytest.raises(ValueError, match="out is"):
        cirq.targeted_left_multiply(left_matrix=m, right_target=common, out=common, target_axes=[2])
    with pytest.raises(ValueError, match="out is"):
        cirq.targeted_left_multiply(left_matrix=m, right_target=common, out=m, target_axes=[2])


def test_targeted_left_multiply_reorders_matrices():
    t = np.eye(4).reshape((2, 2, 2, 2))
    m = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]).reshape((2, 2, 2, 2))

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m, right_target=t, target_axes=[0, 1]), m, atol=1e-8
    )

    np.testing.assert_allclose(
        cirq.targeted_left_multiply(left_matrix=m, right_target=t, target_axes=[1, 0]),
        np.array([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]).reshape((2, 2, 2, 2)),
        atol=1e-8,
    )


def test_targeted_left_multiply_out():
    left = np.array([[2, 3], [5, 7]])
    right = np.array([1, -1])
    out = np.zeros(2)

    result = cirq.targeted_left_multiply(
        left_matrix=left, right_target=right, target_axes=[0], out=out
    )
    assert result is out
    np.testing.assert_allclose(result, np.array([-1, -2]), atol=1e-8)


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
    expected = np.einsum('ij,jklm,ln->iknm', a, b, np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)

    result = cirq.targeted_conjugate_about(a, b, [1])
    expected = np.einsum('ij,kjlm,mn->kiln', a, b, np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_multiple_indices():
    a = np.reshape(np.arange(16) + 1j, (2, 2, 2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0, 1])
    expected = np.einsum('ijkl,klmn,mnop->ijop', a, b, np.transpose(np.conjugate(a), (2, 3, 0, 1)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_multiple_indices_flip_order():
    a = np.reshape(np.arange(16) + 1j, (2, 2, 2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [1, 0], [3, 2])
    expected = np.einsum('ijkl,lknm,mnop->jipo', a, b, np.transpose(np.conjugate(a), (2, 3, 0, 1)))
    np.testing.assert_almost_equal(result, expected)


def test_targeted_conjugate_out():
    a = np.reshape([0, 1, 2j, 3j], (2, 2))
    b = np.reshape(np.arange(16), (2,) * 4)
    buffer = np.empty((2,) * 4, dtype=a.dtype)
    out = np.empty((2,) * 4, dtype=a.dtype)
    result = cirq.targeted_conjugate_about(a, b, [0], buffer=buffer, out=out)
    assert result is out
    expected = np.einsum('ij,jklm,ln->iknm', a, b, np.transpose(np.conjugate(a)))
    np.testing.assert_almost_equal(result, expected)


def test_apply_matrix_to_slices():
    # Output is input.
    with pytest.raises(ValueError, match='out'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(target=target, matrix=np.eye(2), out=target, slices=[0, 1])

    # Wrong matrix size.
    with pytest.raises(ValueError, match='shape'):
        target = np.eye(2)
        _ = cirq.apply_matrix_to_slices(target=target, matrix=np.eye(3), slices=[0, 1])

    # Empty case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(target=np.array(range(5)), matrix=np.eye(0), slices=[]),
        np.array(range(5)),
    )

    # Middle 2x2 of 4x4 case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.eye(4), matrix=np.array([[2, 3], [5, 7]]), slices=[1, 2]
        ),
        np.array([[1, 0, 0, 0], [0, 2, 3, 0], [0, 5, 7, 0], [0, 0, 0, 1]]),
    )

    # Middle 2x2 of 4x4 with opposite order case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.eye(4), matrix=np.array([[2, 3], [5, 7]]), slices=[2, 1]
        ),
        np.array([[1, 0, 0, 0], [0, 7, 5, 0], [0, 3, 2, 0], [0, 0, 0, 1]]),
    )

    # Complicated slices of tensor case.
    np.testing.assert_allclose(
        cirq.apply_matrix_to_slices(
            target=np.array(range(8)).reshape((2, 2, 2)),
            matrix=np.array([[0, 1], [1, 0]]),
            slices=[(0, slice(None), 0), (1, slice(None), 0)],
        ).reshape((8,)),
        [4, 1, 6, 3, 0, 5, 2, 7],
    )

    # Specified output case.
    out = np.zeros(shape=(4,))
    actual = cirq.apply_matrix_to_slices(
        target=np.array([1, 2, 3, 4]), matrix=np.array([[2, 3], [5, 7]]), slices=[1, 2], out=out
    )
    assert actual is out
    np.testing.assert_allclose(actual, np.array([1, 13, 31, 4]))


def test_partial_trace():
    a = np.reshape(np.arange(4), (2, 2))
    b = np.reshape(np.arange(9) + 4, (3, 3))
    c = np.reshape(np.arange(16) + 13, (4, 4))
    tr_a = np.trace(a)
    tr_b = np.trace(b)
    tr_c = np.trace(c)
    tensor = np.reshape(np.kron(a, np.kron(b, c)), (2, 3, 4, 2, 3, 4))

    np.testing.assert_almost_equal(cirq.partial_trace(tensor, []), tr_a * tr_b * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0]), a * tr_b * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [1]), b * tr_a * tr_c)
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [2]), c * tr_a * tr_b)
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0, 1]), np.reshape(np.kron(a, b), (2, 3, 2, 3)) * tr_c
    )
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [1, 2]), np.reshape(np.kron(b, c), (3, 4, 3, 4)) * tr_a
    )
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [0, 2]), np.reshape(np.kron(a, c), (2, 4, 2, 4)) * tr_b
    )
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0, 1, 2]), tensor)

    # permutes indices
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [1, 0]), np.reshape(np.kron(b, a), (3, 2, 3, 2)) * tr_c
    )
    np.testing.assert_almost_equal(
        cirq.partial_trace(tensor, [2, 0, 1]),
        np.reshape(np.kron(c, np.kron(a, b)), (4, 2, 3, 4, 2, 3)),
    )


def test_partial_trace_non_kron():
    tensor = np.zeros((2, 2, 2, 2))
    tensor[0, 0, 0, 0] = 1
    tensor[1, 1, 1, 1] = 4
    np.testing.assert_almost_equal(cirq.partial_trace(tensor, [0]), np.array([[1, 0], [0, 4]]))


def test_partial_trace_invalid_inputs():
    with pytest.raises(ValueError, match='2, 3, 2, 2'):
        cirq.partial_trace(np.reshape(np.arange(2 * 3 * 2 * 2), (2, 3, 2, 2)), [1])
    with pytest.raises(ValueError, match='2'):
        cirq.partial_trace(np.reshape(np.arange(2 * 2 * 2 * 2), (2,) * 4), [2])


def test_sub_state_vector():
    a = np.arange(4) / np.linalg.norm(np.arange(4))
    b = (np.arange(8) + 3) / np.linalg.norm(np.arange(8) + 3)
    c = (np.arange(16) + 1) / np.linalg.norm(np.arange(16) + 1)
    state = np.kron(np.kron(a, b), c).reshape((2, 2, 2, 2, 2, 2, 2, 2, 2))

    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(a, [0, 1], atol=1e-8), a)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(b, [0, 1, 2], atol=1e-8), b)
    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(c, [0, 1, 2, 3], atol=1e-8), c)

    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(state, [0, 1], atol=1e-15), a.reshape((2, 2))
    )
    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(state, [2, 3, 4], atol=1e-15), b.reshape((2, 2, 2))
    )
    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(state, [5, 6, 7, 8], atol=1e-15), c.reshape((2, 2, 2, 2))
    )

    # Output state vector conforms to the shape of the input state vector.
    reshaped_state = state.reshape(-1)
    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(reshaped_state, [0, 1], atol=1e-15), a
    )
    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(reshaped_state, [2, 3, 4], atol=1e-15), b
    )
    assert cirq.equal_up_to_global_phase(
        cirq.sub_state_vector(reshaped_state, [5, 6, 7, 8], atol=1e-15), c
    )

    # Reject factoring for very tight tolerance.
    assert cirq.sub_state_vector(state, [0, 1], default=None, atol=1e-16) is None
    assert cirq.sub_state_vector(state, [2, 3, 4], default=None, atol=1e-16) is None
    assert cirq.sub_state_vector(state, [5, 6, 7, 8], default=None, atol=1e-16) is None

    # Permit invalid factoring for loose tolerance.
    for q1 in range(9):
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1) is not None


def test_sub_state_vector_bad_subset():
    a = cirq.testing.random_superposition(4)
    b = cirq.testing.random_superposition(8)
    state = np.kron(a, b).reshape((2, 2, 2, 2, 2))

    for q1 in range(5):
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1e-8) is None
    for q1 in range(2):
        for q2 in range(2, 5):
            assert cirq.sub_state_vector(state, [q1, q2], default=None, atol=1e-8) is None
    for q3 in range(2, 5):
        assert cirq.sub_state_vector(state, [0, 1, q3], default=None, atol=1e-8) is None
    for q4 in range(2):
        assert cirq.sub_state_vector(state, [2, 3, 4, q4], default=None, atol=1e-8) is None


def test_sub_state_vector_non_kron():
    a = np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
    b = np.array([1, 1]) / np.sqrt(2)
    state = np.kron(a, b).reshape((2, 2, 2, 2))

    for q1 in [0, 1, 2]:
        assert cirq.sub_state_vector(state, [q1], default=None, atol=1e-8) is None
    for q1 in [0, 1, 2]:
        assert cirq.sub_state_vector(state, [q1, 3], default=None, atol=1e-8) is None

    with pytest.raises(ValueError, match='factored'):
        _ = cirq.sub_state_vector(a, [0], atol=1e-8)

    assert cirq.equal_up_to_global_phase(cirq.sub_state_vector(state, [3]), b, atol=1e-8)


def test_sub_state_vector_invalid_inputs():

    # State cannot be expressed as a separable pure state.
    with pytest.raises(ValueError, match='7'):
        cirq.sub_state_vector(np.arange(7), [1, 2], atol=1e-8)

    # State shape does not conform to input requirements.
    with pytest.raises(ValueError, match='shaped'):
        cirq.sub_state_vector(np.arange(16).reshape((2, 4, 2)), [1, 2], atol=1e-8)
    with pytest.raises(ValueError, match='shaped'):
        cirq.sub_state_vector(np.arange(16).reshape((16, 1)), [1, 2], atol=1e-8)

    with pytest.raises(ValueError, match='normalized'):
        cirq.sub_state_vector(np.arange(16), [1, 2], atol=1e-8)

    # Bad choice of input indices.
    state = np.arange(8) / np.linalg.norm(np.arange(8))
    with pytest.raises(ValueError, match='2, 2'):
        cirq.sub_state_vector(state, [1, 2, 2], atol=1e-8)

    state = np.array([1, 0, 0, 0]).reshape((2, 2))
    with pytest.raises(ValueError, match='invalid'):
        cirq.sub_state_vector(state, [5], atol=1e-8)
    with pytest.raises(ValueError, match='invalid'):
        cirq.sub_state_vector(state, [0, 1, 2], atol=1e-8)


def test_partial_trace_of_state_vector_as_mixture_invalid_input():

    with pytest.raises(ValueError, match='7'):
        cirq.partial_trace_of_state_vector_as_mixture(np.arange(7), [1, 2], atol=1e-8)

    with pytest.raises(ValueError, match='normalized'):
        cirq.partial_trace_of_state_vector_as_mixture(np.arange(8), [1], atol=1e-8)

    state = np.arange(8) / np.linalg.norm(np.arange(8))
    with pytest.raises(ValueError, match='repeated axis'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [1, 2, 2], atol=1e-8)

    state = np.array([1, 0, 0, 0]).reshape((2, 2))
    with pytest.raises(IndexError, match='out of range'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [5], atol=1e-8)
    with pytest.raises(IndexError, match='out of range'):
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 2], atol=1e-8)


def mixtures_equal(m1, m2, atol=1e-7):
    for (p1, v1), (p2, v2) in zip(m1, m2):
        if not (
            cirq.approx_eq(p1, p2, atol=atol) and cirq.equal_up_to_global_phase(v1, v2, atol=atol)
        ):
            return False
    return True


def test_partial_trace_of_state_vector_as_mixture_pure_result():
    a = cirq.testing.random_superposition(4)
    b = cirq.testing.random_superposition(8)
    c = cirq.testing.random_superposition(16)
    state = np.kron(np.kron(a, b), c).reshape((2,) * 9)

    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-8),
        ((1.0, a.reshape((2, 2))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4], atol=1e-8),
        ((1.0, b.reshape((2, 2, 2))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [5, 6, 7, 8], atol=1e-8),
        ((1.0, c.reshape((2, 2, 2, 2))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 2, 3, 4], atol=1e-8),
        ((1.0, np.kron(a, b).reshape((2, 2, 2, 2, 2))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1, 5, 6, 7, 8], atol=1e-8),
        ((1.0, np.kron(a, c).reshape((2, 2, 2, 2, 2, 2))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4, 5, 6, 7, 8], atol=1e-8),
        ((1.0, np.kron(b, c).reshape((2, 2, 2, 2, 2, 2, 2))),),
    )

    # Shapes of states in the output mixture conform to the input's shape.
    state = state.reshape(2**9)
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-8), ((1.0, a),)
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [2, 3, 4], atol=1e-8), ((1.0, b),)
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [5, 6, 7, 8], atol=1e-8), ((1.0, c),)
    )

    # Return mixture will defer to numpy.linalg.eigh's builtin tolerance.
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-20), truth, atol=1e-15
    )
    assert not mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-20), truth, atol=1e-16
    )


def test_partial_trace_of_state_vector_as_mixture_pure_result_qudits():
    a = cirq.testing.random_superposition(2)
    b = cirq.testing.random_superposition(3)
    c = cirq.testing.random_superposition(4)
    state = np.kron(np.kron(a, b), c).reshape((2, 3, 4))

    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0], atol=1e-8), ((1.0, a),)
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [1], atol=1e-8), ((1.0, b),)
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [2], atol=1e-8), ((1.0, c),)
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 1], atol=1e-8),
        ((1.0, np.kron(a, b).reshape((2, 3))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [0, 2], atol=1e-8),
        ((1.0, np.kron(a, c).reshape((2, 4))),),
    )
    assert mixtures_equal(
        cirq.partial_trace_of_state_vector_as_mixture(state, [1, 2], atol=1e-8),
        ((1.0, np.kron(b, c).reshape((3, 4))),),
    )


def test_partial_trace_of_state_vector_as_mixture_mixed_result():
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    for q1 in [0, 1]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-8)
        assert mixtures_equal(mixture, truth)

    state = np.array([0, 1, 1, 0, 1, 0, 0, 0]).reshape((2, 2, 2)) / np.sqrt(3)
    truth = ((1 / 3, np.array([0.0, 1.0])), (2 / 3, np.array([1.0, 0.0])))
    for q1 in [0, 1, 2]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-8)
        assert mixtures_equal(mixture, truth)

    state = np.array([1, 0, 0, 0, 0, 0, 0, 1]).reshape((2, 2, 2)) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0])), (0.5, np.array([0, 1])))
    for q1 in [0, 1, 2]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-8)
        assert mixtures_equal(mixture, truth)

    truth = (
        (0.5, np.array([1, 0, 0, 0]).reshape((2, 2))),
        (0.5, np.array([0, 0, 0, 1]).reshape((2, 2))),
    )
    for q1, q2 in [(0, 1), (0, 2), (1, 2)]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1, q2], atol=1e-8)
        assert mixtures_equal(mixture, truth)


def test_partial_trace_of_state_vector_as_mixture_mixed_result_qudits():
    state = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) / np.sqrt(2)
    truth = ((0.5, np.array([1, 0, 0])), (0.5, np.array([0, 0, 1])))
    for q1 in [0, 1]:
        mixture = cirq.partial_trace_of_state_vector_as_mixture(state, [q1], atol=1e-8)
        assert mixtures_equal(mixture, truth)


def test_to_special():
    u = cirq.testing.random_unitary(4)
    su = cirq.to_special(u)
    assert not cirq.is_special_unitary(u)
    assert cirq.is_special_unitary(su)


def test_default_tolerance():
    a, b = cirq.LineQubit.range(2)
    final_state_vector = (
        cirq.Simulator()
        .simulate(cirq.Circuit(cirq.H(a), cirq.H(b), cirq.CZ(a, b), cirq.measure(a)))
        .final_state_vector.reshape((2, 2))
    )
    # Here, we do NOT specify the default tolerance. It is merely to check that the default value
    # is reasonable.
    cirq.sub_state_vector(final_state_vector, [0])


@pytest.mark.parametrize('state_1', [0, 1])
@pytest.mark.parametrize('state_2', [0, 1])
def test_factor_state_vector(state_1: int, state_2: int):
    # Kron two state vectors and apply a phase. Factoring should produce the expected results.
    n = 12
    for i in range(n):
        phase = np.exp(2 * np.pi * 1j * i / n)
        a = cirq.to_valid_state_vector(state_1, 1)
        b = cirq.to_valid_state_vector(state_2, 1)
        c = cirq.linalg.transformations.state_vector_kronecker_product(a, b) * phase
        a1, b1 = cirq.linalg.transformations.factor_state_vector(c, [0], validate=True)
        c1 = cirq.linalg.transformations.state_vector_kronecker_product(a1, b1)
        assert np.allclose(c, c1)

        # All phase goes into a1, and b1 is just the dephased state vector
        assert np.allclose(a1, a * phase)
        assert np.allclose(b1, b)


@pytest.mark.parametrize('num_dimensions', [*range(1, 7)])
def test_transpose_flattened_array(num_dimensions):
    np.random.seed(0)
    for _ in range(10):
        shape = np.random.randint(1, 5, (num_dimensions,)).tolist()
        axes = np.random.permutation(num_dimensions).tolist()
        volume = np.prod(shape)
        A = np.random.permutation(volume)
        want = np.transpose(A.reshape(shape), axes)
        got = linalg.transpose_flattened_array(A, shape, axes).reshape(want.shape)
        assert np.array_equal(want, got)
        got = linalg.transpose_flattened_array(A.reshape(shape), shape, axes).reshape(want.shape)
        assert np.array_equal(want, got)


@pytest.mark.parametrize('shape, result', [((), True), (30 * (1,), True), ((-3, 1, -1), False)])
def test_can_numpy_support_shape(shape: tuple[int, ...], result: bool) -> None:
    assert linalg.can_numpy_support_shape(shape) is result


@pytest.mark.parametrize('coeff', [1, 1j, -1, -1j, 1j**0.5, 1j**0.3])
def test_phase_delta(coeff):
    u1 = cirq.testing.random_unitary(4)
    u2 = u1 * coeff
    np.testing.assert_almost_equal(linalg.phase_delta(u1, u2), coeff)
    np.testing.assert_almost_equal(u1 * linalg.phase_delta(u1, u2), u2)
