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

import itertools

import numpy as np
import pytest

import cirq
import cirq.testing


def test_deprecated():
    with cirq.testing.assert_logs('cirq.to_valid_density_matrix', 'deprecated'):
        _ = cirq.sim.to_valid_density_matrix(0, 1)

    with cirq.testing.assert_logs('cirq.von_neumann_entropy', 'deprecated'):
        _ = cirq.sim.von_neumann_entropy(np.eye(2) / 2)


def test_sample_density_matrix_big_endian():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        sample = cirq.sample_density_matrix(matrix, [2, 1, 0])
        results.append(sample)
    expecteds = [[list(reversed(x))] for x in
                 list(itertools.product([False, True], repeat=3))]
    for result, expected in zip(results, expecteds):
        np.testing.assert_equal(result, expected)


def test_sample_density_matrix_partial_indices():
    for index in range(3):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            np.testing.assert_equal(cirq.sample_density_matrix(matrix, [index]),
                                    [[bool(1 & (x >> (2 - index)))]])

def test_sample_density_matrix_partial_indices_oder():
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        expected = [[bool(1 & (x >> 0)), bool(1 & (x >> 1))]]
        np.testing.assert_equal(cirq.sample_density_matrix(matrix, [2, 1]),
                                expected)


def test_sample_density_matrix_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            expected = [[bool(1 & (x >> (2 - p))) for p in perm]]
            np.testing.assert_equal(cirq.sample_density_matrix(matrix, perm),
                                    expected)


def test_sample_density_matrix():
    state = np.zeros(8, dtype=np.complex64)
    state[0] = 1 / np.sqrt(2)
    state[2] = 1 / np.sqrt(2)
    matrix = cirq.to_valid_density_matrix(state, num_qubits=3)
    for _ in range(10):
        sample = cirq.sample_density_matrix(matrix, [2, 1, 0])
        assert (np.array_equal(sample, [[False, False, False]])
                or np.array_equal(sample, [[False, True, False]]))
    # Partial sample is correct.
    for _ in range(10):
        np.testing.assert_equal(cirq.sample_density_matrix(matrix, [2]),
                                [[False]])
        np.testing.assert_equal(cirq.sample_density_matrix(matrix, [0]),
                                [[False]])


def test_sample_density_matrix_seed():
    density_matrix = 0.5 * np.eye(2)

    samples = cirq.sample_density_matrix(density_matrix, [0],
                                         repetitions=10,
                                         seed=1234)
    assert np.array_equal(samples, [[False], [True], [False], [True], [True],
                                    [False], [False], [True], [True], [True]])

    samples = cirq.sample_density_matrix(density_matrix, [0],
                                         repetitions=10,
                                         seed=np.random.RandomState(1234))
    assert np.array_equal(samples, [[False], [True], [False], [True], [True],
                                    [False], [False], [True], [True], [True]])


def test_sample_empty_density_matrix():
    matrix = np.zeros(shape=())
    np.testing.assert_almost_equal(cirq.sample_density_matrix(matrix, []), [[]])


def test_sample_density_matrix_no_repetitions():
    matrix = cirq.to_valid_density_matrix(0, 3)
    np.testing.assert_almost_equal(
        cirq.sample_density_matrix(matrix, [1], repetitions=0),
        np.zeros(shape=(0, 1)))
    np.testing.assert_almost_equal(
        cirq.sample_density_matrix(matrix, [0, 1], repetitions=0),
        np.zeros(shape=(0, 2)))


def test_sample_density_matrix_repetitions():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            expected = [[bool(1 & (x >> (2 - p))) for p in perm]] * 3

            result = cirq.sample_density_matrix(matrix, perm, repetitions=3)
            np.testing.assert_equal(result, expected)


def test_sample_density_matrix_negative_repetitions():
    matrix = cirq.to_valid_density_matrix(0, 3)
    with pytest.raises(ValueError, match='-1'):
        cirq.sample_density_matrix(matrix, [1], repetitions=-1)


def test_sample_density_matrix_not_square():
    with pytest.raises(ValueError, match='not square'):
        cirq.sample_density_matrix(np.array([1, 0, 0]), [1])


def test_sample_density_matrix_not_power_of_two():
    with pytest.raises(ValueError, match='power of two'):
        cirq.sample_density_matrix(np.ones((3, 3)) / 3, [1])
    with pytest.raises(ValueError, match='power of two'):
        cirq.sample_density_matrix(np.ones((2, 3, 2, 3)) / 6, [1])


def test_sample_density_matrix_higher_powers_of_two():
    with pytest.raises(ValueError, match='powers of two'):
        cirq.sample_density_matrix(np.ones((2, 4, 2, 4)) / 8, [1])


def test_sample_density_matrix_out_of_range():
    matrix = cirq.to_valid_density_matrix(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.sample_density_matrix(matrix, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.sample_density_matrix(matrix, [3])


def test_sample_density_matrix_no_indices():
    matrix = cirq.to_valid_density_matrix(0, 3)
    bits = cirq.sample_density_matrix(matrix, [])
    np.testing.assert_almost_equal(bits, np.zeros(shape=(1, 0)))


def test_sample_density_matrix_validate_qid_shape():
    matrix = cirq.to_valid_density_matrix(0, 3)
    cirq.sample_density_matrix(matrix, [], qid_shape=(2, 2, 2))
    with pytest.raises(ValueError,
                       match='Matrix size does not match qid shape'):
        cirq.sample_density_matrix(matrix, [], qid_shape=(2, 2, 1))
    matrix2 = cirq.to_valid_density_matrix(0, qid_shape=(1, 2, 3))
    cirq.sample_density_matrix(matrix2, [], qid_shape=(1, 2, 3))
    with pytest.raises(ValueError,
                       match='Matrix size does not match qid shape'):
        cirq.sample_density_matrix(matrix2, [], qid_shape=(2, 2, 2))


def test_measure_density_matrix_computational_basis():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(out_matrix, matrix)
    expected = [list(reversed(x)) for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_measure_density_matrix_computational_basis_reversed():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        bits, out_matrix = cirq.measure_density_matrix(matrix, [0, 1, 2])
        results.append(bits)
        np.testing.assert_almost_equal(out_matrix, matrix)
    expected = [list(x) for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_measure_density_matrix_computational_basis_reshaped():
    results = []
    for x in range(8):
        matrix = np.reshape(cirq.to_valid_density_matrix(x, 3), (2,) * 6)
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(out_matrix, matrix)
    expected = [list(reversed(x)) for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_measure_density_matrix_partial_indices():
    for index in range(3):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            bits, out_matrix = cirq.measure_density_matrix(matrix, [index])
            np.testing.assert_almost_equal(out_matrix, matrix)
            assert bits == [bool(1 & (x >> (2 - index)))]


def test_measure_density_matrix_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            bits, out_matrix = cirq.measure_density_matrix(matrix, perm)
            np.testing.assert_almost_equal(matrix, out_matrix)
            assert bits == [bool(1 & (x >> (2 - p))) for p in perm]


def matrix_000_plus_010():
    state = np.zeros(8, dtype=np.complex64)
    state[0] = 1 / np.sqrt(2)
    state[2] = 1j / np.sqrt(2)
    return cirq.to_valid_density_matrix(state, num_qubits=3)


def test_measure_density_matrix_collapse():
    matrix = matrix_000_plus_010()
    for _ in range(10):
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0])
        assert bits in [[False, False, False], [False, True, False]]
        expected = np.zeros(8, dtype=np.complex64)
        if bits[1]:
            expected[2] = 1j
        else:
            expected[0] = 1
        expected_matrix = np.outer(np.conj(expected), expected)
        np.testing.assert_almost_equal(out_matrix, expected_matrix)
        assert out_matrix is not matrix

    # Partial sample is correct.
    for _ in range(10):
        bits, out_matrix = cirq.measure_density_matrix(matrix, [2])
        np.testing.assert_almost_equal(out_matrix, matrix)
        assert bits == [False]

        bits, out_matrix = cirq.measure_density_matrix(matrix, [0])
        np.testing.assert_almost_equal(out_matrix, matrix)
        assert bits == [False]


def test_measure_density_matrix_seed():
    n = 5
    matrix = np.eye(2**n) / 2**n

    bits, out_matrix1 = cirq.measure_density_matrix(matrix, range(n), seed=1234)
    assert bits == [False, False, True, True, False]

    bits, out_matrix2 = cirq.measure_density_matrix(
        matrix, range(n), seed=np.random.RandomState(1234))
    assert bits == [False, False, True, True, False]

    np.testing.assert_allclose(out_matrix1, out_matrix2)


def test_measure_density_matrix_out_is_matrix():
    matrix = matrix_000_plus_010()
    bits, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0],
                                                   out=matrix)
    expected_state = np.zeros(8, dtype=np.complex64)
    expected_state[2 if bits[1] else 0] = 1.0
    expected_matrix = np.outer(np.conj(expected_state), expected_state)
    np.testing.assert_array_almost_equal(out_matrix, expected_matrix)
    assert out_matrix is matrix


def test_measure_state_out_is_not_matrix():
    matrix = matrix_000_plus_010()
    out = np.zeros_like(matrix)
    _, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0], out=out)
    assert out is not matrix
    assert out is out_matrix


def test_measure_density_matrix_not_square():
    with pytest.raises(ValueError, match='not square'):
        cirq.measure_density_matrix(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='not square'):
        cirq.measure_density_matrix(np.array([1, 0, 0, 0]).reshape((2, 1, 2)),
                                    [1],
                                    qid_shape=(2, 1))


def test_measure_density_matrix_not_power_of_two():
    with pytest.raises(ValueError, match='power of two'):
        cirq.measure_density_matrix(np.ones((3, 3)) / 3, [1])
    with pytest.raises(ValueError, match='power of two'):
        cirq.measure_density_matrix(np.ones((2, 3, 2, 3)) / 6, [1])


def test_measure_density_matrix_higher_powers_of_two():
    with pytest.raises(ValueError, match='powers of two'):
        cirq.measure_density_matrix(np.ones((2, 4, 2, 4)) / 8, [1])


def test_measure_density_matrix_tensor_different_left_right_shape():
    with pytest.raises(ValueError, match='not equal'):
        cirq.measure_density_matrix(np.array([1, 0, 0, 0]).reshape(
            (2, 2, 1, 1)), [1],
                                    qid_shape=(2, 1))


def test_measure_density_matrix_out_of_range():
    matrix = cirq.to_valid_density_matrix(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.measure_density_matrix(matrix, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.measure_density_matrix(matrix, [3])


def test_measure_state_no_indices():
    matrix = cirq.to_valid_density_matrix(0, 3)
    bits, out_matrix = cirq.measure_density_matrix(matrix, [])
    assert [] == bits
    np.testing.assert_almost_equal(out_matrix, matrix)



def test_measure_state_no_indices_out_is_matrix():
    matrix = cirq.to_valid_density_matrix(0, 3)
    bits, out_matrix = cirq.measure_density_matrix(matrix, [], out=matrix)
    assert [] == bits
    np.testing.assert_almost_equal(out_matrix, matrix)
    assert out_matrix is matrix


def test_measure_state_no_indices_out_is_not_matrix():
    matrix = cirq.to_valid_density_matrix(0, 3)
    out = np.zeros_like(matrix)
    bits, out_matrix = cirq.measure_density_matrix(matrix, [], out=out)
    assert [] == bits
    np.testing.assert_almost_equal(out_matrix, matrix)
    assert out is out_matrix
    assert out is not matrix


def test_measure_state_empty_density_matrix():
    matrix = np.zeros(shape=())
    bits, out_matrix = cirq.measure_density_matrix(matrix, [])
    assert [] == bits
    np.testing.assert_almost_equal(matrix, out_matrix)


@pytest.mark.parametrize('seed', [17, 35, 48])
def test_to_valid_density_matrix_on_simulator_output(seed):
    circuit = cirq.testing.random_circuit(qubits=5,
                                          n_moments=20,
                                          op_density=0.9,
                                          random_state=seed)
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.simulate(circuit)
    _ = cirq.to_valid_density_matrix(result.final_density_matrix, num_qubits=5)
