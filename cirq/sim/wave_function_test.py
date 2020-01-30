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
"""Tests for wave_function.py"""

import itertools
import pytest

import numpy as np

import cirq


def assert_dirac_notation_numpy(vec, expected, decimals=2):
    assert cirq.dirac_notation(np.array(vec), decimals=decimals) == expected


def assert_dirac_notation_python(vec, expected, decimals=2):
    assert cirq.dirac_notation(vec, decimals=decimals) == expected


def test_state_mixin():
    class TestClass(cirq.StateVectorMixin):
        def state_vector(self) -> np.ndarray:
            return np.array([0, 0, 1, 0])
    qubits = cirq.LineQubit.range(2)
    test = TestClass(qubit_map={qubits[i]: i for i in range(2)})
    assert test.dirac_notation() == '|10⟩'
    np.testing.assert_almost_equal(test.bloch_vector_of(qubits[0]),
                                   np.array([0, 0, -1]))
    np.testing.assert_almost_equal(test.density_matrix_of(qubits[0:1]),
                                   np.array([[0, 0], [0, 1]]))

    assert cirq.qid_shape(TestClass({qubits[i]: 1 - i for i in range(2)
                                    })) == (2, 2)
    assert cirq.qid_shape(
        TestClass({cirq.LineQid(i, i + 1): 2 - i for i in range(3)})) == (3, 2,
                                                                          1)
    assert cirq.qid_shape(TestClass(), 'no shape') == 'no shape'

    with pytest.raises(ValueError, match='Qubit index out of bounds'):
        _ = TestClass({qubits[0]: 1})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: 0, qubits[1]: 0})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: 1, qubits[1]: 1})
    with pytest.raises(ValueError, match='Duplicate qubit index'):
        _ = TestClass({qubits[0]: -1, qubits[1]: 1})


def test_bloch_vector_simple_H_zero():
    sqrt = np.sqrt(0.5)
    H_state = np.array([sqrt, sqrt])

    bloch = cirq.bloch_vector_from_state_vector(H_state, 0)
    desired_simple = np.array([1,0,0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_simple_XH_zero():
    sqrt = np.sqrt(0.5)
    XH_state = np.array([sqrt, sqrt])
    bloch = cirq.bloch_vector_from_state_vector(XH_state, 0)

    desired_simple = np.array([1,0,0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_simple_YH_zero():
    sqrt = np.sqrt(0.5)
    YH_state = np.array([-1.0j * sqrt, 1.0j * sqrt])
    bloch = cirq.bloch_vector_from_state_vector(YH_state, 0)

    desired_simple = np.array([-1,0,0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_simple_ZH_zero():
    sqrt = np.sqrt(0.5)
    ZH_state = np.array([sqrt, -sqrt])
    bloch = cirq.bloch_vector_from_state_vector(ZH_state, 0)

    desired_simple = np.array([-1,0,0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_simple_TH_zero():
    sqrt = np.sqrt(0.5)
    TH_state = np.array([sqrt, 0.5+0.5j])
    bloch = cirq.bloch_vector_from_state_vector(TH_state, 0)

    desired_simple = np.array([sqrt,sqrt,0])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_equal_sqrt3():
    sqrt3 = 1/np.sqrt(3)
    test_state = np.array([0.888074, 0.325058 + 0.325058j])
    bloch = cirq.bloch_vector_from_state_vector(test_state, 0)

    desired_simple = np.array([sqrt3,sqrt3,sqrt3])
    np.testing.assert_array_almost_equal(bloch, desired_simple)


def test_bloch_vector_multi_pure():
    HH_state = np.array([0.5,0.5,0.5,0.5])

    bloch_0 = cirq.bloch_vector_from_state_vector(HH_state, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(HH_state, 1)
    desired_simple = np.array([1,0,0])

    np.testing.assert_array_almost_equal(bloch_1, desired_simple)
    np.testing.assert_array_almost_equal(bloch_0, desired_simple)


def test_bloch_vector_multi_mixed():
    sqrt = np.sqrt(0.5)
    HCNOT_state = np.array([sqrt, 0., 0., sqrt])

    bloch_0 = cirq.bloch_vector_from_state_vector(HCNOT_state, 0)
    bloch_1 = cirq.bloch_vector_from_state_vector(HCNOT_state, 1)
    zero = np.zeros(3)

    np.testing.assert_array_almost_equal(bloch_0, zero)
    np.testing.assert_array_almost_equal(bloch_1, zero)

    RCNOT_state = np.array([0.90612745, -0.07465783j,
        -0.37533028j, 0.18023996])
    bloch_mixed_0 = cirq.bloch_vector_from_state_vector(RCNOT_state, 0)
    bloch_mixed_1 = cirq.bloch_vector_from_state_vector(RCNOT_state, 1)

    true_mixed_0 = np.array([0., -0.6532815, 0.6532815])
    true_mixed_1 = np.array([0., 0., 0.9238795])

    np.testing.assert_array_almost_equal(true_mixed_0, bloch_mixed_0)
    np.testing.assert_array_almost_equal(true_mixed_1, bloch_mixed_1)


def test_bloch_vector_multi_big():
    big_H_state = np.array([0.1767767] * 32)
    desired_simple = np.array([1,0,0])
    for qubit in range(0, 5):
        bloch_i = cirq.bloch_vector_from_state_vector(big_H_state, qubit)
        np.testing.assert_array_almost_equal(bloch_i, desired_simple)


def test_bloch_vector_invalid():
    with pytest.raises(ValueError):
        _ = cirq.bloch_vector_from_state_vector(
            np.array([0.5, 0.5, 0.5]), 0)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(
            np.array([0.5, 0.5,0.5,0.5]), -1)
    with pytest.raises(IndexError):
        _ = cirq.bloch_vector_from_state_vector(
            np.array([0.5, 0.5,0.5,0.5]), 2)


def test_density_matrix():
    test_state = np.array([0.-0.35355339j, 0.+0.35355339j, 0.-0.35355339j,
        0.+0.35355339j, 0.+0.35355339j, 0.-0.35355339j, 0.+0.35355339j,
        0.-0.35355339j])

    full_rho = cirq.density_matrix_from_state_vector(test_state)
    np.testing.assert_array_almost_equal(full_rho,
        np.outer(test_state, np.conj(test_state)))

    rho_one = cirq.density_matrix_from_state_vector(test_state, [1])
    true_one = np.array([[0.5+0.j, 0.5+0.j], [0.5+0.j, 0.5+0.j]])
    np.testing.assert_array_almost_equal(rho_one, true_one)

    rho_two_zero = cirq.density_matrix_from_state_vector(test_state, [0,2])
    true_two_zero = np.array([[ 0.25+0.j, -0.25+0.j, -0.25+0.j,  0.25+0.j],
                             [-0.25+0.j, 0.25+0.j,  0.25+0.j, -0.25+0.j],
                             [-0.25+0.j, 0.25+0.j,  0.25+0.j, -0.25+0.j],
                             [ 0.25+0.j, -0.25+0.j, -0.25+0.j,  0.25+0.j]])
    np.testing.assert_array_almost_equal(rho_two_zero, true_two_zero)

    # two and zero will have same single qubit density matrix.
    rho_two = cirq.density_matrix_from_state_vector(test_state, [2])
    true_two = np.array([[0.5+0.j, -0.5+0.j], [-0.5+0.j, 0.5+0.j]])
    np.testing.assert_array_almost_equal(rho_two, true_two)
    rho_zero = cirq.density_matrix_from_state_vector(test_state, [0])
    np.testing.assert_array_almost_equal(rho_zero, true_two)


def test_density_matrix_invalid():
    bad_state = np.array([0.5,0.5,0.5])
    good_state = np.array([0.5,0.5,0.5,0.5])
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state)
    with pytest.raises(ValueError):
        _ = cirq.density_matrix_from_state_vector(bad_state, [0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1, 0, 1])
    with pytest.raises(IndexError):
        _ = cirq.density_matrix_from_state_vector(good_state, [-1])


def test_dirac_notation():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation_numpy([0, 0], "0")
    assert_dirac_notation_python([1], "|⟩")
    assert_dirac_notation_numpy([sqrt, sqrt], "0.71|0⟩ + 0.71|1⟩")
    assert_dirac_notation_python([-sqrt, sqrt], "-0.71|0⟩ + 0.71|1⟩")
    assert_dirac_notation_numpy([sqrt, -sqrt], "0.71|0⟩ - 0.71|1⟩")
    assert_dirac_notation_python([-sqrt, -sqrt], "-0.71|0⟩ - 0.71|1⟩")
    assert_dirac_notation_numpy([sqrt, 1j * sqrt], "0.71|0⟩ + 0.71j|1⟩")
    assert_dirac_notation_python([sqrt, exp_pi_2], "0.71|0⟩ + (0.5+0.5j)|1⟩")
    assert_dirac_notation_numpy([exp_pi_2, -sqrt], "(0.5+0.5j)|0⟩ - 0.71|1⟩")
    assert_dirac_notation_python([exp_pi_2, 0.5 - 0.5j],
                                 "(0.5+0.5j)|0⟩ + (0.5-0.5j)|1⟩")
    assert_dirac_notation_numpy([0.5, 0.5, -0.5, -0.5],
                                "0.5|00⟩ + 0.5|01⟩ - 0.5|10⟩ - 0.5|11⟩")
    assert_dirac_notation_python([0.71j, 0.71j], "0.71j|0⟩ + 0.71j|1⟩")


def test_dirac_notation_partial_state():
    sqrt = np.sqrt(0.5)
    exp_pi_2 = 0.5 + 0.5j
    assert_dirac_notation_numpy([1, 0], "|0⟩")
    assert_dirac_notation_python([1j, 0], "1j|0⟩")
    assert_dirac_notation_numpy([0, 1], "|1⟩")
    assert_dirac_notation_python([0, 1j], "1j|1⟩")
    assert_dirac_notation_numpy([sqrt, 0, 0, sqrt], "0.71|00⟩ + 0.71|11⟩")
    assert_dirac_notation_python([sqrt, sqrt, 0, 0], "0.71|00⟩ + 0.71|01⟩")
    assert_dirac_notation_numpy([exp_pi_2, 0, 0, exp_pi_2],
                                "(0.5+0.5j)|00⟩ + (0.5+0.5j)|11⟩")
    assert_dirac_notation_python([0, 0, 0, 1], "|11⟩")


def test_dirac_notation_precision():
    sqrt = np.sqrt(0.5)
    assert_dirac_notation_numpy([sqrt, sqrt], "0.7|0⟩ + 0.7|1⟩", decimals=1)
    assert_dirac_notation_python([sqrt, sqrt],
                                 "0.707|0⟩ + 0.707|1⟩",
                                 decimals=3)


def test_to_valid_state_vector():
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex64), 2),
        np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(0, 2),
                                   np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(cirq.to_valid_state_vector(1, 2),
                                   np.array([0.0, 1.0, 0.0, 0.0]))

    v = cirq.to_valid_state_vector([0, 1, 2, 0], qid_shape=(3, 3, 3, 3))
    assert v.shape == (3**4,)
    assert v[6 + 9] == 1

    v = cirq.to_valid_state_vector([False, True, False, False], num_qubits=4)
    assert v.shape == (16,)
    assert v[4] == 1

    v = cirq.to_valid_state_vector([0, 1, 0, 0], num_qubits=2)
    assert v.shape == (4,)
    assert v[1] == 1

    v = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.complex64),
                                   qid_shape=(2, 1))
    assert v.shape == (2,)
    assert v[0] == 1


def test_to_valid_state_vector_creates_new_copy():
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex64)
    out = cirq.to_valid_state_vector(state, 2)
    assert out is not state


def test_invalid_to_valid_state_vector():
    with pytest.raises(ValueError, match="Must specify"):
        _ = cirq.to_valid_state_vector(np.array([1]))

    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(
            np.array([1.0, 0.0], dtype=np.complex64), 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(-1, 2)
    with pytest.raises(ValueError):
        _ = cirq.to_valid_state_vector(5, 2)
    with pytest.raises(TypeError, match='Unrecognized type of STATE_LIKE'):
        _ = cirq.to_valid_state_vector('0000', 2)
    with pytest.raises(TypeError, match='Unrecognized type of STATE_LIKE'):
        _ = cirq.to_valid_state_vector('not an int', 2)
    with pytest.raises(ValueError, match=r'num_qubits != len\(qid_shape\)'):
        _ = cirq.to_valid_state_vector(0, 5, qid_shape=(1, 2, 3))

    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([3], qid_shape=(3,))
    with pytest.raises(ValueError, match='out of bounds'):
        _ = cirq.to_valid_state_vector([-1], qid_shape=(3,))
    with pytest.raises(ValueError, match='but its shape was neither'):
        _ = cirq.to_valid_state_vector([], qid_shape=(3,))
    with pytest.raises(ValueError, match='but its shape was neither'):
        _ = cirq.to_valid_state_vector([0, 1], num_qubits=3)
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector([1, 0], qid_shape=(2, 1))
    with pytest.raises(ValueError, match='ambiguous'):
        _ = cirq.to_valid_state_vector(np.array([1, 0], dtype=np.int64),
                                       qid_shape=(2, 1))


def test_check_state():
    cirq.validate_normalized_state(np.array([0.5, 0.5, 0.5, 0.5],
                                            dtype=np.complex64),
                                   qid_shape=(2, 2))
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(np.array([1, 1], dtype=np.complex64),
                                       qid_shape=(2, 2))
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(np.array([1.0, 0.2, 0.0, 0.0],
                                                dtype=np.complex64),
                                       qid_shape=(2, 2))
    with pytest.raises(ValueError):
        cirq.validate_normalized_state(np.array([1.0, 0.0, 0.0, 0.0],
                                                dtype=np.float64),
                                       qid_shape=(2, 2))


def test_sample_state_big_endian():
    results = []
    for x in range(8):
        state = cirq.to_valid_state_vector(x, 3)
        sample = cirq.sample_state_vector(state, [2, 1, 0])
        results.append(sample)
    expecteds = [[list(reversed(x))] for x in
                 list(itertools.product([False, True], repeat=3))]
    for result, expected in zip(results, expecteds):
        np.testing.assert_equal(result, expected)


def test_sample_state_partial_indices():
    for index in range(3):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            np.testing.assert_equal(cirq.sample_state_vector(state, [index]),
                                    [[bool(1 & (x >> (2 - index)))]])

def test_sample_state_partial_indices_oder():
    for x in range(8):
        state = cirq.to_valid_state_vector(x, 3)
        expected = [[bool(1 & (x >> 0)), bool(1 & (x >> 1))]]
        np.testing.assert_equal(cirq.sample_state_vector(state, [2, 1]),
                                expected)


def test_sample_state_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            expected = [[bool(1 & (x >> (2 - p))) for p in perm]]
            np.testing.assert_equal(cirq.sample_state_vector(state, perm),
                                    expected)


def test_sample_state():
    state = np.zeros(8, dtype=np.complex64)
    state[0] = 1 / np.sqrt(2)
    state[2] = 1 / np.sqrt(2)
    for _ in range(10):
        sample = cirq.sample_state_vector(state, [2, 1, 0])
        assert (np.array_equal(sample, [[False, False, False]])
                or np.array_equal(sample, [[False, True, False]]))
    # Partial sample is correct.
    for _ in range(10):
        np.testing.assert_equal(cirq.sample_state_vector(state, [2]), [[False]])
        np.testing.assert_equal(cirq.sample_state_vector(state, [0]), [[False]])


def test_sample_empty_state():
    state = np.array([1.0])
    np.testing.assert_almost_equal(cirq.sample_state_vector(state, []),
        np.zeros(shape=(1,0)))


def test_sample_no_repetitions():
    state = cirq.to_valid_state_vector(0, 3)
    np.testing.assert_almost_equal(
        cirq.sample_state_vector(state, [1], repetitions=0),
        np.zeros(shape=(0, 1)))
    np.testing.assert_almost_equal(
        cirq.sample_state_vector(state, [1, 2], repetitions=0),
        np.zeros(shape=(0, 2)))


def test_sample_state_repetitions():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            state = cirq.to_valid_state_vector(x, 3)
            expected = [[bool(1 & (x >> (2 - p))) for p in perm]] * 3

            result = cirq.sample_state_vector(state, perm, repetitions=3)
            np.testing.assert_equal(result, expected)


def test_sample_state_seed():
    state = np.ones(2) / np.sqrt(2)

    samples = cirq.sample_state_vector(state, [0], repetitions=10, seed=1234)
    assert np.array_equal(samples, [[False], [True], [False], [True], [True],
                                    [False], [False], [True], [True], [True]])

    samples = cirq.sample_state_vector(state, [0],
                                       repetitions=10,
                                       seed=np.random.RandomState(1234))
    assert np.array_equal(samples, [[False], [True], [False], [True], [True],
                                    [False], [False], [True], [True], [True]])


def test_sample_state_negative_repetitions():
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(ValueError, match='-1'):
        cirq.sample_state_vector(state, [1], repetitions=-1)


def test_sample_state_not_power_of_two():
    with pytest.raises(ValueError, match='3'):
        cirq.sample_state_vector(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='5'):
        cirq.sample_state_vector(np.array([0, 1, 0, 0, 0]), [1])


def test_sample_state_index_out_of_range():
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.sample_state_vector(state, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.sample_state_vector(state, [3])


def test_sample_no_indices():
    state = cirq.to_valid_state_vector(0, 3)
    np.testing.assert_almost_equal(
        cirq.sample_state_vector(state, []), np.zeros(shape=(1, 0)))


def test_sample_no_indices_repetitions():
    state = cirq.to_valid_state_vector(0, 3)
    np.testing.assert_almost_equal(
        cirq.sample_state_vector(state, [], repetitions=2),
        np.zeros(shape=(2, 0)))


def test_measure_state_computational_basis():
    results = []
    for x in range(8):
        initial_state = cirq.to_valid_state_vector(x, 3)
        bits, state = cirq.measure_state_vector(initial_state, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(state, initial_state)
    expected = [list(reversed(x)) for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_measure_state_reshape():
    results = []
    for x in range(8):
        initial_state = np.reshape(cirq.to_valid_state_vector(x, 3), [2] * 3)
        bits, state = cirq.measure_state_vector(initial_state, [2, 1, 0])
        results.append(bits)
        np.testing.assert_almost_equal(state, initial_state)
    expected = [list(reversed(x)) for x in
                list(itertools.product([False, True], repeat=3))]
    assert results == expected


def test_measure_state_partial_indices():
    for index in range(3):
        for x in range(8):
            initial_state = cirq.to_valid_state_vector(x, 3)
            bits, state = cirq.measure_state_vector(initial_state, [index])
            np.testing.assert_almost_equal(state, initial_state)
            assert bits == [bool(1 & (x >> (2 - index)))]


def test_measure_state_partial_indices_order():
    for x in range(8):
        initial_state = cirq.to_valid_state_vector(x, 3)
        bits, state = cirq.measure_state_vector(initial_state, [2, 1])
        np.testing.assert_almost_equal(state, initial_state)
        assert bits == [bool(1 & (x >> 0)), bool(1 & (x >> 1))]


def test_measure_state_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            initial_state = cirq.to_valid_state_vector(x, 3)
            bits, state = cirq.measure_state_vector(initial_state, perm)
            np.testing.assert_almost_equal(state, initial_state)
            assert bits == [bool(1 & (x >> (2 - p))) for p in perm]


def test_measure_state_collapse():
    initial_state = np.zeros(8, dtype=np.complex64)
    initial_state[0] = 1 / np.sqrt(2)
    initial_state[2] = 1 / np.sqrt(2)
    for _ in range(10):
        bits, state = cirq.measure_state_vector(initial_state, [2, 1, 0])
        assert bits in [[False, False, False], [False, True, False]]
        expected = np.zeros(8, dtype=np.complex64)
        expected[2 if bits[1] else 0] = 1.0
        np.testing.assert_almost_equal(state, expected)
        assert state is not initial_state

    # Partial sample is correct.
    for _ in range(10):
        bits, state = cirq.measure_state_vector(initial_state, [2])
        np.testing.assert_almost_equal(state, initial_state)
        assert bits == [False]

        bits, state = cirq.measure_state_vector(initial_state, [0])
        np.testing.assert_almost_equal(state, initial_state)
        assert bits == [False]


def test_measure_state_seed():
    n = 10
    initial_state = np.ones(2**n) / 2**(n / 2)

    bits, state1 = cirq.measure_state_vector(initial_state, range(n), seed=1234)
    np.testing.assert_equal(
        bits,
        [False, False, True, True, False, False, False, True, False, False])

    bits, state2 = cirq.measure_state_vector(initial_state,
                                             range(n),
                                             seed=np.random.RandomState(1234))
    np.testing.assert_equal(
        bits,
        [False, False, True, True, False, False, False, True, False, False])

    np.testing.assert_allclose(state1, state2)


def test_measure_state_out_is_state():
    initial_state = np.zeros(8, dtype=np.complex64)
    initial_state[0] = 1 / np.sqrt(2)
    initial_state[2] = 1 / np.sqrt(2)
    bits, state = cirq.measure_state_vector(initial_state, [2, 1, 0],
                                            out=initial_state)
    expected = np.zeros(8, dtype=np.complex64)
    expected[2 if bits[1] else 0] = 1.0
    np.testing.assert_array_almost_equal(initial_state, expected)
    assert state is initial_state


def test_measure_state_out_is_not_state():
    initial_state = np.zeros(8, dtype=np.complex64)
    initial_state[0] = 1 / np.sqrt(2)
    initial_state[2] = 1 / np.sqrt(2)
    out = np.zeros_like(initial_state)
    _, state = cirq.measure_state_vector(initial_state, [2, 1, 0], out=out)
    assert out is not initial_state
    assert out is state


def test_measure_state_not_power_of_two():
    with pytest.raises(ValueError, match='3'):
        _, _ = cirq.measure_state_vector(np.array([1, 0, 0]), [1])
    with pytest.raises(ValueError, match='5'):
        cirq.measure_state_vector(np.array([0, 1, 0, 0, 0]), [1])


def test_measure_state_index_out_of_range():
    state = cirq.to_valid_state_vector(0, 3)
    with pytest.raises(IndexError, match='-2'):
        cirq.measure_state_vector(state, [-2])
    with pytest.raises(IndexError, match='3'):
        cirq.measure_state_vector(state, [3])


def test_measure_state_no_indices():
    initial_state = cirq.to_valid_state_vector(0, 3)
    bits, state = cirq.measure_state_vector(initial_state, [])
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)


def test_measure_state_no_indices_out_is_state():
    initial_state = cirq.to_valid_state_vector(0, 3)
    bits, state = cirq.measure_state_vector(initial_state, [],
                                            out=initial_state)
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)
    assert state is initial_state


def test_measure_state_no_indices_out_is_not_state():
    initial_state = cirq.to_valid_state_vector(0, 3)
    out = np.zeros_like(initial_state)
    bits, state = cirq.measure_state_vector(initial_state, [], out=out)
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)
    assert state is out
    assert out is not initial_state


def test_measure_state_empty_state():
    initial_state = np.array([1.0])
    bits, state = cirq.measure_state_vector(initial_state, [])
    assert [] == bits
    np.testing.assert_almost_equal(state, initial_state)


class BasicStateVector(cirq.StateVectorMixin):
    def state_vector(self) -> np.ndarray:
        return np.array([0, 1, 0, 0])


def test_step_result_pretty_state():
    step_result = BasicStateVector()
    assert step_result.dirac_notation() == '|01⟩'


def test_step_result_density_matrix():
    q0, q1 = cirq.LineQubit.range(2)

    step_result = BasicStateVector({q0: 0, q1: 1})
    rho = np.array([[0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho,
        step_result.density_matrix_of([q0, q1]))

    np.testing.assert_array_almost_equal(rho,
        step_result.density_matrix_of())

    rho_ind_rev = np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho_ind_rev,
        step_result.density_matrix_of([q1, q0]))

    single_rho = np.array([[0, 0],
                           [0, 1]])
    np.testing.assert_array_almost_equal(single_rho,
        step_result.density_matrix_of([q1]))


def test_step_result_density_matrix_invalid():
    q0, q1 = cirq.LineQubit.range(2)

    step_result = BasicStateVector({q0: 0})

    with pytest.raises(KeyError):
        step_result.density_matrix_of([q1])
    with pytest.raises(KeyError):
        step_result.density_matrix_of('junk')
    with pytest.raises(TypeError):
        step_result.density_matrix_of(0)


def test_step_result_bloch_vector():
    q0, q1 = cirq.LineQubit.range(2)
    step_result = BasicStateVector({q0: 0, q1: 1})
    bloch1 = np.array([0, 0, -1])
    bloch0 = np.array([0, 0, 1])
    np.testing.assert_array_almost_equal(bloch1,
        step_result.bloch_vector_of(q1))
    np.testing.assert_array_almost_equal(bloch0,
        step_result.bloch_vector_of(q0))
