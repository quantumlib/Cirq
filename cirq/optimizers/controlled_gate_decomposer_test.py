# Copyright 2020 The Cirq Developers
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

import pytest

import numpy as np
import scipy.stats

import cirq

_DECOMPOSER_TOFFOLI = cirq.ControlledGateDecomposer(allow_toffoli=True)
_DECOMPOSER_FULL = cirq.ControlledGateDecomposer(allow_toffoli=False)

_CNOT_MATRIX = cirq.CNOT._unitary_()
_CCNOT_MATRIX = cirq.CCNOT._unitary_()


def _test_decompose_x_with(decomposer):
    """Verifies correctness of multi-controlled X decomposition."""
    for total_qubits_count in range(1, 8):
        qubits = cirq.LineQubit.range(total_qubits_count)
        for controls_count in range(0, total_qubits_count):
            gates = decomposer.decompose_x(qubits[:controls_count],
                                           qubits[controls_count],
                                           free_qubits=qubits[controls_count +
                                                              1:])

            circuit1 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit1.append(gates)
            result_matrix = circuit1.unitary()

            circuit2 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit2 += cirq.ControlledGate(
                cirq.X,
                num_controls=controls_count).on(*qubits[0:controls_count + 1])
            expected_matrix = circuit2.unitary()
            assert np.allclose(expected_matrix, result_matrix)


def test_decompose_x():
    _test_decompose_x_with(_DECOMPOSER_TOFFOLI)
    _test_decompose_x_with(_DECOMPOSER_FULL)


def _random_unitary():
    return scipy.stats.unitary_group.rvs(2)


def _random_special_unitary():
    U = _random_unitary()
    return U / np.sqrt(np.linalg.det(U))


def _validate_matrix(u, allow_toffoli=False):
    """Checks that matrix is allowed in decomposition.

    That is, it's any 2x2 matrix or CNOT matrix.
    If allow_toffoli=True, also allow CCNOT matrix.
    """
    if u.shape == (2, 2):
        pass
    elif u.shape == (4, 4):
        assert np.allclose(u, _CNOT_MATRIX)
    elif u.shape == (8, 8):
        assert allow_toffoli and np.allclose(u, _CCNOT_MATRIX)
    else:
        raise AssertionError('Bad matrix shape')


def _test_decompose_with(matrix, controls_count, decomposer):
    qubits = cirq.LineQubit.range(controls_count + 1)
    gates = decomposer.decompose(matrix, qubits[:-1], qubits[-1])
    for gate in gates:
        _validate_matrix(gate._unitary_(), decomposer.allow_toffoli)
    result_matrix = cirq.Circuit(gates).unitary()

    d = 2**(controls_count + 1)
    expected_matrix = np.eye(d, dtype=np.complex128)
    expected_matrix[d - 2:d, d - 2:d] = matrix

    assert np.allclose(expected_matrix, result_matrix)


def _test_decompose(matrix, controls_count):
    """Test decomposition with given 2x2 matrix and number of controls."""
    _test_decompose_with(matrix, controls_count, _DECOMPOSER_FULL)
    _test_decompose_with(matrix, controls_count, _DECOMPOSER_TOFFOLI)


def test_decompose_specific_matrices():
    for gate in [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.I, cirq.T]:
        for controls_count in range(1, 7):
            _test_decompose(gate._unitary_(), controls_count)


def test_decompose_random_unitary():
    for controls_count in range(1, 8):
        _test_decompose(_random_unitary(), controls_count)


def test_decompose_random_special_unitary():
    for controls_count in range(1, 8):
        _test_decompose(_random_special_unitary(), controls_count)


def _decomposition_size(U, controls_count, dec):
    qubits = cirq.LineQubit.range(controls_count + 1)
    return len(dec.decompose(U, qubits[:controls_count], qubits[-1]))


def test_decompose_size_special_unitary_with_toffoli():
    matrix = _random_special_unitary()
    assert _decomposition_size(matrix, 1, _DECOMPOSER_TOFFOLI) == 7
    assert _decomposition_size(matrix, 2, _DECOMPOSER_TOFFOLI) == 20
    assert _decomposition_size(matrix, 3, _DECOMPOSER_TOFFOLI) == 20
    assert _decomposition_size(matrix, 4, _DECOMPOSER_TOFFOLI) == 26
    assert _decomposition_size(matrix, 5, _DECOMPOSER_TOFFOLI) == 38
    for controls_count in range(6, 50):
        assert _decomposition_size(
            matrix, controls_count,
            _DECOMPOSER_TOFFOLI) == 16 * controls_count - 46


def test_decompose_size_special_unitary_without_toffoli():
    matrix = _random_special_unitary()
    assert _decomposition_size(matrix, 1, _DECOMPOSER_FULL) == 7
    assert _decomposition_size(matrix, 2, _DECOMPOSER_FULL) == 20
    assert _decomposition_size(matrix, 3, _DECOMPOSER_FULL) == 48
    assert _decomposition_size(matrix, 4, _DECOMPOSER_FULL) == 106
    assert _decomposition_size(matrix, 5, _DECOMPOSER_FULL) == 254
    for controls_count in range(6, 50):
        assert _decomposition_size(
            matrix, controls_count,
            _DECOMPOSER_FULL) == 112 * controls_count - 302
