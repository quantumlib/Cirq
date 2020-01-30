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

import numpy as np
import scipy.stats

import cirq


def test_decompose_x():
    """Verifies correctness of multi-controlled X decomposition."""
    for total_qubits_count in range(1, 8):
        qubits = cirq.LineQubit.range(total_qubits_count)
        for controls_count in range(0, total_qubits_count):
            gates = cirq.decompose_multi_controlled_x(
                qubits[:controls_count], qubits[controls_count],
                qubits[controls_count + 1:])

            circuit1 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit1.append(gates)
            result_matrix = circuit1.unitary()

            circuit2 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit2 += cirq.ControlledGate(
                cirq.X,
                num_controls=controls_count).on(*qubits[0:controls_count + 1])
            expected_matrix = circuit2.unitary()
            assert np.allclose(expected_matrix, result_matrix)


def _random_unitary():
    return scipy.stats.unitary_group.rvs(2)


def _random_special_unitary():
    U = _random_unitary()
    return U / np.sqrt(np.linalg.det(U))


def _count_operations(operations):
    """Counts single-qubit, CNOT and CCNOT gates.

    Also validates that there are no other gates."""
    count_2x2 = 0
    count_cnot = 0
    count_ccnot = 0
    for operation in operations:
        u = cirq.unitary(operation)
        if u.shape == (2, 2):
            count_2x2 += 1
        elif u.shape == (4, 4):
            assert np.allclose(u, cirq.unitary(cirq.CNOT))
            count_cnot += 1
        elif u.shape == (8, 8):
            assert np.allclose(u, cirq.unitary(cirq.CCNOT))
            count_ccnot += 1
    return count_2x2, count_cnot, count_ccnot


def _test_decompose(matrix, controls_count):
    qubits = cirq.LineQubit.range(controls_count + 1)
    operations = cirq.decompose_multi_controlled_unitary(
        matrix, qubits[:-1], qubits[-1])
    _count_operations(operations)
    result_matrix = cirq.Circuit(operations).unitary()

    expected_matrix = cirq.Circuit([
        cirq.MatrixGate(matrix).on(qubits[-1]).controlled_by(*qubits[:-1])
    ]).unitary()

    assert np.allclose(expected_matrix, result_matrix)


def test_decompose_specific_matrices():
    for gate in [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.I, cirq.T, cirq.S]:
        for controls_count in range(0, 7):
            _test_decompose(cirq.unitary(gate), controls_count)


def test_decompose_random_unitary():
    np.random.seed(0)
    for controls_count in range(0, 5):
        for _ in range(10):
            _test_decompose(_random_unitary(), controls_count)
    for controls_count in range(5, 8):
        _test_decompose(_random_unitary(), controls_count)


def test_decompose_random_special_unitary():
    np.random.seed(0)
    for controls_count in range(0, 5):
        for _ in range(10):
            _test_decompose(_random_special_unitary(), controls_count)
    for controls_count in range(5, 8):
        _test_decompose(_random_special_unitary(), controls_count)


def _decomposition_size(U, controls_count):
    qubits = cirq.LineQubit.range(controls_count + 1)
    operations = cirq.decompose_multi_controlled_unitary(
        U, qubits[:controls_count], qubits[-1])
    return _count_operations(operations)


def test_decompose_size_special_unitary():
    matrix = _random_special_unitary()
    assert _decomposition_size(matrix, 0) == (1, 0, 0)
    assert _decomposition_size(matrix, 1) == (3, 2, 0)
    assert _decomposition_size(matrix, 2) == (8, 8, 0)
    assert _decomposition_size(matrix, 3) == (8, 6, 2)
    assert _decomposition_size(matrix, 4) == (24, 18, 4)
    assert _decomposition_size(matrix, 5) == (40, 30, 12)
    for i in range(6, 20):
        assert _decomposition_size(matrix,
                                   i) == (64 * i - 312, 48 * i - 234, 16)
