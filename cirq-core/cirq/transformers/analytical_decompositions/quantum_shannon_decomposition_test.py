# Copyright 2023 The Cirq Developers
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

import cirq
from cirq.ops import common_gates
from cirq.transformers.analytical_decompositions.quantum_shannon_decomposition import (
    _multiplexed_cossin,
    _nth_gray,
    _msb_demuxer,
    _single_qubit_decomposition,
    quantum_shannon_decomposition,
)

import pytest
import numpy as np
from scipy.stats import unitary_group


@pytest.mark.parametrize('n_qubits', list(range(1, 8)))
def test_random_qsd_n_qubit(n_qubits):
    U = unitary_group.rvs(2**n_qubits)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(n_qubits)]
    circuit = cirq.Circuit(quantum_shannon_decomposition(qubits, U))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-9)
    # Test all operations in gate set
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all(isinstance(op.gate, gates) for op in circuit.all_operations())


def test_qsd_n_qubit_errors():
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(3)]
    with pytest.raises(ValueError, match="shaped numpy array"):
        cirq.Circuit(quantum_shannon_decomposition(qubits, np.eye(9)))
    with pytest.raises(ValueError, match="is_unitary"):
        cirq.Circuit(quantum_shannon_decomposition(qubits, np.ones((8, 8))))


def test_random_single_qubit_decomposition():
    U = unitary_group.rvs(2)
    qubit = cirq.NamedQubit('q0')
    circuit = cirq.Circuit(_single_qubit_decomposition(qubit, U))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-9)
    # Test all operations in gate set
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all(isinstance(op.gate, gates) for op in circuit.all_operations())


def test_msb_demuxer():
    U1 = unitary_group.rvs(4)
    U2 = unitary_group.rvs(4)
    U_full = np.kron([[1, 0], [0, 0]], U1) + np.kron([[0, 0], [0, 1]], U2)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(3)]
    circuit = cirq.Circuit(_msb_demuxer(qubits, U1, U2))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(U_full, circuit.unitary(), atol=1e-9)
    # Test all operations in gate set
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all(isinstance(op.gate, gates) for op in circuit.all_operations())


def test_multiplexed_cossin():
    angle_1 = np.random.random_sample() * 2 * np.pi
    angle_2 = np.random.random_sample() * 2 * np.pi
    c1, s1 = np.cos(angle_1), np.sin(angle_1)
    c2, s2 = np.cos(angle_2), np.sin(angle_2)
    multiplexed_ry = [[c1, 0, -s1, 0], [0, c2, 0, -s2], [s1, 0, c1, 0], [0, s2, 0, c2]]
    multiplexed_ry = np.array(multiplexed_ry)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(2)]
    circuit = cirq.Circuit(_multiplexed_cossin(qubits, [angle_1, angle_2]))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(multiplexed_ry, circuit.unitary(), atol=1e-9)
    # Test all operations in gate set
    gates = (common_gates.Rz, common_gates.Ry, common_gates.ZPowGate, common_gates.CXPowGate)
    assert all(isinstance(op.gate, gates) for op in circuit.all_operations())


@pytest.mark.parametrize(
    'n, gray',
    [
        (0, 0),
        (1, 1),
        (2, 3),
        (3, 2),
        (4, 6),
        (5, 7),
        (6, 5),
        (7, 4),
        (8, 12),
        (9, 13),
        (10, 15),
        (11, 14),
        (12, 10),
        (13, 11),
        (14, 9),
        (15, 8),
    ],
)
def test_nth_gray(n, gray):
    assert _nth_gray(n) == gray
