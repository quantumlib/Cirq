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


@pytest.mark.xfail(reason='#6765')
@pytest.mark.parametrize('n_qubits', list(range(1, 8)))
def test_random_qsd_n_qubit(n_qubits):
    U = unitary_group.rvs(2**n_qubits)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(n_qubits)]
    circuit = cirq.Circuit(quantum_shannon_decomposition(qubits, U))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(U, circuit.unitary(), atol=1e-9)
    # Test all operations have at most 2 qubits.
    assert all(cirq.num_qubits(op) <= 2 for op in circuit.all_operations())


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
    # Test all operations have at most 2 qubits.
    assert all(cirq.num_qubits(op) <= 2 for op in circuit.all_operations())


def test_msb_demuxer():
    U1 = unitary_group.rvs(4)
    U2 = unitary_group.rvs(4)
    U_full = np.kron([[1, 0], [0, 0]], U1) + np.kron([[0, 0], [0, 1]], U2)
    qubits = [cirq.NamedQubit(f'q{i}') for i in range(3)]
    circuit = cirq.Circuit(_msb_demuxer(qubits, U1, U2))
    # Test return is equal to inital unitary
    assert cirq.approx_eq(U_full, circuit.unitary(), atol=1e-9)
    # Test all operations have at most 2 qubits.
    assert all(cirq.num_qubits(op) <= 2 for op in circuit.all_operations())


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


def test_ghz_circuit_decomposes():
    # Test case from #6725
    ghz_circuit = cirq.Circuit(cirq.H(cirq.q(0)), cirq.CNOT(cirq.q(0), cirq.q(1)))
    ghz_unitary = cirq.unitary(ghz_circuit)
    decomposed_circuit = cirq.Circuit(
        quantum_shannon_decomposition(cirq.LineQubit.range(2), ghz_unitary)
    )
    new_unitary = cirq.unitary(decomposed_circuit)
    np.testing.assert_allclose(new_unitary, ghz_unitary, atol=1e-6)


def test_qft_decomposes():
    # Test case from #6666
    qs = cirq.LineQubit.range(4)
    qft_circuit = cirq.Circuit(cirq.qft(*qs))
    qft_unitary = cirq.unitary(qft_circuit)
    decomposed_circuit = cirq.Circuit(quantum_shannon_decomposition(qs, qft_unitary))
    new_unitary = cirq.unitary(decomposed_circuit)
    np.testing.assert_allclose(new_unitary, qft_unitary, atol=1e-6)


# Cliffords test the different corner cases of the ZYZ decomposition.
@pytest.mark.parametrize(
    ['gate', 'num_ops'],
    [
        (cirq.I, 0),
        (cirq.X, 2),  # rz & ry
        (cirq.Y, 1),  # ry
        (cirq.Z, 1),  # rz
        (cirq.H, 2),  # rz & ry
        (cirq.S, 1),  # rz & ry
    ],
)
def test_cliffords(gate, num_ops):
    desired_unitary = cirq.unitary(gate)
    shannon_circuit = cirq.Circuit(quantum_shannon_decomposition((cirq.q(0),), desired_unitary))
    new_unitary = cirq.unitary(shannon_circuit)
    assert len([*shannon_circuit.all_operations()]) == num_ops
    if num_ops:
        np.testing.assert_allclose(new_unitary, desired_unitary)


@pytest.mark.parametrize('gate', [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S])
def test_cliffords_with_global_phase(gate):
    global_phase = np.exp(1j * np.random.choice(np.linspace(0.1, 2 * np.pi, 10)))
    desired_unitary = cirq.unitary(gate) * global_phase
    shannon_circuit = cirq.Circuit(quantum_shannon_decomposition((cirq.q(0),), desired_unitary))
    new_unitary = cirq.unitary(shannon_circuit)
    np.testing.assert_allclose(new_unitary, desired_unitary)


def test_global_phase():
    global_phase = np.exp(1j * np.random.choice(np.linspace(0, 2 * np.pi, 10)))
    shannon_circuit = cirq.Circuit(
        quantum_shannon_decomposition((cirq.q(0),), np.eye(2) * global_phase)
    )
    new_unitary = cirq.unitary(shannon_circuit)
    np.testing.assert_allclose(np.diag(new_unitary), global_phase)


@pytest.mark.parametrize('gate', [cirq.CZ, cirq.CNOT, cirq.XX, cirq.YY, cirq.ZZ])
def test_two_qubit_gate(gate):
    global_phase = np.exp(1j * np.random.choice(np.linspace(0, 2 * np.pi, 10)))
    desired_unitary = cirq.unitary(gate) * global_phase
    shannon_circuit = cirq.Circuit(
        quantum_shannon_decomposition(cirq.LineQubit.range(2), desired_unitary)
    )
    new_unitary = cirq.unitary(shannon_circuit)
    np.testing.assert_allclose(new_unitary, desired_unitary, atol=1e-6)


@pytest.mark.parametrize('gate', [cirq.CCNOT, cirq.qft(*cirq.LineQubit.range(3))])
def test_three_qubit_gate(gate):
    global_phase = np.exp(1j * np.random.choice(np.linspace(0, 2 * np.pi, 10)))
    desired_unitary = cirq.unitary(gate) * global_phase
    shannon_circuit = cirq.Circuit(
        quantum_shannon_decomposition(cirq.LineQubit.range(3), desired_unitary)
    )
    new_unitary = cirq.unitary(shannon_circuit)
    np.testing.assert_allclose(new_unitary, desired_unitary, atol=1e-6)


@pytest.mark.xfail(reason='#6765')
def test_qft5():
    global_phase = np.exp(1j * np.random.choice(np.linspace(0, 2 * np.pi, 10)))
    desired_unitary = cirq.unitary(cirq.qft(*cirq.LineQubit.range(5))) * global_phase
    shannon_circuit = cirq.Circuit(
        quantum_shannon_decomposition(cirq.LineQubit.range(5), desired_unitary)
    )
    new_unitary = cirq.unitary(shannon_circuit)
    np.testing.assert_allclose(new_unitary, desired_unitary, atol=1e-6)
