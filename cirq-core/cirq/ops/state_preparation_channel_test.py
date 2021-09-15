# Copyright 2021 The Cirq Developers
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
import cirq
import pytest


@pytest.mark.parametrize(
    'state',
    np.array(
        [
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [3, 5, 2, 7],
            [0.7823, 0.12323, 0.4312, 0.12321],
            [23, 43, 12, 19],
            [1j, 0, 0, 0],
            [1j, 0, 0, 1j],
            [1j, -1j, -1j, 1j],
            [1 + 1j, 0, 0, 0],
            [1 + 1j, 0, 1 + 1j, 0],
            [3 + 1j, 5 + 8j, 21, 0.85j],
        ]
    ),
)
def test_state_prep_channel_kraus(state):
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(state)(qubits[0], qubits[1])
    state = state / np.linalg.norm(state)
    np.testing.assert_almost_equal(
        cirq.kraus(gate),
        (
            np.array([state, np.zeros(4), np.zeros(4), np.zeros(4)]).T,
            np.array([np.zeros(4), state, np.zeros(4), np.zeros(4)]).T,
            np.array([np.zeros(4), np.zeros(4), state, np.zeros(4)]).T,
            np.array([np.zeros(4), np.zeros(4), np.zeros(4), state]).T,
        ),
    )


def test_state_prep_channel_kraus_small():
    gate = cirq.StatePreparationChannel(np.array([0.0, 1.0]))(cirq.LineQubit(0))
    np.testing.assert_almost_equal(
        cirq.kraus(gate), (np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 1.0]]))
    )
    assert cirq.has_kraus(gate)
    assert not cirq.has_mixture(gate)

    gate = cirq.StatePreparationChannel(np.array([1.0, 0.0]))(cirq.LineQubit(0))
    np.testing.assert_almost_equal(
        cirq.kraus(gate), (np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 1.0], [0.0, 0.0]]))
    )
    assert cirq.has_kraus(gate)
    assert not cirq.has_mixture(gate)


def test_state_prep_gate_printing():
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(np.array([1, 0, 0, 1]) / np.sqrt(2))
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(gate(qubits[0], qubits[1]))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───H───@───StatePreparation[1]───
          │   │
1: ───────X───StatePreparation[2]───
""",
    )


@pytest.mark.parametrize('name', ['Prep', 'S'])
def test_state_prep_gate_printing_with_name(name):
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationChannel(np.array([1, 0, 0, 1]) / np.sqrt(2), name=name)
    circuit.append(cirq.H(qubits[0]))
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(gate(qubits[0], qubits[1]))
    cirq.testing.assert_has_diagram(
        circuit,
        f"""
0: ───H───@───{name}[1]───
          │   │
1: ───────X───{name}[2]───
""",
    )


def test_gate_params():
    state = np.array([1, 0, 0, 0], dtype=np.complex64)
    gate = cirq.StatePreparationChannel(state)
    assert gate.num_qubits() == 2
    assert not gate._has_unitary_()
    assert gate._has_kraus_()
    assert (
        repr(gate)
        == 'cirq.StatePreparationChannel(np.array([(1+0j), 0j, 0j, 0j], dtype=np.complex128))'
    )


def test_gate_error_handling():
    with pytest.raises(ValueError, match='`target_state` must be a 1d numpy array.'):
        cirq.StatePreparationChannel(np.eye(2))
    with pytest.raises(ValueError, match=f'Matrix width \\(5\\) is not a power of 2'):
        cirq.StatePreparationChannel(np.ones(shape=5))


def test_equality_of_gates():
    state = np.array([1, 0, 0, 0], dtype=np.complex64)
    gate_1 = cirq.StatePreparationChannel(state)
    gate_2 = cirq.StatePreparationChannel(state)
    assert gate_1 == gate_2, "Equal state not leading to same gate"
    assert not gate_1 == state, "Incompatible objects shouldn't be equal"
