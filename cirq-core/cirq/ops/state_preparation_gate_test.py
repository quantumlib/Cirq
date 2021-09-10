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
    'target_state',
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
def test_state_prep_gate(target_state):
    gate = cirq.StatePreparationGate(target_state)
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [
            cirq.H(qubits[0]),
            cirq.CNOT(qubits[0], qubits[1]),
            gate(qubits[0], qubits[1]),
        ]
    )
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits).final_state_vector
    assert np.allclose(result, target_state / np.linalg.norm(target_state))


def test_state_prep_gate_printing():
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.StatePreparationGate(np.array([1, 0, 0, 1]) / np.sqrt(2))
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
    gate = cirq.StatePreparationGate(np.array([1, 0, 0, 1]) / np.sqrt(2), name=name)
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
    gate = cirq.StatePreparationGate(state)
    assert gate.num_qubits() == 2
    assert not gate._has_unitary_()
    assert (
        repr(gate)
        == 'cirq.StatePreparationGate(np.array([(1+0j), 0j, 0j, 0j], dtype=np.complex128))'
    )


def test_gate_error_handling():
    with pytest.raises(ValueError, match='`target_state` must be a 1d numpy array.'):
        cirq.StatePreparationGate(np.eye(2))
    with pytest.raises(ValueError, match=f'Matrix width \\(5\\) is not a power of 2'):
        cirq.StatePreparationGate(np.ones(shape=5))
