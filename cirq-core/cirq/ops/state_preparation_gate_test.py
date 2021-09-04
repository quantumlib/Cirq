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

import numpy as np

import cirq


def test_state_prep_gate():
    state = np.array([1, 0, 0, 1]) / np.sqrt(2)
    gate = cirq.PrepareState(state)
    qubits = cirq.LineQubit.range(2)

    circuit = cirq.Circuit([
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        gate(qubits[0], qubits[1]),
    ])

    simulator = cirq.Simulator()
    result = simulator.simulate(circuit, qubit_order=qubits).final_state_vector
    assert np.allclose(result, state)


def test_state_prep_gate_printing():
    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(2)
    gate = cirq.PrepareState(np.array([1, 0, 0, 1]) / np.sqrt(2))
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
