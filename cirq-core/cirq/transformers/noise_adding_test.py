# Copyright 2024 The Cirq Developers
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
import cirq.transformers.noise_adding as na


def test_noise_adding():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.Circuit(cirq.CZ(*qubits[:2]), cirq.CZ(*qubits[2:])) * 10
    transformed_circuit_p0 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 0.0)
    assert transformed_circuit_p0 == circuit
    transformed_circuit_p1 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 1.0)
    assert len(transformed_circuit_p1) == 20
    transformed_circuit_p0_03 = na.add_depolarizing_noise_to_two_qubit_gates(circuit, 0.03)
