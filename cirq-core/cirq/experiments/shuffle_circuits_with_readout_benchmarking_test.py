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

import cirq

def test_qubit_extraction():
    qubits = cirq.LineQubit.range(10)
    circuits = []
    circuits += [cirq.Circuit(cirq.X(qubits[0]))]
    circuits+= [cirq.Circuit(cirq.I(qubits[1]))]
    circuits+= [cirq.Circuit(cirq.H(qubits[2]))]
    circuits+= [cirq.Circuit(cirq.CNOT(qubits[3], qubits[4]))]
    num_random_bitstrings = 5
    result = cirq.experiments.run_shuffled_with_readout_benchmarking(
        circuits=circuits, num_random_bitstrings=num_random_bitstrings
    )
    # Check that the number of circuits is correct
    assert len(result) == 5
    # Check that the number of qubits and gates in each circuit is correct
    for res_circuit in result:
        assert res_circuit.all_qubits() == set(qubits[:5]) # All circuits operate on the expected qubits
        assert len(res_circuit) == 2  # Each circuit has the correct number of moments (X/I gates + measurement)
