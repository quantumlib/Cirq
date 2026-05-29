# Copyright 2026 The Cirq Developers
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

import cirq.transformers as ct
from cirq import devices, ops, circuits
import  cirq.experiments.random_quantum_circuit_generation as rcg

def test_lightcone_filter():
    qubits = devices.GridQubit.rect(5,5)
    depth = 5
    circuit = rcg.random_rotations_between_grid_interaction_layers_circuit(qubits, depth, seed=0)
    circuit += circuits.Circuit(ops.M(qubits[:10]))
    assert [len(moment) for moment in circuit] == [25, 10, 25, 10, 25, 10, 25, 10, 25, 10, 25, 1]
    new_circuit = ct.lightcone_filter(circuit)
    assert [len(moment) for moment in new_circuit] == [18, 8, 13, 5, 10, 4, 10, 4, 10, 4, 10, 1]