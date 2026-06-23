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

import cirq.experiments.random_quantum_circuit_generation as rcg
import cirq.transformers as ct
from cirq import circuits, devices, ops


def test_lightcone_filter():
    qubits = devices.GridQubit.square(5)
    terminal_measured_qubits = set(qubits[:10])
    depth = 5
    circuit = rcg.random_rotations_between_grid_interaction_layers_circuit(qubits, depth, seed=0)
    circuit.append(ops.M(terminal_measured_qubits))
    # add a midcircuit measurement:
    circuit = circuit[:5] + circuits.Circuit(ops.M(devices.GridQubit(3, 0))) + circuit[5:]
    circuit_filtered = ct.lightcone_filter(circuit)

    assert [len(moment) for moment in circuit] == [25, 10, 25, 10, 25, 1, 10, 25, 10, 25, 10, 25, 1]
    assert [len(moment) for moment in circuit_filtered] == [
        19,
        8,
        15,
        6,
        11,
        1,
        4,
        10,
        4,
        10,
        4,
        10,
        1,
    ]
    for old_moment, new_moment in zip(circuit, circuit_filtered):
        assert set(new_moment.operations).issubset(old_moment.operations)
        for op in set(old_moment.operations).difference(new_moment.operations):
            assert terminal_measured_qubits.isdisjoint(op.qubits)
