# Copyright 2018 Google LLC
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
from cirq.examples import generate_supremacy_circuit
from cirq.google import Exp11Gate


def test_generate_supremacy_circuit():
    device = cirq.google.Foxtail

    circuit = generate_supremacy_circuit(device, cz_depth=6)
    # Circuit should have 6 layers of 3 plus a final layer of 2.
    assert len(circuit.moments) == 20

    # For this chip, by cz-depth 6 there should be one CZ on each edge.
    op_counts = {}
    for m in circuit.moments:
        for op in m.operations:
            op_counts[op] = op_counts.get(op, 0) + 1
    for q1 in device.qubits:
        for q2 in device.neighbors_of(q1):
            assert op_counts[Exp11Gate().on(q1, q2)] == 1
