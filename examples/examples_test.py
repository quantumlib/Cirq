# Copyright 2018 The Cirq Developers
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
import examples.bell_inequality
import examples.bernstein_vazirani
import examples.grover
import examples.place_on_bristlecone
import examples.hello_qubit
import examples.quantum_fourier_transform
import examples.bcs_mean_field
import examples.supremacy
import examples.phase_estimator
import examples.basic_arithmetic
from examples.supremacy import generate_supremacy_circuit


def test_generate_supremacy_circuit():
    device = cirq.google.Foxtail

    circuit = examples.supremacy.generate_supremacy_circuit(device, cz_depth=6)
    # Circuit should have 6 layers of 2 plus a final layer of 1 plus measures.
    assert len(circuit) == 14

    # For this chip, by cz-depth 6 there should be one CZ on each edge.
    op_counts = {}
    for m in circuit:
        for op in m.operations:
            op_counts[op] = op_counts.get(op, 0) + 1
    for q1 in device.qubits:
        for q2 in device.neighbors_of(q1):
            assert op_counts[cirq.google.Exp11Gate().on(q1, q2)] == 1


def test_generate_supremacy_circuit_seeding():
    device = cirq.google.Foxtail

    circuit1 = examples.supremacy.generate_supremacy_circuit(
        device, cz_depth=6, seed=42)
    circuit2 = examples.supremacy.generate_supremacy_circuit(
        device, cz_depth=6, seed=42)
    circuit3 = examples.supremacy.generate_supremacy_circuit(
        device, cz_depth=6, seed=43)

    assert circuit1 == circuit2
    assert circuit1 != circuit3


def test_example_runs_bernstein_vazirani():
    examples.bernstein_vazirani.main()

    # Check empty oracle case. Cover both biases.
    a = cirq.NamedQubit('a')
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], False)) == []
    assert list(examples.bernstein_vazirani.make_oracle(
        [], a, [], True)) == [cirq.X(a)]


def test_example_runs_hello_line():
    examples.place_on_bristlecone.main()


def test_example_runs_hello_qubit():
    examples.hello_qubit.main()


def test_example_runs_bell_inequality():
    examples.bell_inequality.main()


def test_example_runs_quantum_fourier_transform():
    examples.quantum_fourier_transform.main()


def test_example_runs_bcs_mean_field():
    examples.bcs_mean_field.main()


def test_example_runs_grover():
    examples.grover.main()


def test_example_runs_phase_estimator():
    examples.phase_estimator.main()


def test_example_runs_basic_arithmetic():
    examples.basic_arithmetic.main(n=2)
