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
import cirq.transformers.randomized_measurements as rand_meas
import pytest


def test_randomized_measurements_appends_two_moments_on_returned_circuit():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit_pre = cirq.Circuit(
        [cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)]
    )
    num_moments_pre = len(circuit_pre.moments)

    # Append randomized measurements to subsystem
    unitary_ensembles = ['pauli', 'clifford', 'cue']
    for u in unitary_ensembles:
        circuit_post = rand_meas.RandomizedMeasurements()(circuit_pre, unitary_ensemble=u)
        num_moments_post = len(circuit_post.moments)
        assert num_moments_post == num_moments_pre + 2


def test_append_randomized_measurements_leaves_qubits_not_in_specified_subsystem_unchanged():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    circuit = rand_meas.RandomizedMeasurements(subsystem=(0, 1))(circuit)
    # assert latter subsystems were not changed.
    assert circuit.operation_at(q2, 4) is None
    assert circuit.operation_at(q3, 4) is None


def test_append_randomized_measurements_leaves_qubits_not_in_noncontinuous_subsystem_unchanged():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    circuit = rand_meas.RandomizedMeasurements(subsystem=(0, 2))(circuit)

    # assert latter subsystems were not changed.
    assert circuit.operation_at(q1, 4) is None
    assert circuit.operation_at(q3, 4) is None


def test_exception():
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    with pytest.raises(ValueError):
        rand_meas.RandomizedMeasurements()(circuit, unitary_ensemble="coe")
