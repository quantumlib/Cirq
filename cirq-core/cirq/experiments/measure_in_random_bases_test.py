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
import cirq.experiments.measure_in_random_bases as mrb


def test_append_randomized_measurements_generates_n_circuits():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements
    circuits = mrb.append_randomized_measurements(circuit)

    assert len(circuits) == 4  # num of qubits


def test_append_randomized_measurements_appends_two_moments_to_end_of_circuit():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    num_moments_pre = len(circuit.moments)

    # Append randomized measurements
    circuits = mrb.append_randomized_measurements(circuit)

    for circuit in circuits:
        num_moments_post = len(circuit.moments)
        assert num_moments_post == num_moments_pre + 2  # 1 random gate and 1 measurement gate


def test_append_randomized_measurements_generates_n_circuits_when_passed_subystem_arg():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, subsystem=(0, 1))

    assert len(circuits) == 4


def test_append_randomized_measurements_leaves_qubits_not_in_specified_subsystem_unchanged():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, subsystem=(0, 1))

    for circuit in circuits:
        # assert latter subsystems were not changed.
        assert circuit.operation_at(q2, 4) == cirq.I(q2)
        assert circuit.operation_at(q3, 4) == cirq.I(q3)


def test_append_randomized_measurements_generates_circuits_only_for_passed_qubit_mapping():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, qubits=[q0, q1])

    assert len(circuits) == 2


def test_append_randomized_measurements_appends_two_moments_to_specified_qubits():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])
    num_moments_pre = len(circuit.moments)

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, qubits=[q0, q1])

    for circuit in circuits:
        num_moments_post = len(circuit.moments)
        assert num_moments_post == num_moments_pre + 2  # 1 random gate and 1 measurement gate


def test_append_randomized_measurements_with_num_unitaries_generates_k_circuits():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])
    num_unitaries = 3

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, num_unitaries=num_unitaries)

    assert len(circuits) == num_unitaries


def test_append_randomized_measurements_with_num_unitaries_appends_two_moments_on_each_circuit():
    # Create a 4-qubit circuit
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CNOT(q0, q1), cirq.CNOT(q1, q2), cirq.CNOT(q2, q3)])
    num_moments_pre = len(circuit.moments)
    num_unitaries = 3

    # Append randomized measurements to subsystem
    circuits = mrb.append_randomized_measurements(circuit, num_unitaries=num_unitaries)

    for circuit in circuits:
        num_moments_post = len(circuit.moments)
        assert num_moments_post == num_moments_pre + 2
