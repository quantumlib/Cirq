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

from math import exp

import pytest
from google.protobuf.text_format import Merge

import cirq
from cirq.testing import assert_equivalent_op_tree
import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models import (
    simple_noise_from_calibration_metrics,
)

# Fake calibration data object.
_CALIBRATION_DATA = Merge(
    """
    timestamp_ms: 1579214873,
    metrics: [{
        name: 'xeb',
        targets: ['0_0', '0_1'],
        values: [{
            double_val: .9999
        }]
    }, {
        name: 'xeb',
        targets: ['0_0', '1_0'],
        values: [{
            double_val: .9998
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_0'],
        values: [{
            double_val: .001
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['0_1'],
        values: [{
            double_val: .002
        }]
    }, {
        name: 'single_qubit_rb_pauli_error_per_gate',
        targets: ['1_0'],
        values: [{
            double_val: .003
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['0_0'],
        values: [{
            double_val: .004
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['0_1'],
        values: [{
            double_val: .005
        }]
    }, {
        name: 'single_qubit_readout_separation_error',
        targets: ['1_0'],
        values: [{
            double_val: .006
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_0'],
        values: [{
            double_val: .007
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['0_1'],
        values: [{
            double_val: .008
        }]
    }, {
        name: 'single_qubit_idle_t1_micros',
        targets: ['1_0'],
        values: [{
            double_val: .009
        }]
    }]
""",
    v2.metrics_pb2.MetricsSnapshot(),
)

DEPOL_001 = 0.001 * 4 / 3
DEPOL_002 = 0.002 * 4 / 3
DEPOL_003 = 0.003 * 4 / 3


def test_noise_from_metrics_requires_type():
    # Attempt to generate a noise model without specifying a noise type.
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    with pytest.raises(ValueError, match='At least one error type must be specified.'):
        simple_noise_from_calibration_metrics(calibration=calibration)


def test_noise_from_metrics_unsupported():
    # Attempt to generate a damping noise model (not yet supported).
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    with pytest.raises(NotImplementedError, match='Gate damping is not yet supported.'):
        simple_noise_from_calibration_metrics(calibration=calibration, damping_noise=True)


def test_per_qubit_depol_noise_from_data():
    # Generate the depolarization noise model from calibration data.
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    noise_model = simple_noise_from_calibration_metrics(calibration=calibration, depol_noise=True)

    # Create the circuit and apply the noise model.
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment([cirq.Z(qubits[1]).with_tags(cirq.VirtualTag())]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly to construct expected output.
    expected_program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.DepolarizingChannel(DEPOL_001).on(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment(
            [
                cirq.DepolarizingChannel(DEPOL_001).on(qubits[0]),
                cirq.DepolarizingChannel(DEPOL_002).on(qubits[1]),
            ]
        ),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.DepolarizingChannel(DEPOL_001).on(qubits[0]),
                cirq.DepolarizingChannel(DEPOL_003).on(qubits[2]),
            ]
        ),
        cirq.Moment([cirq.Z(qubits[1]).with_tags(cirq.VirtualTag())]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    assert_equivalent_op_tree(expected_program, noisy_circuit)


def test_per_qubit_readout_error_from_data():
    # Generate the readout error noise model from calibration data.
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    noise_model = simple_noise_from_calibration_metrics(
        calibration=calibration, readout_error_noise=True
    )

    # Create the circuit and apply the noise model.
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly to construct expected output.
    expected_program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.BitFlipChannel(0.004).on(qubits[0]),
                cirq.BitFlipChannel(0.005).on(qubits[1]),
                cirq.BitFlipChannel(0.006).on(qubits[2]),
            ]
        ),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    assert_equivalent_op_tree(expected_program, noisy_circuit)


def test_per_qubit_readout_decay_from_data():
    # Generate the readout decay noise model from calibration data.
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    noise_model = simple_noise_from_calibration_metrics(
        calibration=calibration, readout_decay_noise=True
    )

    # Create the circuit and apply the noise model.
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly to construct expected output.
    decay_prob = [1 - exp(-1 / 0.007), 1 - exp(-1 / 0.008), 1 - exp(-1 / 0.009)]
    expected_program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment([cirq.AmplitudeDampingChannel(decay_prob[i]).on(qubits[i]) for i in range(3)]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    assert_equivalent_op_tree(expected_program, noisy_circuit)


def test_per_qubit_combined_noise_from_data():
    # Generate the combined noise model from calibration data.
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    noise_model = simple_noise_from_calibration_metrics(
        calibration=calibration,
        depol_noise=True,
        readout_error_noise=True,
        readout_decay_noise=True,
    )

    # Create the circuit and apply the noise model.
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly to construct expected output.
    decay_prob = [1 - exp(-1 / 0.007), 1 - exp(-1 / 0.008), 1 - exp(-1 / 0.009)]
    expected_program = cirq.Circuit(
        cirq.Moment([cirq.H(qubits[0])]),
        cirq.Moment([cirq.DepolarizingChannel(DEPOL_001).on(qubits[0])]),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]),
        cirq.Moment(
            [
                cirq.DepolarizingChannel(DEPOL_001).on(qubits[0]),
                cirq.DepolarizingChannel(DEPOL_002).on(qubits[1]),
            ]
        ),
        cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]),
        cirq.Moment(
            [
                cirq.DepolarizingChannel(DEPOL_001).on(qubits[0]),
                cirq.DepolarizingChannel(DEPOL_003).on(qubits[2]),
            ]
        ),
        cirq.Moment([cirq.AmplitudeDampingChannel(decay_prob[i]).on(qubits[i]) for i in range(3)]),
        cirq.Moment(
            [
                cirq.BitFlipChannel(0.004).on(qubits[0]),
                cirq.BitFlipChannel(0.005).on(qubits[1]),
                cirq.BitFlipChannel(0.006).on(qubits[2]),
            ]
        ),
        cirq.Moment(
            [
                cirq.measure(qubits[0], key='q0'),
                cirq.measure(qubits[1], key='q1'),
                cirq.measure(qubits[2], key='q2'),
            ]
        ),
    )
    assert_equivalent_op_tree(expected_program, noisy_circuit)
