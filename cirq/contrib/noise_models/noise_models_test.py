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
import cirq.contrib.noise_models as ccn
from cirq import ops
from cirq.devices.noise_model_test import _assert_equivalent_op_tree


def test_depol_noise():
    noise_model = ccn.DepolarizingNoiseModel(depol_prob=0.005)
    qubits = cirq.LineQubit.range(2)
    moment = cirq.Moment(
        [
            cirq.X(qubits[0]),
            cirq.Y(qubits[1]),
        ]
    )
    noisy_mom = noise_model.noisy_moment(moment, system_qubits=qubits)
    assert len(noisy_mom) == 2
    assert noisy_mom[0] == moment
    for g in noisy_mom[1]:
        assert isinstance(g.gate, cirq.DepolarizingChannel)


# Composes depolarization noise with readout noise.
def test_readout_noise_after_moment():
    program = cirq.Circuit()
    qubits = cirq.LineQubit.range(3)
    program.append(
        [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[2])]
    )
    program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    # Use noise model to generate circuit
    depol_noise = ccn.DepolarizingNoiseModel(depol_prob=0.01)
    readout_noise = ccn.ReadoutNoiseModel(bitflip_prob=0.05)
    noisy_circuit = cirq.Circuit(depol_noise.noisy_moments(program, qubits))
    noisy_circuit = cirq.Circuit(readout_noise.noisy_moments(noisy_circuit, qubits))

    # Insert channels explicitly
    true_noisy_program = cirq.Circuit()
    true_noisy_program.append([cirq.H(qubits[0])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[0], qubits[1])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[1], qubits[2])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append(
        [cirq.BitFlipChannel(0.05).on(q).with_tags(ops.VirtualTag()) for q in qubits]
    )
    true_noisy_program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ]
    )
    _assert_equivalent_op_tree(true_noisy_program, noisy_circuit)


# Composes depolarization, damping, and readout noise (in that order).
def test_decay_noise_after_moment():
    program = cirq.Circuit()
    qubits = cirq.LineQubit.range(3)
    program.append(
        [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[2])]
    )
    program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    # Use noise model to generate circuit
    depol_noise = ccn.DepolarizingNoiseModel(depol_prob=0.01)
    readout_noise = ccn.ReadoutNoiseModel(bitflip_prob=0.05)
    damping_noise = ccn.DampedReadoutNoiseModel(decay_prob=0.02)
    noisy_circuit = cirq.Circuit(depol_noise.noisy_moments(program, qubits))
    noisy_circuit = cirq.Circuit(damping_noise.noisy_moments(noisy_circuit, qubits))
    noisy_circuit = cirq.Circuit(readout_noise.noisy_moments(noisy_circuit, qubits))

    # Insert channels explicitly
    true_noisy_program = cirq.Circuit()
    true_noisy_program.append([cirq.H(qubits[0])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[0], qubits[1])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[1], qubits[2])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append(
        [cirq.AmplitudeDampingChannel(0.02).on(q).with_tags(ops.VirtualTag()) for q in qubits]
    )
    true_noisy_program.append(
        [cirq.BitFlipChannel(0.05).on(q).with_tags(ops.VirtualTag()) for q in qubits]
    )
    true_noisy_program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ]
    )
    _assert_equivalent_op_tree(true_noisy_program, noisy_circuit)


# Test the aggregate noise models.
def test_aggregate_readout_noise_after_moment():
    program = cirq.Circuit()
    qubits = cirq.LineQubit.range(3)
    program.append(
        [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[2])]
    )
    program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    # Use noise model to generate circuit
    noise_model = ccn.DepolarizingWithReadoutNoiseModel(depol_prob=0.01, bitflip_prob=0.05)
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly
    true_noisy_program = cirq.Circuit()
    true_noisy_program.append([cirq.H(qubits[0])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[0], qubits[1])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[1], qubits[2])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.BitFlipChannel(0.05).on(q) for q in qubits])
    true_noisy_program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ]
    )
    _assert_equivalent_op_tree(true_noisy_program, noisy_circuit)


def test_aggregate_decay_noise_after_moment():
    program = cirq.Circuit()
    qubits = cirq.LineQubit.range(3)
    program.append(
        [cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[2])]
    )
    program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )

    # Use noise model to generate circuit
    noise_model = ccn.DepolarizingWithDampedReadoutNoiseModel(
        depol_prob=0.01, decay_prob=0.02, bitflip_prob=0.05
    )
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))

    # Insert channels explicitly
    true_noisy_program = cirq.Circuit()
    true_noisy_program.append([cirq.H(qubits[0])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[0], qubits[1])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.CNOT(qubits[1], qubits[2])])
    true_noisy_program.append(
        [cirq.DepolarizingChannel(0.01).on(q) for q in qubits],
        strategy=cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    true_noisy_program.append([cirq.AmplitudeDampingChannel(0.02).on(q) for q in qubits])
    true_noisy_program.append([cirq.BitFlipChannel(0.05).on(q) for q in qubits])
    true_noisy_program.append(
        [
            cirq.measure(qubits[0], key='q0'),
            cirq.measure(qubits[1], key='q1'),
            cirq.measure(qubits[2], key='q2'),
        ]
    )
    _assert_equivalent_op_tree(true_noisy_program, noisy_circuit)
