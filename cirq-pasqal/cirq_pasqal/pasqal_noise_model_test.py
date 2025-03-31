# Copyright 2020 The Cirq Developers
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
import pytest

import cirq
from cirq.ops import NamedQubit
from cirq_pasqal import PasqalDevice, PasqalNoiseModel


def test_noise_model_init():
    noise_model = PasqalNoiseModel(PasqalDevice([]))
    assert noise_model.noise_op_dict == {
        str(cirq.ops.YPowGate()): cirq.ops.depolarize(1e-2),
        str(cirq.ops.ZPowGate()): cirq.ops.depolarize(1e-2),
        str(cirq.ops.XPowGate()): cirq.ops.depolarize(1e-2),
        str(cirq.ops.HPowGate(exponent=1)): cirq.ops.depolarize(1e-2),
        str(cirq.ops.PhasedXPowGate(phase_exponent=0)): cirq.ops.depolarize(1e-2),
        str(cirq.ops.CNotPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
        str(cirq.ops.CZPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
        str(cirq.ops.CCXPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
        str(cirq.ops.CCZPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
    }
    with pytest.raises(TypeError, match="noise model varies between Pasqal's devices."):
        PasqalNoiseModel(cirq.devices.UNCONSTRAINED_DEVICE)


def test_noisy_moments():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(qubits=p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    circuit.append(cirq.ops.CZ(p_qubits[0], p_qubits[1]))
    circuit.append(cirq.ops.Z(p_qubits[1]))
    p_circuit = cirq.Circuit(circuit)

    n_mts = []
    for moment in p_circuit._moments:
        n_mts.append(noise_model.noisy_moment(moment, p_qubits))

    assert n_mts == [
        [
            cirq.ops.CZ.on(NamedQubit('q0'), NamedQubit('q1')),
            cirq.depolarize(p=0.03).on(NamedQubit('q0')),
            cirq.depolarize(p=0.03).on(NamedQubit('q1')),
        ],
        [cirq.ops.Z.on(NamedQubit('q1')), cirq.depolarize(p=0.01).on(NamedQubit('q1'))],
    ]


def test_default_noise():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(qubits=p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    Gate_l = cirq.ops.CZPowGate(exponent=2)
    circuit.append(Gate_l.on(p_qubits[0], p_qubits[1]))
    p_circuit = cirq.Circuit(circuit)
    n_mts = []
    for moment in p_circuit._moments:
        n_mts.append(noise_model.noisy_moment(moment, p_qubits))

    assert n_mts == [
        [
            cirq.ops.CZPowGate(exponent=2).on(NamedQubit('q0'), NamedQubit('q1')),
            cirq.depolarize(p=0.05).on(NamedQubit('q0')),
            cirq.depolarize(p=0.05).on(NamedQubit('q1')),
        ]
    ]


def test_get_op_string():
    p_qubits = cirq.NamedQubit.range(2, prefix='q')
    p_device = PasqalDevice(p_qubits)
    noise_model = PasqalNoiseModel(p_device)
    circuit = cirq.Circuit()
    circuit.append(cirq.ops.HPowGate(exponent=0.5).on(p_qubits[0]))

    with pytest.raises(ValueError, match='Got unknown operation:'):
        for moment in circuit._moments:
            _ = noise_model.noisy_moment(moment, p_qubits)

    with pytest.raises(ValueError, match='Got unknown operation:'):
        _ = noise_model.get_op_string(circuit)
