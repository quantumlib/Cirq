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

import pytest

import cirq
from cirq import X, Y, Z, XX, Circuit

from cirq.aqt import AQTSimulator
from cirq.aqt.aqt_device import get_aqt_device
from cirq.aqt.aqt_device import AQTNoiseModel


def test_simulator_no_circ():
    with pytest.raises(RuntimeError):
        sim = AQTSimulator(num_qubits=1)
        sim.simulate_samples(1)


def test_ms_crosstalk_n_noise():
    num_qubits = 4
    noise_mod = AQTNoiseModel()
    device, qubits = get_aqt_device(num_qubits)
    circuit = Circuit(device=device)
    circuit.append(XX(qubits[1], qubits[2]) ** 0.5)
    for moment in circuit.moments:
        noisy_moment = noise_mod.noisy_moment(moment, qubits)
    assert noisy_moment == [
        (cirq.XX ** 0.5).on(cirq.LineQubit(1), cirq.LineQubit(2)),
        cirq.depolarize(p=0.01).on(cirq.LineQubit(1)),
        cirq.depolarize(p=0.01).on(cirq.LineQubit(2)),
        (cirq.XX ** 0.015).on(cirq.LineQubit(1), cirq.LineQubit(0)),
        (cirq.XX ** 0.015).on(cirq.LineQubit(1), cirq.LineQubit(3)),
        (cirq.XX ** 0.015).on(cirq.LineQubit(2), cirq.LineQubit(0)),
        (cirq.XX ** 0.015).on(cirq.LineQubit(2), cirq.LineQubit(3)),
    ]


def test_x_crosstalk_n_noise():
    num_qubits = 4
    noise_mod = AQTNoiseModel()
    device, qubits = get_aqt_device(num_qubits)
    circuit = Circuit(device=device)
    circuit.append(Y(qubits[1]) ** 0.5)
    circuit.append(Z(qubits[1]) ** 0.5)
    circuit.append(X(qubits[1]) ** 0.5)
    for moment in circuit.moments:
        noisy_moment = noise_mod.noisy_moment(moment, qubits)
    assert noisy_moment == [
        (cirq.X ** 0.5).on(cirq.LineQubit(1)),
        cirq.depolarize(p=0.001).on(cirq.LineQubit(1)),
        (cirq.X ** 0.015).on(cirq.LineQubit(0)),
        (cirq.X ** 0.015).on(cirq.LineQubit(2)),
    ]
