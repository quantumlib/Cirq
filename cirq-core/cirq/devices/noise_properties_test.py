# Copyright 2021 The Cirq Developers
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

from typing import List, Tuple
import cirq

from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_properties import NoiseProperties, NoiseModelFromNoiseProperties
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG


# These properties are for testing purposes only - they are not representative
# of device behavior for any existing hardware.
class SampleNoiseProperties(NoiseProperties):
    def __init__(self, system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]):
        self.qubits = system_qubits
        self.qubit_pairs = qubit_pairs

    def build_noise_models(self):
        add_h = InsertionNoiseModel({OpIdentifier(cirq.Gate, q): cirq.H(q) for q in self.qubits})
        add_iswap = InsertionNoiseModel(
            {OpIdentifier(cirq.Gate, *qs): cirq.ISWAP(*qs) for qs in self.qubit_pairs}
        )
        return [add_h, add_iswap]


def test_sample_model():
    q0, q1 = cirq.LineQubit.range(2)
    props = SampleNoiseProperties([q0, q1], [(q0, q1), (q1, q0)])
    model = NoiseModelFromNoiseProperties(props)
    circuit = cirq.Circuit(
        cirq.X(q0), cirq.CNOT(q0, q1), cirq.Z(q1), cirq.measure(q0, q1, key='meas')
    )
    noisy_circuit = circuit.with_noise(model)
    expected_circuit = cirq.Circuit(
        cirq.Moment(cirq.X(q0).with_tags(PHYSICAL_GATE_TAG)),
        cirq.Moment(cirq.H(q0)),
        cirq.Moment(cirq.CNOT(q0, q1).with_tags(PHYSICAL_GATE_TAG)),
        cirq.Moment(cirq.ISWAP(q0, q1)),
        cirq.Moment(cirq.Z(q1).with_tags(PHYSICAL_GATE_TAG)),
        cirq.Moment(cirq.H(q1)),
        cirq.Moment(cirq.measure(q0, q1, key='meas')),
        cirq.Moment(cirq.H(q0), cirq.H(q1)),
    )
    assert noisy_circuit == expected_circuit


def test_deprecated_virtual_predicate():
    q0, q1 = cirq.LineQubit.range(2)
    props = SampleNoiseProperties([q0, q1], [(q0, q1), (q1, q0)])
    model = NoiseModelFromNoiseProperties(props)
    with cirq.testing.assert_deprecated("Use is_virtual", deadline="v0.16"):
        _ = model.virtual_predicate(cirq.X(q0))
