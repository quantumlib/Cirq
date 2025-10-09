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

from __future__ import annotations

from typing import TYPE_CHECKING

import cirq
import cirq.testing
from cirq.devices.insertion_noise_model import InsertionNoiseModel
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties, NoiseProperties
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG

if TYPE_CHECKING:
    from cirq.protocols.json_serialization import ObjectFactory


# These properties are for testing purposes only - they are not representative
# of device behavior for any existing hardware.
@cirq.value_equality
class SampleNoiseProperties(NoiseProperties):
    def __init__(self, system_qubits: list[cirq.Qid], qubit_pairs: list[tuple[cirq.Qid, cirq.Qid]]):
        self.qubits = system_qubits
        self.qubit_pairs = qubit_pairs

    def build_noise_models(self) -> list[cirq.NoiseModel]:
        add_h = InsertionNoiseModel({OpIdentifier(cirq.Gate, q): cirq.H(q) for q in self.qubits})
        add_iswap = InsertionNoiseModel(
            {OpIdentifier(cirq.Gate, *qs): cirq.ISWAP(*qs) for qs in self.qubit_pairs}
        )
        return [add_h, add_iswap]

    def _value_equality_values_(self):
        return (self.qubits, self.qubit_pairs)

    def _json_dict_(self) -> dict[str, object]:
        return {'system_qubits': self.qubits, 'qubit_pairs': self.qubit_pairs}

    @classmethod
    def _from_json_dict_(cls, system_qubits, qubit_pairs, **kwargs):
        return cls(system_qubits=system_qubits, qubit_pairs=[tuple(pair) for pair in qubit_pairs])


def test_sample_model() -> None:
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


def custom_resolver(cirq_type: str) -> ObjectFactory | None:
    if cirq_type == "SampleNoiseProperties":
        return SampleNoiseProperties
    return None


def test_noise_model_from_noise_properties_json() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    props = SampleNoiseProperties([q0, q1], [(q0, q1), (q1, q0)])
    model = NoiseModelFromNoiseProperties(props)
    resolvers = [custom_resolver] + cirq.DEFAULT_RESOLVERS
    cirq.testing.assert_json_roundtrip_works(model, resolvers=resolvers)
