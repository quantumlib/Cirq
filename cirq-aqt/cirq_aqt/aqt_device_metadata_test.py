# Copyright 2022 The Cirq Developers
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

"""Tests for AQTDeviceMetadata."""

from typing import List

import pytest

import cirq
from cirq_aqt.aqt_device_metadata import AQTDeviceMetadata
from cirq_aqt.aqt_target_gateset import AQTTargetGateset


@pytest.fixture
def qubits() -> List[cirq.LineQubit]:
    return cirq.LineQubit.range(5)


@pytest.fixture
def metadata(qubits) -> AQTDeviceMetadata:
    return AQTDeviceMetadata(
        qubits=qubits,
        measurement_duration=cirq.Duration(millis=100),
        twoq_gates_duration=cirq.Duration(millis=200),
        oneq_gates_duration=cirq.Duration(millis=10),
    )


def test_aqtdevice_metadata(metadata, qubits):
    assert metadata.qubit_set == frozenset(qubits)
    assert set(qubits) == set(metadata.nx_graph.nodes())
    edges = metadata.nx_graph.edges()
    assert len(edges) == 10
    assert all(q0 != q1 for q0, q1 in edges)
    assert AQTTargetGateset() == metadata.gateset
    assert len(metadata.gate_durations) == 4


def test_aqtdevice_duration_of(metadata, qubits):
    q0, q1 = qubits[:2]
    ms = cirq.Duration(millis=1)
    assert metadata.duration_of(cirq.Z(q0)) == 10 * ms
    assert metadata.duration_of(cirq.measure(q0)) == 100 * ms
    assert metadata.duration_of(cirq.measure(q0, q1)) == 100 * ms
    assert metadata.duration_of(cirq.XX(q0, q1)) == 200 * ms
    with pytest.raises(ValueError, match="Unsupported gate type"):
        metadata.duration_of(cirq.I(q0))


def test_repr(metadata):
    cirq.testing.assert_equivalent_repr(metadata, setup_code='import cirq\nimport cirq_aqt\n')
