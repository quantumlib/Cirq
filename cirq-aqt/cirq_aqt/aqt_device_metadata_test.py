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

import cirq
from cirq_aqt.aqt_device_metadata import AQTDeviceMetadata
from cirq_aqt.aqt_target_gateset import AQTTargetGateset


def test_aqtdevice_metadata():
    qubits = cirq.LineQubit.range(5)
    metadata = AQTDeviceMetadata(qubits)
    assert metadata.qubit_set == frozenset(qubits)
    assert set(qubits) == set(metadata.nx_graph.nodes())
    edges = metadata.nx_graph.edges()
    assert len(edges) == 10
    assert all(q0 != q1 for q0, q1 in edges)
    assert AQTTargetGateset() == metadata.gateset


def test_repr():
    qubits = cirq.LineQubit.range(3)
    metadata = AQTDeviceMetadata(qubits)
    cirq.testing.assert_equivalent_repr(metadata, setup_code='import cirq\nimport cirq_aqt\n')
