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


"""DeviceMetadata for ion trap device with mutually linked qubits placed on a line.
"""

from typing import Iterable

import networkx as nx

import cirq
from cirq_aqt import aqt_target_gateset


class AQTDeviceMetadata(cirq.DeviceMetadata):
    """Hardware metadata for ion trap device with all-connected qubits placed on a line."""

    def __init__(self, qubits: Iterable['cirq.LineQubit']):
        """Create metadata object for AQTDevice.

        Args:
            qubits: Iterable of `cirq.Qid`s that exist on the device.
        """

        graph = nx.Graph()
        graph.add_edges_from([(a, b) for a in qubits for b in qubits if a != b], directed=False)
        super().__init__(qubits, graph)
        self._gateset = aqt_target_gateset.AQTTargetGateset()

    @property
    def gateset(self) -> 'cirq.Gateset':
        """Returns the `cirq.Gateset` of supported gates on this device."""
        return self._gateset

    def __repr__(self) -> str:
        return (
            f'cirq_aqt.aqt_device_metadata.AQTDeviceMetadata('
            f'qubits={sorted(self.qubit_set)!r}'
            f')'
        )
