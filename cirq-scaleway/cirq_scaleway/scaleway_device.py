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

from .scaleway_client import QaaSClient


class ScalewayDevice(cirq.devices.Device):
    def __init__(
        self, client: QaaSClient, id: str, name: str, version: str, num_qubits: int, metadata: str
    ) -> None:
        self._id = id
        self._client = client
        self._version = version
        self._num_qubits = num_qubits
        self._name = name
        self._metadata = metadata

    @property
    def id(self):
        return self._id

    @property
    def availability(self):
        resp = self._client.get_platform(self._id)

        return resp.get("availability", None)

    @property
    def name(self):
        return self._name

    @property
    def num_qubits(self):
        return self._num_qubits

    @property
    def metadata(self):
        return self._metadata
