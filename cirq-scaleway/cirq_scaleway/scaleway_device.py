# Copyright 2024 Scaleway
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

from typing import Union, Optional
from pytimeparse.timeparse import timeparse

from .scaleway_session import ScalewaySession
from .scaleway_client import QaaSClient


class ScalewayDevice(cirq.devices.Device):
    def __init__(
        self, client: QaaSClient, id: str, name: str, version: str, num_qubits: int, metadata: str
    ) -> None:
        self.__id = id
        self.__client = client
        self.__version = version
        self.__num_qubits = num_qubits
        self.__name = name
        self.__metadata = metadata

    @property
    def id(self):
        return self.__id

    @property
    def availability(self):
        resp = self.__client.get_platform(self.__id)

        return resp.get("availability")

    @property
    def name(self):
        return self.__name

    @property
    def num_qubits(self):
        return self.__num_qubits

    @property
    def metadata(self):
        return self.__metadata

    @property
    def version(self):
        return self.__version

    def start_session(
        self,
        name: Optional[str] = "qsim-session-from-cirq",
        deduplication_id: Optional[str] = "qsim-session-from-cirq",
        max_duration: Union[int, str] = "1h",
        max_idle_duration: Union[int, str] = "20m",
    ) -> str:
        if isinstance(max_duration, str):
            max_duration = f"{timeparse(max_duration)}s"

        if isinstance(max_idle_duration, str):
            max_idle_duration = f"{timeparse(max_idle_duration)}s"

        session_id = self.__client.create_session(
            name,
            platform_id=self.id,
            deduplication_id=deduplication_id,
            max_duration=max_duration,
            max_idle_duration=max_idle_duration,
        )

        return ScalewaySession(client=self.__client, id=session_id, name=name)
