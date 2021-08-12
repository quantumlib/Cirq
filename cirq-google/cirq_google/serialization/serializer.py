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

import abc
from typing import Dict, List, Optional

import cirq
from cirq_google.api import v2


class Serializer(metaclass=abc.ABCMeta):
    """Interface for serialization."""

    @abc.abstractmethod
    def serialize(
        self,
        program: cirq.Circuit,
        msg: Optional[v2.program_pb2.Program] = None,
        *,
        arg_function_language: Optional[str] = None,
    ) -> v2.program_pb2.Program:
        """Serialize."""

    def deserialize(
        self, proto: v2.program_pb2.Program, device: Optional[cirq.Device] = None
    ) -> cirq.Circuit:
        """Deserialize a Circuit from a cirq_google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq_google.api.v2.Program proto.
            device: If the proto is for a schedule, a device is required
                Otherwise optional.

        Returns:
            The deserialized Circuit, with a device if device was
            not None.
        """
