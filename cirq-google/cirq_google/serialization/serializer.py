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
from typing import Optional

import cirq
from cirq_google.api import v2


class Serializer(metaclass=abc.ABCMeta):
    """Interface for serialization."""

    def __init__(self, gate_set_name: str):
        self._gate_set_name = gate_set_name

    @property
    def name(self):
        """The name of the serializer."""
        return self._gate_set_name

    @abc.abstractmethod
    def serialize(
        self,
        program: cirq.AbstractCircuit,
        msg: Optional[v2.program_pb2.Program] = None,
        *,
        arg_function_language: Optional[str] = None,
    ) -> v2.program_pb2.Program:
        """Serialize a Circuit to cirq_google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
            msg: An optional proto object to populate with the serialization
                results.
            arg_function_language: The `arg_function_language` field from
                `Program.Language`.
        """

    @abc.abstractmethod
    def deserialize(self, proto: v2.program_pb2.Program) -> cirq.Circuit:
        """Deserialize a Circuit from a cirq_google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq_google.api.v2.Program proto.

        Returns:
            The deserialized Circuit.
        """
