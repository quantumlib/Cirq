# Copyright 2025 The Cirq Developers
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
from typing import Any, List

from cirq_google.api import v2


class TagDeserializer(abc.ABC):
    """Generic supertype for tag deserializers.

    Each tag deserializer describes how to deserialize a specific
    set of tag protos.
    """

    @abc.abstractmethod
    def can_deserialize_proto(self, proto: v2.program_pb2.Tag) -> bool:
        """Whether the given tag can be serialized by this serializer."""

    @abc.abstractmethod
    def from_proto(
        self,
        proto: v2.program_pb2.Tag,
        *,
        constants: List[v2.program_pb2.Constant],
        deserialized_constants: List[Any],
    ) -> Any:
        """Converts a proto-formatted operation into a Cirq operation.

        Args:
            proto: The proto object to be deserialized.
            constants: The list of Constant protos referenced by constant
                table indices in `proto`.
            deserialized_constants: The deserialized contents of `constants`.

        Returns:
            The deserialized operation represented by `proto`.
        """
