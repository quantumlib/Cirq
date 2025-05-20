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

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from cirq_google.api import v2


class TagSerializer(abc.ABC):
    """Generic supertype for tag serializers.

    Each tag serializer describes how to serialize a specific type of
    tags to its corresponding proto format.
    """

    @abc.abstractmethod
    def to_proto(
        self,
        tag,
        msg=None,
        *,
        constants: List[v2.program_pb2.Constant],
        raw_constants: Dict[Any, int],
    ) -> Optional[v2.program_pb2.Tag]:
        """Converts tag to proto using this serializer.

        If self.can_serialize_tag(tag) == False, this should return None.

        Args:
            tag: The tag to be serialized.
            msg: An optional proto object to populate with the serialization
                results.
            constants: The list of previously-serialized Constant protos.
            raw_constants: A map raw objects to their respective indices in
                `constants`.

        Returns:
            The proto-serialized version of `tag`. If `msg` was provided, it is
            the returned object.
        """

    @abc.abstractmethod
    def can_serialize_tag(self, tag: Any) -> bool:
        """Whether the given tag can be serialized by this serializer."""
