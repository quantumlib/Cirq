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

from typing import Optional

import attrs

from cirq_google.api.v2 import program_pb2


SUPPORTED_DD_PROTOCOLS = frozenset(
    [
        "X",  # An even number of X
        "Y",  # An even number of Y
        "XY4",  # Repetitions of XYXY blocks.
        "XY8",  # Repetitions of XYXYYXYX blocks.
    ]
)


@attrs.frozen
class DynamicalDecouplingTag:
    """A tag to indicate using DD to fill qubit time.

    Attributes:
        protocol: The name of the decoupling protocol (eg 'X', 'XY4').
    """

    protocol: str = attrs.field()  # Which DD protocol to use.

    @protocol.validator
    def _validate_protocol(self, attribute, value):
        assert value in SUPPORTED_DD_PROTOCOLS

    def to_proto(self, msg: Optional[program_pb2.Tag] = None) -> program_pb2.Tag:
        if msg is None:
            msg = program_pb2.Tag()
        msg.dynamical_decoupling.protocol = self.protocol
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag) -> 'DynamicalDecouplingTag':
        if msg.WhichOneof("tag") != "dynamical_decoupling":
            raise ValueError(f"Message is not a DynamicalDecouplingTag, {msg}")
        return DynamicalDecouplingTag(protocol=msg.dynamical_decoupling.protocol)
