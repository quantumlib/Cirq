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

import attrs

from typing import Optional
import cirq_google.api.v2.program_pb2 as v2


SUPPORTED_DD_PROTOCOLS = frozenset(["X", "Y"])


@attrs.frozen
class DynamicalDecouplingTag:
    "A tag to indicate using DD to fill qubit time."

    protocol: str = attrs.field()  # Which DD protocol to use.

    @protocol.validator
    def _validate_protocol(self, attribute, value):
        assert value in SUPPORTED_DD_PROTOCOLS

    def to_proto(
        self, msg: Optional[v2.DynamicalDecouplingTag] = None
    ) -> v2.DynamicalDecouplingTag:
        if msg is None:
            msg = v2.DynamicalDecouplingTag()
        msg.protocol = self.protocol
        return msg

    @staticmethod
    def from_proto(msg: v2.DynamicalDecouplingTag) -> 'DynamicalDecouplingTag':
        return DynamicalDecouplingTag(protocol=msg.protocol)
