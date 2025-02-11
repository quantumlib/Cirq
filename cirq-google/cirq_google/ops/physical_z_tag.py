# Copyright 2020 The Cirq Developers
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
"""A class that can be used to denote a physical Z gate."""
from typing import Any, Dict, Optional

import cirq
from cirq_google.api.v2 import program_pb2


class PhysicalZTag:
    """Class to add as a tag onto an Operation to denote a Physical Z operation.

    By default, all Z rotations on Google devices are considered to be virtual.
    When performing the Z operation, the device will update its internal phase
    tracking mechanisms, essentially commuting it forwards through the circuit
    until it hits a non-commuting operation (Such as a sqrt(iSwap)).

    When applied to a Z rotation operation, this tag indicates to the hardware
    that an actual physical operation should be done instead.  This class can
    only be applied to instances of `cirq.ZPowGate`.  If applied to other gates
    (such as PhasedXZGate), this class will have no effect.
    """

    def __str__(self) -> str:
        return 'PhysicalZTag()'

    def __repr__(self) -> str:
        return 'cirq_google.PhysicalZTag()'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __eq__(self, other) -> bool:
        return isinstance(other, PhysicalZTag)

    def __hash__(self) -> int:
        return 123

    def to_proto(self, msg: Optional[program_pb2.Tag] = None) -> program_pb2.Tag:
        if msg is None:
            msg = program_pb2.Tag()
        msg.physical_z.SetInParent()
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag) -> 'PhysicalZTag':
        return PhysicalZTag()
