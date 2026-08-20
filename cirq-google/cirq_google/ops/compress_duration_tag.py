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

"""A class that can be used to denote a physical Z gate."""

from __future__ import annotations

from typing import Any

import cirq
from cirq_google.api.v2 import program_pb2


class CompressDurationTag:
    """Class to direct hardware to compress operations of zero duration.

    By default, parameters of gates that could lead to no-ops do not
    change the original duration of the gate.  For instance, X**0
    will cause a wait of the same duration as an X gate.  (This is so
    that parameter sweeps do not inadvertently change the duration of
    the circuit, leading to artifacts in data).

    When applied to an operation, this will cause the operation to
    compress to zero duration if possible.  Currently, this will affect
    gates with angles of zero.

    This can also affect PhasedXZGates to turn them into virtual Z gates
    if the resulting gate has a Z phase but no X component.
    """

    def __str__(self) -> str:
        return 'CompressDurationTag()'

    def __repr__(self) -> str:
        return 'cirq_google.CompressDurationTag()'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __eq__(self, other) -> bool:
        return isinstance(other, CompressDurationTag)

    def __hash__(self) -> int:
        return 456789

    def to_proto(self, msg: program_pb2.Tag | None = None) -> program_pb2.Tag:
        if msg is None:
            msg = program_pb2.Tag()
        msg.compress_duration.SetInParent()
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag) -> CompressDurationTag:
        return CompressDurationTag()
