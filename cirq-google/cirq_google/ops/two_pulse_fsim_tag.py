# Copyright 2026 The Cirq Developers
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

"""A class that can be used to denote FSim gate implementation using two pulses."""

from __future__ import annotations

from typing import Any

import cirq
from cirq_google.api.v2 import program_pb2


class TwoPulseFSimTag:
    """A tag class to denote FSim gate implementation using two pulses.

    If FSimGate is tagged with this class, the translation will become
    a two pulse implementation.
    """

    def __str__(self) -> str:
        return 'TwoPulseFSimTag()'

    def __repr__(self) -> str:
        return 'cirq_google.TwoPulseFSimTag()'

    def _json_dict_(self) -> dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __eq__(self, other) -> bool:
        return isinstance(other, TwoPulseFSimTag)

    def __hash__(self) -> int:
        return hash("TwoPulseFSimTag")

    def to_proto(self, msg: program_pb2.Tag | None = None) -> program_pb2.Tag:
        if msg is None:
            msg = program_pb2.Tag()
        msg.two_pulse_fsim.SetInParent()
        return msg

    @staticmethod
    def from_proto(msg: program_pb2.Tag) -> TwoPulseFSimTag:
        if msg.WhichOneof("tag") != "two_pulse_fsim":
            raise ValueError(f"Message is not a TwoPulseFSimTag, {msg}")
        return TwoPulseFSimTag()
