# Copyright 2022 The Cirq Developers
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

from typing import Dict, Tuple, TypeVar

from google.protobuf import any_pb2
from google.protobuf.message import Message

import cirq

M = TypeVar('M', bound=Message)

# Bundled Z phase errors in the format:
#
#   {gate_type: {angle_type: {qubit_pair: error}}}
#
# where gate_type is "syc" or "sqrt_iswap", angle_type is "zeta" or "gamma",
# and "qubit_pair" is a tuple of qubits.
ZPhaseDataType = Dict[str, Dict[str, Dict[Tuple[cirq.Qid, ...], float]]]


def pack_any(message: Message) -> any_pb2.Any:
    """Packs a message into an Any proto.

    Returns the packed Any proto.
    """
    packed = any_pb2.Any()
    packed.Pack(message)
    return packed


def unpack_any(message: any_pb2.Any, out: M) -> M:
    message.Unpack(out)
    return out
