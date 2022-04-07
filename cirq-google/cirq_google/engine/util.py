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

import inspect
from typing import TypeVar

from google.protobuf import any_pb2
from google.protobuf.message import Message

import cirq

M = TypeVar('M', bound=Message)


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


def deprecated_gate_set_parameter(func):
    """Decorates a function that takes a deprecated 'gate_set' parameter."""
    signature = inspect.signature(func)
    gate_set_param = signature.parameters['gate_set']
    assert gate_set_param.default is None  # Must be optional and default to None.
    idx = list(signature.parameters).index('gate_set')

    decorator = cirq._compat.deprecated_parameter(
        deadline='v0.15',
        fix='Remove the gate_set parameter.',
        parameter_desc='gate_set',
        match=lambda args, kwargs: 'gate_set' in kwargs
        or (gate_set_param.kind != inspect.Parameter.KEYWORD_ONLY and len(args) > idx),
    )
    return decorator(func)
