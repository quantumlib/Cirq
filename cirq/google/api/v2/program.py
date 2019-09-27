# Copyright 2019 The Cirq Developers
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

from typing import TYPE_CHECKING

from cirq import devices

if TYPE_CHECKING:
    import cirq


def qubit_to_proto_id(q: devices.GridQubit) -> str:
    """Return a proto id for a `cirq.GridQubit`"""
    return '{}_{}'.format(q.row, q.col)


def qubit_from_proto_id(proto_id: str) -> 'cirq.GridQubit':
    """Parse a proto id string to a `cirq.GirdQubit`"""
    row, col = proto_id.split('_')
    return devices.GridQubit(row=int(row), col=int(col))
