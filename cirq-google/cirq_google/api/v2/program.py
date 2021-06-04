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
import re
from typing import TYPE_CHECKING

import cirq

if TYPE_CHECKING:
    import cirq

GRID_QUBIT_ID_PATTERN = r'^q?(-?\d+)_(-?\d+)$'


def qubit_to_proto_id(q: cirq.Qid) -> str:
    """Return a proto id for a `cirq.Qid`.

    For `cirq.GridQubit`s this id `{row}_{col}` where `{row}` is the integer
    row of the grid qubit, and `{col}` is the integer column of the qubit.

    For `cirq.NamedQubit`s this id is the name.

    For `cirq.LineQubit`s this is string of the `x` attribute.
    """
    if isinstance(q, cirq.GridQubit):
        return f'{q.row}_{q.col}'
    elif isinstance(q, cirq.NamedQubit):
        return q.name
    elif isinstance(q, cirq.LineQubit):
        return f'{q.x}'
    else:
        raise ValueError(f'Qubits of type {type(q)} do not support proto id')


def qubit_from_proto_id(proto_id: str) -> cirq.Qid:
    """Return a `cirq.Qid` for a proto id.

    Proto IDs of the form {int}_{int} are parsed as GridQubits.

    Proto IDs of the form {int} are parsed as LineQubits.

    All other proto IDs are parsed as NamedQubits. Note that this will happily
    accept any string; for circuits which explicitly use Grid or LineQubits,
    prefer one of the specialized methods below.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.Qid` corresponding to the proto id.
    """
    num_coords = len(proto_id.split('_'))
    if num_coords == 2:
        try:
            grid_q = grid_qubit_from_proto_id(proto_id)
            return grid_q
        except ValueError:
            pass  # Not a grid qubit.
    elif num_coords == 1:
        try:
            line_q = line_qubit_from_proto_id(proto_id)
            return line_q
        except ValueError:
            pass  # Not a line qubit.

    # named_qubit_from_proto has no failure condition.
    named_q = named_qubit_from_proto_id(proto_id)
    return named_q


def grid_qubit_from_proto_id(proto_id: str) -> cirq.GridQubit:
    """Parse a proto id to a `cirq.GridQubit`.

    Proto ids for grid qubits are of the form `{row}_{col}` where `{row}` is
    the integer row of the grid qubit, and `{col}` is the integer column of
    the qubit.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.GridQubit` corresponding to the proto id.

    Raises:
        ValueError: If the string not of the correct format.
    """

    match = re.match(GRID_QUBIT_ID_PATTERN, proto_id)
    if match is None:
        raise ValueError(
            f'GridQubit proto id must be of the form [q]<int>_<int> but was {proto_id}'
        )
    row, col = match.groups()
    return cirq.GridQubit(row=int(row), col=int(col))


def line_qubit_from_proto_id(proto_id: str) -> cirq.LineQubit:
    """Parse a proto id to a `cirq.LineQubit`.

    Proto ids for line qubits are integer strings representing the `x`
    attribute of the line qubit.

    Args:
        proto_id: The id to convert.

    Returns:
        A `cirq.LineQubit` corresponding to the proto id.

    Raises:
        ValueError: If the string is not an integer.
    """
    try:
        return cirq.LineQubit(x=int(proto_id))
    except ValueError:
        raise ValueError(f'Line qubit proto id must be an int but was {proto_id}')


def named_qubit_from_proto_id(proto_id: str) -> cirq.NamedQubit:
    """Parse a proto id to a `cirq.NamedQubit'

    This simply returns a `cirq.NamedQubit` with a name equal to `proto_id`.
    """
    return cirq.NamedQubit(proto_id)
