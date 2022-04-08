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

from typing import Optional, overload, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import cirq


@overload
def q(_x: int) -> 'cirq.LineQubit':
    ...


@overload
def q(_x: int, _y: int) -> 'cirq.GridQubit':
    ...


@overload
def q(_x: str) -> 'cirq.NamedQubit':
    ...


def q(
    _x: Union[int, str], _y: Optional[int] = None
) -> Union['cirq.LineQubit', 'cirq.GridQubit', 'cirq.NamedQubit']:
    """Constructs a qubit id of the appropriate type based on args.

    This is shorthand for constructing qubit ids of common types:
    >>> cirq.q(1) == cirq.LineQubit(1)
    >>> cirq.q(1, 2) == cirq.GridQubit(1, 2)
    >>> cirq.q("foo") == cirq.NamedQubit("foo")

    Note that arguments should be treated as positional only, even
    though this is only enforceable in python 3.8 or later.

    Args:
        _x: First arg.
        _y: Second arg.

    Returns:
        cirq.LineQubit if called with one integer arg.
        cirq.GridQubit if called with two integer args.
        cirq.NamedQubit if called with one string arg.

    Raises:
        ValueError: if called with invalid arguments.
    """
    import cirq  # avoid circular import

    if _y is None:
        if isinstance(_x, int):
            return cirq.LineQubit(_x)
        elif isinstance(_x, str):
            return cirq.NamedQubit(_x)
    else:
        if isinstance(_x, int) and isinstance(_y, int):
            return cirq.GridQubit(_x, _y)
    raise ValueError(f"Could not construct qubit: args={(_x, _y)}")
