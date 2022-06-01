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

from typing import overload, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import cirq


@overload
def q(__x: int) -> 'cirq.LineQubit':
    ...


@overload
def q(__row: int, __col: int) -> 'cirq.GridQubit':
    ...


@overload
def q(__name: str) -> 'cirq.NamedQubit':
    ...


def q(*args: Union[int, str]) -> Union['cirq.LineQubit', 'cirq.GridQubit', 'cirq.NamedQubit']:
    """Constructs a qubit id of the appropriate type based on args.

    This is shorthand for constructing qubit ids of common types:
    >>> cirq.q(1) == cirq.LineQubit(1)
    >>> cirq.q(1, 2) == cirq.GridQubit(1, 2)
    >>> cirq.q("foo") == cirq.NamedQubit("foo")

    Note that arguments should be treated as positional only, even
    though this is only enforceable in python 3.8 or later.

    Args:
        *args: One or two ints, or a single str, as described above.

    Returns:
        cirq.LineQubit if called with one integer arg.
        cirq.GridQubit if called with two integer args.
        cirq.NamedQubit if called with one string arg.

    Raises:
        ValueError: if called with invalid arguments.
    """
    import cirq  # avoid circular import

    if len(args) == 1:
        if isinstance(args[0], int):
            return cirq.LineQubit(args[0])
        elif isinstance(args[0], str):
            return cirq.NamedQubit(args[0])
    elif len(args) == 2:
        if isinstance(args[0], int) and isinstance(args[1], int):
            return cirq.GridQubit(args[0], args[1])
    raise ValueError(f"Could not construct qubit: args={args}")
