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
def q(x: int) -> 'cirq.LineQubit':
    ...

@overload
def q(x: int, y: int) -> 'cirq.GridQubit':
    ...

@overload
def q(x: str) -> 'cirq.NamedQubit':
    ...

def q(x: Union[int, str], y: Optional[int] = None) -> Union['cirq.LineQubit', 'cirq.GridQubit', 'cirq.NamedQubit']:
    """Constructs a qubit id of the appropriate type based on args.

    This is shorthand for constructing qubit ids of common types:
    >>> cirq.q(1) == cirq.LineQubit(1)
    >>> cirq.q(1, 2) == cirq.GridQubit(1, 2)
    >>> cirq.q("foo") == cirq.NamedQubit("foo")

    Args:
        x: First arg.
        y: Second arg.

    Returns:
        cirq.LineQubit if called with one integer arg.
        cirq.GridQubit if called with two integer args.
        cirq.NamedQubit if called with one string arg.

    Raises:
        ValueError if called with invalid arguments.
    """
    import cirq  # avoid circular import

    if y is None:
        if isinstance(x, int):
            return cirq.LineQubit(x)
        elif isinstance(x, str):
            return cirq.NamedQubit(x)
    else:
        if isinstance(x, int) and isinstance(y, int):
            return cirq.GridQubit(x, y)
    raise ValueError(f"Could not construct qubit: args={(x, y)}")
