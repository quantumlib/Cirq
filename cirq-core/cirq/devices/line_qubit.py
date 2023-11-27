# Copyright 2018 The Cirq Developers
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

import abc
import functools
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TYPE_CHECKING, Union
from typing_extensions import Self

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


@functools.total_ordering
class _BaseLineQid(ops.Qid):
    """The base class for `LineQid` and `LineQubit`."""

    _x: int
    _dimension: int
    _hash: Optional[int] = None

    def __getstate__(self):
        # Don't save hash when pickling; see #3777.
        state = self.__dict__
        if "_hash" in state:
            state = state.copy()
            del state["_hash"]
        return state

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._x, self._dimension))
        return self._hash

    def __eq__(self, other):
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseLineQid):
            return self._x == other._x and self._dimension == other._dimension
        return NotImplemented

    def __ne__(self, other):
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseLineQid):
            return self._x != other._x or self._dimension != other._dimension
        return NotImplemented

    def _comparison_key(self):
        return self._x

    @property
    def x(self) -> int:
        return self._x

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'LineQid':
        return LineQid(self._x, dimension)

    def is_adjacent(self, other: 'cirq.Qid') -> bool:
        """Determines if two qubits are adjacent line qubits.

        Args:
            other: `cirq.Qid` to test for adjacency.

        Returns: True iff other and self are adjacent.
        """
        return isinstance(other, _BaseLineQid) and abs(self._x - other._x) == 1

    def neighbors(self, qids: Optional[Iterable[ops.Qid]] = None) -> Set['_BaseLineQid']:
        """Returns qubits that are potential neighbors to this LineQubit

        Args:
            qids: optional Iterable of qubits to constrain neighbors to.
        """
        return {q for q in [self - 1, self + 1] if qids is None or q in qids}

    @abc.abstractmethod
    def _with_x(self, x: int) -> Self:
        """Returns a qubit with the same type but a different value of `x`."""

    def __add__(self, other: Union[int, Self]) -> Self:
        if isinstance(other, _BaseLineQid):
            if self._dimension != other._dimension:
                raise TypeError(
                    "Can only add LineQids with identical dimension. "
                    f"Got {self._dimension} and {other._dimension}"
                )
            return self._with_x(x=self._x + other._x)
        if not isinstance(other, int):
            raise TypeError(f"Can only add ints and {type(self).__name__}. Instead was {other}")
        return self._with_x(self._x + other)

    def __sub__(self, other: Union[int, Self]) -> Self:
        if isinstance(other, _BaseLineQid):
            if self._dimension != other._dimension:
                raise TypeError(
                    "Can only subtract LineQids with identical dimension. "
                    f"Got {self._dimension} and {other._dimension}"
                )
            return self._with_x(x=self._x - other._x)
        if not isinstance(other, int):
            raise TypeError(
                f"Can only subtract ints and {type(self).__name__}. Instead was {other}"
            )
        return self._with_x(self._x - other)

    def __radd__(self, other: int) -> Self:
        return self + other

    def __rsub__(self, other: int) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        return self._with_x(-self._x)

    def __complex__(self) -> complex:
        return complex(self._x)

    def __float__(self) -> float:
        return float(self._x)

    def __int__(self) -> int:
        return int(self._x)


class LineQid(_BaseLineQid):
    """A qid on a 1d lattice with nearest-neighbor connectivity.

    `LineQid`s have a single attribute, and integer coordinate 'x', which
    identifies the qids location on the line. `LineQid`s are ordered by
    this integer.

    One can construct new `cirq.LineQid`s by adding or subtracting integers:

    >>> cirq.LineQid(1, dimension=2) + 3
    cirq.LineQid(4, dimension=2)

    >>> cirq.LineQid(2, dimension=3) - 1
    cirq.LineQid(1, dimension=3)

    """

    def __init__(self, x: int, dimension: int) -> None:
        """Initializes a line qid at the given x coordinate.

        Args:
            x: The x coordinate.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        self.validate_dimension(dimension)
        self._x = x
        self._dimension = dimension

    def _with_x(self, x: int) -> 'LineQid':
        return LineQid(x, dimension=self._dimension)

    @staticmethod
    def range(*range_args, dimension: int) -> List['LineQid']:
        """Returns a range of line qids.

        Args:
            *range_args: Same arguments as python's built-in range method.
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of line qids.
        """
        return [LineQid(i, dimension=dimension) for i in range(*range_args)]

    @staticmethod
    def for_qid_shape(qid_shape: Sequence[int], start: int = 0, step: int = 1) -> List['LineQid']:
        """Returns a range of line qids for each entry in `qid_shape` with
        matching dimension.

        Args:
            qid_shape: A sequence of dimensions for each `LineQid` to create.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
        return [
            LineQid(start + step * i, dimension=dimension) for i, dimension in enumerate(qid_shape)
        ]

    @staticmethod
    def for_gate(val: Any, start: int = 0, step: int = 1) -> List['LineQid']:
        """Returns a range of line qids with the same qid shape as the gate.

        Args:
            val: Any value that supports the `cirq.qid_shape` protocol.  Usually
                a gate.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
        # Avoids circular import.
        from cirq.protocols.qid_shape_protocol import qid_shape

        return LineQid.for_qid_shape(qid_shape(val), start=start, step=step)

    def __repr__(self) -> str:
        return f"cirq.LineQid({self._x}, dimension={self._dimension})"

    def __str__(self) -> str:
        return f"q({self._x}) (d={self._dimension})"

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f"{self._x} (d={self._dimension})",))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['x', 'dimension'])


class LineQubit(_BaseLineQid):
    """A qubit on a 1d lattice with nearest-neighbor connectivity.

    LineQubits have a single attribute, and integer coordinate 'x', which
    identifies the qubits location on the line. LineQubits are ordered by
    this integer.

    One can construct new `cirq.LineQubit`s by adding or subtracting integers:

    >>> cirq.LineQubit(1) + 3
    cirq.LineQubit(4)

    >>> cirq.LineQubit(2) - 1
    cirq.LineQubit(1)

    """

    _dimension = 2

    def __init__(self, x: int) -> None:
        """Initializes a line qubit at the given x coordinate.

        Args:
            x: The x coordinate.
        """
        self._x = x

    def _with_x(self, x: int) -> 'LineQubit':
        return LineQubit(x)

    def _cmp_tuple(self):
        cls = LineQid if type(self) is LineQubit else type(self)
        # Must be the same as Qid._cmp_tuple but with cls in place of
        # type(self).
        return (cls.__name__, repr(cls), self._comparison_key(), self._dimension)

    @staticmethod
    def range(*range_args) -> List['LineQubit']:
        """Returns a range of line qubits.

        Args:
            *range_args: Same arguments as python's built-in range method.

        Returns:
            A list of line qubits.
        """
        return [LineQubit(i) for i in range(*range_args)]

    def __repr__(self) -> str:
        return f"cirq.LineQubit({self._x})"

    def __str__(self) -> str:
        return f"q({self._x})"

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f"{self._x}",))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['x'])
