# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Any, List, Sequence, TypeVar

import abc

from cirq import ops, protocols

TSelf = TypeVar('TSelf', bound='_BaseLineQid')


@functools.total_ordering
class _BaseLineQid(ops.Qid):
    """The base class for `LineQid` and `LineQubit`."""

    def __init__(self, x: int) -> None:
        """Initializes a line qubit at the given x coordinate."""
        self.x = x

    def _comparison_key(self):
        return self.x

    def with_levels(self, levels: int) -> 'LineQid':
        return LineQid(self.x, levels)

    def is_adjacent(self, other: ops.Qid) -> bool:
        """Determines if two qubits are adjacent line qubits."""
        return isinstance(other, _BaseLineQid) and abs(self.x - other.x) == 1

    @abc.abstractmethod
    def _with_x(self: TSelf, x: int) -> TSelf:
        '''Returns a qubit with the same type but a different value of `x`.'''

    def __add__(self: TSelf, other: int) -> TSelf:
        if not isinstance(other, int):
            raise TypeError('Can only add ints and {}. Instead was {}'.format(
                type(self).__name__, other))
        return self._with_x(self.x + other)

    def __sub__(self: TSelf, other: int) -> TSelf:
        if not isinstance(other, int):
            raise TypeError('Can only subtract ints and {}. Instead was {}'
                            ''.format(type(self).__name__, other))
        return self._with_x(self.x - other)

    def __radd__(self: TSelf, other: int) -> TSelf:
        return self + other

    def __rsub__(self: TSelf, other: int) -> TSelf:
        return -self + other

    def __neg__(self: TSelf) -> TSelf:
        return self._with_x(-self.x)


class LineQid(_BaseLineQid):
    """A qid on a 1d lattice with nearest-neighbor connectivity.

    `LineQid`s have a single attribute, and integer coordinate 'x', which
    identifies the qids location on the line. `LineQid`s are ordered by
    this integer.

    One can construct new `LineQid`s by adding or subtracting integers:

        >>> cirq.LineQid(1, levels=2) + 3
        cirq.LineQid(4, levels=2)

        >>> cirq.LineQid(2, levels=3) - 1
        cirq.LineQid(1, levels=3)
    """

    def __init__(self, x: int, levels: int) -> None:
        """Initializes a line qid at the given x coordinate.

        Args:
            x: The x coordinate.
            levels: The number of quantum levels.
        """
        super().__init__(x)
        self._levels = levels
        self.validate_levels(levels)

    @property
    def levels(self):
        return self._levels

    def _with_x(self, x: int) -> 'LineQid':
        return LineQid(x, levels=self.levels)

    @staticmethod
    def range(*range_args, levels: int) -> List['LineQid']:
        """Returns a range of line qids.

        Args:
            *range_args: Same arguments as python's built-in range method.
            levels: The number of quantum levels.

        Returns:
            A list of line qids.
        """
        return [LineQid(i, levels=levels) for i in range(*range_args)]

    @staticmethod
    def for_qid_shape(qid_shape: Sequence[int], start: int = 0,
                      step: int = 1) -> List['LineQid']:
        """Returns a range of line qids for each entry in `qid_shape` with
        matching levels.

        Args:
            qid_shape: A sequence of levels for for each `LineQid` to create.
            start: The x coordinate of the first `LineQid`.
            step: The amount to increment each x coordinate.
        """
        return [
            LineQid(start + step * i, levels=levels)
            for i, levels in enumerate(qid_shape)
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

    def __repr__(self):
        return 'cirq.LineQid({}, levels={})'.format(self.x, self.levels)

    def __str__(self):
        return '{!s} (d={})'.format(self.x, self.levels)

    def _json_dict_(self):
        return protocols.to_json_dict(self, ['x', 'levels'])


class LineQubit(_BaseLineQid):
    """A qubit on a 1d lattice with nearest-neighbor connectivity.

    LineQubits have a single attribute, and integer coordinate 'x', which
    identifies the qubits location on the line. LineQubits are ordered by
    this integer.

    One can construct new LineQubits by adding or subtracting integers:

        >>> cirq.LineQubit(1) + 3
        cirq.LineQubit(4)

        >>> cirq.LineQubit(2) - 1
        cirq.LineQubit(1)
    """

    @property
    def levels(self) -> int:
        return 2

    def _with_x(self, x: int) -> 'LineQubit':
        return LineQubit(x)

    @staticmethod
    def range(*range_args) -> List['LineQubit']:
        """Returns a range of line qubits.

        Args:
            *range_args: Same arguments as python's built-in range method.

        Returns:
            A list of line qubits.
        """
        return [LineQubit(i) for i in range(*range_args)]

    def __repr__(self):
        return 'cirq.LineQubit({})'.format(self.x)

    def __str__(self):
        return '{}'.format(self.x)

    def _json_dict_(self):
        return protocols.to_json_dict(self, ['x'])
