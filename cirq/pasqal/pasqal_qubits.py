# Copyright 2020 The Cirq Developers
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
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from numpy import sqrt, isclose

import cirq


class ThreeDGridQubit(cirq.ops.Qid):
    """A qubit on a 3d lattice.

    ThreeDGridQubits use row-column-layer ordering:

        ThreeDGridQubit(0, 0, 0) < ThreeDGridQubit(0, 0, 1)
        < ThreeDGridQubit(0, 1, 0)< ThreeDGridQubit(1, 0, 0)
        < ThreeDGridQubit(0, 1, 1)< ThreeDGridQubit(1, 0, 1)
        < ThreeDGridQubit(1, 1, 0)< ThreeDGridQubit(1, 1, 1)

    New ThreeDGridQubit can be constructed by adding or subtracting tuples

        >>> cirq.pasqal.ThreeDGridQubit(2, 3, 4) + (3, 1, 6)
        pasqal.ThreeDGridQubit(5, 4, 10)

        >>> cirq.pasqal.ThreeDGridQubit(2, 3, 4) - (1, 2, 2)
        pasqal.ThreeDGridQubit(1, 1, 2)
    """

    def __init__(self, row: int, col: int, lay: int):
        self.row = row
        self.col = col
        self.lay = lay

    def _comparison_key(self):
        return self.row, self.col, self.lay

    @property
    def dimension(self) -> int:
        return 2

    def is_adjacent(self, other: cirq.ops.Qid) -> bool:
        """Determines if two qubits are adjacent qubits."""
        return isclose(self.distance(other), 1)

    def distance(self, other: cirq.ops.Qid) -> float:
        """Returns the distance between two qubits in a 3D grid."""
        if not isinstance(other, ThreeDGridQubit):
            raise TypeError(
                "Can compute distance to another ThreeDGridQubit, but {}".
                format(other))
        return sqrt((self.row - other.row)**2 + (self.col - other.col)**2 +
                    (self.lay - other.lay)**2)

    def neighbors(self, qids: Optional[Iterable[cirq.ops.Qid]] = None
                 ) -> Set['ThreeDGridQubit']:
        """Returns qubits that are potential neighbors to this ThreeDGridQubit

        Args:
            qids: optional Iterable of qubits to constrain neighbors to.
        """
        neighbors = set()
        for q in [
                self + (0, 0, 1), self + (0, 1, 0), self + (1, 0, 0),
                self + (0, 0, -1), self + (0, -1, 0), self + (-1, 0, 0)
        ]:
            if qids is None or q in qids:
                neighbors.add(q)
        return neighbors

    @staticmethod
    def cube(diameter: int, top: int = 0, left: int = 0,
             upper: int = 0) -> List['ThreeDGridQubit']:
        """Returns a cube of ThreeDGridQubits.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row
            upper: Column number of the uppermost layer

        Returns:
            A list of ThreeDGridQubits filling in a square grid
        """
        return ThreeDGridQubit.parallelep(diameter,
                                          diameter,
                                          diameter,
                                          top=top,
                                          left=left,
                                          upper=upper)

    @staticmethod
    def parallelep(rows: int,
                   cols: int,
                   lays: int,
                   top: int = 0,
                   left: int = 0,
                   upper: int = 0) -> List['ThreeDGridQubit']:
        """Returns a parallelepiped of ThreeDGridQubits.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of ThreeDGridQubits filling in a rectangular grid
        """
        return [
            ThreeDGridQubit(row, col, lay) for row in range(top, top + rows)
            for col in range(left, left + cols)
            for lay in range(upper, upper + lays)
        ]

    @staticmethod
    def square(diameter: int, top: int = 0,
               left: int = 0) -> List['ThreeDGridQubit']:
        """Returns a square of ThreeDGridQubits.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of ThreeDGridQubits filling in a square grid
        """
        return ThreeDGridQubit.rect(diameter, diameter, top=top, left=left)

    @staticmethod
    def rect(rows: int, cols: int, top: int = 0,
             left: int = 0) -> List['ThreeDGridQubit']:
        """Returns a rectangle of ThreeDGridQubits.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of ThreeDGridQubits filling in a rectangular grid
        """
        return [
            ThreeDGridQubit(row, col, 0)
            for row in range(top, top + rows)
            for col in range(left, left + cols)
        ]

    def __repr__(self) -> str:
        return f'pasqal.ThreeDGridQubit({self.row}, {self.col}, {self.lay})'

    def __str__(self) -> str:
        return f'({self.row}, {self.col}, {self.lay})'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['row', 'col', 'lay'])

    def __add__(self, other: Tuple[int, int, int]) -> 'ThreeDGridQubit':
        if not (isinstance(other, tuple) and len(other) == 3 and
                all(isinstance(x, int) for x in other)):
            raise TypeError(
                'Can only add tuples of length 3. Was {}'.format(other))
        return ThreeDGridQubit(row=self.row + other[0],
                               col=self.col + other[1],
                               lay=self.lay + other[2])

    def __sub__(self, other: Tuple[int, int, int]) -> 'ThreeDGridQubit':
        if not (isinstance(other, tuple) and len(other) == 3 and
                all(isinstance(x, int) for x in other)):
            raise TypeError(
                'Can only subtract tuples of length 3. Was {}'.format(other))
        return ThreeDGridQubit(row=self.row - other[0],
                               col=self.col - other[1],
                               lay=self.lay - other[2])

    def __radd__(self, other: Tuple[int, int, int]) -> 'ThreeDGridQubit':
        return self + other

    def __rsub__(self, other: Tuple[int, int, int]) -> 'ThreeDGridQubit':
        return -self + other

    def __neg__(self) -> 'ThreeDGridQubit':
        return ThreeDGridQubit(row=-self.row, col=-self.col, lay=-self.lay)
