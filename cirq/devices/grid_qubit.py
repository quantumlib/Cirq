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

import functools
from typing import (Any, Dict, Iterable, List, Optional, Tuple, Set, TypeVar,
                    TYPE_CHECKING)

import abc

import numpy as np

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq

TSelf = TypeVar('TSelf', bound='_BaseGridQid')  # type: ignore


@functools.total_ordering  # type: ignore
class _BaseGridQid(ops.Qid):
    """The Base class for `GridQid` and `GridQubit`."""

    def __init__(self, row: int, col: int):
        self._row = row
        self._col = col

    def _comparison_key(self):
        return self._row, self._col

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    def with_dimension(self, dimension: int) -> 'GridQid':
        return GridQid(self.row, self.col, dimension=dimension)

    def is_adjacent(self, other: 'cirq.Qid') -> bool:
        """Determines if two qubits are adjacent qubits."""
        return (isinstance(other, GridQubit) and
                abs(self.row - other.row) + abs(self.col - other.col) == 1)

    def neighbors(self, qids: Optional[Iterable[ops.Qid]] = None
                 ) -> Set['_BaseGridQid']:
        """Returns qubits that are potential neighbors to this GridQid

        Args:
            qids: optional Iterable of qubits to constrain neighbors to.
        """
        neighbors = set()
        for q in [self + (0, 1), self + (1, 0), self + (-1, 0), self + (0, -1)]:
            if qids is None or q in qids:
                neighbors.add(q)
        return neighbors

    @abc.abstractmethod
    def _with_row_col(self: TSelf, row: int, col: int) -> TSelf:
        """Returns a qid with the same type but a different coordinate."""

    def __add__(self: TSelf, other: Tuple[int, int]) -> 'TSelf':
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(
                    "Can only add GridQids with identical dimension. "
                    f"Got {self.dimension} and {other.dimension}")
            return self._with_row_col(row=self.row + other.row,
                                      col=self.col + other.col)
        if not (isinstance(other, (tuple, np.ndarray)) and len(other) == 2 and
                all(isinstance(x, (int, np.integer)) for x in other)):
            raise TypeError('Can only add integer tuples of length 2 to '
                            f'{type(self).__name__}. Instead was {other}')
        return self._with_row_col(row=self.row + other[0],
                                  col=self.col + other[1])

    def __sub__(self: TSelf, other: Tuple[int, int]) -> 'TSelf':
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(
                    "Can only subtract GridQids with identical dimension. "
                    f"Got {self.dimension} and {other.dimension}")
            return self._with_row_col(row=self.row - other.row,
                                      col=self.col - other.col)
        if not (isinstance(other, (tuple, np.ndarray)) and len(other) == 2 and
                all(isinstance(x, (int, np.integer)) for x in other)):
            raise TypeError("Can only subtract integer tuples of length 2 to "
                            f"{type(self).__name__}. Instead was {other}")
        return self._with_row_col(row=self.row - other[0],
                                  col=self.col - other[1])

    def __radd__(self: TSelf, other: Tuple[int, int]) -> 'TSelf':
        return self + other

    def __rsub__(self: TSelf, other: Tuple[int, int]) -> 'TSelf':
        return -self + other

    def __neg__(self: TSelf) -> 'TSelf':
        return self._with_row_col(row=-self.row, col=-self.col)


class GridQid(_BaseGridQid):
    """A qid on a 2d square lattice

    GridQid uses row-major ordering:

        GridQid(0, 0, dimension=2) < GridQid(0, 1, dimension=2)
        < GridQid(1, 0, dimension=2) < GridQid(1, 1, dimension=2)

    New GridQid can be constructed by adding or subtracting tuples or numpy
    arrays

        >>> cirq.GridQid(2, 3, dimension=2) + (3, 1)
        cirq.GridQid(5, 4, dimension=2)

        >>> cirq.GridQid(2, 3, dimension=2) - (1, 2)
        cirq.GridQid(1, 1, dimension=2)

        >>> cirq.GridQid(2, 3, dimension=2) + np.array([3, 1], dtype=int)
        cirq.GridQid(5, 4, dimension=2)
    """

    def __init__(self, row: int, col: int, *, dimension: int) -> None:
        """Initializes a grid qid at the given row, col coordinate

        Args:
            row: the row coordinate
            col: the column coordinate
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        super().__init__(row, col)
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def dimension(self):
        return self._dimension

    def _with_row_col(self, row: int, col: int) -> 'GridQid':
        return GridQid(row, col, dimension=self.dimension)

    @staticmethod
    def square(diameter: int, top: int = 0, left: int = 0, *,
               dimension: int) -> List['GridQid']:
        """Returns a square of GridQid.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of GridQid filling in a square grid
        """
        return GridQid.rect(diameter,
                            diameter,
                            top=top,
                            left=left,
                            dimension=dimension)

    @staticmethod
    def rect(rows: int,
             cols: int,
             top: int = 0,
             left: int = 0,
             *,
             dimension: int) -> List['GridQid']:
        """Returns a rectangle of GridQid.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.

        Returns:
            A list of GridQid filling in a rectangular grid
        """
        return [
            GridQid(row, col, dimension=dimension)
            for row in range(top, top + rows)
            for col in range(left, left + cols)
        ]

    @staticmethod
    def from_diagram(diagram: str, dimension: int) -> List['GridQid']:
        """Parse ASCII art device layout into info about qids and
        connectivity. As an example, the below diagram will create a list of
        GridQid in a pyramid structure.
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA

        You can use any character other than a hyphen to mark a qid. As an
        example, the qids for the Bristlecone device could be represented by
        the below diagram. This produces a diamond-shaped grid of qids, and
        qids with the same letter correspond to the same readout line.

        .....AB.....
        ....ABCD....
        ...ABCDEF...
        ..ABCDEFGH..
        .ABCDEFGHIJ.
        ABCDEFGHIJKL
        .CDEFGHIJKL.
        ..EFGHIJKL..
        ...GHIJKL...
        ....IJKL....
        .....KL.....

        Args:
            diagram: String representing the qid layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQid corresponding to qids in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        coords = _ascii_diagram_to_coords(diagram)
        return [GridQid(*c, dimension=dimension) for c in coords]

    def __repr__(self) -> str:
        return f"cirq.GridQid({self.row}, {self.col}, " \
               f"dimension={self.dimension})"

    def __str__(self) -> str:
        return f"({self.row}, {self.col}) (d={self.dimension})"

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['row', 'col', 'dimension'])


class GridQubit(_BaseGridQid):
    """A qubit on a 2d square lattice.

    GridQubits use row-major ordering:

        GridQubit(0, 0) < GridQubit(0, 1) < GridQubit(1, 0) < GridQubit(1, 1)

    New GridQubits can be constructed by adding or subtracting tuples

        >>> cirq.GridQubit(2, 3) + (3, 1)
        cirq.GridQubit(5, 4)

        >>> cirq.GridQubit(2, 3) - (1, 2)
        cirq.GridQubit(1, 1)

        >>> cirq.GridQubit(2, 3,) + np.array([3, 1], dtype=int)
        cirq.GridQubit(5, 4)
    """

    @property
    def dimension(self) -> int:
        return 2

    def _with_row_col(self, row: int, col: int):
        return GridQubit(row, col)

    def _cmp_tuple(self):
        cls = GridQid if type(self) is GridQubit else type(self)
        # Must be same as Qid._cmp_tuple but with cls in place of type(self).
        return (cls.__name__, repr(cls), self._comparison_key(), self.dimension)

    @staticmethod
    def square(diameter: int, top: int = 0, left: int = 0) -> List['GridQubit']:
        """Returns a square of GridQubits.

        Args:
            diameter: Length of a side of the square
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of GridQubits filling in a square grid
        """
        return GridQubit.rect(diameter, diameter, top=top, left=left)

    @staticmethod
    def rect(rows: int, cols: int, top: int = 0,
             left: int = 0) -> List['GridQubit']:
        """Returns a rectangle of GridQubits.

        Args:
            rows: Number of rows in the rectangle
            cols: Number of columns in the rectangle
            top: Row number of the topmost row
            left: Column number of the leftmost row

        Returns:
            A list of GridQubits filling in a rectangular grid
        """
        return [
            GridQubit(row, col)
            for row in range(top, top + rows)
            for col in range(left, left + cols)
        ]

    @staticmethod
    def from_diagram(diagram: str) -> List['GridQubit']:
        """Parse ASCII art device layout into info about qubits and
        connectivity. As an example, the below diagram will create a list of
        GridQubit in a pyramid structure.
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA

        You can use any character other than a hyphen to mark a qubit. As an
        example, the qubits for the Bristlecone device could be represented by
        the below diagram. This produces a diamond-shaped grid of qids, and
        qids with the same letter correspond to the same readout line.

        .....AB.....
        ....ABCD....
        ...ABCDEF...
        ..ABCDEFGH..
        .ABCDEFGHIJ.
        ABCDEFGHIJKL
        .CDEFGHIJKL.
        ..EFGHIJKL..
        ...GHIJKL...
        ....IJKL....
        .....KL.....

        Args:
            diagram: String representing the qubit layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQubit corresponding to qubits in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        coords = _ascii_diagram_to_coords(diagram)
        return [GridQubit(*c) for c in coords]

    def __repr__(self) -> str:
        return f"cirq.GridQubit({self.row}, {self.col})"

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['row', 'col'])


def _ascii_diagram_to_coords(diagram: str) -> List[Tuple[int, int]]:
    """Parse ASCII art device layout into info about qids coordinates

    Args:
        diagram: String representing the qid layout. Each line represents
            a row. Alphanumeric characters are assigned as qid.
            Dots ('.'), dashes ('-'), and spaces (' ') are treated as
            empty locations in the grid. If diagram has characters other
            than alphanumerics, spacers, and newlines ('\n'), an error will
            be thrown. The top-left corner of the diagram will be have
            coordinate (0,0).

    Returns:
        A list of Tuples corresponding to the coordinates for qids in the
        provided diagram

    Raises:
        ValueError: If the input string contains an invalid character.
    """
    lines = diagram.strip().split('\n')
    no_qid_characters = ['.', '-', ' ']
    qid_coords = []
    for row, line in enumerate(lines):
        for col, c in enumerate(line.strip()):
            if c not in no_qid_characters:
                if not c.isalnum():
                    raise ValueError("Input string has invalid character")
                qid_coords.append((row, col))
    return qid_coords
