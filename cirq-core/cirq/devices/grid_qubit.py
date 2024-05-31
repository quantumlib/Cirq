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
import weakref
from typing import Any, Dict, Iterable, List, Optional, Tuple, Set, TYPE_CHECKING, Union
from typing_extensions import Self

import numpy as np

from cirq import ops, protocols

if TYPE_CHECKING:
    import cirq


@functools.total_ordering
class _BaseGridQid(ops.Qid):
    """The Base class for `GridQid` and `GridQubit`."""

    _row: int
    _col: int
    _dimension: int
    _comp_key: Optional[Tuple[int, int]] = None
    _hash: Optional[int] = None

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((self._row, self._col, self._dimension))
        return self._hash

    def __eq__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            return self is other or (
                self._row == other._row
                and self._col == other._col
                and self._dimension == other._dimension
            )
        return NotImplemented

    def __ne__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            return self is not other and (
                self._row != other._row
                or self._col != other._col
                or self._dimension != other._dimension
            )
        return NotImplemented

    def __lt__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            k0, k1 = self._comparison_key(), other._comparison_key()
            return k0 < k1 or (k0 == k1 and self._dimension < other._dimension)
        return super().__lt__(other)

    def __le__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            k0, k1 = self._comparison_key(), other._comparison_key()
            return k0 < k1 or (k0 == k1 and self._dimension <= other._dimension)
        return super().__le__(other)

    def __ge__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            k0, k1 = self._comparison_key(), other._comparison_key()
            return k0 > k1 or (k0 == k1 and self._dimension >= other._dimension)
        return super().__ge__(other)

    def __gt__(self, other) -> bool:
        # Explicitly implemented for performance (vs delegating to Qid).
        if isinstance(other, _BaseGridQid):
            k0, k1 = self._comparison_key(), other._comparison_key()
            return k0 > k1 or (k0 == k1 and self._dimension > other._dimension)
        return super().__gt__(other)

    def _comparison_key(self):
        if self._comp_key is None:
            self._comp_key = self._row, self._col
        return self._comp_key

    @property
    def row(self) -> int:
        return self._row

    @property
    def col(self) -> int:
        return self._col

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> 'GridQid':
        return GridQid(self._row, self._col, dimension=dimension)

    def is_adjacent(self, other: 'cirq.Qid') -> bool:
        """Determines if two qubits are adjacent qubits."""
        return (
            isinstance(other, GridQubit)
            and abs(self._row - other._row) + abs(self._col - other._col) == 1
        )

    def neighbors(self, qids: Optional[Iterable[ops.Qid]] = None) -> Set['_BaseGridQid']:
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
    def _with_row_col(self, row: int, col: int) -> Self:
        """Returns a qid with the same type but a different coordinate."""

    def __complex__(self) -> complex:
        return self._col + 1j * self._row

    def __add__(self, other: Union[Tuple[int, int], Self]) -> Self:
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(
                    "Can only add GridQids with identical dimension. "
                    f"Got {self.dimension} and {other.dimension}"
                )
            return self._with_row_col(row=self._row + other._row, col=self._col + other._col)
        if not (
            isinstance(other, (tuple, np.ndarray))
            and len(other) == 2
            and all(isinstance(x, (int, np.integer)) for x in other)
        ):
            raise TypeError(
                'Can only add integer tuples of length 2 to '
                f'{type(self).__name__}. Instead was {other}'
            )
        return self._with_row_col(row=self._row + other[0], col=self._col + other[1])

    def __sub__(self, other: Union[Tuple[int, int], Self]) -> Self:
        if isinstance(other, _BaseGridQid):
            if self.dimension != other.dimension:
                raise TypeError(
                    "Can only subtract GridQids with identical dimension. "
                    f"Got {self.dimension} and {other.dimension}"
                )
            return self._with_row_col(row=self._row - other._row, col=self._col - other._col)
        if not (
            isinstance(other, (tuple, np.ndarray))
            and len(other) == 2
            and all(isinstance(x, (int, np.integer)) for x in other)
        ):
            raise TypeError(
                "Can only subtract integer tuples of length 2 to "
                f"{type(self).__name__}. Instead was {other}"
            )
        return self._with_row_col(row=self._row - other[0], col=self._col - other[1])

    def __radd__(self, other: Tuple[int, int]) -> Self:
        return self + other

    def __rsub__(self, other: Tuple[int, int]) -> Self:
        return -self + other

    def __neg__(self) -> Self:
        return self._with_row_col(row=-self._row, col=-self._col)


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

    # Cache of existing GridQid instances, returned by __new__ if available.
    # Holds weak references so instances can still be garbage collected.
    _cache = weakref.WeakValueDictionary[Tuple[int, int, int], 'cirq.GridQid']()

    def __new__(cls, row: int, col: int, *, dimension: int) -> 'cirq.GridQid':
        """Creates a grid qid at the given row, col coordinate

        Args:
            row: the row coordinate
            col: the column coordinate
            dimension: The dimension of the qid's Hilbert space, i.e.
                the number of quantum levels.
        """
        key = (row, col, dimension)
        inst = cls._cache.get(key)
        if inst is None:
            cls.validate_dimension(dimension)
            inst = super().__new__(cls)
            inst._row = row
            inst._col = col
            inst._dimension = dimension
            cls._cache[key] = inst
        return inst

    def __getnewargs_ex__(self):
        """Returns a tuple of (args, kwargs) to pass to __new__ when unpickling."""
        return (self._row, self._col), {"dimension": self._dimension}

    def _with_row_col(self, row: int, col: int) -> 'GridQid':
        return GridQid(row, col, dimension=self._dimension)

    @staticmethod
    def square(diameter: int, top: int = 0, left: int = 0, *, dimension: int) -> List['GridQid']:
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
        return GridQid.rect(diameter, diameter, top=top, left=left, dimension=dimension)

    @staticmethod
    def rect(
        rows: int, cols: int, top: int = 0, left: int = 0, *, dimension: int
    ) -> List['GridQid']:
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
        """Parse ASCII art device layout into a device.

        As an example, the below diagram will create a list of GridQid in a
        pyramid structure.


        ```
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA
        ```

        You can use any character other than a hyphen, period or space to mark a
        qid. As an example, the qids for a Bristlecone device could be
        represented by the below diagram. This produces a diamond-shaped grid of
        qids, and qids with the same letter correspond to the same readout line.

        ```
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
        ```

        Args:
            diagram: String representing the qid layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0, 0).

            dimension: The dimension of the qubits in the `cirq.GridQid`s used
                in this construction.

        Returns:
            A list of `cirq.GridQid`s corresponding to qids in the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        coords = _ascii_diagram_to_coords(diagram)
        return [GridQid(*c, dimension=dimension) for c in coords]

    def __repr__(self) -> str:
        return f"cirq.GridQid({self._row}, {self._col}, dimension={self._dimension})"

    def __str__(self) -> str:
        return f"q({self._row}, {self._col}) (d={self._dimension})"

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(
            wire_symbols=(f"({self._row}, {self._col}) (d={self._dimension})",)
        )

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

    _dimension = 2

    # Cache of existing GridQubit instances, returned by __new__ if available.
    # Holds weak references so instances can still be garbage collected.
    _cache = weakref.WeakValueDictionary[Tuple[int, int], 'cirq.GridQubit']()

    def __new__(cls, row: int, col: int) -> 'cirq.GridQubit':
        """Creates a grid qubit at the given row, col coordinate

        Args:
            row: the row coordinate
            col: the column coordinate
        """
        key = (row, col)
        inst = cls._cache.get(key)
        if inst is None:
            inst = super().__new__(cls)
            inst._row = row
            inst._col = col
            cls._cache[key] = inst
        return inst

    def __getnewargs__(self):
        """Returns a tuple of args to pass to __new__ when unpickling."""
        return (self._row, self._col)

    def _with_row_col(self, row: int, col: int) -> 'GridQubit':
        return GridQubit(row, col)

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
    def rect(rows: int, cols: int, top: int = 0, left: int = 0) -> List['GridQubit']:
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
        """Parse ASCII art into device layout info.

        As an example, the below diagram will create a list of
        GridQubit in a pyramid structure.

        ```
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA
        ```

        You can use any character other than a hyphen, period or space to mark
        a qubit. As an example, the qubits for a Bristlecone device could be
        represented by the below diagram. This produces a diamond-shaped grid
        of qids, and qids with the same letter correspond to the same readout
        line.

        ```
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
        ```

        Args:
            diagram: String representing the qubit layout. Each line represents
                a row. Alphanumeric characters are assigned as qid.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\\n'), an error will
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
        return f"cirq.GridQubit({self._row}, {self._col})"

    def __str__(self) -> str:
        return f"q({self._row}, {self._col})"

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        return protocols.CircuitDiagramInfo(wire_symbols=(f"({self._row}, {self._col})",))

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
