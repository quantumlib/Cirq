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


from typing import Dict, List, Tuple

from cirq import ops


class GridQubit(ops.Qid):
    """A qubit on a 2d square lattice.

    GridQubits use row-major ordering:

        GridQubit(0, 0) < GridQubit(0, 1) < GridQubit(1, 0) < GridQubit(1, 1)

    New GridQubits can be constructed by adding or subtracting tuples

        >>> cirq.GridQubit(2, 3) + (3, 1)
        cirq.GridQubit(5, 4)

        >>> cirq.GridQubit(2, 3) - (1, 2)
        cirq.GridQubit(1, 1)
    """

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def _comparison_key(self):
        return self.row, self.col

    def is_adjacent(self, other: ops.Qid) -> bool:
        """Determines if two qubits are adjacent qubits."""
        return (isinstance(other, GridQubit) and
                abs(self.row - other.row) + abs(self.col - other.col) == 1)

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
        GridQubits in a pyramid structure.
        ---A---
        --AAA--
        -AAAAA-
        AAAAAAA

        You can use any character other than a hyphen to mark a qubit. As an
        example, the qubits for the Bristlecone device could be represented by
        the below diagram. This produces a diamond-shaped grid of qubits, and
        qubits with the same letter correspond to the same readout line.

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
                a row. Alphanumeric characters are assigned as qubits.
                Dots ('.'), dashes ('-'), and spaces (' ') are treated as
                empty locations in the grid. If diagram has characters other
                than alphanumerics, spacers, and newlines ('\n'), an error will
                be thrown. The top-left corner of the diagram will be have
                coordinate (0,0).

        Returns:
            A list of GridQubits corresponding to the provided diagram

        Raises:
            ValueError: If the input string contains an invalid character.
        """
        lines = diagram.strip().split('\n')
        no_qubit_characters = ['.', '-', ' ']
        qubits = []
        for row, line in enumerate(lines):
            for col, c in enumerate(line.strip()):
                if c not in no_qubit_characters:
                    if not c.isalnum():
                        raise ValueError("Input string has invalid character")
                    qubits.append(GridQubit(row, col))
        return qubits

    def __repr__(self):
        return 'cirq.GridQubit({}, {})'.format(self.row, self.col)

    def __str__(self):
        return '({}, {})'.format(self.row, self.col)

    def __add__(self, other: Tuple[int, int]) -> 'GridQubit':
        if not (isinstance(other, tuple) and len(other) == 2 and
                all(isinstance(x, int) for x in other)):
            raise TypeError(
                'Can only add tuples of length 2 to GridQubits. Was {}'.format(
                    other))
        return GridQubit(row=self.row + other[0], col=self.col + other[1])

    def __sub__(self, other: Tuple[int, int]) -> 'GridQubit':
        if not (isinstance(other, tuple) and len(other) == 2 and
                all(isinstance(x, int) for x in other)):
            raise TypeError(
                'Can only subtract tuples of length 2 to GridQubits. Was {}'.
                format(other))
        return GridQubit(row=self.row - other[0], col=self.col - other[1])

    def __radd__(self, other: Tuple[int, int]) -> 'GridQubit':
        return self + other

    def __rsub__(self, other: Tuple[int, int]) -> 'GridQubit':
        return -self + other

    def __neg__(self) -> 'GridQubit':
        return GridQubit(row=-self.row, col=-self.col)

    def to_proto_dict(self, v2_proto=False) -> Dict:
        """Return the proto in dictionary form."""
        # TODO: Deprecate v1 proto method.
        return {
            'row': self.row,
            'col': self.col,
        }

    def proto_id(self) -> str:
        return '{}_{}'.format(self.row, self.col)

    @staticmethod
    def from_proto_dict(proto_dict: Dict) -> 'GridQubit':
        """Proto dict must have 'row' and 'col' keys."""
        # TODO: Deprecate v1 proto method.
        if 'row' not in proto_dict or 'col' not in proto_dict:
            raise ValueError(
                'Proto dict does not contain row or col: {}'.format(proto_dict))
        return GridQubit(row=proto_dict['row'], col=proto_dict['col'])

    @staticmethod
    def from_proto_id(proto_id: str) -> 'GridQubit':
        row, col = proto_id.split('_')
        return GridQubit(row=int(row), col=int(col))
