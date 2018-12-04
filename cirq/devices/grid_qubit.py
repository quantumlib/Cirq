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


from typing import Dict

from cirq import ops


class GridQubit(ops.QubitId):
    """A qubit on a 2d square lattice.

    GridQubits use row-major ordering:

        GridQubit(0, 0) < GridQubit(0, 1) < GridQubit(1, 0) < GridQubit(1, 1)
    """

    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def _comparison_key(self):
        return self.row, self.col

    def is_adjacent(self, other: ops.QubitId) -> bool:
        """Determines if two qubits are adjacent qubits."""
        return (isinstance(other, GridQubit) and
                abs(self.row - other.row) + abs(self.col - other.col) == 1)

    def __repr__(self):
        return 'cirq.GridQubit({}, {})'.format(self.row, self.col)

    def __str__(self):
        return '({}, {})'.format(self.row, self.col)

    def to_proto_dict(self) -> Dict:
        """Return the proto in dictionary form."""
        return {
            'row': self.row,
            'col': self.col,
        }

    @staticmethod
    def from_proto_dict(proto_dict: Dict) -> 'GridQubit':
        """Proto dict must have 'row' and 'col' keys."""
        if 'row' not in proto_dict or 'col' not in proto_dict:
            raise ValueError(
                'Proto dict does not contain row or col: {}'.format(proto_dict))
        return GridQubit(row=proto_dict['row'], col=proto_dict['col'])
