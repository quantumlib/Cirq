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

import operator

from cirq.api.google.v1 import operations_pb2
from cirq.ops import QubitId


class XmonQubit(QubitId):
    """A qubit at a location on an xmon chip.

    XmonQubits use row-major ordering:

        XmonQubit(0, 0) < XmonQubit(0, 1) < XmonQubit(1, 0) < XmonQubit(1, 1)
    """

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def is_adjacent(self, other: QubitId) -> bool:
        """Determines if two qubits are adjacent xmon qubits."""
        return (isinstance(other, XmonQubit) and
                abs(self.row - other.row) + abs(self.col - other.col) == 1)

    def _compare(self, other, op):
        if not isinstance(other, type(self)):
            return NotImplemented
        return op((self.row, self.col), (other.row, other.col))

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        return self._compare(other, operator.lt)

    def __gt__(self, other):
        return self._compare(other, operator.gt)

    def __le__(self, other):
        return self._compare(other, operator.le)

    def __ge__(self, other):
        return self._compare(other, operator.ge)

    def __hash__(self):
        return hash((XmonQubit, self.row, self.col))

    def __repr__(self):
        return 'XmonQubit({}, {})'.format(self.row, self.col)

    def __str__(self):
        return '({}, {})'.format(self.row, self.col)

    def to_proto(
            self, out: operations_pb2.Qubit = None) -> operations_pb2.Qubit:
        """Return the proto form, mutating supplied form if supplied."""
        if out is None:
            out = operations_pb2.Qubit()
        out.row = self.row
        out.col = self.col
        return out

    @staticmethod
    def from_proto(q: operations_pb2.Qubit) -> 'XmonQubit':
        return XmonQubit(row=q.row, col=q.col)
