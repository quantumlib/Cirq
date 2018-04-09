# Copyright 2018 Google LLC
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
import re

from cirq.api.google.v1 import operations_pb2
from cirq.ops import QubitId


class XmonQubit(QubitId):
    """A qubit at a location on an xmon chip."""

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def is_adjacent(self, other: 'XmonQubit'):
        return abs(self.row - other.row) + abs(self.col - other.col) == 1

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((XmonQubit, self.row, self.col))

    def __repr__(self):
        return 'XmonQubit({}, {})'.format(self.row, self.col)

    def __str__(self):
        return '({}, {})'.format(self.row, self.col)

    @staticmethod
    def try_parse_from_ascii(text):
        if re.match('\\(\\s*\\d+,\\s*\\d+\\s*\\)', text):
            a, b = text[1:-1].split(',')
            return XmonQubit(int(a.strip()), int(b.strip()))
        return None

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
