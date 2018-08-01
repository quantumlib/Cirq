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

from typing import List, Iterable

from cirq.circuits import TextDiagramDrawer
from cirq.devices import GridQubit


LineSequence = List[GridQubit]


class NotFoundError(Exception):
    pass


class GridQubitLineTuple(tuple):
    """A contiguous non-overlapping sequence of adjacent grid qubits."""

    @staticmethod
    def best_of(lines: Iterable[LineSequence],
                length: int) -> 'GridQubitLineTuple':
        lines = list(lines)
        longest = max(lines, key=len) if lines else []
        if len(longest) < length:
            raise NotFoundError('No line placement with desired length found.')
        return GridQubitLineTuple(longest[:length])

    def __str__(self):
        diagram = TextDiagramDrawer()
        dx = min(q.col for q in self)
        dy = min(q.row for q in self)

        for q in self:
            diagram.write(q.col - dx, q.row - dy, str(q))

        for q1, q2 in zip(self, self[1:]):
            diagram.grid_line(q1.col - dx, q1.row - dy,
                              q2.col - dx, q2.row - dy,
                              True)

        return diagram.render(horizontal_spacing=2,
                              vertical_spacing=1,
                              use_unicode_characters=True)
