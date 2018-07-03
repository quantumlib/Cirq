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

from typing import Dict, List, Optional
from cirq.google import XmonQubit
from cirq.line import LineQubit


class LineSequence:

    def __init__(self,
                 qubits: List[LineQubit],
                 line: List[XmonQubit]) -> None:
        self._line = line
        self._qubits = qubits

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._qubits == other._qubits and self._line == other._line

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((tuple(self._qubits), tuple(self._line)))

    def map(self) -> Dict[LineQubit, XmonQubit]:
        if len(self._line) < len(self._qubits):
            raise ValueError('Line sequence is too short')
        return {self._qubits[i]: self._line[i] for i in range(len(self._line))}


class LinePlacement:

    def __init__(self, lines: List[LineSequence]) -> None:
        self.lines = lines

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.lines == other.lines

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self.lines))

    def get(self):
        """Gives the preferred result.

        Returns:
            The preferred, error-efficient sequence found.
        """
        # TODO: Introduce error-cost optimizations.
        return self.longest()

    def longest(self) -> Optional[LineSequence]:
        """Gives the longest sequence found.

        Returns:
            The longest sequence from the sequences list. If more than one
            longest sequence exist, the first one is returned. None is returned
            for empty list.
        """
        if self.lines:
            return max(self.lines, key=lambda sequence: len(sequence))
        return None
