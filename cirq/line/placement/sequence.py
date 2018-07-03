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

from typing import List, Optional

from cirq.google import XmonQubit


class NotFoundError(Exception):
    pass


class LineSequence:

    def __init__(self, line: List[XmonQubit]) -> None:
        self.line = line

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.line == other.line

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(tuple(self.line))


class LinePlacement:

    def __init__(self, length: int, lines: List[LineSequence]) -> None:
        self.length = length
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
        """Retrieves the preferred line placement.

        Returns:
             The preferred, best line placement found.

        Raises:
            NotFoundError: When no line satisfying requirements was found.
        """
        best = self.longest()
        if len(best.line) < self.length:
            raise NotFoundError('No line placement with desired length found')
        return best

    def longest(self) -> Optional[LineSequence]:
        """Gives the longest sequence found.

        Returns:
            The longest sequence found. If more than one longest sequence
            exist, the first one is returned. None is returned if there are no
            sequences found.
        """
        if self.lines:
            return max(self.lines, key=lambda sequence: len(sequence.line))
        return None
