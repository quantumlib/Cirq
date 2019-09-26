# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import Cell


class ExplicitOperationsCell(Cell):
    """A quirk cell with known body operations and basis change operations."""

    def __init__(self,
                 operations: Iterable[ops.Operation],
                 basis_change: Iterable[ops.Operation] = ()):
        self._operations = tuple(operations)
        self._basis_change = tuple(basis_change)

    def basis_change(self) -> 'cirq.OP_TREE':
        return self._basis_change

    def operations(self) -> 'cirq.OP_TREE':
        return self._operations

    def controlled_by(self, qubit: 'cirq.Qid'):
        return ExplicitOperationsCell(
            [op.controlled_by(qubit) for op in self._operations],
            self._basis_change)
