# Copyright 2019 The Cirq Developers
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
from typing import Iterator, TYPE_CHECKING

from cirq import ops
from cirq.contrib.quirk.cells.cell import CellMaker, ExplicitOperationsCell

if TYPE_CHECKING:
    import cirq


def generate_all_scalar_cell_makers() -> Iterator[CellMaker]:
    yield _scalar("NeGate", ops.GlobalPhaseOperation(-1))
    yield _scalar("i", ops.GlobalPhaseOperation(1j))
    yield _scalar("-i", ops.GlobalPhaseOperation(-1j))
    yield _scalar("√i", ops.GlobalPhaseOperation(1j**0.5))
    yield _scalar("√-i", ops.GlobalPhaseOperation((-1j)**0.5))


def _scalar(identifier: str, operation: 'cirq.Operation') -> CellMaker:
    return CellMaker(identifier,
                     size=1,
                     maker=lambda _: ExplicitOperationsCell([operation]))
