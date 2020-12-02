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
from typing import Iterator, Optional, cast, Iterable, TYPE_CHECKING

from cirq import ops
from cirq.interop.quirk.cells.cell import CellMaker, ExplicitOperationsCell

if TYPE_CHECKING:
    import cirq


def generate_all_measurement_cell_makers() -> Iterator[CellMaker]:
    yield _measurement("Measure")
    yield _measurement("ZDetector")
    yield _measurement("YDetector", basis_change=ops.X ** -0.5)
    yield _measurement("XDetector", basis_change=ops.Y ** 0.5)


def _measurement(identifier: str, basis_change: Optional['cirq.Gate'] = None) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        maker=lambda args: ExplicitOperationsCell(
            [ops.measure(*args.qubits, key=f'row={args.row},col={args.col}')],
            basis_change=cast(
                Iterable['cirq.Operation'], [basis_change.on(*args.qubits)] if basis_change else ()
            ),
        ),
    )
