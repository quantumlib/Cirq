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
from typing import Iterator, Optional, cast, Iterable

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.cell import CellMaker, ExplicitOperationsCell


def generate_all_measurement_cell_makers() -> Iterator[CellMaker]:
    yield reg_measurement("Measure")
    yield reg_measurement("ZDetector")
    yield reg_measurement("YDetector", basis_change=ops.X**-0.5)
    yield reg_measurement("XDetector", basis_change=ops.Y**0.5)


def reg_measurement(identifier: str,
                    basis_change: Optional['cirq.Gate'] = None) -> CellMaker:
    return CellMaker(
        identifier=identifier,
        size=1,
        maker=lambda args: ExplicitOperationsCell(
            [ops.measure(*args.qubits, key=f'row={args.row},col={args.col}')],
            basis_change=cast(Iterable['cirq.Operation'], [basis_change.on(*args.qubits)]
            if basis_change else ())))
