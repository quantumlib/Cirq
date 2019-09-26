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
from typing import Iterator, Optional

import cirq
from cirq import ops
from cirq.contrib.quirk.cells.explicit_operations_cell import ExplicitOperationsCell
from cirq.contrib.quirk.cells.arithmetic_cells import \
    generate_all_arithmetic_cells
from cirq.contrib.quirk.cells.cell import (
    CellMaker,
)
from cirq.contrib.quirk.cells.control_cells import generate_all_control_cells
from cirq.contrib.quirk.cells.frequency_space_cells import \
    generate_all_frequency_space_cells
from cirq.contrib.quirk.cells.ignored_cells import generate_all_ignored_cells
from cirq.contrib.quirk.cells.input_cells import (
    generate_all_input_cells
)
from cirq.contrib.quirk.cells.input_rotation_cells import \
    generate_all_input_rotation_cells
from cirq.contrib.quirk.cells.qubit_permutation_cells import \
    generate_all_qubit_permutation_cells
from cirq.contrib.quirk.cells.scalar_cells import generate_all_scalar_cells
from cirq.contrib.quirk.cells.single_qubit_rotation_cells import \
    generate_all_single_qubit_rotation_cells
from cirq.contrib.quirk.cells.swap_cell import (
    generate_all_swap_cells)
from cirq.contrib.quirk.cells.unsupported_cells import \
    generate_all_unsupported_cells


def generate_all_measurement_cells() -> Iterator[CellMaker]:
    yield reg_measurement("Measure")
    yield reg_measurement("ZDetector")
    yield reg_measurement("YDetector", basis_change=ops.X**-0.5)
    yield reg_measurement("XDetector", basis_change=ops.Y**0.5)


def reg_measurement(identifier: str, basis_change: Optional['cirq.Gate'] = None) -> CellMaker:
    return CellMaker(
        identifier, 1, lambda args: ExplicitOperationsCell(
            [ops.measure(*args.qubits, key=f'row={args.row},col={args.col}')],
            basis_change=[basis_change.on(*args.qubits)]
            if basis_change else ()))
