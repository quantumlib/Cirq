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

"""This module defines building blocks for parsing Quirk circuits."""


from cirq.contrib.quirk.cells.all_cells import (
    generate_all_cells
)

from cirq.contrib.quirk.cells.arithmetic_cells import (
    ArithmeticCell,
    QuirkArithmeticOperation,
)

from cirq.contrib.quirk.cells.cell import (
    Cell,
    CELL_SIZES,
    CellMaker,
    CellMakerArgs,
)

from cirq.contrib.quirk.cells.control_cells import (
    ControlCell,
    ParityControlCell,
)

from cirq.contrib.quirk.cells.explicit_operations_cell import (
    ExplicitOperationsCell,
)

from cirq.contrib.quirk.cells.input_cells import (
    InputCell,
    SetDefaultInputCell,
)

from cirq.contrib.quirk.cells.input_rotation_cells import (
    InputRotationCell,
    QuirkInputRotationOperation,
)

from cirq.contrib.quirk.cells.qubit_permutation_cells import (
    QuirkQubitPermutationOperation,
)

from cirq.contrib.quirk.cells.swap_cell import (
    SwapCell,
)
