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

"""Types for representing and methods for manipulating circuit operation trees.
"""

from cirq.ops.common_gates import (
    CNOT,
    CNotGate,
    CZ,
    H,
    HGate,
    ISWAP,
    ISwapGate,
    measure,
    measure_each,
    MeasurementGate,
    Rot11Gate,
    RotXGate,
    RotYGate,
    RotZGate,
    S,
    SWAP,
    SwapGate,
    T,
    X,
    Y,
    Z,
)
from cirq.ops.controlled_gate import (
    ControlledGate,
)
from cirq.ops.eigen_gate import (
    EigenGate,
)
from cirq.ops.gate_features import (
    BoundedEffect,
    CompositeGate,
    CompositeOperation,
    ExtrapolatableEffect,
    InterchangeableQubitsGate,
    KnownMatrix,
    ParameterizableEffect,
    PhaseableEffect,
    ReversibleEffect,
    SingleQubitGate,
    TextDiagrammable,
    TextDiagramInfo,
    TextDiagramInfoArgs,
    ThreeQubitGate,
    TwoQubitGate,
)
from cirq.ops.gate_operation import (
    GateOperation,
)
from cirq.ops.qubit_order import (
    QubitOrder,
)
from cirq.ops.qubit_order_or_list import (
    QubitOrderOrList,
)
from cirq.ops.matrix_gates import (
    SingleQubitMatrixGate,
    TwoQubitMatrixGate,
)
from cirq.ops.op_tree import (
    OP_TREE,
    flatten_op_tree,
    freeze_op_tree,
    transform_op_tree,
)
from cirq.ops.raw_types import (
    Gate,
    NamedQubit,
    Operation,
    QubitId,
)
from cirq.ops.reversible_composite_gate import (
    inverse,
    ReversibleCompositeGate,
)
from cirq.ops.three_qubit_gates import (
    CCX,
    CCZ,
    CSWAP,
    FREDKIN,
    TOFFOLI,
)
