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
from cirq.ops.gate_features import (
    BoundedEffectGate,
    CompositeGate,
    ExtrapolatableGate,
    KnownMatrixGate,
    PhaseableGate,
    ReversibleGate,
    SelfInverseGate,
    SingleQubitGate,
    TextDiagrammableGate,
    TwoQubitGate,
)
from cirq.ops.qubit_order import (
    QubitOrder,
    QubitOrderOrList,
)
from cirq.ops.line_qubit import (
    LineQubit,
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
    InterchangeableQubitsGate,
    NamedQubit,
    Operation,
    QubitId,
)
from cirq.ops.reversible_composite_gate import (
    inverse_of_invertible_op_tree,
    ReversibleCompositeGate,
)
