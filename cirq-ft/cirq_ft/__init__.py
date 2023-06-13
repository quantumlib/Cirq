# Copyright 2023 The Cirq Developers
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

from cirq_ft._version import __version__
from cirq_ft.algos import (
    QROM,
    AdditionGate,
    AddMod,
    And,
    ApplyGateToLthQubit,
    ContiguousRegisterGate,
    GenericSelect,
    LessThanEqualGate,
    LessThanGate,
    MultiControlPauli,
    MultiTargetCNOT,
    MultiTargetCSwap,
    MultiTargetCSwapApprox,
    PrepareHubbard,
    PrepareOracle,
    PrepareUniformSuperposition,
    ProgrammableRotationGateArray,
    ProgrammableRotationGateArrayBase,
    QubitizationWalkOperator,
    ReflectionUsingPrepare,
    SelectedMajoranaFermionGate,
    SelectHubbard,
    SelectOracle,
    SelectSwapQROM,
    StatePreparationAliasSampling,
    SwapWithZeroGate,
    UnaryIterationGate,
    unary_iteration,
)
from cirq_ft.infra import (
    GateWithRegisters,
    GreedyQubitManager,
    Register,
    Registers,
    SelectionRegister,
    SelectionRegisters,
    TComplexity,
    map_clean_and_borrowable_qubits,
    t_complexity,
    testing,
)
