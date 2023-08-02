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

from cirq_ft.algos.and_gate import And
from cirq_ft.algos.apply_gate_to_lth_target import ApplyGateToLthQubit
from cirq_ft.algos.arithmetic_gates import (
    AdditionGate,
    AddMod,
    ContiguousRegisterGate,
    LessThanEqualGate,
    LessThanGate,
    SingleQubitCompare,
    BiQubitsMixer,
)
from cirq_ft.algos.generic_select import GenericSelect
from cirq_ft.algos.hubbard_model import PrepareHubbard, SelectHubbard
from cirq_ft.algos.multi_control_multi_target_pauli import MultiControlPauli, MultiTargetCNOT
from cirq_ft.algos.prepare_uniform_superposition import PrepareUniformSuperposition
from cirq_ft.algos.programmable_rotation_gate_array import (
    ProgrammableRotationGateArray,
    ProgrammableRotationGateArrayBase,
)
from cirq_ft.algos.qrom import QROM
from cirq_ft.algos.qubitization_walk_operator import QubitizationWalkOperator
from cirq_ft.algos.reflection_using_prepare import ReflectionUsingPrepare
from cirq_ft.algos.select_and_prepare import PrepareOracle, SelectOracle
from cirq_ft.algos.select_swap_qrom import SelectSwapQROM
from cirq_ft.algos.selected_majorana_fermion import SelectedMajoranaFermionGate
from cirq_ft.algos.state_preparation import StatePreparationAliasSampling
from cirq_ft.algos.swap_network import MultiTargetCSwap, MultiTargetCSwapApprox, SwapWithZeroGate
from cirq_ft.algos.unary_iteration_gate import UnaryIterationGate, unary_iteration
