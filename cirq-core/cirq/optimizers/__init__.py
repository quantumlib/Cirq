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

"""Classes and methods that optimize quantum circuits."""

from cirq.optimizers.align_left import (
    AlignLeft,
)

from cirq.optimizers.align_right import (
    AlignRight,
)

from cirq.optimizers.drop_empty_moments import (
    DropEmptyMoments,
)

from cirq.optimizers.drop_negligible import (
    DropNegligible,
)

from cirq.optimizers.convert_to_cz_and_single_gates import (
    ConvertToCzAndSingleGates,
)

from cirq.optimizers.eject_phased_paulis import (
    EjectPhasedPaulis,
)

from cirq.optimizers.eject_z import (
    EjectZ,
)

from cirq.optimizers.expand_composite import (
    ExpandComposite,
)

from cirq.optimizers.merge_interactions import (
    MergeInteractions,
)

from cirq.optimizers.merge_interactions_to_sqrt_iswap import (
    MergeInteractionsToSqrtIswap,
)

from cirq.optimizers.merge_single_qubit_gates import (
    merge_single_qubit_gates_into_phased_x_z,
    merge_single_qubit_gates_into_phxz,
    MergeSingleQubitGates,
)

from cirq.optimizers.synchronize_terminal_measurements import (
    SynchronizeTerminalMeasurements,
)

from cirq.transformers.stratify import stratified_circuit

from cirq.transformers.analytical_decompositions import (
    compute_cphase_exponents_for_fsim_decomposition,
    decompose_cphase_into_two_fsim,
    decompose_clifford_tableau_to_operations,
    decompose_multi_controlled_x,
    decompose_multi_controlled_rotation,
    decompose_two_qubit_interaction_into_four_fsim_gates,
    is_negligible_turn,
    prepare_two_qubit_state_using_cz,
    prepare_two_qubit_state_using_sqrt_iswap,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z,
    single_qubit_matrix_to_phxz,
    single_qubit_op_to_framed_phase_form,
    three_qubit_matrix_to_operations,
    two_qubit_matrix_to_cz_operations,
    two_qubit_matrix_to_diagonal_and_cz_operations,
    two_qubit_matrix_to_sqrt_iswap_operations,
)

from cirq import _compat

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.clifford_decomposition",
    old_parent="cirq.optimizers",
    old_child="clifford_decomposition",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.cphase_to_fsim",
    old_parent="cirq.optimizers",
    old_child="cphase_to_fsim",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.controlled_gate_decomposition",
    old_parent="cirq.optimizers",
    old_child="controlled_gate_decomposition",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.single_qubit_decompositions",
    old_parent="cirq.optimizers",
    old_child="decompositions",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.three_qubit_decomposition",
    old_parent="cirq.optimizers",
    old_child="three_qubit_decomposition",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.two_qubit_to_cz",
    old_parent="cirq.optimizers",
    old_child="two_qubit_decompositions",
    deadline="v0.16",
    create_attribute=True,
)


_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.two_qubit_to_fsim",
    old_parent="cirq.optimizers",
    old_child="two_qubit_to_fsim",
    deadline="v0.16",
    create_attribute=True,
)


_compat.deprecated_submodule(
    new_module_name="cirq.transformers.analytical_decompositions.two_qubit_to_sqrt_iswap",
    old_parent="cirq.optimizers",
    old_child="two_qubit_to_sqrt_iswap",
    deadline="v0.16",
    create_attribute=True,
)

_compat.deprecated_submodule(
    new_module_name="cirq.transformers.stratify",
    old_parent="cirq.optimizers",
    old_child="stratify",
    deadline="v0.16",
    create_attribute=True,
)
