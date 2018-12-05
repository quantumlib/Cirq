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

"""Circuit transformation utilities."""

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

from cirq.optimizers.merge_single_qubit_gates import (
    MergeSingleQubitGates,
    merge_single_qubit_gates_into_phased_x_z,
)

from cirq.optimizers.decompositions import (
    is_negligible_turn,
    single_qubit_matrix_to_gates,
    single_qubit_matrix_to_pauli_rotations,
    single_qubit_matrix_to_phased_x_z,
    single_qubit_op_to_framed_phase_form,
)

from cirq.optimizers.two_qubit_decompositions import (
    two_qubit_matrix_to_operations,
)
