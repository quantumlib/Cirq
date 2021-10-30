# Copyright 2020 The Cirq Developers
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

"""Tools and methods for quantum information science."""

from cirq.qis.channels import (
    choi_to_kraus,
    choi_to_superoperator,
    kraus_to_choi,
    kraus_to_superoperator,
    operation_to_choi,
    operation_to_superoperator,
    superoperator_to_choi,
    superoperator_to_kraus,
)

from cirq.qis.clifford_tableau import CliffordTableau

from cirq.qis.measures import (
    entanglement_fidelity,
    fidelity,
    von_neumann_entropy,
)

from cirq.qis.states import (
    bloch_vector_from_state_vector,
    density_matrix,
    density_matrix_from_state_vector,
    dirac_notation,
    eye_tensor,
    infer_qid_shape,
    one_hot,
    QUANTUM_STATE_LIKE,
    QuantumState,
    quantum_state,
    STATE_VECTOR_LIKE,
    to_valid_density_matrix,
    to_valid_state_vector,
    validate_density_matrix,
    validate_indices,
    validate_qid_shape,
    validate_normalized_state_vector,
)
