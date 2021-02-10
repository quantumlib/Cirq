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

from cirq.qis.measures import (
    fidelity,
    von_neumann_entropy,
)

from cirq.qis.states import (
    bloch_vector_from_state_vector,
    density_matrix,
    density_matrix_from_state_vector,
    dirac_notation,
    eye_tensor,
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
    validate_normalized_state,
    validate_normalized_state_vector,
)
