# Copyright 2019 The Cirq Developers
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

from typing import Tuple, Type, Union

import numpy as np

from cirq import protocols
from cirq.sim import wave_function

def to_valid_density_matrix(
    density_matrix_rep: Union[int, np.ndarray, Tuple[float, np.ndarray]],
    num_qubits: int,
    dtype: Type[np.number] = np.complex64) -> np.ndarray:
    """
    """
    if isinstance(density_matrix_rep, tuple):
        protocols.validate_mixture()

    if isinstance(density_matrix_rep,
                  np.ndarray) and density_matrix_rep.ndim == 2:
        try:
            np.linalg.cholesky(density_matrix_rep)
        except np.linalg.LinAlgError:
            raise ValueError('The density matrix was not positive definite.')
        if not np.isclose(np.trace(density_matrix_rep), 1.0):
            raise ValueError(
                'Density matrix did not have trace 1 but instead {}'.format(
                    np.trace(density_matrix_rep)))
        if density_matrix_rep.shape != (2 ** num_qubits, 2 ** num_qubits):
            raise ValueError(
                'Density max was not square and of size 2 ** num_qubit, '
                'instead was {}'.format(density_matrix_rep.shape))
        return density_matrix_rep
    state_vector = wave_function.to_valid_state_vector(density_matrix_rep,
                                                       num_qubits, dtype)
    return np.outer(np.conj(state_vector), state_vector)
