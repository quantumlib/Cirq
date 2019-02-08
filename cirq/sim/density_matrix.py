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
"""Code to handle density matrices."""

from typing import Type, Union

import numpy as np

from cirq.sim import wave_function


def to_valid_density_matrix(
    density_matrix_rep: Union[int, np.ndarray],
    num_qubits: int,
    dtype: Type[np.number] = np.complex64) -> np.ndarray:
    """Verifies the density_matrix_rep is valid and converts it to ndarray form.

    This method is used to support passing a matrix, a vector (wave function),
    or a computational basis state as a representation of a state.

    Args:
        density_matrix_rep: If an numpy array, if it is of rank 2 (a matrix),
            then this is the density matrix. If it is a numpy array of rank 1
            (a vector) then this is a wave function. If this is an int,
            then this is the computation basis state.
        num_qubits: The number of qubits for the density matrix. The
            density_matrix_rep must be valid for this number of qubits.
        dtype: The numpy dtype of the density matrix, will be used when creating
            the state for a computational basis state (int), or validated
            against if density_matrix_rep is a numpy array.

    Returns:
        A numpy matrix corresponding to the density matrix on the given number
        of qubits.

    Raises:
        ValueError if the density_matrix_rep is not valid.
    """
    if (isinstance(density_matrix_rep, np.ndarray)
        and density_matrix_rep.ndim == 2):
        if density_matrix_rep.shape != (2 ** num_qubits, 2 ** num_qubits):
            raise ValueError(
                'Density matrix was not square and of size 2 ** num_qubit, '
                'instead was {}'.format(density_matrix_rep.shape))
        if not np.allclose(density_matrix_rep,
                           np.transpose(np.conj(density_matrix_rep))):
            raise ValueError('The density matrix is not hermitian.')
        if not np.isclose(np.trace(density_matrix_rep), 1.0):
            raise ValueError(
                'Density matrix did not have trace 1 but instead {}'.format(
                    np.trace(density_matrix_rep)))
        if density_matrix_rep.dtype != dtype:
            raise ValueError(
                'Density matrix had dtype {} but expected {}'.format(
                    density_matrix_rep.dtype, dtype))
        if not np.all(np.linalg.eigvalsh(density_matrix_rep) > -1e-8):
            raise ValueError('The density matrix is not positive semidefinite.')
        return density_matrix_rep

    state_vector = wave_function.to_valid_state_vector(density_matrix_rep,
                                                       num_qubits, dtype)
    return np.outer(state_vector, np.conj(state_vector))
