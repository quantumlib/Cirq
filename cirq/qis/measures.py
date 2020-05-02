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
"""Measures on and between quantum states and operations."""

import numpy as np
import scipy
import scipy.stats


def _sqrt_positive_semidefinite_matrix(mat: np.ndarray) -> np.ndarray:
    """Square root of a positive semidefinite matrix."""
    eigs, vecs = scipy.linalg.eigh(mat)
    # Zero out small negative entries
    eigs = np.maximum(eigs, np.zeros(eigs.shape, dtype=eigs.dtype))
    return vecs @ (np.sqrt(eigs) * vecs).T.conj()


def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Fidelity of two quantum states.

    The fidelity of two density matrices ρ and σ is defined as

        trace(sqrt(sqrt(ρ) σ sqrt(ρ)))^2.

    The given states can be state vectors or density matrices.

    Args:
        state1: The first state.
        state2: The second state.
    """
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        # Both state vectors
        return np.abs(np.vdot(state1, state2))**2
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        # state1 is a state vector and state2 is a density matrix
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        # state1 is a density matrix and state2 is a state vector
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # Both density matrices
        state1_sqrt = _sqrt_positive_semidefinite_matrix(state1)
        eigs = scipy.linalg.eigvalsh(state1_sqrt @ state2 @ state1_sqrt)
        # Zero out small negative entries
        eigs = np.maximum(eigs, np.zeros(eigs.shape, dtype=eigs.dtype))
        trace = np.sum(np.sqrt(eigs))
        return trace**2
    raise ValueError('The given arrays must be one- or two-dimensional. '
                     f'Got shapes {state1.shape} and {state2.shape}.')


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """Calculates von Neumann entropy of density matrix in bits.
    Args:
        density_matrix: The density matrix.
    Returns:
        The calculated von Neumann entropy.
    """
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    return scipy.stats.entropy(abs(eigenvalues), base=2)
