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
"""Distance measures between quantum states and operations."""

import numbers

import numpy as np
import scipy

from cirq import linalg


def fidelity(state1: np.ndarray, state2: np.ndarray, *, rtol=1e-5,
             atol=1e-8) -> float:
    """Fidelity of two quantum states.

    The fidelity of two density matrices ρ and σ is defined as

        trace(sqrt(sqrt(ρ) σ sqrt(ρ)))^2.

    The given states can be state vectors or density matrices. If two density
    matrices are given, this checks if they approximately commute, and if so,
    computes the fidelity in a more efficient way. The relative and absolute
    numerical tolerances used for the commutativity check can be specified by
    rtol and atol, respectively.

    Args:
        state1: The first state.
        state2: The second state.
        rtol: The per-matrix-entry relative tolerance used to check whether
            two density matrices commute.
        atol: The per-matrix-entry absolute tolerance used to check whether
            two density matrices commute.
    """
    if len(state1.shape) == 1 and len(state2.shape) == 1:
        # Both state vectors
        return np.abs(np.dot(np.conjugate(state1), state2))**2
    elif len(state1.shape) == 1 and len(state2.shape) == 2:
        # state1 is a state vector and state2 is a density matrix
        return np.real(np.conjugate(state1) @ state2 @ state1)
    elif len(state1.shape) == 2 and len(state2.shape) == 1:
        # state1 is a density matrix and state2 is a state vector
        return np.real(np.conjugate(state2) @ state1 @ state2)
    elif len(state1.shape) == 2 and len(state2.shape) == 2:
        # Both density matrices
        if linalg.matrix_commutes(state1, state2, rtol=rtol, atol=atol):
            # Fidelity of commuting matrices can be computed about twice as fast
            return np.real(np.trace(scipy.linalg.sqrtm(state1 @ state2))**2)
        state1_sqrt = scipy.linalg.sqrtm(state1)
        return np.real(
            np.trace(scipy.linalg.sqrtm(state1_sqrt @ state2 @ state1_sqrt))**2)
    else:
        raise ValueError('The given arrays must be one- or two-dimensional. '
                         f'Got shapes {state1.shape} and {state2.shape}.')
