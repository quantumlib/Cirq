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

from typing import cast, Optional, Tuple

import numpy as np
import numpy.typing as npt

from cirq import protocols
from cirq.ops import DensePauliString


def _argmax(V: npt.NDArray) -> Tuple[int, float]:
    """Returns a tuple (index of max number, max number)."""
    V = (V * V.conj()).real
    idx_max = np.argmax(V)
    V[idx_max] = 0
    return cast(int, idx_max), np.max(V)


def _validate_decomposition(decomposition: DensePauliString, U: npt.NDArray, eps: float) -> bool:
    """Returns whether the max absolute value of the elementwise difference is less than eps."""
    got = protocols.unitary(decomposition)
    return np.abs(got - U).max() < eps


def _fast_walsh_hadamard_transform(V: npt.NDArray) -> None:
    """Fast Walsh–Hadamard Transform of an array."""
    m = len(V)
    n = m.bit_length() - 1
    for h in [2**i for i in range(n)]:
        for i in range(0, m, h * 2):
            for j in range(i, i + h):
                x = V[j]
                y = V[j + h]
                V[j] = x + y
                V[j + h] = x - y


def _conjugate_with_hadamard(U: npt.NDArray) -> npt.NDArray:
    """Applies HcUH in O(n4^n) instead of O(8^n)."""

    U = np.copy(U.T)
    for i in range(U.shape[1]):
        _fast_walsh_hadamard_transform(U[:, i])
    U = U.T
    for i in range(U.shape[1]):
        _fast_walsh_hadamard_transform(U[:, i])
    return U


def unitary_to_pauli_string(U: npt.NDArray, eps: float = 1e-15) -> Optional[DensePauliString]:
    """Attempts to find a pauli string (with possible phase) equivalent to U up to eps.

        Based on this answer https://shorturl.at/aA079.
        Let x_mask be the index of the maximum number of the first column of U
        and z_mask be the index of the maximum number of the first column of H†UH
        each of these indicies is n-bits long where U is 2^n x 2^n.

        These two indices/masks encode in binary the indices of the qubits that
        have I, X, Y, Z acting on them as follows:
        x_mask[i] == 1 and z_mask[i] == 0: X acts on the ith qubit
        x_mask[i] == 0 and z_mask[i] == 1: Z acts on the ith qubit
        x_mask[i] == 1 and z_mask[i] == 1: Y acts on the ith qubit
        x_mask[i] == 0 and z_mask[i] == 0: I acts on the ith qubit

    Args:
        U: A square array whose dimension is a power of 2.
        eps: numbers smaller than `eps` are considered zero.

    Returns:
        A DensePauliString of None.

    Raises:
        ValueError: if U is not square with a power of 2 dimension.
    """

    if len(U.shape) != 2 or U.shape[0] != U.shape[1]:
        raise ValueError(f'Input has a non-square shape {U}')
    n = U.shape[0].bit_length() - 1
    if U.shape[0] != 2**n:
        raise ValueError(f'Input dimension {U.shape[0]} isn\'t a power of 2')

    x_msk, second_largest = _argmax(U[:, 0])
    if second_largest > eps:
        return None
    U_z = _conjugate_with_hadamard(U)
    z_msk, second_largest = _argmax(U_z[:, 0])
    if second_largest > eps:
        return None

    def select(i):
        """Returns the gate that acts on the ith qubit."""
        has_x = (x_msk >> i) & 1
        has_z = (z_msk >> i) & 1
        # The mapping is:
        #   - has_x and not has_z => X
        #   - not has_x and has_z => Z
        #   - has_x and has_z => Y
        #   - not has_x and not has_z => I
        gate_table = ['IX', 'ZY']
        return gate_table[has_z][has_x]

    decomposition = DensePauliString(''.join(select(i) for i in reversed(range(n))))

    guess = protocols.unitary(decomposition)
    if np.abs(guess[x_msk, 0]) < eps:
        return None
    phase = U[x_msk, 0] / guess[x_msk, 0]
    phase /= np.abs(phase)  # Make sure |phase| = 1 to avoid rounding issues.

    decomposition = DensePauliString(
        ''.join(select(i) for i in reversed(range(n))), coefficient=phase
    )

    if not _validate_decomposition(decomposition, U, eps):
        return None

    return decomposition
