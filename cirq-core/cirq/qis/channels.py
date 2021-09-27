# Copyright 2021 The Cirq Developers
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
"""Tools for analyzing and manipulating quantum channels."""
from typing import Sequence

import numpy as np

from cirq import protocols
from cirq._compat import deprecated


def kraus_to_choi(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the unique Choi matrix corresponding to a Kraus representation of a channel."""
    d = np.prod(kraus_operators[0].shape, dtype=np.int64)
    c = np.zeros((d, d), dtype=np.complex128)
    for k in kraus_operators:
        v = np.reshape(k, d)
        c += np.outer(v, v.conj())
    return c


@deprecated(deadline='v0.14', fix='use cirq.kraus_to_superoperator instead')
def kraus_to_channel_matrix(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the matrix representation of the linear map with given Kraus operators."""
    return kraus_to_superoperator(kraus_operators)


def kraus_to_superoperator(kraus_operators: Sequence[np.ndarray]) -> np.ndarray:
    """Returns the matrix representation of the linear map with given Kraus operators."""
    d_out, d_in = kraus_operators[0].shape
    m = np.zeros((d_out * d_out, d_in * d_in), dtype=np.complex128)
    for k in kraus_operators:
        m += np.kron(k, k.conj())
    return m


def operation_to_choi(operation: 'protocols.SupportsKraus') -> np.ndarray:
    r"""Returns the unique Choi matrix associated with an operation .

    Choi matrix J(E) of a linear map E: L(H1) -> L(H2) which takes linear operators
    on Hilbert space H1 to linear operators on Hilbert space H2 is defined as

        $$
        J(E) = (E \otimes I)(|\phi\rangle\langle\phi|)
        $$

    where $|\phi\rangle = \sum_i|i\rangle|i\rangle$ is the unnormalized maximally
    entangled state and I: L(H1) -> L(H1) is the identity map. Note that J(E) is
    a square matrix with d1*d2 rows and columns where d1 = dim H1 and d2 = dim H2.

    Args:
        operation: Quantum channel.
    Returns:
        Choi matrix corresponding to operation.
    """
    return kraus_to_choi(protocols.kraus(operation))


@deprecated(deadline='v0.14', fix='use cirq.operation_to_superoperator instead')
def operation_to_channel_matrix(operation: 'protocols.SupportsKraus') -> np.ndarray:
    """Returns the matrix representation of an operation in standard basis.

    Let E: L(H1) -> L(H2) denote a linear map which takes linear operators on Hilbert space H1
    to linear operators on Hilbert space H2 and let d1 = dim H1 and d2 = dim H2. Also, let Fij
    denote an operator whose matrix has one in ith row and jth column and zeros everywhere else.
    Note that d1-by-d1 operators Fij form a basis of L(H1). Similarly, d2-by-d2 operators Fij
    form a basis of L(H2). This function returns the matrix of E in these bases.

    Args:
        operation: Quantum channel.
    Returns:
        Matrix representation of operation.
    """
    return operation_to_superoperator(operation)


def operation_to_superoperator(operation: 'protocols.SupportsKraus') -> np.ndarray:
    """Returns the matrix representation of an operation in standard basis.

    Let E: L(H1) -> L(H2) denote a linear map which takes linear operators on Hilbert space H1
    to linear operators on Hilbert space H2 and let d1 = dim H1 and d2 = dim H2. Also, let Fij
    denote an operator whose matrix has one in ith row and jth column and zeros everywhere else.
    Note that d1-by-d1 operators Fij form a basis of L(H1). Similarly, d2-by-d2 operators Fij
    form a basis of L(H2). This function returns the matrix of E in these bases.

    Args:
        operation: Quantum channel.
    Returns:
        Matrix representation of operation.
    """
    return kraus_to_superoperator(protocols.kraus(operation))
