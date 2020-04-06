# Copyright 2020 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from cirq.linalg import is_unitary
from cirq.ops import YY
from cirq.protocols import unitary

_YY = unitary(YY)


def _to_special(u: np.ndarray) -> np.ndarray:
    """Converts a unitary matrix to a special unitary matrix.

    All unitary matrices u have |det(u)| = 1.
    Also for all d dimensional unitary matrix u, and scalar s:
        det(u * s) = det(u) * s^(d)
    To find a special unitary matrix from u:
        u * det(u)^{-1/d}

    Args:
        u: the unitary matrix
    Returns:
        the special unitary matrix
    """
    return u * (np.linalg.det(u) ** (-1 / len(u)))


def _gamma(u: np.ndarray) -> np.ndarray:
    """Gamma function to convert u to the magic basis.

    See Definition IV.1 in Shende et al. "Minimal Universal Two-Qubit CNOT-based
     Circuits." https://arxiv.org/abs/quant-ph/0308033

    Args:
        u: a member of SU(4)
    Returns:
        u @ yy @ u.T @ yy, where yy = Y ⊗ Y
    """

    return u @ _YY @ u.T @ _YY


def _is_three_cnot_two_qubit_unitary(u: np.ndarray) -> bool:
    """Returns true if U requires 3 CNOT/CZ gates.

    See Proposition III.1, III.2, III.3 in Shende et al. “Recognizing Small-
    Circuit Structure in Two-Qubit Operators and Timing Hamiltonians to Compute
    Controlled-Not Gates”.  https://arxiv.org/abs/quant-ph/0308045

    Args:
        u: a two-qubit unitary
    Returns:
        the number of two-qubit gates required to implement the unitary
    """
    assert np.shape(u) == (4, 4)
    assert is_unitary(u)

    poly = np.poly(_gamma(_to_special(u)))
    return not np.alltrue(np.isclose(0, np.imag(poly)))
