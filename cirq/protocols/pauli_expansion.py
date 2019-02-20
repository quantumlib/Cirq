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

"""Protocol for obtaining expansion of linear operators in Pauli basis."""

from typing import Any, Union

import numpy as np

from cirq.linalg import operator_spaces
from cirq.protocols.unitary import unitary
from cirq.type_workarounds import NotImplementedType


def pauli_expansion(val: Any) -> Union[np.ndarray, NotImplementedType]:
    """Returns coefficients of the expansion of val in the Pauli basis.

    Args:
        val: The value whose Pauli expansion is to returned.

    Returns:
        If `val` has a _pauli_expansion_ method, then its result is
        returned. Otherwise, if `val` has a single-qubit unitary then
        that unitary is expanded in the Pauli basis and coefficients
        are returned. Otherwise, NotImplemented is returned.
    """
    method = getattr(val, '_pauli_expansion_', None)
    result = NotImplemented if method is None else method()

    if result is not NotImplemented and result is not None:
        return result

    matrix = unitary(val, default=None)
    if matrix is not None and matrix.shape == (2, 2):
        return operator_spaces.expand_in_basis(
            matrix, operator_spaces.PAULI_BASIS)

    return NotImplemented
