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

from typing import Any, Dict, Optional

import numpy as np

from cirq.linalg import operator_spaces
from cirq.protocols.unitary import unitary


RaiseTypeErrorIfNotProvided = {}  # type: Dict[str, complex]


def _make_basis_for_unitary(
        unitary: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
    """Returns Pauli basis for unitary or None if unitary is too large."""
    basis = operator_spaces.PAULI_BASIS
    if unitary.shape == (2, 2):
        return basis
    basis = operator_spaces.kron_bases(basis, operator_spaces.PAULI_BASIS)
    if unitary.shape == (4, 4):
        return basis
    basis = operator_spaces.kron_bases(basis, operator_spaces.PAULI_BASIS)
    if unitary.shape == (8, 8):
        return basis
    return None


def _filter_coefficients(expansion: Dict[str, complex],
                         tolerance: float) -> Dict[str, complex]:
    """Drops insignificant coefficients."""
    return {n: c for n, c in expansion.items() if abs(c) > tolerance}


def pauli_expansion(
    val: Any,
    *,
    default: Optional[Dict[str, complex]] = RaiseTypeErrorIfNotProvided,
    tolerance: float = 1e-9
) -> Optional[Dict[str, complex]]:
    """Returns coefficients of the expansion of val in the Pauli basis.

    Args:
        val: The value whose Pauli expansion is to returned.
        default: Determines what happens when `val` does not have methods that
            allow Pauli expansion to be obtained (see below). If set, the value
            is returned in that case. Otherwise, TypeError is raised.
        tolerance: Ignore coefficients whose absolute value is smaller than
            this.

    Returns:
        If `val` has a _pauli_expansion_ method, then its result is returned.
        Otherwise, if `val` has a small unitary then that unitary is expanded
        in the Pauli basis and coefficients are returned. Otherwise, if default
        is set to None or other value then default is returned. Otherwise,
        TypeError is raised.

    Raises:
        TypeError if `val` has none of the methods necessary to obtain its Pauli
        expansion and no default value has been provided.
    """
    method = getattr(val, '_pauli_expansion_', None)
    expansion = NotImplemented if method is None else method()

    if expansion is not NotImplemented:
        return _filter_coefficients(expansion, tolerance)

    matrix = unitary(val, default=None)
    if matrix is None:
        if default is RaiseTypeErrorIfNotProvided:
            raise TypeError('No Pauli expansion for object {} of type {}'
                    .format(val, type(val)))
        return default

    basis = _make_basis_for_unitary(matrix)
    if basis is None:
        if default is RaiseTypeErrorIfNotProvided:
            raise TypeError('Unitary too large for Pauli expansion.')
        return default

    expansion = operator_spaces.expand_in_basis(matrix, basis)
    return _filter_coefficients(expansion, tolerance)
