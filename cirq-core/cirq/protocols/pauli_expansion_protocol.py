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

from typing import Any, TypeVar, Union
from typing_extensions import Protocol

from cirq import value
from cirq._doc import doc_private
from cirq.linalg import operator_spaces
from cirq.protocols import qid_shape_protocol, unitary_protocol

TDefault = TypeVar('TDefault')

RaiseTypeErrorIfNotProvided = value.LinearDict({})  # type: value.LinearDict[str]


class SupportsPauliExpansion(Protocol):
    """An object that knows its expansion in the Pauli basis."""

    @doc_private
    def _pauli_expansion_(self) -> value.LinearDict[str]:
        """Efficiently obtains expansion of self in the Pauli basis.

        Returns:
            Linear dict keyed by name of Pauli basis element. The names
            consist of n captal letters from the set 'I', 'X', 'Y', 'Z'
            where n is the number of qubits. For example, 'II', 'IX' and
            'XY' are valid Pauli names in the two-qubit case.
        """


def pauli_expansion(
    val: Any,
    *,
    default: Union[value.LinearDict[str], TDefault] = RaiseTypeErrorIfNotProvided,
    atol: float = 1e-9,
) -> Union[value.LinearDict[str], TDefault]:
    """Returns coefficients of the expansion of val in the Pauli basis.

    Args:
        val: The value whose Pauli expansion is to returned.
        default: Determines what happens when `val` does not have methods that
            allow Pauli expansion to be obtained (see below). If set, the value
            is returned in that case. Otherwise, TypeError is raised.
        atol: Ignore coefficients whose absolute value is smaller than this.

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
        return expansion.clean(atol=atol)

    # Don't attempt to derive the pauli expansion if this is a qudit gate
    if not all(d == 2 for d in qid_shape_protocol.qid_shape(val, default=())):
        if default is RaiseTypeErrorIfNotProvided:
            raise TypeError(f'No Pauli expansion for object {val} of type {type(val)}')
        return default

    matrix = unitary_protocol.unitary(val, default=None)
    if matrix is None:
        if default is RaiseTypeErrorIfNotProvided:
            raise TypeError(f'No Pauli expansion for object {val} of type {type(val)}')
        return default

    num_qubits = matrix.shape[0].bit_length() - 1
    basis = operator_spaces.kron_bases(operator_spaces.PAULI_BASIS, repeat=num_qubits)

    expansion = operator_spaces.expand_matrix_in_orthogonal_basis(matrix, basis)
    return expansion.clean(atol=atol)
