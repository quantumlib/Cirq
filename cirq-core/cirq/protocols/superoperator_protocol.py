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

"""Protocol and methods for obtaining matrix representation of quantum channels."""

from typing import Any, Tuple, TypeVar, Union
from typing_extensions import Protocol

import numpy as np

from cirq._doc import doc_private
from cirq.protocols import kraus_protocol
from cirq.qis.channels import kraus_to_superoperator
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the superoperator protocol to determine whether
# or not the caller provided a 'default' argument. It must be of type np.ndarray to ensure
# the method has the correct type signature in that case. It is checked for using `is`, so
# it won't have a false positive if the user provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])

TDefault = TypeVar('TDefault')


class SupportsSuperoperator(Protocol):
    """An object that may be described using a superoperator."""

    @doc_private
    def _superoperator_(self) -> Union[np.ndarray, NotImplementedType]:
        """Matrix of the superoperator corresponding to this quantum channel.

        Every quantum channel is a linear map between the vector spaces of input and output
        density matrices and therefore admits a representation as a matrix. The matrix depends
        on the choice of bases in the input and output vector spaces. This function returns
        the matrix of self in the standard basis. Assuming self acts on density matrices of
        a d-dimensional system, the matrix of the superoperator has d**2 rows and d**2 columns
        corresponding to the fact that a vectorized density matrix is a d**2-dimensional vector.

        Note that this is different from the Choi matrix. This is easiest to notice in the
        general case of a quantum channel mapping density matrices of a d1-dimensional system
        to density matrices of a d2-dimensional system. The Choi matrix is then a square matrix
        with d1*d2 rows and d1*d2 columns. On the other hand, the superoperator matrix has
        d2*d2 rows and d1*d1 columns. Note that this implementation assumes d1=d2, so the
        Choi matrix and the superoperator matrix have the same shape. Nevertheless, they are
        generally different matrices in this case, too.

        Returns:
            Superoperator matrix of the channel, or NotImplemented if there is no such
            matrix.
        """

    @doc_private
    def _has_superoperator_(self) -> bool:
        """Whether this value has a superoperator representation.

        This method is used by the `cirq.has_superoperator` protocol. If this
        method is not present, or returns NotImplemented, the protocol tries
        to find the superoperator via other means and returns True if any of
        the available fallback avenues succeed. Otherwise, `cirq.has_superoperator`
        returns False.

        Returns:
            True if the value has a superoperator representation, False otherwise.
        """


def superoperator(
    val: Any, default: Any = RaiseTypeErrorIfNotProvided
) -> Union[np.ndarray, TDefault]:
    """Matrix of the superoperator corresponding to this quantum channel.

    Every quantum channel is a linear map between the vector spaces of input and output
    density matrices and therefore admits a representation as a matrix. The matrix depends
    on the choice of bases in the input and output vector spaces. This function returns
    the matrix of self in the standard basis. Assuming self acts on density matrices of
    a d-dimensional system, the matrix of the superoperator has d**2 rows and d**2 columns
    corresponding to the fact that a vectorized density matrix is a d**2-dimensional vector.

    Note that this is different from the Choi matrix. This is easiest to notice in the
    general case of a quantum channel mapping density matrices of a d1-dimensional system
    to density matrices of a d2-dimensional system. The Choi matrix is then a square matrix
    with d1*d2 rows and d1*d2 columns. On the other hand, the superoperator matrix has
    d2*d2 rows and d1*d1 columns. Note that this implementation assumes d1=d2, so the
    Choi matrix and the superoperator matrix have the same shape. Nevertheless, they are
    generally different matrices in this case, too.

    Args:
        val: The object whose superoperator representation is to be found.
        default: Determines the fallback behavior when no superoperator representation
            is found for `val`. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a `_superoperator_` method and its result is not NotImplemented,
        that result is returned. Otherwise, the function attempts to obtain the Kraus
        representation and compute the superoperator from it. If this is unsuccessful
        and a default value was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _superoperator_, kraus_, _mixture_ or _unitary_
            method (or they returned NotImplemented) and also no default value was
            specified.
    """
    so_getter = getattr(val, '_superoperator_', None)
    so_result = NotImplemented if so_getter is None else so_getter()
    if so_result is not NotImplemented:
        return so_result

    kraus_result = kraus_protocol.kraus(val, default=None)
    if kraus_result is not None:
        return kraus_to_superoperator(kraus_result)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if so_getter is None:
        raise TypeError(
            "object of type '{}' has no _superoperator_ method and we could find no Kraus "
            "representation for it.".format(type(val))
        )

    raise TypeError(
        "object of type '{}' does have a _superoperator_ method, but it returned "
        "NotImplemented and we could find no Kraus representation for it.".format(type(val))
    )


def has_superoperator(val: Any, *, allow_decompose: bool = True) -> bool:
    """Returns whether val has a superoperator representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant decompositions from
            being performed, see `cirq.has_kraus` for more details. Defaults to True.
            When False, the decomposition strategy for determining the result is skipped.

    Returns:
        If `val` has a `_has_superoperator_` method and its result is not NotImplemented,
        that result is returned. Otherwise, if `cirq.has_kraus` protocol returns True then
        True is returned. Finally, if `cirq.superoperator` returns a non-default value
        then True is returned. Otherwise, False is returned.
    """
    so_getter = getattr(val, '_has_superoperator_', None)
    so_result = NotImplemented if so_getter is None else so_getter()
    if so_result is not NotImplemented:
        return so_result

    if kraus_protocol.has_kraus(val, allow_decompose=allow_decompose):
        return True

    return superoperator(val, default=None) is not None
