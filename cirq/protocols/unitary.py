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

from typing import Any, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the unitary method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# np.ndarray to ensure the method has the correct type signature in that case.
# It is checked for using `is`, so it won't have a false positive if the user
# provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class SupportsUnitary(Protocol):
    """An object that may be describable by a unitary matrix."""

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        """A unitary matrix describing this value, e.g. the matrix of a gate.

        This method is used by the global `cirq.unitary` method. If this method
        is not present, or returns NotImplemented, it is assumed that the
        receiving object doesn't have a unitary matrix (resulting in a TypeError
        or default result when calling `cirq.unitary` on it). (The ability to
        return NotImplemented is useful when a class cannot know if it has a
        matrix until runtime, e.g. cirq.X**c normally has a matrix but
        cirq.X**cirq.Symbol('a') doesn't.)

        The order of cells in the matrix is always implicit with respect to the
        object being called. For example, for gates the matrix must be ordered
        with respect to the list of qubits that the gate is applied to. For
        operations, the matrix is ordered to match the list returned by its
        `qubits` attribute. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        Returns:
            A unitary matrix describing this value, or NotImplemented if there
            is no such matrix.
        """

    def _has_unitary_(self) -> bool:
        """Whether this value has a unitary matrix representation.

        This method is used by the global `cirq.has_unitary` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _unitary_ with a default value, or False if neither exist.

        Returns:
          True if the value has a unitary matrix representation, False
          otherwise.
        """


def unitary(val: Any,
            default: TDefault = RaiseTypeErrorIfNotProvided
            ) -> Union[np.ndarray, TDefault]:
    """Returns a unitary matrix describing the given value.

    Args:
        val: The value to describe with a unitary matrix.
        default: Determines the fallback behavior when `val` doesn't have
            a unitary matrix. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.

    Returns:
        If `val` has a _unitary_ method and its result is not NotImplemented,
        that result is returned. Otherwise, if a default value was specified,
        the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _unitary_ method (or that method
            returned NotImplemented) and also no default value was specified.
    """
    getter = getattr(val, '_unitary_', None)
    result = NotImplemented if getter is None else getter()

    if result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if getter is None:
        raise TypeError("object of type '{}' "
                        "has no _unitary_ method.".format(type(val)))
    raise TypeError("object of type '{}' does have a _unitary_ method, "
                    "but it returned NotImplemented.".format(type(val)))


def has_unitary(val: Any) -> bool:
    """Returns whether the value has a unitary matrix representation.

    Returns:
        If `val` has a _has_unitary_ method and its result is not
        NotImplemented, that result is returned. Otherwise, if the value
        has a _unitary_ method return if that has a non-default value.
        Returns False if neither function exists.
    """
    getter = getattr(val, '_has_unitary_', None)
    result = NotImplemented if getter is None else getter()

    if result is not NotImplemented:
        return result

    # No _has_unitary_ function, use _unitary_ instead
    return unitary(val, None) is not None
