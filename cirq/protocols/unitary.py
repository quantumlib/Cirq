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

from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq

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
        cirq.X**sympy.Symbol('a') doesn't.)

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
        that result is returned. Otherwise, if `val` is a cirq.Gate or
        cirq.Operation, decomposition will be attempted and the resulting
        unitary is returned if unitaries exist for all operations of the
        decompostion. If the result is still NotImplemented and a default value
        was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a _unitary_ method (or that method
            returned NotImplemented) and also no default value was specified.
    """
    from cirq import Gate, Operation  # HACK: Avoids circular dependencies.

    getter = getattr(val, '_unitary_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result

    # Fallback to decomposition for gates and operations
    if isinstance(val, (Gate, Operation)):
        decomposed_unitary = _decompose_and_get_unitary(val)
        if decomposed_unitary is not None:
            return decomposed_unitary

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
        NotImplemented, that result is returned. Otherwise, if `val` is a
        cirq.Gate or cirq.Operation, a decomposition is attempted and the
        resulting unitary is returned if has_unitary is True for all operations
        of the decompostion. Otherwise, if the value has a _unitary_ method
        return if that has a non-default value. Returns False if neither
        function exists.
    """
    from cirq.protocols.decompose import (decompose_once,
                                          decompose_once_with_qubits)
    # HACK: Avoids circular dependencies.
    from cirq import Gate, Operation, LineQubit

    getter = getattr(val, '_has_unitary_', None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented:
        return result

    # Fallback to explicit _unitary_
    unitary_getter = getattr(val, '_unitary_', None)
    if unitary_getter is not None and unitary_getter() is not NotImplemented:
        return True

    # Fallback to decomposition for gates and operations
    if isinstance(val, Gate):
        # Since gates don't know about qubits, we need to create some
        decomposed_val = decompose_once_with_qubits(val,
            LineQubit.range(val.num_qubits()),
            default=None)
        if decomposed_val is not None:
            return all(has_unitary(v) for v in decomposed_val)
    elif isinstance(val, Operation):
        decomposed_val = decompose_once(val, default=None)
        if decomposed_val is not None:
            return all(has_unitary(v) for v in decomposed_val)

    # Finally, fallback to full unitary method, including decomposition
    return unitary(val, None) is not None


def _decompose_and_get_unitary(val: Union['cirq.Operation', 'cirq.Gate']
                               ) -> np.ndarray:
    """Try to decompose a cirq.Operation or cirq.Gate, and return its unitary
    if it exists.

    Returns:
        If `val` can be decomposed into unitaries, calculate the resulting
        unitary and return it. If it doesn't exist, None is returned.
    """
    from cirq.protocols.apply_unitary import apply_unitary, ApplyUnitaryArgs
    from cirq.protocols.decompose import (decompose_once,
                                          decompose_once_with_qubits)
    from cirq import Gate, LineQubit, Operation

    if isinstance(val, Operation):
        qubits = val.qubits
        decomposed_val = decompose_once(val, default=None)
    elif isinstance(val, Gate):
        # Since gates don't know about qubits, we need to create some
        qubits = tuple(LineQubit.range(val.num_qubits()))
        decomposed_val = decompose_once_with_qubits(val,
                                                    qubits,
                                                    default=None)

    if decomposed_val is not None:
        # Calculate the resulting unitary (if it exists)
        n = len(qubits)
        state = np.eye(1 << n, dtype=np.complex128)
        state.shape = (2,) * (2 * n)
        buffer = np.zeros(state.shape, dtype=np.complex128)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        result = state
        for op in decomposed_val:
            indices = [qubit_map[q] for q in op.qubits]
            result = apply_unitary(
                unitary_value=op,
                args=ApplyUnitaryArgs(state, buffer, indices),
                default=None)
            if result is None:
                return None
            if result is buffer:
                buffer = state
            state = result
        if result is not None:
            return result.reshape((1 << n, 1 << n))
