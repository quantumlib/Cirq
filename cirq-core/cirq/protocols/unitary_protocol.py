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

from typing import (
    Any,
    TypeVar,
    Union,
    Optional,
)

import numpy as np
from typing_extensions import Protocol

from cirq import qis
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.protocols.apply_unitary_protocol import (
    ApplyUnitaryArgs,
    apply_unitaries,
)
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits,
)
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the unitary method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# np.ndarray to ensure the method has the correct type signature in that case.
# It is checked for using `is`, so it won't have a false positive if the user
# provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided: np.ndarray = np.array([])

TDefault = TypeVar('TDefault')


class SupportsUnitary(Protocol):
    """An object that may be describable by a unitary matrix."""

    @doc_private
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

    @doc_private
    def _has_unitary_(self) -> bool:
        """Whether this value has a unitary matrix representation.

        This method is used by the global `cirq.has_unitary` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _unitary_ with a default value, or False if neither exist.

        Returns:
            True if the value has a unitary matrix representation, False
            otherwise.
        """


def unitary(
    val: Any, default: TDefault = RaiseTypeErrorIfNotProvided
) -> Union[np.ndarray, TDefault]:
    """Returns a unitary matrix describing the given value.

    The matrix is determined by any one of the following techniques:

    - The value has a `_unitary_` method that returns something besides None or
        NotImplemented. The matrix is whatever the method returned.
    - The value has a `_decompose_` method that returns a list of operations,
        and each operation in the list has a unitary effect. The matrix is
        created by aggregating the sub-operations' unitary effects.
    - The value has an `_apply_unitary_` method, and it returns something
        besides None or NotImplemented. The matrix is created by applying
        `_apply_unitary_` to an identity matrix.

    If none of these techniques succeeds, it is assumed that `val` doesn't have
    a unitary effect. The order in which techniques are attempted is
    unspecified.

    Args:
        val: The value to describe with a unitary matrix.
        default: Determines the fallback behavior when `val` doesn't have
            a unitary effect. If `default` is not set, a TypeError is raised. If
            `default` is set to a value, that value is returned.

    Returns:
        If `val` has a unitary effect, the corresponding unitary matrix.
        Otherwise, if `default` is specified, it is returned.

    Raises:
        TypeError: `val` doesn't have a unitary effect and no default value was
            specified.
    """
    strats = [
        _strat_unitary_from_unitary,
        _strat_unitary_from_apply_unitary,
        _strat_unitary_from_decompose,
    ]
    for strat in strats:
        result = strat(val)
        if result is None:
            break
        if result is not NotImplemented:
            return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "cirq.unitary failed. "
        "Value doesn't have a (non-parameterized) unitary effect.\n"
        "\n"
        "type: {}\n"
        "value: {!r}\n"
        "\n"
        "The value failed to satisfy any of the following criteria:\n"
        "- A `_unitary_(self)` method that returned a value "
        "besides None or NotImplemented.\n"
        "- A `_decompose_(self)` method that returned a "
        "list of unitary operations.\n"
        "- An `_apply_unitary_(self, args) method that returned a value "
        "besides None or NotImplemented.".format(type(val), val)
    )


def _strat_unitary_from_unitary(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _unitary_ method."""
    getter = getattr(val, '_unitary_', None)
    if getter is None:
        return NotImplemented
    return getter()


def _strat_unitary_from_apply_unitary(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _apply_unitary_ method."""
    # Check for the magic method.
    method = getattr(val, '_apply_unitary_', None)
    if method is None:
        return NotImplemented

    # Get the qid_shape.
    val_qid_shape = qid_shape_protocol.qid_shape(val, None)
    if val_qid_shape is None:
        return NotImplemented

    # Apply unitary effect to an identity matrix.
    state = qis.eye_tensor(val_qid_shape, dtype=np.complex128)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(len(val_qid_shape))))

    if result is NotImplemented or result is None:
        return result
    state_len = np.prod(val_qid_shape, dtype=np.int64)
    return result.reshape((state_len, state_len))


def _strat_unitary_from_decompose(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _decompose_ method."""
    # Check if there's a decomposition.
    operations, qubits, val_qid_shape = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented

    # Apply sub-operations' unitary effects to an identity matrix.
    state = qis.eye_tensor(val_qid_shape, dtype=np.complex128)
    buffer = np.empty_like(state)
    result = apply_unitaries(
        operations, qubits, ApplyUnitaryArgs(state, buffer, range(len(val_qid_shape))), None
    )

    # Package result.
    if result is None:
        return None
    state_len = np.prod(val_qid_shape, dtype=np.int64)
    return result.reshape((state_len, state_len))
