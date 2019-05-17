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
    TYPE_CHECKING,
    Any,
    TypeVar,
    Union,
    Optional,
    Tuple,
    List,
    Sequence,
)

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
        _strat_unitary_from_unitary, _strat_unitary_from_apply_unitary,
        _strat_unitary_from_decompose
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
        "besides None or NotImplemented.".format(type(val), val))


def _strat_unitary_from_unitary(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _unitary_ method."""
    getter = getattr(val, '_unitary_', None)
    if getter is None:
        return NotImplemented
    result = getter()
    if result is NotImplemented:
        return None
    return result


def _strat_unitary_from_apply_unitary(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _apply_unitary_ method."""
    from cirq.protocols.apply_unitary import ApplyUnitaryArgs
    from cirq import ops

    # Check for the magic method.
    method = getattr(val, '_apply_unitary_', None)
    if method is None:
        return NotImplemented

    # Infer number of qubits.
    if isinstance(val, ops.Gate):
        n = val.num_qubits()
    elif isinstance(val, ops.Operation):
        n = len(val.qubits)
    else:
        return NotImplemented

    # Apply unitary effect to an identity matrix.
    state = np.eye(1 << n, dtype=np.complex128)
    state.shape = (2,) * (2 * n)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(n)))

    # Package result.
    if result is NotImplemented or result is None:
        return None
    return result.reshape((1 << n, 1 << n))


def _strat_unitary_from_decompose(val: Any) -> Optional[np.ndarray]:
    """Attempts to compute a value's unitary via its _decompose_ method."""
    from cirq.protocols.apply_unitary import ApplyUnitaryArgs, apply_unitaries

    # Check for the magic method.
    operations, qubits = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented

    # Apply sub-operations' unitary effects to an identity matrix.
    n = len(qubits)
    state = np.eye(1 << n, dtype=np.complex128)
    state.shape = (2,) * (2 * n)
    buffer = np.empty_like(state)
    result = apply_unitaries(operations, qubits,
                             ApplyUnitaryArgs(state, buffer, range(n)), None)

    # Package result.
    if result is None:
        return None
    return result.reshape((1 << n, 1 << n))


def has_unitary(val: Any) -> bool:
    """Determines whether the value has a unitary effect.

    A value has a unitary effect iff any of the following conditions are met:

    - It has a `_has_unitary_` method that returns True.
    - It has a `_unitary_` method that doesn't return None or NotImplemented.
    - It has an `_apply_unitary_` method that doesn't return None or
        NotImplemented.
    - It has a `_decompose_` method that returns an `OP_TREE` of operations, and
        each operation in the list has a unitary effect.

    It is assumed that, when multiple methods are present, they will be
    consistent with each other.

    If the given value does not implement any of the specified methods, the
    default result is False.

    Args:
        The value that may or may not have a unitary effect.

    Returns:
        Whether or not `val` has a unitary effect.
    """
    strats = [
        _strat_has_unitary_from_has_unitary, _strat_has_unitary_from_decompose,
        _strat_has_unitary_from_apply_unitary, _strat_has_unitary_from_unitary
    ]
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result

    # If you can't tell that it's unitary, it's not unitary.
    return False


def _strat_has_unitary_from_has_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _has_unitary_ method."""
    getter = getattr(val, '_has_unitary_', None)
    result = NotImplemented if getter is None else getter()
    if result is NotImplemented:
        return None
    return result


def _strat_has_unitary_from_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _unitary_ method."""
    getter = getattr(val, '_unitary_', None)
    if getter is None:
        return None
    result = getter()
    return result is not NotImplemented and result is not None


def _strat_has_unitary_from_decompose(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _decompose_ method."""
    operations, qubits = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return None
    has_unitaries = [has_unitary(op) for op in operations]
    if any(v is None or v is NotImplemented for v in has_unitaries):
        return None
    return all(has_unitaries)


def _strat_has_unitary_from_apply_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _apply_unitary_ method.
    """
    from cirq.protocols.apply_unitary import apply_unitary, ApplyUnitaryArgs
    from cirq import linalg, line, ops

    method = getattr(val, '_apply_unitary_', None)
    if method is None:
        return None
    if isinstance(val, ops.Gate):
        val = val.on(*line.LineQubit.range(val.num_qubits()))
    if not isinstance(val, ops.Operation):
        return None

    n = len(val.qubits)
    state = linalg.one_hot(shape=(2,) * n, dtype=np.complex64)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(n)))
    return result is not None and result is not NotImplemented


def _try_decompose_into_operations_and_qubits(
        val: Any
) -> Tuple[Optional[List['cirq.Operation']], Sequence['cirq.Qid']]:
    """Returns the value's decomposition (if any) and the qubits it applies to.
    """
    from cirq.protocols.decompose import (decompose_once,
                                          decompose_once_with_qubits)
    from cirq import LineQubit, Gate, Operation

    if isinstance(val, Gate):
        # Gates don't specify qubits, and so must be handled specially.
        qubits = LineQubit.range(val.num_qubits())
        return decompose_once_with_qubits(val, qubits, None), qubits

    if isinstance(val, Operation):
        return decompose_once(val, None), val.qubits

    return None, ()
