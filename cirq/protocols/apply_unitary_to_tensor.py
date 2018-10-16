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

"""A protocol for implementing high performance unitary left-multiplies."""


from typing import Any, Union, Sequence, TypeVar

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.unitary import unitary


# This is a special indicator value used by the apply_unitary_to_tensor method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class SupportsApplyUnitaryToTensor(Protocol):
    """An object that can be efficiently left-multiplied into tensors."""

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplemented]:
        """Left-multiplies a unitary effect onto a tensor with good performance.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        The target may represent a wavefunction, a unitary matrix, or some other
        tensor. Implementations will work in all of these cases as long as they
        correctly focus on only operating on the given axes.

        Args:
            target_tensor: The input tensor that needs to be left-multiplied by
                the unitary effect of the receiving object. The tensor will
                have the shape (2, 2, 2, ..., 2). It usually corresponds to
                a multi-qubit superposition, but it could also be a multi-qubit
                unitary transformation or some other concept.
            available_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            axes: Which axes the unitary effect is being applied to (e.g. the
                qubits that the gate is operating on).

        Returns:
            If the receiving object is not able to apply its unitary effect,
            NotImplemented should be returned.

            If the receiving object is able to work inline, it should directly
            mutate target_tensor and then return target_tensor. The caller will
            understand this to mean that the result is in target_tensor.

            If the receiving object is unable to work inline, it can write its
            output over available_buffer and then return available_buffer. The
            caller will understand this to mean that the result is in
            available_buffer (and so what was available_buffer will become
            target_tensor in the next call, and vice versa).

            The receiving object is also permitted to allocate a new
            numpy.ndarray and return that as its result.
        """


def apply_unitary_to_tensor(val: Any,
                            target_tensor: np.ndarray,
                            available_buffer: np.ndarray,
                            axes: Sequence[int],
                            default: TDefault = RaiseTypeErrorIfNotProvided
                            ) -> Union[np.ndarray, TDefault]:
    """High performance left-multiplication of a unitary effect onto a tensor.

    If `val` defines an _apply_unitary_to_tensor_ method, that method will be
    used to apply `val`'s unitary effect to the target tensor. Otherwise, if
    `val` defines a _unitary_ method, its unitary matrix will be retrieved and
    applied using a generic method. Otherwise the application fails, and either
    an exception is raised or the specified default value is returned.

        The target may represent a wavefunction, a unitary matrix, or some other
        tensor. Implementations will work in all of these cases as long as they
        correctly focus on only operating on the given axes. See also:
        `cirq.slice_for_qubits_equal_to(axes, int)`, which does the correct
        thing in all these cases.

    Args:
        val: The value with a unitary effect to apply to the target tensor.
        target_tensor: The input tensor that needs to be left-multiplied by
            the unitary effect of `val`. Note that this value may be mutated
            inline into the output. The tensor will have the shape
            (2, 2, 2, ..., 2). target_tensor may correspond to a multi-qubit
            superposition (with each axis being a qubit), a multi-qubit unitary
            transformation (with some axes being qubit inputs and others being
            qubit outputs), or some other concept.
        available_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor. Note that the output may be written
            into this buffer.
        axes: Which axes the unitary effect is being applied to (e.g. the
            qubits that the gate is operating on). For example, a CNOT being
            applied to qubits #4 and #2 of a circuit would result in
            axes=(4, 2).
        default: What should be returned if `val` doesn't have a unitary effect.
            If not specified, a TypeError is raised instead of returning
            a default value.

    Returns:
        If the receiving object is not able to apply its unitary effect,
        the specified default value is returned (or a TypeError is raised).

        If the receiving object was able to work inline, directly
        mutating target_tensor it will return target_tensor. The caller is
        responsible for checking if the result is target_tensor.

        If the receiving object wrote its output over available_buffer, the
        result will be available_buffer. The caller is responsible for
        checking if the result is available_buffer (and e.g. swapping
        the buffer for the target tensor before the next call).

        The receiving object may also write its output over a new buffer
        that it created, in which case that new array is returned.

    Raises:
        TypeError: `val` doesn't have a unitary effect and `default` wasn't
            specified.
    """

    # Check if the specialized method is present.
    getter = getattr(val, '_apply_unitary_to_tensor_', None)
    if getter is not None:
        result = getter(target_tensor, available_buffer, axes)
        if result is not NotImplemented:
            return result

    # Fallback to using the object's _unitary_ matrix.
    matrix = unitary(val, None)
    if matrix is not None:
        return linalg.targeted_left_multiply(
            matrix.astype(target_tensor.dtype).reshape((2,) * (2 * len(axes))),
            target_tensor,
            axes,
            out=available_buffer)

    # Don't know how to apply. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError("object of type '{}' "
                    "has no _apply_unitary_to_tensor_ "
                    "or _unitary_ methods "
                    "(or they returned NotImplemented).".format(type(val)))
