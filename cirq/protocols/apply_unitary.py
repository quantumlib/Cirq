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


from typing import Any, Union, TypeVar, Tuple, Iterable

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.unitary import unitary
from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the apply_unitary method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class ApplyUnitaryArgs:
    """Arguments for performing an efficient left-multiplication by a unitary.

    The receiving object is expected to mutate `target_tensor` so that it
    contains the state after multiplication, and then return `target_tensor`.
    Alternatively, if workspace is required, the receiving object can overwrite
    `available_buffer` with the results and return `available_buffer`. Or, if
    the receiving object is attempting to be simple instead of fast, it can
    create an entirely new array and return that.

    Attributes:
        target_tensor: The input tensor that needs to be left-multiplied by
            the unitary effect of the receiving object. The tensor will
            have the shape (2, 2, 2, ..., 2). It usually corresponds to
            a multi-qubit superposition, but it could also be a multi-qubit
            unitary transformation or some other concept.
        available_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor.
        axes: Which axes the unitary effect is being applied to (e.g. the
            qubits that the gate is operating on).
    """

    def __init__(self,
                 target_tensor: np.ndarray,
                 available_buffer: np.ndarray,
                 axes: Iterable[int]):
        """

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
        """
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.axes = tuple(axes)

    def subspace_index(self, little_endian_bits_int: int
                       ) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
        """An index for the subspace where the target axes equal a value.

        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order.

        Returns:
            A value that can be used to index into `target_tensor` and
            `available_buffer`, and manipulate only the part of Hilbert space
            corresponding to a given bit assignment.

        Example:
            If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
            then this method will return the following when given
            `little_endian_bits=0b01`:

                `(slice(None), 0, slice(None), 1, Ellipsis)`

            Therefore the following two lines would be equivalent:

                args.target_tensor[args.subspace_index(0b01)] += 1

                args.target_tensor[:, 0, :, 1] += 1
        """
        return linalg.slice_for_qubits_equal_to(self.axes,
                                                little_endian_bits_int)


class SupportsApplyUnitary(Protocol):
    """An object that can be efficiently left-multiplied into tensors."""

    def _apply_unitary_(self, args: ApplyUnitaryArgs
                        ) -> Union[np.ndarray, None, NotImplementedType]:
        """Left-multiplies a unitary effect onto a tensor with good performance.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        The target may represent a wave function, a unitary matrix, or some
        other tensor. Implementations will work in all of these cases as long as
        they correctly focus on only operating on the given axes.

        Args:
            args: A `cirq.ApplyUnitaryArgs` object with the `args.target_tensor`
                to operate on, an `args.available_workspace` buffer to use as
                temporary workspace, and the `args.axes` of the tensor to target
                with the unitary operation. Note that this method is permitted
                (and in fact expected) to mutate `args.target_tensor` and
                `args.available_workspace`.

        Returns:
            If the receiving object is not able to apply its unitary effect,
            None or NotImplemented should be returned.

            If the receiving object is able to work inline, it should directly
            mutate `args.target_tensor` and then return `args.target_tensor`.
            The caller will understand this to mean that the result is in
            `args.target_tensor`.

            If the receiving object is unable to work inline, it can write its
            output over `args.available_buffer` and then return
            `args.available_buffer`. The caller will understand this to mean
            that the result is in `args.available_buffer` (and so what was
            `args.available_buffer` will become `args.target_tensor` in the next
            call, and vice versa).

            The receiving object is also permitted to allocate a new
            numpy.ndarray and return that as its result.
        """


def apply_unitary(unitary_value: Any,
                  args: ApplyUnitaryArgs,
                  default: TDefault = RaiseTypeErrorIfNotProvided
                  ) -> Union[np.ndarray, TDefault]:
    """High performance left-multiplication of a unitary effect onto a tensor.

    If `unitary_value` defines an `_apply_unitary_` method, that method will be
    used to apply `unitary_value`'s unitary effect to the target tensor.
    Otherwise, if `unitary_value` defines a `_unitary_` method, its unitary
    matrix will be retrieved and applied using a generic method. Otherwise the
    application fails, and either an exception is raised or the specified
    default value is returned.

    Args:
        unitary_value: The value with a unitary effect to apply to the target.
        args: A mutable `cirq.ApplyUnitaryArgs` object describing the target
            tensor, available workspace, and axes to operate on. The attributes
            of this object will be mutated as part of computing the result.
        default: What should be returned if `unitary_value` doesn't have a
            unitary effect. If not specified, a TypeError is raised instead of
            returning a default value.

    Returns:
        If the receiving object is not able to apply its unitary effect,
        the specified default value is returned (or a TypeError is raised). If
        this occurs, then `target_tensor` should not have been mutated.

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
        TypeError: `unitary_value` doesn't have a unitary effect and `default`
            wasn't specified.
    """

    # Check if the specialized method is present.
    func = getattr(unitary_value, '_apply_unitary_', None)
    if func is not None:
        result = func(args)
        if result is not NotImplemented and result is not None:
            return result

    # Fallback to using the object's _unitary_ matrix.
    matrix = unitary(unitary_value, None)
    if matrix is not None:
        # Special case for single-qubit operations.
        if matrix.shape == (2, 2):
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            return linalg.apply_matrix_to_slices(args.target_tensor,
                                                 matrix,
                                                 [zero, one],
                                                 out=args.available_buffer)

        # Fallback to np.einsum for the general case.
        return linalg.targeted_left_multiply(
            matrix.astype(args.target_tensor.dtype).reshape(
                (2,) * (2 * len(args.axes))),
            args.target_tensor,
            args.axes,
            out=args.available_buffer)

    # Don't know how to apply. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' has no _apply_unitary_ or _unitary_ methods "
        "(or they returned None or NotImplemented).".format(
            type(unitary_value)))
