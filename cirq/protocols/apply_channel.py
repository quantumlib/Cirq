# Copyright 2019 The Cirq Developers
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

"""A protocol for implementing high performance channel evolutions."""

from typing import Any, Union, TypeVar, Tuple, Iterable

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.apply_unitary import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.channel import channel
from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the apply_channel method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class Axes:
    """An enum for the two types of axes, row and column indices.

    For a tensor with shape (d_1_,...d_k), some subset of the first
    of these indices correspond to row indices, and the rest of these
    indices correspond to column indices.  These are row and column, in the
    sense that they describe a matrix where the row indices could be
    reshaped to form a row, and similar for the column indices.
    """
    ROW = 1
    COLUMN = 2


class ApplyChannelArgs:
    r"""Arguments for efficiently performing a channel.

    A channel performs the mapping
        $$
        X \rightarrow \sum_i A_i X A_i^\dagger
        $$
    for operators $A_i$ that satisfy the normalization condition
        $$
        \sum_i A_i^\dagger A_i = I.
        $$

    The receiving object is expected to mutate `target_tensor` so that it
    contains the density matrix after multiplication, and then return
    `target_tensor`. Alternatively, if workspace is required,
    the receiving object can overwrite `out_buffer` with the results
    and return `out_buffer`. Or, if the receiving object is attempting to
    be simple instead of fast, it can create an entirely new array and
    return that.

    Attributes:
        target_tensor: The input tensor that needs to be left and right
            multiplied and summed, representing the effect of the channel.
            The tensor will have the shape (2, 2, 2, ..., 2). It usually
            corresponds to a multi-qubit density matrix, with the first
            n indices corresponding to the rows of the density matrix and
            the last n indices corresponding to the columns of the density
            matrix.
        out_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor. If buffers are used, the result should
            end up in this buffer. It is the responsibility of calling code
            to notice if the result is this buffer.
        auxiliary_buffer0: Pre-allocated workspace with the same shape and dtype
            as the target tensor.
        auxiliary_buffer1: Pre-allocated workspace with the same shape
            and dtype as the target tensor.
        left_axes: Which axes to multiply the left action of the channel upon.
        right_axes: Which axes to multiply the right action of the channel upon.
    """

    def __init__(self,
            target_tensor: np.ndarray,
            out_buffer: np.ndarray,
            auxiliary_buffer0: np.ndarray,
            auxiliary_buffer1: np.ndarray,
            left_axes: Iterable[int],
            right_axes: Iterable[int]):
        """

        Args:
            target_tensor: The input tensor that needs to be left and right
                multiplied and summed representing the effect of the channel.
                The tensor will have the shape (2, 2, 2, ..., 2). It usually
                corresponds to a multi-qubit density matrix, with the first
                n indices corresponding to the rows of the density matrix and
                the last n indices corresponding to the columns of the density
                matrix.
            out_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor. If buffers are used, the result
                should end up in this buffer. It is the responsibility of
                calling code to notice if the result is this buffer.
            auxiliary_buffer0: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            auxiliary_buffer1: Pre-allocated workspace with the same shape
                and dtype as the target tensor.
            left_axes: Which axes to multiply the left action of the channel
                upon.
            right_axes: Which axes to multiply the right action of the channel
                upon.
        """
        self.target_tensor = target_tensor
        self.out_buffer = out_buffer
        self.auxiliary_buffer0 = auxiliary_buffer0
        self.auxiliary_buffer1 = auxiliary_buffer1
        self.left_axes = tuple(left_axes)
        self.right_axes = tuple(right_axes)

    def subspace_index(
            self,
            little_endian_bits_int: int,
            axes: int = Axes.ROW) -> Tuple[
        Union[slice, int, 'ellipsis'], ...]:
        """An index for the subspace where the target axes equal a value.

        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order.

        Returns:
            A value that can be used to index into `target_tensor` and
            `out_buffer`, and manipulate only the part of Hilbert space
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
        axes_indices = self.left_axes if axes == Axes.ROW else self.right_axes
        return linalg.slice_for_qubits_equal_to(axes_indices,
                                                little_endian_bits_int)


class SupportsApplyChannel(Protocol):
    """An object that can efficiently implement a channel."""

    def _apply_channel_(self, args: ApplyChannelArgs
    ) -> Union[np.ndarray, None, NotImplementedType]:
        """Efficiently applies a channel.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        a workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        Args:
            args: A `cirq.ApplyChannelArgs` object with the `args.target_tensor`
                to operate on, an `args.out_buffer`, 'args.auxiliary_buffer0`
                and `args.auxiliary_buffer1` buffers to use as temporary
                workspace, and the `args.left_axes` and `args.right_axes` of
                the tensor to target with the unitary operation. Note that
                this method is permitted (and in fact expected) to mutate
                `args.target_tensor` and the given buffers.

        Returns:
            If the receiving object is not able to apply a chanel, None
            or NotImplemented should be returned.

            If the receiving object is able to work inline, it should directly
            mutate `args.target_tensor` and then return `args.target_tensor`.
            The caller will understand this to mean that the result is in
            `args.target_tensor`.

            If the receiving object is unable to work inline, it can write its
            output over `args.out_buffer` and then return `args.out_buffer`.
            The caller will understand this to mean that the result is in
            `args.out_buffer` (and so what was `args.out` will become
            `args.target_tensor` in the next call, and vice versa).

            The receiving object is also permitted to allocate a new
            numpy.ndarray and return that as its result.
        """


def apply_channel(val: Any,
        args: ApplyChannelArgs,
        default: TDefault = RaiseTypeErrorIfNotProvided
) -> Union[np.ndarray, TDefault]:
    """High performance evolution under a channel evolution.

    If `val` defines an `_apply_channel_` method, that method will be
    used to apply `val`'s channel effect to the target tensor. Otherwise, if
    `val` defines an `_apply_unitary_` method, that method will be used to
    apply `val`s channel effect to the target tensor.  Otherwise, if `val`
    returns a non-default channel with `cirq.channel`, that channel will be
    applied using a generic method.  If none of these cases apply, an
    exception is raised or the specified default value is returned.


    Args:
        val: The value with a channel to apply to the target.
        args: A mutable `cirq.ApplyChannelArgs` object describing the target
            tensor, available workspace, and left and right axes to operate on.
            The attributes of this object will be mutated as part of computing
            the result.
        default: What should be returned if `val` doesn't have a channel. If
            not specified, a TypeError is raised instead of returning a default
            value.

    Returns:
        If the receiving object is not able to apply a channel,
        the specified default value is returned (or a TypeError is raised). If
        this occurs, then `target_tensor` should not have been mutated.

        If the receiving object was able to work inline, directly
        mutating `target_tensor` it will return `target_tensor`. The caller is
        responsible for checking if the result is `target_tensor`.

        If the receiving object wrote its output over `out_buffer`, the
        result will be `out_buffer`. The caller is responsible for
        checking if the result is `out_buffer` (and e.g. swapping
        the buffer for the target tensor before the next call).

        The receiving object may also write its output over a new buffer
        that it created, in which case that new array is returned.

    Raises:
        TypeError: `val` doesn't have a channel and `default` wasn't specified.
    """

    # Check if the specialized method is present.
    func = getattr(val, '_apply_channel_', None)
    if func is not None:
        result = func(args)
        if result is not NotImplemented and result is not None:
            return result

    # Possibly use apply_unitary.
    left_args = ApplyUnitaryArgs(target_tensor=args.target_tensor,
                                 available_buffer=args.auxiliary_buffer0,
                                 axes=args.left_axes)
    left_result = apply_unitary(val, left_args, None)
    if left_result is not None:
        right_args = ApplyUnitaryArgs(
                target_tensor=np.conjugate(left_result),
                available_buffer=args.out_buffer,
                axes=args.right_axes)
        right_result = apply_unitary(val, right_args)
        np.conjugate(right_result, out=right_result)
        return right_result

    # Fallback to using the object's _channel_ matrices.
    krauss = tuple(channel(val, None))
    if krauss is not None:
        # Special case for single-qubit operations.
        args.out_buffer[:] = 0
        np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)

        if krauss[0].shape == (2, 2):
            zero_left = args.subspace_index(0, axes=Axes.ROW)
            one_left = args.subspace_index(1, axes=Axes.ROW)
            zero_right = args.subspace_index(0, axes=Axes.COLUMN)
            one_right = args.subspace_index(1, axes=Axes.COLUMN)
            for krauss_op in krauss:
                np.copyto(dst=args.target_tensor,
                          src=args.auxiliary_buffer0)
                linalg.apply_matrix_to_slices(
                    args.target_tensor,
                    krauss_op,
                    [zero_left, one_left],
                    out=args.auxiliary_buffer1)
                # No need to transpose as we are acting on the tensor
                # representation of matrix, so transpose is done for us.
                linalg.apply_matrix_to_slices(
                    args.auxiliary_buffer1,
                    np.conjugate(krauss_op),
                    [zero_right, one_right],
                    out=args.target_tensor)
                args.out_buffer += args.target_tensor
            return args.out_buffer

        # Fallback to np.einsum for the general case.
        for krauss_op in krauss:
            np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
            krauss_tensor = np.reshape(
                krauss_op.astype(args.target_tensor.dtype),
                (2,) * len(args.left_axes) * 2)
            linalg.targeted_left_multiply(
                    krauss_tensor,
                    args.target_tensor,
                    args.left_axes,
                    out=args.auxiliary_buffer1)
            # No need to transpose as we are acting on the tensor
            # representation of matrix, so transpose is done for us.
            linalg.targeted_left_multiply(
                    np.conjugate(krauss_tensor),
                    args.auxiliary_buffer1,
                    args.right_axes,
                    out=args.target_tensor)
            args.out_buffer += args.target_tensor
        return args.out_buffer

    # Don't know how to apply. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
            "object of type '{}' has no _apply_channel_, _apply_unitary_, "
            "_unitary_, or _channel_ methods (or they returned None or "
            "NotImplemented).".format(type(val)))
