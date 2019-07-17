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

from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.apply_unitary import (
    apply_unitary, ApplyUnitaryArgs, _incorporate_result_into_buffer)
from cirq.protocols.channel import channel
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the apply_channel method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


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
        """Args for apply channel.

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
        self._target_tensor = target_tensor
        self._out_buffer = out_buffer
        self._auxiliary_buffer0 = auxiliary_buffer0
        self._auxiliary_buffer1 = auxiliary_buffer1
        self._left_axes = tuple(left_axes)
        self._right_axes = tuple(right_axes)

    @property
    def target_tensor(self) -> np.ndarray:
        return self._target_tensor

    @target_tensor.setter
    def target_tensor(self, other):
        if other is not self._target_tensor:
            raise AttributeError("can't set attribute")

    @property
    def out_buffer(self) -> np.ndarray:
        return self._out_buffer

    @out_buffer.setter
    def out_buffer(self, other):
        if other is not self._out_buffer:
            raise AttributeError("can't set attribute")

    @property
    def auxiliary_buffer0(self) -> np.ndarray:
        return self._auxiliary_buffer0

    @auxiliary_buffer0.setter
    def auxiliary_buffer0(self, other):
        if other is not self._auxiliary_buffer0:
            raise AttributeError("can't set attribute")

    @property
    def auxiliary_buffer1(self) -> np.ndarray:
        return self._auxiliary_buffer1

    @auxiliary_buffer1.setter
    def auxiliary_buffer1(self, other):
        if other is not self._auxiliary_buffer1:
            raise AttributeError("can't set attribute")

    @property
    def left_axes(self) -> np.ndarray:
        return self._left_axes

    @property
    def right_axes(self) -> np.ndarray:
        return self._right_axes

    def _for_channel_with_qid_shape(self, indices: Iterable[int],
                                    qid_shape: Tuple[int, ...]
                                   ) -> 'ApplyChannelArgs':
        """Creates a sliced and transposed view of `self` appropriate for a
        channel with shape `qid_shape` on qubits with the given indices.

        Example:
            sub_args = args._for_channel_with_qid_shape(indices, (2, 2, 2))
            # Slice where the first qubit is |1>.
            #                           (left axes) (right axes)
            sub_args.target_tensor[..., 1, :, :,    1, :, :     ]

        Args:
            indices: Integer indices into `self.left_axes` and `self.right_axes`
                specifying which qubits the channel applies to.
            qid_shape: The qid shape of the channel, the expected number of
                quantum levels in each qubit the channel applies to.

        Returns: A new `ApplyChannelArgs` where `sub_args.target_tensor` and
            `sub_args.out_buffer` are sliced and transposed views of
            `self.target_tensor` and `self.out_buffer` respectively.
        """
        slices = [slice(0, size) for size in qid_shape]
        sub_left_axes = [self.left_axes[i] for i in indices]
        sub_right_axes = [self.right_axes[i] for i in indices]
        axis_set = set(sub_left_axes)
        axis_set.update(sub_right_axes)
        other_axes = [
            axis for axis in range(len(self.target_tensor.shape))
            if axis not in axis_set
        ]
        ordered_axes = (*other_axes, *sub_left_axes, *sub_right_axes)
        # Transpose sub_left_axes+sub_right_axes to the end of the shape and
        # slice them.
        target_tensor = self.target_tensor.transpose(*ordered_axes)[(
            ..., *slices, *slices)]
        out_buffer = self.out_buffer.transpose(*ordered_axes)[(
            ..., *slices, *slices)]
        aux0 = self.auxiliary_buffer0.transpose(*ordered_axes)[(
            ..., *slices, *slices)]
        aux1 = self.auxiliary_buffer1.transpose(*ordered_axes)[(
            ..., *slices, *slices)]
        new_left_axes = range(len(other_axes),
                              len(other_axes)+len(sub_left_axes))
        new_right_axes = range(len(other_axes)+len(sub_left_axes),
                               len(ordered_axes))
        return ApplyChannelArgs(target_tensor, out_buffer, aux0, aux1,
                                new_left_axes, new_right_axes)


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

        Note that it is an error for the return object to be either of the
        auxiliary buffers, and the method will raise an AssertionError if
        this contract is violated.

        The receiving object may also write its output over a new buffer
        that it created, in which case that new array is returned.

    Raises:
        TypeError: `val` doesn't have a channel and `default` wasn't specified.
        AssertionError: if the
    """
    # Possibly use specialized method `_apply_channel_`.
    result = _strat_apply_channel_from_apply_channel(val, args)
    if result is not None and result is not NotImplemented:
        return result

    # Possibly use `apply_unitary`.
    result = _apply_unitary(val, args)
    if result is not None:
        return result

    # Fallback to using the object's `_channel_` matrices.
    krauss = channel(val, None)
    if krauss is not None:
        return _apply_krauss(krauss, args)

    # Don't know how to apply channel. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
            "object of type '{}' has no _apply_channel_, _apply_unitary_, "
            "_unitary_, or _channel_ methods (or they returned None or "
            "NotImplemented).".format(type(val)))


def _strat_apply_channel_from_apply_channel(val: Any,
                                            args: ApplyChannelArgs
                                           ) -> Optional[np.ndarray]:
    func = getattr(val, '_apply_channel_', None)
    if func is None:
        return NotImplemented
    op_qid_shape = qid_shape_protocol.qid_shape(val,
                                                (2,) * len(args.left_axes))
    sub_args = args._for_channel_with_qid_shape(range(len(op_qid_shape)),
                                                op_qid_shape)
    sub_result = func(sub_args)
    if sub_result is NotImplemented or sub_result is None:
        return sub_result
    def err_str(buf_num_str):
        return (
            "Object of type '{}' returned a result object equal to "
            "auxiliary_buffer{}. This type violates the contract "
            "that appears in apply_channel's documentation.".format(
                type(val), buf_num_str))
    assert sub_result is not sub_args.auxiliary_buffer0, err_str('0')
    assert sub_result is not sub_args.auxiliary_buffer1, err_str('1')
    return _incorporate_result_into_target(args, sub_args, sub_result)


def _apply_unitary(val: Any, args: 'ApplyChannelArgs') -> Optional[np.ndarray]:
    """Attempt to use `apply_unitary` and return the result.

    If `val` does not support `apply_unitary` returns None.
    """
    left_args = ApplyUnitaryArgs(target_tensor=args.target_tensor,
                                 available_buffer=args.auxiliary_buffer0,
                                 axes=args.left_axes)
    left_result = apply_unitary(val, left_args, None)
    if left_result is None:
        return None
    right_args = ApplyUnitaryArgs(
            target_tensor=np.conjugate(left_result),
            available_buffer=args.out_buffer,
            axes=args.right_axes)
    right_result = apply_unitary(val, right_args)
    np.conjugate(right_result, out=right_result)
    return right_result


def _apply_krauss(krauss: Union[Tuple[np.ndarray], Sequence[Any]],
        args: 'ApplyChannelArgs') -> np.ndarray:
    """Directly apply the kraus operators to the target tensor."""
    # Initialize output.
    args.out_buffer[:] = 0
    # Stash initial state into buffer0.
    np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)

    # Special case for single-qubit operations.
    if krauss[0].shape == (2, 2):
        return _apply_krauss_single_qubit(krauss, args)
    # Fallback to np.einsum for the general case.
    return _apply_krauss_multi_qubit(krauss, args)


def _apply_krauss_single_qubit(krauss: Union[Tuple[Any], Sequence[Any]],
        args: 'ApplyChannelArgs') -> np.ndarray:
    """Use slicing to apply single qubit channel."""
    zero_left = linalg.slice_for_qubits_equal_to(args.left_axes, 0)
    one_left = linalg.slice_for_qubits_equal_to(args.left_axes, 1)
    zero_right = linalg.slice_for_qubits_equal_to(args.right_axes, 0)
    one_right = linalg.slice_for_qubits_equal_to(args.right_axes, 1)
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


def _apply_krauss_multi_qubit(krauss: Union[Tuple[Any], Sequence[Any]],
        args: 'ApplyChannelArgs') -> np.ndarray:
    """Use numpy's einsum to apply a multi-qubit channel."""
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


def _incorporate_result_into_target(args: 'ApplyChannelArgs',
                                    sub_args: 'ApplyChannelArgs',
                                    sub_result: np.ndarray):
    """Takes the result of calling `_apply_channel_` on `sub_args` and
    copies it back into `args.target_tensor` or `args.out_buffer` as
    necessary to return the result of applying the channel to the full args.

    Args:
        args: The original args.
        sub_args: A version of `args` with transposed and sliced views of
            it's tensors.
        sub_result: The result of calling an object's `_apply_channel_`
            method on `sub_args`.  A transposed subspace of the desired
            result.

    Returns: The full result tensor after applying the channel.  Either
        `args.target_tensor` or `args.out_buffer`.
    """
    return _incorporate_result_into_buffer(args.target_tensor,
                                           args.out_buffer,
                                           sub_args.target_tensor,
                                           sub_args.out_buffer,
                                           sub_result)
