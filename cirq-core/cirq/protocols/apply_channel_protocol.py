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
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the apply_channel method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided: np.ndarray = np.array([])

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

    def __init__(
        self,
        target_tensor: np.ndarray,
        out_buffer: np.ndarray,
        auxiliary_buffer0: np.ndarray,
        auxiliary_buffer1: np.ndarray,
        left_axes: Iterable[int],
        right_axes: Iterable[int],
    ):
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
        self.target_tensor = target_tensor
        self.out_buffer = out_buffer
        self.auxiliary_buffer0 = auxiliary_buffer0
        self.auxiliary_buffer1 = auxiliary_buffer1
        self.left_axes = tuple(left_axes)
        self.right_axes = tuple(right_axes)


class SupportsApplyChannel(Protocol):
    """An object that can efficiently implement a channel."""

    @doc_private
    def _apply_channel_(
        self, args: ApplyChannelArgs
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


def apply_channel(
    val: Any, args: ApplyChannelArgs, default: TDefault = RaiseTypeErrorIfNotProvided
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
        ValueError: Different left and right shapes of `args.target_tensor`
            selected by `left_axes` and `right_axes` or `qid_shape(val)` doesn't
            equal the left and right shapes.
        AssertionError: `_apply_channel_` returned an auxiliary buffer.
    """
    # Verify that val has the same qid shape as the selected axes of the density
    # matrix tensor.
    val_qid_shape = qid_shape_protocol.qid_shape(val, (2,) * len(args.left_axes))
    left_shape = tuple(args.target_tensor.shape[i] for i in args.left_axes)
    right_shape = tuple(args.target_tensor.shape[i] for i in args.right_axes)
    if left_shape != right_shape:
        raise ValueError(
            'Invalid target_tensor shape or selected axes. '
            'The selected left and right shape of target_tensor '
            'are not equal. Got {!r} and {!r}.'.format(left_shape, right_shape)
        )
    if val_qid_shape != left_shape:
        raise ValueError(
            'Invalid channel qid shape is not equal to the '
            'selected left and right shape of target_tensor. '
            'Got {!r} but expected {!r}.'.format(val_qid_shape, left_shape)
        )

    # Check if the specialized method is present.
    if hasattr(val, '_apply_channel_'):
        result = val._apply_channel_(args)
        if result is not NotImplemented and result is not None:

            def err_str(buf_num_str):
                return (
                    "Object of type '{}' returned a result object equal to "
                    "auxiliary_buffer{}. This type violates the contract "
                    "that appears in apply_channel's documentation.".format(type(val), buf_num_str)
                )

            assert result is not args.auxiliary_buffer0, err_str('0')
            assert result is not args.auxiliary_buffer1, err_str('1')
            return result

    # Possibly use `apply_unitary`.
    result = _apply_unitary(val, args)
    if result is not None:
        return result

    # Fallback to using the object's `_kraus_` matrices.
    ks = kraus(val, None)
    if ks is not None:
        return _apply_kraus(ks, args)

    # Don't know how to apply channel. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' has no _apply_channel_, _apply_unitary_, "
        "_unitary_, or _kraus_ methods (or they returned None or "
        "NotImplemented).".format(type(val))
    )


def _apply_unitary(val: Any, args: 'ApplyChannelArgs') -> Optional[np.ndarray]:
    """Attempt to use `apply_unitary` and return the result.

    If `val` does not support `apply_unitary` returns None.
    """
    left_args = ApplyUnitaryArgs(
        target_tensor=args.target_tensor,
        available_buffer=args.auxiliary_buffer0,
        axes=args.left_axes,
    )
    left_result = apply_unitary(val, left_args, None)
    if left_result is None:
        return None
    right_args = ApplyUnitaryArgs(
        target_tensor=np.conjugate(left_result),
        available_buffer=args.out_buffer,
        axes=args.right_axes,
    )
    right_result = apply_unitary(val, right_args)
    np.conjugate(right_result, out=right_result)
    return right_result


def _apply_kraus(
    kraus: Union[Tuple[np.ndarray], Sequence[Any]], args: 'ApplyChannelArgs'
) -> np.ndarray:
    """Directly apply the kraus operators to the target tensor."""
    # Initialize output.
    args.out_buffer[:] = 0
    # Stash initial state into buffer0.
    np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)

    # Special case for single-qubit operations.
    if len(args.left_axes) == 1 and kraus[0].shape == (2, 2):
        return _apply_kraus_single_qubit(kraus, args)
    # Fallback to np.einsum for the general case.
    return _apply_kraus_multi_qubit(kraus, args)


def _apply_kraus_single_qubit(
    kraus: Union[Tuple[Any], Sequence[Any]], args: 'ApplyChannelArgs'
) -> np.ndarray:
    """Use slicing to apply single qubit channel.  Only for two-level qubits."""
    zero_left = linalg.slice_for_qubits_equal_to(args.left_axes, 0)
    one_left = linalg.slice_for_qubits_equal_to(args.left_axes, 1)
    zero_right = linalg.slice_for_qubits_equal_to(args.right_axes, 0)
    one_right = linalg.slice_for_qubits_equal_to(args.right_axes, 1)
    for kraus_op in kraus:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
        linalg.apply_matrix_to_slices(
            args.target_tensor, kraus_op, [zero_left, one_left], out=args.auxiliary_buffer1
        )
        # No need to transpose as we are acting on the tensor
        # representation of matrix, so transpose is done for us.
        linalg.apply_matrix_to_slices(
            args.auxiliary_buffer1,
            np.conjugate(kraus_op),
            [zero_right, one_right],
            out=args.target_tensor,
        )
        args.out_buffer += args.target_tensor
    return args.out_buffer


def _apply_kraus_multi_qubit(
    kraus: Union[Tuple[Any], Sequence[Any]], args: 'ApplyChannelArgs'
) -> np.ndarray:
    """Use numpy's einsum to apply a multi-qubit channel."""
    qid_shape = tuple(args.target_tensor.shape[i] for i in args.left_axes)
    for kraus_op in kraus:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
        kraus_tensor = np.reshape(kraus_op.astype(args.target_tensor.dtype), qid_shape * 2)
        linalg.targeted_left_multiply(
            kraus_tensor, args.target_tensor, args.left_axes, out=args.auxiliary_buffer1
        )
        # No need to transpose as we are acting on the tensor
        # representation of matrix, so transpose is done for us.
        linalg.targeted_left_multiply(
            np.conjugate(kraus_tensor),
            args.auxiliary_buffer1,
            args.right_axes,
            out=args.target_tensor,
        )
        args.out_buffer += args.target_tensor
    return args.out_buffer
