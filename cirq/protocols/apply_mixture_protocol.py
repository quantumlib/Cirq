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
"""A protocol for implementing high performance mixture evolutions."""

from typing import Any, Iterable, Optional, TypeVar, Union

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.apply_unitary_protocol import (
    apply_unitary,
    ApplyUnitaryArgs,
)

from cirq.protocols.mixture import mixture
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the apply_mixture method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class ApplyMixtureArgs:
    """Arguments for performing a mixture of unitaries.

    The receiving object is expected to mutate `target_tensor` so that it
    contains the density matrix after applying the mixture then return
    `target_tensor`. Alternatively, if workspace is required,
    the receiving object can overwrite `out_buffer` with the results
    and return `out_buffer`. Or, if the receiving object is attempting to
    be simple instead of fast, it can create an entirely new array and
    return that.

    Attributes:
        target_tensor: The input tensor that needs to be left and right
            multiplied and summed, representing the effect of the mixture.
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
        left_axes: Which axes to multiply the left action of the mixture upon.
        right_axes: Which axes to multiply the right action of the mixture upon.
    """

    def __init__(self, target_tensor: np.ndarray, out_buffer: np.ndarray,
                 auxiliary_buffer0: np.ndarray, auxiliary_buffer1: np.ndarray,
                 left_axes: Iterable[int], right_axes: Iterable[int]):
        """Args for apply mixture.

        Args:
            target_tensor: The input tensor that needs to be left and right
                multiplied and summed representing the effect of the mixture.
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
            left_axes: Which axes to multiply the left action of the mixture
                upon.
            right_axes: Which axes to multiply the right action of the mixture
                upon.
        """
        self.target_tensor = target_tensor
        self.out_buffer = out_buffer
        self.auxiliary_buffer0 = auxiliary_buffer0
        self.auxiliary_buffer1 = auxiliary_buffer1
        self.left_axes = tuple(left_axes)
        self.right_axes = tuple(right_axes)


class SupportsApplyMixture(Protocol):
    """An object that can efficiently implement a mixture."""

    def _apply_mixture_(self, args: ApplyMixtureArgs
        ) -> Union[np.ndarray, None, NotImplementedType]:
        """Efficiently applies a mixture.

        This method is given both the target tensor and workspace of the same
        shape and dtype. The method then either performs inline modifications of
        the target tensor and returns it, or writes its output into the
        a workspace tensor and returns that. This signature makes it possible to
        write specialized simulation methods that run without performing large
        allocations, significantly increasing simulation performance.

        Args:
            args: A `cirq.ApplyMixtureArgs` object with the `args.target_tensor`
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


def apply_mixture(val: Any,
                  args: ApplyMixtureArgs,
                  default: TDefault = RaiseTypeErrorIfNotProvided
                 ) -> Union[np.ndarray, TDefault]:
    """High performance evolution under a mixture of unitaries evolution.

    If `val` defines an `_apply_mixture_` method, that method will be
    used to apply `val`'s mixture effect to the target tensor. Failing that
    if `val` defines an `_apply_unitary_` method, that will be used to apply
    val's mixture effect. Otherwise, the information returned from `_mixture_`
    will be used to implement the mixture effect to the target tensor. If none
    of these cases apply, an exception is raised or the specified default value
    is returned.


    Args:
        val: The value with a mixture to apply to the target.
        args: A mutable `cirq.ApplyMixtureArgs` object describing the target
            tensor, available workspace, and left and right axes to operate on.
            The attributes of this object will be mutated as part of computing
            the result.
        default: What should be returned if `val` doesn't have a mixture. If
            not specified, a TypeError is raised instead of returning a default
            value.

    Returns:
        If the receiving object is not able to apply a mixture,
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
        TypeError: `val` doesn't have a mixture and `default` wasn't specified.
        ValueError: Different left and right shapes of `args.target_tensor`
            selected by `left_axes` and `right_axes` or `qid_shape(val)` doesn't
            equal the left and right shapes.
        AssertionError: `_apply_mixture_` returned an auxiliary buffer.
    """

    # Verify that val has the same qid shape as the selected axes of the density
    # matrix tensor.
    val_qid_shape = qid_shape_protocol.qid_shape(val,
                                                 (2,) * len(args.left_axes))
    left_shape = tuple(args.target_tensor.shape[i] for i in args.left_axes)
    right_shape = tuple(args.target_tensor.shape[i] for i in args.right_axes)
    if left_shape != right_shape:
        raise ValueError('Invalid target_tensor shape or selected axes. '
                         'The selected left and right shape of target_tensor '
                         'are not equal. Got {!r} and {!r}.'.format(
                             left_shape, right_shape))
    if val_qid_shape != left_shape:
        raise ValueError('Invalid mixture qid shape is not equal to the '
                         'selected left and right shape of target_tensor. '
                         'Got {!r} but expected {!r}.'.format(
                             val_qid_shape, left_shape))

    # Check if the specialized method is present.
    func = getattr(val, '_apply_mixture_', None)
    if func is not None:
        result = func(args)
        if result is not NotImplemented and result is not None:

            def err_str(buf_num_str):
                return ("Object of type '{}' returned a result object equal to "
                        "auxiliary_buffer{}. This type violates the contract "
                        "that appears in apply_mixture's documentation.".format(
                            type(val), buf_num_str))

            assert result is not args.auxiliary_buffer0, err_str('0')
            assert result is not args.auxiliary_buffer1, err_str('1')
            return result

    # Possibly use `apply_unitary`.
    result = _apply_unitary(val, args)
    if result is not None:
        return result

    # Fallback to using the object's `_mixture_` matrices.
    prob_mix = mixture(val, None)
    if prob_mix is not None:
        return _apply_mixture(prob_mix, args)

    # Don't know how to apply mixture. Fallback to specified default behavior.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' has no _apply_mixture_, _apply_unitary_, "
        "_unitary_, or _mixture_ methods (or they returned None or "
        "NotImplemented).".format(type(val)))


def _apply_unitary(val: Any, args: 'ApplyMixtureArgs') -> Optional[np.ndarray]:
    """Attempt to use `apply_unitary` and return the result.

    If `val` does not support `apply_unitary` returns None.
    """
    left_args = ApplyUnitaryArgs(target_tensor=args.target_tensor,
                                 available_buffer=args.auxiliary_buffer0,
                                 axes=args.left_axes)
    left_result = apply_unitary(val, left_args, None)
    if left_result is None:
        return None
    right_args = ApplyUnitaryArgs(target_tensor=np.conjugate(left_result),
                                  available_buffer=args.auxiliary_buffer0,
                                  axes=args.right_axes)
    right_result = apply_unitary(val, right_args)
    np.conjugate(right_result, out=right_result)
    return right_result


def _apply_unitary_fallback(
    val: np.ndarray, args: 'ApplyMixtureArgs') -> Optional[np.ndarray]:
    """Use regular linalg matrix multiplication.

    If `val` does not support `apply_unitary` returns None.
    """
    qid_shape = tuple(args.target_tensor.shape[i] for i in args.left_axes)
    matrix_tensor = np.reshape(val.astype(args.target_tensor.dtype),
                               qid_shape * 2)
    linalg.targeted_left_multiply(matrix_tensor,
                                  args.target_tensor,
                                  args.left_axes,
                                  out=args.auxiliary_buffer0)
    # No need to transpose as we are acting on the tensor
    # representation of matrix, so transpose is done for us.
    linalg.targeted_left_multiply(np.conjugate(matrix_tensor),
                                  args.auxiliary_buffer0,
                                  args.right_axes,
                                  out=args.target_tensor)
    return args.target_tensor


def _apply_mixture(val: Any, args: 'ApplyMixtureArgs') -> Optional[np.ndarray]:
    """Attempt to use unitary matrices in _mixture_ and return the result."""
    args.out_buffer[:] = 0
    np.copyto(dst=args.auxiliary_buffer1, src=args.target_tensor)
    for prob, op in val:
        np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer1)
        right_result = _apply_unitary(op, args)
        if right_result is None:
            right_result = _apply_unitary_fallback(op, args)

        args.out_buffer += prob * right_result

    return args.out_buffer
