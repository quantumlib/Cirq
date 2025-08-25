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

"""Protocol and methods for obtaining Kraus representation of quantum channels."""

from __future__ import annotations

import warnings
from types import NotImplementedType
from typing import Any, Protocol, Sequence, TypeVar

import numpy as np

from cirq import protocols, qis
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.protocols.mixture_protocol import has_mixture
from cirq.protocols.unitary_protocol import unitary

# This is a special indicator value used by the channel method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# Sequence[np.ndarray] to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive
# if the user provides a different (np.array([]),) value.
RaiseTypeErrorIfNotProvided: tuple[np.ndarray] = (np.array([]),)


TDefault = TypeVar('TDefault')


class SupportsKraus(Protocol):
    """An object that may be describable as a quantum channel."""

    @doc_private
    def _kraus_(self) -> Sequence[np.ndarray] | NotImplementedType:
        r"""A list of Kraus matrices describing the quantum channel.

        These matrices are the terms in the operator sum representation of a
        quantum channel. If the returned matrices are ${A_0,A_1,..., A_{r-1}}$,
        then this describes the channel:
            $$
            \rho \rightarrow \sum_{k=0}^{r-1} A_k \rho A_k^\dagger
            $$
        These matrices are required to satisfy the trace preserving condition
            $$
            \sum_{k=0}^{r-1} A_k^\dagger A_k = I
            $$
        where $I$ is the identity matrix. The matrices $A_k$ are sometimes
        called Kraus or noise operators.

        This method is used by the global `cirq.channel` method. If this method
        or the _unitary_ method is not present, or returns NotImplement,
        it is assumed that the receiving object doesn't have a channel
        (resulting in a TypeError or default result when calling `cirq.channel`
        on it). (The ability to return NotImplemented is useful when a class
        cannot know if it is a channel until runtime.)

        The order of cells in the matrices is always implicit with respect to
        the object being called. For example, for GateOperations these matrices
        must be ordered with respect to the list of qubits that the channel is
        applied to. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        Returns:
            A list of matrices describing the channel (Kraus operators), or
            NotImplemented if there is no such matrix.
        """

    @doc_private
    def _has_kraus_(self) -> bool:
        """Whether this value has a Kraus representation.

        This method is used by the global `cirq.has_channel` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to similarly checking `cirq.has_mixture` or `cirq.has_unitary`. If none
        of these are present or return NotImplemented, then `cirq.has_channel`
        will fall back to checking whether `cirq.channel` has a non-default
        value. Otherwise `cirq.has_channel` returns False.

        Returns:
            True if the value has a channel representation, False otherwise.
        """


def _strat_kraus_from_apply_channel(val: Any, atol: float) -> tuple[np.ndarray, ...] | None:
    """Attempts to compute a value's Kraus operators via its _apply_channel_ method.
    This is very expensive (O(16^N)), so only do this as a last resort.

    Args:
        val: value to calculate kraus channels from.
        atol: Absolute tolerance for super-operator calculation.
            Matrices with all entries less than this will be dropped."""
    method = getattr(val, '_apply_channel_', None)
    if method is None:
        return None

    qid_shape = protocols.qid_shape(val)

    eye = qis.eye_tensor(qid_shape * 2, dtype=np.complex128)
    buffer = np.empty_like(eye)
    buffer.fill(float('nan'))
    superop = protocols.apply_channel(
        val=val,
        args=protocols.ApplyChannelArgs(
            target_tensor=eye,
            out_buffer=buffer,
            auxiliary_buffer0=buffer.copy(),
            auxiliary_buffer1=buffer.copy(),
            left_axes=list(range(len(qid_shape))),
            right_axes=list(range(len(qid_shape), len(qid_shape) * 2)),
        ),
        default=None,
    )
    if superop is None or superop is NotImplemented:
        return None
    n = np.prod(qid_shape) ** 2
    # Note that super-operator calculations can be numerically unstable
    # and we want to avoid returning kraus channels with "almost zero"
    # components
    kraus_ops = qis.superoperator_to_kraus(superop.reshape((n, n)), atol=atol)
    return tuple(kraus_ops)


def kraus(
    val: Any, default: Any = RaiseTypeErrorIfNotProvided, atol: float = 1e-6
) -> tuple[np.ndarray, ...] | TDefault:
    r"""Returns a list of matrices describing the channel for the given value.

    These matrices are the terms in the operator sum representation of
    a quantum channel. If the returned matrices are ${A_0,A_1,..., A_{r-1}}$,
    then this describes the channel:
        $$
        \rho \rightarrow \sum_{k=0}^{r-1} A_k \rho A_k^\dagger
        $$
    These matrices are required to satisfy the trace preserving condition
        $$
        \sum_{k=0}^{r-1} A_k^\dagger A_k = I
        $$
    where $I$ is the identity matrix. The matrices $A_k$ are sometimes called
    Kraus or noise operators.

    Args:
        val: The value to describe by a channel.
        default: Determines the fallback behavior when `val` doesn't have
            a channel. If `default` is not set, a TypeError is raised. If
            default is set to a value, that value is returned.
        atol: If calculating Kraus channels from channels, use this tolerance
            for determining whether a super-operator is all zeros.

    Returns:
        If `val` has a `_kraus_` method and its result is not NotImplemented,
        that result is returned. Otherwise, if `val` has a `_mixture_` method
        and its results is not NotImplement a tuple made up of channel
        corresponding to that mixture being a probabilistic mixture of unitaries
        is returned.  Otherwise, if `val` has a `_unitary_` method and
        its result is not NotImplemented a tuple made up of that result is
        returned. Otherwise, if a default value was specified, the default
        value is returned.

    Raises:
        TypeError: `val` doesn't have a _kraus_ or _unitary_ method (or that
            method returned NotImplemented) and also no default value was
            specified.
    """
    channel_getter = getattr(val, '_channel_', None)
    if channel_getter is not None:
        warnings.warn(  # pragma: no cover
            '_channel_ is deprecated and will be removed in cirq 0.13, rename to _kraus_',
            DeprecationWarning,
        )

    kraus_getter = getattr(val, '_kraus_', None)
    kraus_result = NotImplemented if kraus_getter is None else kraus_getter()
    if kraus_result is not NotImplemented:
        return tuple(kraus_result)

    mixture_getter = getattr(val, '_mixture_', None)
    mixture_result = NotImplemented if mixture_getter is None else mixture_getter()
    if mixture_result is not NotImplemented and mixture_result is not None:
        return tuple(np.sqrt(p) * unitary(u) for p, u in mixture_result)

    unitary_getter = getattr(val, '_unitary_', None)
    unitary_result = NotImplemented if unitary_getter is None else unitary_getter()
    if unitary_result is not NotImplemented and unitary_result is not None:
        return (unitary_result,)

    if has_unitary(val):
        return (unitary(val),)

    channel_result = NotImplemented if channel_getter is None else channel_getter()
    if channel_result is not NotImplemented:
        return tuple(channel_result)  # pragma: no cover

    # Last-resort fallback: try to derive Kraus from _apply_channel_.
    # Note: _apply_channel can lead to kraus being called again, so if default
    # is None, this can trigger an infinite loop.
    if default is not None:
        result = _strat_kraus_from_apply_channel(val, atol)
        if result is not None:
            return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if kraus_getter is None and unitary_getter is None and mixture_getter is None:
        raise TypeError(
            f"object of type '{type(val)}' has no _kraus_ or _mixture_ or _unitary_ method."
        )

    raise TypeError(
        f"object of type '{type(val)}' does have a _kraus_, _mixture_ or "
        "_unitary_ method, but it returned NotImplemented."
    )


def has_kraus(val: Any, *, allow_decompose: bool = True) -> bool:
    """Returns whether the value has a Kraus representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When False, the decomposition strategy for determining
            the result is skipped.

    Returns:
        If `val` has a `_has_kraus_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if `val` has a
        `_has_mixture_` method and its result is not NotImplemented, that
        result is returned. Otherwise if `val` has a `_has_unitary_` method
        and its results is not NotImplemented, that result is returned.
        Otherwise, if the value has a _kraus_ method return if that
        has a non-default value. Returns False if none of these functions
        exists.
    """
    kraus_getter = getattr(val, '_has_kraus_', None)
    result = NotImplemented if kraus_getter is None else kraus_getter()
    if result is not NotImplemented:
        return result

    result = has_mixture(val, allow_decompose=False)
    if result is not NotImplemented and result:
        return result

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all(has_kraus(val) for val in operations)

    # No has methods, use `_kraus_` or delegates instead.
    return kraus(val, None) is not None
