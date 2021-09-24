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

"""Protocol and methods for quantum channels."""

from typing import Any, Sequence, Tuple, TypeVar, Union, List
import warnings

import numpy as np
from typing_extensions import Protocol

from cirq._compat import (
    deprecated,
    deprecated_class,
)
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits, decompose
from cirq.protocols.mixture_protocol import mixture, has_mixture
from cirq.protocols.unitary_protocol import unitary
from cirq.protocols.has_unitary_protocol import has_unitary

from cirq.ops.raw_types import Qid

from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the channel method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# Sequence[np.ndarray] to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive
# if the user provides a different (np.array([]),) value.
RaiseTypeErrorIfNotProvided = (np.array([]),)


TDefault = TypeVar("TDefault")


@deprecated_class(deadline="v0.13", fix="use cirq.SupportsKraus instead")
class SupportsChannel(Protocol):
    pass


class SupportsKraus(Protocol):
    """An object that may be describable as a quantum channel."""

    @doc_private
    def _kraus_(self) -> Union[Sequence[np.ndarray], NotImplementedType]:
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


@deprecated(deadline="v0.13", fix="use cirq.kraus instead")
def channel(
    val: Any, default: Any = RaiseTypeErrorIfNotProvided
) -> Union[Tuple[np.ndarray, ...], TDefault]:
    return kraus(val, default=default)


def kraus(
    val: Any, default: Any = RaiseTypeErrorIfNotProvided
) -> Union[Tuple[np.ndarray, ...], TDefault]:
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
    channel_getter = getattr(val, "_channel_", None)
    if channel_getter is not None:
        warnings.warn(
            "_channel_ is deprecated and will be removed in cirq 0.13, rename to _kraus_",
            DeprecationWarning,
        )

    channel_result = NotImplemented if channel_getter is None else channel_getter()
    if channel_result is not NotImplemented:
        return tuple(channel_result)

    _, kraus_result = _strat_kraus_from_kraus(val)
    if kraus_result is not None and kraus_result is not NotImplemented:
        return kraus_result

    mixture_result = mixture(val, None)
    if mixture_result is not None and mixture_result is not NotImplemented:
        return tuple(np.sqrt(p) * u for p, u in mixture_result)

    unitary_result = unitary(val, None)
    if unitary_result is not None and unitary_result is not NotImplemented:
        return (unitary_result,)

    decomposed = decompose(val)

    if decomposed != [val]:
        qubits: List[Qid] = []
        for x in decomposed:
            qubits.extend(x.qubits)

        qubits = sorted(list(set(qubits)))
        kraus_list = list(map(lambda x: _kraus_tensor(x, qubits, default), decomposed))
        if not any([_check_equality(x, default) for x in kraus_list]):
            kraus_result = kraus_list[0]
            for i in range(1, len(kraus_list)):
                kraus_result = [op_2.dot(op_1) for op_1 in kraus_result for op_2 in kraus_list[i]]

            return tuple(kraus_result)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if not any(
        getattr(val, instance, None) is not None
        for instance in ["_kraus_", "_unitary_", "_mixture_"]
    ):
        raise TypeError(
            "object of type '{}' has no _kraus_ or _mixture_ or "
            "_unitary_ method.".format(type(val))
        )

    raise TypeError(
        "object of type '{}' does have a _kraus_, _mixture_ or "
        "_unitary_ method, but it returned NotImplemented.".format(type(val))
    )


@deprecated(deadline="v0.13", fix="use cirq.has_kraus instead")
def has_channel(val: Any, *, allow_decompose: bool = True) -> bool:
    return has_kraus(val, allow_decompose=allow_decompose)


def has_kraus(val: Any, *, allow_decompose: bool = True) -> bool:
    """Determines whether the value has a Kraus representation.

    Determines whether `val` has a Kraus representation by attempting
    the following strategies:

    1. Try to use `val.has_channel_()`.
        Case a) Method not present or returns `None`.
            Continue to next strategy.
        Case b) Method returns `True`.
            Kraus.

    2. Try to use `val._kraus_()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns a 3D array.
            Kraus.

    3. Try to use `cirq.mixture()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns a 3D array.
            Kraus.

    4. Try to use `cirq.unitary()`.
        Case a) Method not present or returns `NotImplemented`.
            No Kraus.
        Case b) Method returns a 3D array.
            Kraus.

    5. If decomposition is allowed apply recursion and check.

    If all the above methods fail then it is assumed to have no Kraus
    representation.

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
        Whether or not `val` has a Kraus representation.
    """
    channel_getter = getattr(val, "_has_channel_", None)
    if channel_getter is not None:
        warnings.warn(
            "_has_channel_ is deprecated and will be removed in cirq 0.13, rename to _has_kraus_",
            DeprecationWarning,
        )

    result = NotImplemented if channel_getter is None else channel_getter()
    if result is not NotImplemented and result:
        return True

    for instance in [has_unitary, has_mixture]:
        result = instance(val)
        if result is not NotImplemented and result:
            return True

    getter = getattr(val, "_has_kraus_", None)
    result = NotImplemented if getter is None else getter()
    if result is not NotImplemented and result:
        return True

    strats = [_strat_kraus_from_kraus]
    if any(strat(val)[1] is not None and strat(val)[1] is not NotImplemented for strat in strats):
        return True

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all(has_kraus(val) for val in operations)

    return False


def _check_equality(x, y):
    if type(x) != type(y):
        return False
    if type(x) not in [list, tuple, np.ndarray]:
        return x == y
    if type(x) == np.ndarray:
        return x.shape == y.shape and np.all(x == y)
    return False if len(x) != len(y) else all([_check_equality(a, b) for a, b in zip(x, y)])


def _kraus_tensor(op, qubits, default):
    kraus_list = kraus(op, default)
    if _check_equality(kraus_list, default):
        return default

    val = None
    op_q = op.qubits
    found = False
    for i in range(len(qubits)):
        if qubits[i] in op_q:
            if not found:
                found = True
                if val is None:
                    val = kraus_list
                else:
                    val = tuple([np.kron(x, y) for x in val for y in kraus_list])

        elif val is None:
            val = (np.identity(2),)
        else:
            val = tuple([np.kron(x, np.identity(2)) for x in val])

    return val


def _strat_kraus_from_kraus(val: Any):
    """Attempts to compute the value's kraus via its _kraus_ method."""
    kraus_getter = getattr(val, "_kraus_", None)
    kraus_result = NotImplemented if kraus_getter is None else kraus_getter()
    if kraus_result is not NotImplemented:
        return kraus_getter, tuple(kraus_result)

    return kraus_getter, kraus_result
