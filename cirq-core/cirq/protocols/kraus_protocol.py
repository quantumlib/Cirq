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

from typing import Any, Sequence, Tuple, TypeVar, Union
import warnings

import numpy as np
from typing_extensions import Protocol

from cirq._compat import (
    deprecated,
    deprecated_class,
)
from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits,
)
from cirq.protocols import mixture_protocol
from cirq.type_workarounds import NotImplementedType


# This is a special indicator value used by the channel method to determine
# whether or not the caller provided a 'default' argument. It must be of type
# Sequence[np.ndarray] to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive
# if the user provides a different (np.array([]),) value.
RaiseTypeErrorIfNotProvided = (np.array([]),)


TDefault = TypeVar('TDefault')


@deprecated_class(deadline='v0.13', fix='use cirq.SupportsKraus instead')
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


@deprecated(deadline='v0.13', fix='use cirq.kraus instead')
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
    result = _gettr_helper(val, ['_kraus_', '_channel_'])
    if result is not None and result is not NotImplemented:
        return result

    mixture_result = mixture_protocol.mixture(val, None)
    if mixture_result is not None and mixture_result is not NotImplemented:
        return tuple(np.sqrt(p) * u for p, u in mixture_result)

    decomposed, qubits, _ = _try_decompose_into_operations_and_qubits(val)

    if decomposed is not None and decomposed != [val]:
        limit = (4 ** np.prod(len(qubits))) ** 2

        kraus_list = list(map(lambda x: _kraus_tensor(x, qubits, default), decomposed))
        if not any([_check_equality(x, default) for x in kraus_list]):
            kraus_result = kraus_list[0]
            for i in range(1, len(kraus_list)):
                kraus_result = [op_2.dot(op_1) for op_1 in kraus_result for op_2 in kraus_list[i]]
                assert (
                    len(kraus_result) < limit
                ), f"{val} kraus decomposition had combinatorial explosion."
            return tuple(kraus_result)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if _gettr_helper(val, ['_kraus_', '_unitary_', '_mixture_']) is None:
        raise TypeError(
            "object of type '{}' has no _kraus_ or _mixture_ or "
            "_unitary_ method.".format(type(val))
        )

    raise TypeError(
        "object of type '{}' does have a _kraus_, _mixture_ or "
        "_unitary_ method, but it returned NotImplemented.".format(type(val))
    )


@deprecated(deadline='v0.13', fix='use cirq.has_kraus instead')
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

    4. If decomposition is allowed apply recursion and check.

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
    result = _gettr_helper(val, ['_has_kraus_', '_has_channel_', '_kraus_', '_channel_'])
    if result is not None and result is not NotImplemented and result:
        return True

    if mixture_protocol.has_mixture(val, allow_decompose=False):
        return True

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all(has_kraus(val) for val in operations)

    # No has methods, use `_kraus_` or delegates instead.
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


def _gettr_helper(val: Any, gett_str_list: Sequence[str]):
    notImplementedFlag = False
    for gettr_str in gett_str_list:
        gettr = getattr(val, gettr_str, None)
        if gettr is None:
            continue
        if 'channel' in gettr_str:
            warnings.warn(
                f'{gettr_str} is deprecated and will be removed in cirq 0.13, rename to '
                f'{gettr_str.replace("channel", "kraus")}',
                DeprecationWarning,
            )
        result = gettr()
        if result is NotImplemented:
            notImplementedFlag = True
        elif result is not None:
            return result

    if notImplementedFlag:
        return NotImplemented
    return None
