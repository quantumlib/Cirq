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
"""Protocol for objects that are mixtures (probabilistic combinations)."""
from typing import Any, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.protocols import has_unitary_protocol, unitary_protocol, decompose_protocol
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ((0.0, []),)  # type: Sequence[Tuple[float, Any]]


class SupportsMixture(Protocol):
    """An object that decomposes into a probability distribution of unitaries."""

    @doc_private
    def _mixture_(self) -> Union[Sequence[Tuple[float, Any]], NotImplementedType]:
        """Decompose into a probability distribution of unitaries.

        This method is used by the global `cirq.mixture` method.

        A mixture is described by an iterable of tuples of the form

            (probability of unitary, unitary as numpy array)

        The probability components of the tuples must sum to 1.0 and be between
        0 and 1 (inclusive).

        Returns:
            A list of (probability, unitary) pairs.
        """

    @doc_private
    def _has_mixture_(self) -> bool:
        """Whether this value has a mixture representation.

        This method is used by the global `cirq.has_mixture` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _mixture_ with a default value, or False if neither exist.

        Returns:
          True if the value has a mixture representation, Falseotherwise.
        """


def mixture(
    val: Any, default: Any = RaiseTypeErrorIfNotProvided
) -> Sequence[Tuple[float, np.ndarray]]:
    """Return a sequence of tuples representing a probabilistic unitary.

    A mixture is described by an iterable of tuples of the form

        (probability of unitary, unitary as numpy array)

    The probability components of the tuples must sum to 1.0 and be
    non-negative.

    Args:
        val: The value to decompose into a mixture of unitaries.
        default: A default value if val does not support mixture.

    Returns:
        An iterable of tuples of size 2. The first element of the tuple is a
        probability (between 0 and 1) and the second is the object that occurs
        with that probability in the mixture. The probabilities will sum to 1.0.

    Raises:
        TypeError: If `val` has no `_mixture_` or `_unitary_` mehod, or if it
            does and this method returned `NotImplemented`.
    """

    mixture_result = _gettr_helper(val, ['_mixture_'])
    if mixture_result is not None and mixture_result is not NotImplemented:
        return mixture_result

    unitary_result = unitary_protocol.unitary(val, None)
    if unitary_result is not None and unitary_result is not NotImplemented:
        return ((1.0, unitary_result),)

    decomposed, qubits, _ = decompose_protocol._try_decompose_into_operations_and_qubits(val)

    # serial concatenation
    if decomposed is not None and decomposed != [val]:
        limit = (4 ** np.prod(len(qubits))) ** 2

        mixture_list = list(map(lambda x: _mixture_tensor(x, qubits, default), decomposed))
        if not any([_check_equality(x, default) for x in mixture_list]):
            mixture_result = mixture_list[0]
            for i in range(1, len(mixture_list)):
                mixture_result = [
                    _product_mixture_pair(op_1, op_2)
                    for op_1 in mixture_result
                    for op_2 in mixture_list[i]
                ]
                assert (
                    len(mixture_result) < limit
                ), f"{val} mixture decomposition had combinatorial explosion."
            else:
                return tuple(mixture_result)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if _gettr_helper(val, ['_unitary_', '_mixture_']) is None:
        raise TypeError(f"object of type '{type(val)}' has no _mixture_ or _unitary_ method.")

    raise TypeError(
        "object of type '{}' does have a _mixture_ or _unitary_ "
        "method, but it returned NotImplemented.".format(type(val))
    )


def has_mixture(val: Any, *, allow_decompose: bool = True) -> bool:
    """Returns whether the value has a mixture representation.

    Args:
        val: The value to check.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When false, the decomposition strategy for determining
            the result is skipped.

    Returns:
        If `val` has a `_has_mixture_` method and its result is not
        NotImplemented, that result is returned. Otherwise, if the value
        has a `_mixture_` method return True if that has a non-default value.
        Returns False if neither function exists.
    """
    result = _gettr_helper(val, ['_has_mixture_', '_mixture_'])
    if result is not None and result is not NotImplemented and result:
        return True

    if has_unitary_protocol.has_unitary(val, allow_decompose=False):
        return True

    if allow_decompose:
        operations, _, _ = decompose_protocol._try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all(has_mixture(val) for val in operations)

    return False


def validate_mixture(supports_mixture: SupportsMixture):
    """Validates that the mixture's tuple are valid probabilities."""
    mixture_tuple = mixture(supports_mixture, None)
    if mixture_tuple is None:
        raise TypeError(f'{supports_mixture}_mixture did not have a _mixture_ method')

    def validate_probability(p, p_str):
        if p < 0:
            raise ValueError(f'{p_str} was less than 0.')
        elif p > 1:
            raise ValueError(f'{p_str} was greater than 1.')

    total = 0.0
    for p, val in mixture_tuple:
        validate_probability(p, f"{str(val)}'s probability")
        total += p
    if not np.isclose(total, 1.0):
        raise ValueError("Sum of probabilities of a mixture was not 1.0")


def _check_equality(x, y):
    if type(x) != type(y):
        return False
    if type(x) not in [list, tuple, np.ndarray]:
        return x == y
    if type(x) == np.ndarray:
        return x.shape == y.shape and np.all(x == y)
    return False if len(x) != len(y) else all([_check_equality(a, b) for a, b in zip(x, y)])


def _tensor_mixture_pair(x, y):
    p_new = x[0] * y[0]
    mat_new = np.kron(x[1], y[1])
    return (p_new, mat_new)


def _product_mixture_pair(x, y):
    p_new = x[0] * y[0]
    mat_new = y[1].dot(x[1])
    return (p_new, mat_new)


def _mixture_tensor(op, qubits, default):
    mixture_list = mixture(op, default)
    if _check_equality(mixture_list, default):
        return default

    val = None
    op_q = op.qubits
    found = False
    for i in range(len(qubits)):
        if qubits[i] in op_q:
            if not found:
                found = True
                if val is None:
                    val = mixture_list
                else:
                    val = tuple([_tensor_mixture_pair(x, y) for x in val for y in mixture_list])

        elif val is None:
            val = ((1, np.identity(2)),)
        else:
            val = tuple([_tensor_mixture_pair(x, (1, np.identity(2))) for x in val])

    return val


def _gettr_helper(val: Any, gett_str_list: Sequence[str]):
    notImplementedFlag = False
    for gettr_str in gett_str_list:
        gettr = getattr(val, gettr_str, None)
        if gettr is None:
            continue
        result = gettr()
        if result is NotImplemented:
            notImplementedFlag = True
        elif result is not None:
            return result

    if notImplementedFlag:
        return NotImplemented
    return None
