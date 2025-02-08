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

from types import NotImplementedType
from typing import Any, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.protocols.has_unitary_protocol import has_unitary
from cirq.protocols.unitary_protocol import unitary

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided: Sequence[Tuple[float, Any]] = ((0.0, []),)


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
        TypeError: If `val` has no `_mixture_` or `_unitary_` method, or if it
            does and this method returned `NotImplemented`.
    """

    mixture_getter = getattr(val, '_mixture_', None)
    result = NotImplemented if mixture_getter is None else mixture_getter()
    if result is not NotImplemented and result is not None:
        return tuple((p, u if isinstance(u, np.ndarray) else unitary(u)) for p, u in result)

    unitary_getter = getattr(val, '_unitary_', None)
    result = NotImplemented if unitary_getter is None else unitary_getter()
    if result is not NotImplemented:
        return ((1.0, result),)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if mixture_getter is None and unitary_getter is None:
        raise TypeError(f"object of type '{type(val)}' has no _mixture_ or _unitary_ method.")

    raise TypeError(
        f"object of type '{type(val)}' does have a _mixture_ or _unitary_ "
        "method, but it returned NotImplemented."
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
    mixture_getter = getattr(val, '_has_mixture_', None)
    result = NotImplemented if mixture_getter is None else mixture_getter()
    if result is not NotImplemented:
        return result

    if has_unitary(val, allow_decompose=False):
        return True

    if allow_decompose:
        operations, _, _ = _try_decompose_into_operations_and_qubits(val)
        if operations is not None:
            return all(has_mixture(val) for val in operations)

    # No _has_mixture_ or _has_unitary_ function, use _mixture_ instead.
    return mixture(val, None) is not None


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
        validate_probability(p, f"{val}'s probability")
        total += p
    if not np.isclose(total, 1.0):
        raise ValueError('Sum of probabilities of a mixture was not 1.0')
