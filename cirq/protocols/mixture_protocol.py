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
import functools
from typing import Any, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Protocol

from cirq._compat import deprecated
from cirq._doc import document
from cirq.protocols.has_unitary_protocol import has_unitary

from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ((0.0, []),)  # type: Sequence[Tuple[float, Any]]


class SupportsMixture(Protocol):
    """An object that decomposes into a probability distribution of unitaries.
    """

    @document
    def _mixture_(self
                 ) -> Union[Sequence[Tuple[float, Any]], NotImplementedType]:
        """Decompose into a probability distribution of unitaries.

        This method is used by the global `cirq.mixture` method.

        A mixture is described by an iterable of tuples of the form

            (probability of unitary, unitary as numpy array)

        The probability components of the tuples must sum to 1.0 and be between
        0 and 1 (inclusive).

        Returns:
            A list of (probability, unitary) pairs.
        """

    @document
    def _has_mixture_(self) -> bool:
        """Whether this value has a mixture representation.

        This method is used by the global `cirq.has_mixture` method.  If this
        method is not present, or returns NotImplemented, it will fallback
        to using _mixture_ with a default value, or False if neither exist.

        Returns:
          True if the value has a mixture representation, Falseotherwise.
        """


def mixture(val: Any, default: Any = RaiseTypeErrorIfNotProvided
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
    """

    mixture_getter = getattr(val, '_mixture_', None)
    result = NotImplemented if mixture_getter is None else mixture_getter()
    if result is not NotImplemented:
        return result

    unitary_getter = getattr(val, '_unitary_', None)
    result = NotImplemented if unitary_getter is None else unitary_getter()
    if result is not NotImplemented:
        return ((1.0, result),)

    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if mixture_getter is None and unitary_getter is None:
        raise TypeError(
            "object of type '{}' has no _mixture_ or _unitary_ method.".format(
                type(val)))

    raise TypeError("object of type '{}' does have a _mixture_ or _unitary_ "
                    "method, but it returned NotImplemented.".format(type(val)))


def has_mixture(val: Any) -> bool:
    """Returns whether the value has a mixture representation.

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

    result = has_unitary(val)
    if result is not NotImplemented and result:
        return result

    # No _has_mixture_ or _has_unitary_ function, use _mixture_ instead.
    return mixture(val, None) is not None


@deprecated(deadline='v0.10.0',
            fix='Use "cirq.mixture" instead.',
            name='mixture_channel')
@functools.wraps(mixture)
def mixture_channel(val: Any, default: Any = RaiseTypeErrorIfNotProvided
                   ) -> Sequence[Tuple[float, np.ndarray]]:
    return mixture(val, default)


@deprecated(deadline='v0.10.0',
            fix='Use "cirq.has_mixture" instead.',
            name='has_mixture_channel')
@functools.wraps(has_mixture)
def has_mixture_channel(val: Any) -> bool:
    return has_mixture(val)


def validate_mixture(supports_mixture: SupportsMixture):
    """Validates that the mixture's tuple are valid probabilities."""
    mixture_tuple = mixture(supports_mixture, None)
    if mixture_tuple is None:
        raise TypeError('{}_mixture did not have a _mixture_ method'.format(
            supports_mixture))

    def validate_probability(p, p_str):
        if p < 0:
            raise ValueError('{} was less than 0.'.format(p_str))
        elif p > 1:
            raise ValueError('{} was greater than 1.'.format(p_str))

    total = 0.0
    for p, val in mixture_tuple:
        validate_probability(p, '{}\'s probability'.format(str(val)))
        total += p
    if not np.isclose(total, 1.0):
        raise ValueError('Sum of probabilities of a mixture was not 1.0')
