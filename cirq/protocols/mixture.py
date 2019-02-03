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

from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ((0.0, []),)  # type: Sequence[Tuple[float, Any]]


class SupportsMixture(Protocol):
    """An object that may be describable as a probabilistic combination.

    A mixture is described by an iterable of tuples of the form

        (probability of object, object)

    The probability components of the tuples must sum to 1.0 and be between
    0 and 1 (inclusive).
    """

def _mixture_(self) -> Union[Sequence[Tuple[float, Any]], NotImplementedType]:
    pass


def mixture(
    val: Any,
    default: Any = RaiseTypeErrorIfNotProvided) -> Sequence[Tuple[float, Any]]:
    """Return a iterable of the tuples representing a probabilistic combination.

    A mixture is described by an iterable of tuples of the form

        (probability of object, object)

    The probability components of the tuples must sum to 1.0 and be between
    0 and 1 (inclusive).

    Args:
        val: The value whose mixture is being computed.
        default: A default value if val does not support mixture.

    Returns:
        An iterable of tuples of size 2. The first element of the tuple is a
        probability (between 0 and 1) and th second is the object that occurs
        with that probability in the mixture. The probabilities will sum to 1.0.
    """

    getter = getattr(val, '_mixture_', None)
    result = NotImplemented if getter is None else getter()

    if result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default

    if getter is None:
        raise TypeError(
            "object of type '{}' has no _mixture_ method.".format(type(val)))

    raise TypeError("object of type '{}' does have a _mixture_ method, "
                    "but it returned NotImplemented.".format(type(val)))


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

