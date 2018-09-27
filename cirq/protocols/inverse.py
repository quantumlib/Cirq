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

from typing import Any, Union

import collections

from cirq import value

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided = []


def inverse(val: Any, default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns the inverse `val**-1` of the given value, if defined.

    An object can define an inverse by defining a __pow__(self, exponent) method
    that returns something besides NotImplemented when given the exponent -1.
    The inverse of iterables is by default defined to be the iterable's items,
    each inverted, in reverse order.

    Args:
        val: The value (or iterable of invertable values) to invert.
        default: Determines the fallback behavior when `val` doesn't have
            an inverse defined. If `default` is not set, a TypeError is raised.
            If `default` is set to a value, that value is returned.

    Returns:
        If `val` has a __pow__ method that returns something besides
        NotImplemented when given an exponent of -1, that result is returned.
        Otherwise, if `val` is iterable, the result is a tuple with the same
        items as `val` but in reverse order and with each item inverted.
        Otherwise, if a `default` argument was specified, it is returned.

    Raises:
        TypeError: `val` doesn't have a __pow__ method, or that method returned
            NotImplemented when given -1. Furthermore `val` isn't an
            iterable containing invertible items. Also, no `default` argument
            was specified.
    """
    # Check if object defines an inverse via __pow__.
    raiser = getattr(val, '__pow__', None)
    result = NotImplemented if raiser is None else raiser(-1)
    if result is not NotImplemented:
        return result

    # Maybe it's an iterable of invertable items?
    # Note: we avoid str because 'a'[0] == 'a', which creates an infinite loop.
    if isinstance(val, collections.Iterable) and not isinstance(val, str):
        unique_indicator = []
        results = tuple(inverse(e, unique_indicator) for e in val)
        if all(e is not unique_indicator for e in results):
            return results[::-1]

    # Can't invert.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' isn't invertable. "
        "It has no __pow__ method (or the method returned NotImplemented) "
        "and it isn't an iterable of invertable objects.".format(type(val)))


def extrapolate(val: Any,
                factor: Union[float, value.Symbol],
                default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns `val**factor` of the given value, if defined.

    Values define an extrapolation by defining a __pow__(self, exponent) method.
    Note that the method may return NotImplemented to indicate a particular
    extrapolation can't be done.

    Args:
        val: The value or iterable of values to invert.
        factor: The extrapolation factor. For example, if this is 0.5 and val
            is a gate then the caller is asking for a square root of the gate.
        default: Determines the fallback behavior when `val` doesn't have
            an extrapolation defined. If `default` is not set and that occurs,
            a TypeError is raised instead.

    Returns:
        If `val` has a __pow__ method that returns something besides
        NotImplemented, that result is returned. Otherwise, if a default value
        was specified, the default value is returned.

    Raises:
        TypeError: `val` doesn't have a __pow__ method (or that method returned
            NotImplemented) and no `default` value was specified.
    """
    raiser = getattr(val, '__pow__', None)
    result = NotImplemented if raiser is None else raiser(factor)
    if result is not NotImplemented:
        return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' isn't iterable and has no __pow__ method (or it "
        "returned NotImplemented for an exponent of -1).".format(type(val)))
