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

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
from cirq import value

RaiseTypeErrorIfNotProvided = []


def inverse(val: Any, default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns the inverse `val**-1` of the given value, if defined.

    Values define an inverse by defining a __pow__(self, exponent) method that
    returns something besides NotImplemented when given the exponent -1. When
    the given value is a list or other iterable, the inverse is defined to
    be a tuple with the iterable's items each inverted and in reverse order.

    Args:
        val: The value or iterable of values to invert.
        default: Determines the fallback behavior when `val` doesn't have
            an inverse defined. If `default` is not set, a TypeError is raised.
            If `default` is set to a value, that value is returned.

    Returns:
        If `val` has an __pow__ method that returns something besides
        NotImplemented when given an exponent of -1, that result is returned.
        Otherwise, if `val` is iterable, the result is a reversed tuple with
        inverted items. Otherwise, if a default value was specified, the default
        value is returned.

    Raises:
        TypeError: `val` doesn't have a __pow__ method, or that method returned
            NotImplemented when given -1. Furthermore `val` isn't an
            iterable containing invertible items. Finally, no `default` value
            was specified.
    """
    raiser = getattr(val, '__pow__', None)
    result = NotImplemented if raiser is None else raiser(-1)
    if result is not NotImplemented:
        return result

    if isinstance(val, collections.Iterable):
        return tuple(inverse(e) for e in reversed(list(val)))

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        "object of type '{}' isn't iterable and has no __pow__ method (or it "
        "returned NotImplemented for an exponent of -1).".format(type(val)))


def extrapolate(val: Any,
                factor: Union[float, value.Symbol],
                default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns the inverse `val**-1` of the given value, if defined.

    Values define an inverse by defining a __pow__(self, exponent) method that
    returns something besides NotImplemented when given the exponent -1. When
    the given value is a list or other iterable, the inverse is defined to
    be a tuple with the iterable's items each inverted and in reverse order.

    Args:
        val: The value or iterable of values to invert.
        default: Determines the fallback behavior when `val` doesn't have
            an inverse defined. If `default` is not set, a TypeError is raised.
            If `default` is set to a value, that value is returned.

    Returns:
        If `val` has an __pow__ method that returns something besides
        NotImplemented when given an exponent of -1, that result is returned.
        Otherwise, if `val` is iterable, the result is a reversed tuple with
        inverted items. Otherwise, if a default value was specified, the default
        value is returned.

    Raises:
        TypeError: `val` doesn't have a __pow__ method, or that method returned
            NotImplemented when given -1. Furthermore `val` isn't an
            iterable containing invertible items. Finally, no `default` value
            was specified.
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
