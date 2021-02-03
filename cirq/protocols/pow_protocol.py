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

from typing import Any, overload, TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    import cirq

# This is a special indicator value used by the pow method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided: Any = ([],)

TDefault = TypeVar('TDefault')


# pylint: disable=function-redefined, redefined-builtin
@overload
def pow(val: 'cirq.Gate', exponent: Any) -> 'cirq.Gate':
    pass


@overload
def pow(val: 'cirq.Operation', exponent: Any) -> 'cirq.Operation':
    pass


@overload
def pow(val: 'cirq.Gate', exponent: Any, default: TDefault) -> Union[TDefault, 'cirq.Gate']:
    pass


@overload
def pow(
    val: 'cirq.Operation', exponent: Any, default: TDefault
) -> Union[TDefault, 'cirq.Operation']:
    pass


@overload
def pow(val: 'cirq.Circuit', exponent: int, default: TDefault) -> Union[TDefault, 'cirq.Circuit']:
    pass


@overload
def pow(val: Any, exponent: Any, default: TDefault) -> Any:
    pass


def pow(val: Any, exponent: Any, default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns `val**factor` of the given value, if defined.

    Values define an extrapolation by defining a __pow__(self, exponent) method.
    Note that the method may return NotImplemented to indicate a particular
    extrapolation can't be done.

    Args:
        val: The value or iterable of values to invert.
        exponent: The extrapolation factor. For example, if this is 0.5 and val
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
    result = NotImplemented if raiser is None else raiser(exponent)
    if result is not NotImplemented:
        return result

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if raiser is None:
        raise TypeError("object of type '{}' has no __pow__ method.".format(type(val)))
    raise TypeError(
        "object of type '{}' does have a __pow__ method, "
        "but it returned NotImplemented.".format(type(val))
    )


# pylint: enable=function-redefined, redefined-builtin
