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

from typing import Any, List, overload, Tuple, TYPE_CHECKING, TypeVar, Union, Iterable

from cirq import ops

if TYPE_CHECKING:
    import cirq

# This is a special indicator value used by the inverse method to determine
# whether or not the caller provided a 'default' argument.
RaiseTypeErrorIfNotProvided: Tuple[List[Any]] = ([],)

TDefault = TypeVar('TDefault')


# pylint: disable=function-redefined
@overload
def inverse(val: 'cirq.Gate') -> 'cirq.Gate':
    pass


@overload
def inverse(val: 'cirq.Operation') -> 'cirq.Operation':
    pass


@overload
def inverse(val: 'cirq.Circuit') -> 'cirq.Circuit':
    pass


@overload
def inverse(val: 'cirq.OP_TREE') -> 'cirq.OP_TREE':
    pass


@overload
def inverse(val: 'cirq.Gate', default: TDefault) -> Union[TDefault, 'cirq.Gate']:
    pass


@overload
def inverse(val: 'cirq.Operation', default: TDefault) -> Union[TDefault, 'cirq.Operation']:
    pass


@overload
def inverse(val: 'cirq.Circuit', default: TDefault) -> Union[TDefault, 'cirq.Circuit']:
    pass


@overload
def inverse(val: 'cirq.OP_TREE', default: TDefault) -> Union[TDefault, 'cirq.OP_TREE']:
    pass


def inverse(val: Any, default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns the inverse `val**-1` of the given value, if defined.

    An object can define an inverse by defining a __pow__(self, exponent) method
    that returns something besides NotImplemented when given the exponent -1.
    The inverse of iterables is by default defined to be the iterable's items,
    each inverted, in reverse order.

    Args:
        val: The value (or iterable of invertible values) to invert.
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

    # pylint: disable=not-callable
    result = NotImplemented if raiser is None else raiser(-1)
    if result is not NotImplemented:
        return result

    # Maybe it's an iterable of invertible items?
    # Note: we avoid str because 'a'[0] == 'a', which creates an infinite loop.
    if isinstance(val, Iterable) and not isinstance(val, (str, ops.Operation)):
        unique_indicator: List[Any] = []
        results = tuple(inverse(e, unique_indicator) for e in val)
        if all(e is not unique_indicator for e in results):
            return results[::-1]

    # Can't invert.
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        f"object of type '{type(val)}' isn't invertible. "
        "It has no __pow__ method (or the method returned NotImplemented) "
        "and it isn't an iterable of invertible objects."
    )


# pylint: enable=function-redefined
