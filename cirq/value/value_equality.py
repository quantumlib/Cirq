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

"""Defines `@cirq.value_equality`, for easy __eq__/__hash__ methods."""

from typing import Union, Callable, overload, Any

from typing_extensions import Protocol


class _SupportsValueEquality(Protocol):
    """An object decorated with the value equality decorator."""

    def _value_equality_values_(self) -> Any:
        """Returns a value or values that define the identity of this object.

        For example, a Point2D would be defined by the tuple (x, y) and so it
        would return `(x, y)` from this method.

        The decorated class is responsible for implementing this method.

        Returns:
            Values used when determining if the receiving object is equal to
            another object.
        """
        pass

    def _value_equality_values_cls_(self) -> Any:
        """Automatically implemented by the `cirq.value_equality` decorator.

        This method encodes the logic used to determine whether or not objects
        that have the same equivalence values but different types are considered
        to be equal. By default, this returns the decorated type. But there is
        an option (`distinct_child_types`) to make it return `type(self)`
        instead.

        Returns:
            Type used when determining if the receiving object is equal to
            another object.
        """
        pass


def _value_equality_eq(self: _SupportsValueEquality,
                       other: _SupportsValueEquality) -> bool:
    cls_self = self._value_equality_values_cls_()
    if not isinstance(other, cls_self):
        return NotImplemented
    cls_other = other._value_equality_values_cls_()
    if cls_self != cls_other:
        return False
    return self._value_equality_values_() == other._value_equality_values_()


def _value_equality_ne(self: _SupportsValueEquality,
                       other: _SupportsValueEquality) -> bool:
    return not self == other


def _value_equality_hash(self: _SupportsValueEquality) -> int:
    return hash((self._value_equality_values_cls_(),
                 self._value_equality_values_()))


# pylint: disable=function-redefined
@overload
def value_equality(cls: type,
                   *,
                   unhashable: bool = False,
                   distinct_child_types: bool = False
                   ) -> type:
    pass


@overload
def value_equality(*,
                   unhashable: bool = False,
                   distinct_child_types: bool = False
                   ) -> Callable[[type], type]:
    pass


def value_equality(cls: type = None,
                   *,
                   unhashable: bool = False,
                   distinct_child_types: bool = False
                   ) -> Union[Callable[[type], type], type]:
    """Implements __eq__/__ne__/__hash__ via a _value_equality_values_ method.

    _value_equality_values_ is a method that the decorated class must implement.

    Note that the type of the decorated value is included as part of the value
    equality values. This is so that completely separate classes with identical
    equality values (e.g. a Point2D and a Vector2D) don't compare as equal.
    Further note that this means that child types of the decorated type will be
    considered equal to each other, though this behavior can be changed via
    the 'distinct_child_types` argument. The type logic is implemented behind
    the scenes by a `_value_equality_values_cls_` method added to the class.

    Args:
        cls: The type to decorate. Automatically passed in by python when using
            the @cirq.value_equality decorator notation on a class.
        unhashable: When set, the __hash__ method will be set to None instead of
            to a hash of the equality class and equality values. Useful for
            mutable types such as dictionaries.
        distinct_child_types: When set, classes that inherit from the decorated
            class will not be considered equal to it. Also, different child
            classes will not be considered equal to each other. Useful for when
            the decorated class is an abstract class or trait that is helping to
            define equality for many conceptually distinct concrete classes.
    """

    # If keyword arguments were specified, python invokes the decorator method
    # without a `cls` argument, then passes `cls` into the result.
    if cls is None:
        return lambda deferred_cls: value_equality(
            deferred_cls,
            unhashable=unhashable,
            distinct_child_types=distinct_child_types)

    getter = getattr(cls, '_value_equality_values_', None)
    if getter is None:
        raise TypeError('The @cirq.value_equality decorator requires a '
                        '_value_equality_values_ method to be defined.')

    if distinct_child_types:
        setattr(cls, '_value_equality_values_cls_', lambda self: type(self))
    else:
        setattr(cls, '_value_equality_values_cls_', lambda self: cls)
    setattr(cls, '__hash__', None if unhashable else _value_equality_hash)
    setattr(cls, '__eq__', _value_equality_eq)
    setattr(cls, '__ne__', _value_equality_ne)

    return cls
# pylint: enable=function-redefined
