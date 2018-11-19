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


"""Defines `@cirq.class_equality`, for `class_euqals` support."""

from typing import Any, Callable, Union, overload

from typing_extensions import Protocol


class _SupportsClassEquality(Protocol):
    """An object decorated with the class equality values decorator."""

    def _class_equality_values_(self) -> Any:
        """Automatically implemented by the `cirq.class_equality` decorator.

        This method encodes the logic used to determine whether different types
        are considered to be note equal. By default, this returns the decorated
        type. But there is an option (`distinct_child_types`) to make it return
        `type(self)` instead.

        Returns:
            Type used when determining if the receiving object is equal to
            another object.
        """
        pass


def class_equals(a: Any, b: Any) -> bool:
    """
    Compares if two classes are type equivalent.

    Both `a` and `b` must implement `_SupportsClassEquality` protocol which is
    implemented by `class_equality` or `value_equality` decorator.

    Args:
        a: First value to compare.
        b: Second value to compare.

    Returns:
        True if classes are type equivalent, False otherwise. `NotImplemented`
        is returned if either class does not support `_SupportsClassEquality`
        protocol.
    """
    a_getter = getattr(a, '_class_equality_values_', None)
    if a_getter is None:
        return NotImplemented
    cls_a = a_getter()

    b_getter = getattr(b, '_class_equality_values_', None)
    if b_getter is None:
        return NotImplemented
    cls_b = b_getter()

    return cls_a == cls_b


# pylint: disable=function-redefined
@overload
def class_equality(cls: type, *, distinct_child_types: bool = False) -> type:
    pass


@overload
def class_equality(*,
                   distinct_child_types: bool = False
                   ) -> Callable[[type], type]:
    pass


def class_equality(cls: type = None,
                   *,
                   distinct_child_types: bool = False
                   ) -> Union[Callable[[type], type], type]:
    """Implements `_class_equality_values_` method.

    Exposes correct type of the decorated class which can be used as a part of
    equality check through `class_equals` function.

    Child types of the decorated type will be considered equal to each other,
    though this behavior can be changed via the `distinct_child_types` argument.
    The type logic is implemented behind the scenes by a
    `_class_equality_values_` method added to the class.

    Args:
        cls: The type to decorate. Automatically passed in by python when using
            the @cirq.class_equality decorator notation on a class.
        distinct_child_types: When set, classes that inherit from the decorated
            class will not be considered equal to it. Also, different child
            classes will not be considered equal to each other. Useful for when
            the decorated class is an abstract class or trait that is helping to
            define equality for many conceptually distinct concrete classes.
    """

    # If keyword arguments were specified, python invokes the decorator method
    # without a `cls` argument, then passes `cls` into the result.
    if cls is None:
        return lambda deferred_cls: class_equality(
            deferred_cls,
            distinct_child_types=distinct_child_types)

    if distinct_child_types:
        setattr(cls, '_class_equality_values_', lambda self: type(self))
    else:
        setattr(cls, '_class_equality_values_', lambda self: cls)

    return cls
# pylint: enable=function-redefined