# Copyright 2018 Google LLC
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

"""Ways to wrap values so that they match desired interfaces."""

import inspect

from typing import Any  # pylint: disable=unused-import
from typing import Callable, Dict, Optional, Type, TypeVar

from cirq.extension.potential_implementation import PotentialImplementation

T_ACTUAL = TypeVar('T_ACTUAL')
T_DESIRED = TypeVar('T_DESIRED')


class Extensions:
    """Specifies ways to wrap values so that they match desired interfaces."""

    def __init__(
            self,
            desired_to_actual_to_wrapper: Optional[Dict[
                Type[T_DESIRED],
                Dict[Type[T_ACTUAL],
                     Callable[[T_ACTUAL],
                              Optional[T_DESIRED]]]]]=None
            ) -> None:
        """Specifies extensions.

        Args:
            desired_to_actual_to_wrapper: A dictionary of dictionaries. The
                top-level dictionary is keyed by desired type. The second-level
                dictionaries map from actual type to wrapper methods. For
                example, the arg value {Printable: {str: wrap_string}}
                indicates that to get a Printable from a string you use the
                result of passing the string into wrap_string.
        """
        self._desired_to_actual_to_wrapper = (
            {}
            if desired_to_actual_to_wrapper is None
            else desired_to_actual_to_wrapper
        )  # type: Dict[Type[Any], Dict[Any, Callable[[Any], Optional[Any]]]]

    def add_cast(self,
                 desired_type: Type[T_DESIRED],
                 actual_type: Type[T_ACTUAL],
                 conversion: Callable[[T_ACTUAL], Optional[T_DESIRED]],
                 also_add_inherited_conversions: bool = True,
                 overwrite_existing: bool = False) -> None:
        """Adds a way to turn one type of thing into another.

        Args:
            desired_type: The type that the casting caller wants.
            actual_type: The type of the value that the casting caller has.
            conversion: A function that takes the value the casting  caller
                has, and returns a value that is an instance of the type the
                casting caller wants (or else acts like an instance of that
                type; it may not literally be an instance).
            also_add_inherited_conversions: Whether or not to also use the
                given conversion method to convert from the given actual type
                to desired types that the given desired type derives from
                (unless instances of the actual type are already instances of
                the alternate desired types).
            overwrite_existing: Normally, this method will fail if a redundant
                conversion is specified, either directly or via an inheritance
                relation. If this argument is set to True, the existing
                conversions are overwritten instead.
        """
        all_desired_types = [desired_type]
        if also_add_inherited_conversions:
            all_desired_types.extend(
                other_desired_type
                for other_desired_type in inspect.getmro(desired_type)[1:]
                if not issubclass(actual_type, other_desired_type))

        if not overwrite_existing:
            for t in all_desired_types:
                if self._have(t, actual_type):
                    raise ValueError(
                        'Already have a way to cast {} into {}.'.format(
                            actual_type, t))

        for t in all_desired_types:
            if t not in self._desired_to_actual_to_wrapper:
                self._desired_to_actual_to_wrapper[t] = {}
            self._desired_to_actual_to_wrapper[t][actual_type] = conversion

    def _have(self,
              desired_type: Type[T_DESIRED],
              actual_type: Type[T_ACTUAL]) -> bool:
        return (
            desired_type in self._desired_to_actual_to_wrapper and
            actual_type in self._desired_to_actual_to_wrapper[desired_type]
        )

    def can_cast(self,
                 actual_value: T_ACTUAL,
                 desired_type: Type[T_DESIRED]) -> bool:
        """Is it possible to turn the given value into the desired type?

        Args:
            actual_value: The value that the caller has.
            desired_type: The type that the caller wants.

        Returns:
            True if the cast will work, False otherwise.
        """
        return self.try_cast(actual_value, desired_type) is not None

    def try_cast(self,
                 actual_value: T_ACTUAL,
                 desired_type: Type[T_DESIRED]) -> Optional[T_DESIRED]:
        """Represents the given value as the desired type, if possible.

        Returns None if no wrapper method is found, and the value isn't already
        an instance of the desired type. Wrapper methods, which are keyed by
        type, are searched for in the same type order as the standard method
        resolution order.

        Args:
            actual_value: The value to be represented as the desired type.
            desired_type: The type of value that the caller wants.

        Returns:
            A value of the desired type, or else None.
        """
        actual_to_wrapper = self._desired_to_actual_to_wrapper.get(
            desired_type)
        if actual_to_wrapper:
            for actual_type in inspect.getmro(type(actual_value)):
                wrapper = actual_to_wrapper.get(actual_type)
                if wrapper:
                    wrapped = wrapper(actual_value)
                    if wrapped is not None:
                        return wrapped

        if isinstance(actual_value, desired_type):
            return actual_value

        if isinstance(actual_value, PotentialImplementation):
            return actual_value.try_cast_to(desired_type)

        return None

    def cast(self,
             actual_value: T_ACTUAL,
             desired_type: Type[T_DESIRED]) -> T_DESIRED:
        """Represents the given value as the desired type, if possible.

        Fails if no wrapper method is found, and the value isn't already an
        instance of the desired type. Wrapper methods, which are keyed by type,
        are searched for in the same type order as the standard method
        resolution order.

        Args:
            actual_value: The value to be represented as the desired type.
            desired_type: The type of value that the caller wants.

        Returns:
            A value of the desired type, if possible.

        Raises:
            TypeError: The value is not of the correct type, and there was no
                extension specified for exposing its type or parent types as
                the desired type.
        """
        result = self.try_cast(actual_value, desired_type)
        if result is None:
            raise TypeError('Expected a {} but got {}'.format(
                desired_type,
                repr(actual_value)))
        return result
