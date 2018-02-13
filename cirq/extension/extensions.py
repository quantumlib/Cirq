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

from typing import Dict, Type, Callable, TypeVar, Union, Optional

from cirq.extension.potential_implementation import PotentialImplementation

T_ACTUAL = TypeVar('TActual')
T_DESIRED = TypeVar('TDesired')


class Extensions:
    """Specifies ways to wrap values so that they match desired interfaces."""

    def __init__(
            self,
            desired_to_actual_to_wrapper: Optional[Dict[
                Type[T_DESIRED],
                Dict[Type[T_ACTUAL], Callable[[T_ACTUAL], T_DESIRED]]]]=None):
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
            else desired_to_actual_to_wrapper)

    def try_cast(self,
                 actual_value: T_ACTUAL,
                 desired_type: Type[T_DESIRED]
                 ) -> Union[type(None), T_DESIRED]:
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
                    return wrapper(actual_value)

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
