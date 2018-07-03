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

"""Casting methods that are aware of PotentialImplementation."""

from typing import Type, TypeVar, Optional

from cirq.extension import extensions

T_ACTUAL = TypeVar('T_ACTUAL')
T_DESIRED = TypeVar('T_DESIRED')


def cast(desired_type: Type[T_DESIRED],
         actual_value: T_ACTUAL) -> T_DESIRED:
    """Equivalent to the default cirq.Extensions()'s cast method.

    Fails if the value isn't already an instance of the desired type, and
    either doesn't implement `PotentialImplementation` or returns None when
    asked to cast itself to the desired type.

    Beware that replacing `typing.cast` with this method sometimes work, but is
    not always correct. It will still convince mypy that the output is of the
    desired type, but this method is not a no-op at runtime. It actually
    checks and throws on failure. More seriously, this method can't handle
    generic type parameters.

    Args:
        desired_type: The type of value that the caller wants.
        actual_value: The value to be represented as the desired type.

    Returns:
        A value of the desired type, if possible.

    Raises:
        TypeError: The value could not be exposed as the desired type.
    """
    return extensions.Extensions().cast(desired_type=desired_type,
                                        actual_value=actual_value)


def try_cast(desired_type: Type[T_DESIRED],
             actual_value: T_ACTUAL) -> Optional[T_DESIRED]:
    """Equivalent to the default cirq.Extensions()'s try_cast method.

    Returns None if the value isn't already an instance of the desired type, and
    either doesn't implement `PotentialImplementation` or returns None when
    asked to cast itself to the desired type. Otherwise returns a wrapped value
    of the desired type.

    Args:
        desired_type: The type of value that the caller wants.
        actual_value: The value to be represented as the desired type.

    Returns:
        A value of the desired type, or else None.
    """
    return extensions.Extensions().try_cast(desired_type=desired_type,
                                            actual_value=actual_value)


def can_cast(desired_type: Type[T_DESIRED],
             actual_value: T_ACTUAL) -> bool:
    """Equivalent to the default cirq.Extensions()'s can_cast method.

    Args:
        actual_value: The value that the caller has.
        desired_type: The type that the caller wants.

    Returns:
        True if the cast will work, False otherwise.
    """
    return extensions.Extensions().can_cast(desired_type=desired_type,
                                            actual_value=actual_value)
