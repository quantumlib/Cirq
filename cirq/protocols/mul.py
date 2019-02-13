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

from typing import Any

# This is a special indicator value used to determine whether or not the caller
# provided a 'default' argument.
RaiseTypeErrorIfNotProvided = ([],)  # type: Any


def mul(lhs: Any, rhs: Any, default: Any = RaiseTypeErrorIfNotProvided) -> Any:
    """Returns lhs * rhs, or else a default if the operator is not implemented.

    This method is mostly used by __pow__ methods trying to return
    NotImplemented instead of causing a TypeError.

    Args:
        lhs: Left hand side of the multiplication.
        rhs: Right hand side of the multiplication.
        default: Default value to return if the multiplication is not defined.
            If not default is specified, a type error is raised when the
            multiplication fails.

    Returns:
        The product of the two inputs, or else the default value if the product
        is not defined, or else raises a TypeError if no default is defined.

    Raises:
        TypeError:
            lhs doesn't have __mul__ or it returned NotImplemented
            AND lhs doesn't have __rmul__ or it returned NotImplemented
            AND a default value isn't specified.
    """
    # Use left-hand-side's __mul__.
    left_mul = getattr(lhs, '__mul__', None)
    result = NotImplemented if left_mul is None else left_mul(rhs)

    # Fallback to right-hand-side's __rmul__.
    if result is NotImplemented:
        right_mul = getattr(rhs, '__rmul__', None)
        result = NotImplemented if right_mul is None else right_mul(lhs)

    # Output.
    if result is not NotImplemented:
        return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(
        type(lhs), type(rhs)))
# pylint: enable=function-redefined, redefined-builtin
