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

import numpy as np

from typing_extensions import Protocol


class SupportsApproximateEquality(Protocol):
    """Object which can be compared approximately."""

    def _approx_eq_(
            self,
            other: Any,
            *,
            atol: Union[int, float]
        ) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for approximate
        comparison with other types.

        Args:
            other: Target object for approximate comparison.
            atol: The minimum absolute tolerance. See np.isclose() documentation
                  for details.

        Returns:
            True if objects are approximately equal, False otherwise. Returns
            NotImplemented when approximate equality is not implemented for
            given types.
        """


def approx_eq(val: Any, other: Any, *, atol: Union[int, float] = 1e-8) -> bool:
    """Approximately compares two objects.

    If `val` implements SupportsApproxEquality protocol then it is invoked and
    takes precedence over all other checks:
     - For primitive numeric types `int` and `float` approximate equality is
       delegated to math.isclose().
     - For complex primitive type the real and imaginary parts are treated
       independently and compared using math.isclose().
     - For `val` and `other` both iterable of the same length, consecutive
       elements are compared recursively. Types of `val` and `other` does not
       necessarily needs to match each other. They just need to be iterable and
       have the same structure.

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        True if objects are approximately equal, False otherwise.
    """

    # Check if val defines approximate equality via _approx_eq_. This takes
    # precedence over all other overloads.
    approx_eq_getter = getattr(val, '_approx_eq_', None)
    if approx_eq_getter is not None:
        result = approx_eq_getter(other, atol)
        if result is not NotImplemented:
            return result

    # The same for other to make approx_eq symmetric.
    other_approx_eq_getter = getattr(other, '_approx_eq_', None)
    if other_approx_eq_getter is not None:
        result = other_approx_eq_getter(val, atol)
        if result is not NotImplemented:
            return result

    # Compare primitive types directly.
    if isinstance(val, (int, float)):
        if not isinstance(other, (int, float)):
            return False
        return _isclose(val, other, atol=atol)

    if isinstance(val, complex):
        if not isinstance(other, complex):
            return False
        return _isclose(val, other, atol=atol)

    # Try to compare source and target recursively, assuming they're iterable.
    result = _approx_eq_iterables(val, other, atol=atol)

    # Fallback to __eq__() when anything else fails.
    if result is NotImplemented:
        return val == other
    return result


def _approx_eq_iterables(val: Any, other: Any, *,
                         atol: Union[int, float]) -> bool:
    """Iterates over arguments and calls approx_eq recursively.

    Types of `val` and `other` does not necessarily needs to match each other.
    They just need to be iterable of the same length and have the same
    structure, approx_eq() will be called on each consecutive element of `val`
    and `other`.

    Args:
        val: Source for approximate comparison.
        other: Target for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details.

    Returns:
        True if objects are approximately equal, False otherwise. Returns
        NotImplemented when approximate equality is not implemented for given
        types.
    """

    def get_iter(iterable):
        try:
            return iter(iterable)
        except TypeError:
            return None

    val_it = get_iter(val)
    other_it = get_iter(other)

    if val_it is not None and other_it is not None:
        while True:
            try:
                val_next = next(val_it)
            except StopIteration:
                try:
                    next(other_it)
                    return False
                except StopIteration:
                    return True

            try:
                other_next = next(other_it)
            except StopIteration:
                return False

            result = approx_eq(val_next, other_next, atol=atol)
            if result is not True:
                return result

    return NotImplemented


def _isclose(a: Any, b: Any, *, atol: Union[int, float]) -> bool:
    """Convenience wrapper around np.isclose."""
    return True if np.isclose([a], [b], atol=atol, rtol=0.0)[0] else False
