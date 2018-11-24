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

import sys

from typing import Any
from typing_extensions import Protocol


class SupportsApproximateEquality(Protocol):
    """Object which can be compared approximately."""

    def _approx_eq_(
            self,
            other: Any,
            *,
            rel_tol: float,
            abs_tol: float
        ) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for approximate
        comparison with other types.

        Args:
            other: Target object for approximate comparison.
            rel_tol: The relative tolerance. See math.isclose() documentation
                for details.
            abs_tol: The minimum absolute tolerance. See math.isclose()
                documentation for details.

        Returns:
            True if objects are approximately equal, False otherwise. Returns
            NotImplemented when approximate equality is not implemented for
            given types.
        """
        pass


def approx_eq(
        val: Any,
        other: Any,
        rel_tol: float = 1e-09,
        abs_tol: float = 0.0) -> bool:
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
        rel_tol: The relative tolerance. See math.isclose() documentation for
            details.
        abs_tol: The minimum absolute tolerance. See math.isclose()
            documentation for details.

    Returns:
        True if objects are approximately equal, False otherwise.
    """

    # Check if val defines approximate equality via _approx_eq_. This takes
    # precedence over all other overloads.
    approx_eq_getter = getattr(val, '_approx_eq_', None)
    if approx_eq_getter is not None:
        result = approx_eq_getter(other, rel_tol, abs_tol)
        if result is not NotImplemented:
            return result

    # The same for other to make approx_eq symmetric.
    other_approx_eq_getter = getattr(other, '_approx_eq_', None)
    if other_approx_eq_getter is not None:
        result = other_approx_eq_getter(val, rel_tol, abs_tol)
        if result is not NotImplemented:
            return result

    # Compare primitive types directly.
    if isinstance(val, (int, float)):
        if not isinstance(other, (int, float)):
            return False
        return _isclose(val, other, rel_tol=rel_tol, abs_tol=abs_tol)

    # For complex types, treat real and imaginary parts independently.
    if isinstance(val, complex):
        if not isinstance(other, complex):
            return False
        return _isclose(
            val.real,
            other.real,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        ) and _isclose(
            val.imag,
            other.imag,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )

    # Try to compare source and target recursively, assuming they're iterable.
    result = _approx_eq_iterables(val, other, rel_tol=rel_tol, abs_tol=abs_tol)

    # Fallback to __eq__() when anything else fails.
    if result is NotImplemented:
        return val == other
    return result


def _approx_eq_iterables(
        val: Any,
        other: Any,
        *,
        rel_tol: float,
        abs_tol: float) -> bool:
    """Iterates over arguments and calls approx_eq recursively.

    Types of `val` and `other` does not necessarily needs to match each other.
    They just need to be iterable of the same length and have the same
    structure, approx_eq() will be called on each consecutive element of `val`
    and `other`.

    Args:
        val: Source for approximate comparison.
        other: Target for approximate comparison.
        rel_tol: The relative tolerance. See math.isclose() documentation for
            details.
        abs_tol: The minimum absolute tolerance. See math.isclose()
            documentation for details.

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

            result = approx_eq(
                val_next,
                other_next,
                rel_tol=rel_tol,
                abs_tol=abs_tol)
            if result is not True:
                return result

    return NotImplemented


# Definition of _isclose().
#
# For Python >= 3.5 delegates to math.isclose(), which is symmetric. For older
# version delegates to np.isclose() in such a way that this method is symmetric
# in a and b.
#
# The Python >= 3.5 is more restrictive since it's based on formula
#   abs(a-b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
# and Python < 3.5 is based on formula
#   abs(a-b) <= abs_tol + rel_tol * max(abs(a), abs(b))),
# provided that abs_tol >= 0 and rel_tol >= 0.
if (sys.version_info.major, sys.version_info.minor) >= (3, 5):
    import math

    def _isclose(a: Any, b: Any, *, rel_tol: float, abs_tol: float) -> bool:
        """Approximate comparison for primitive numerical values."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
else:
    import numpy as np

    def _isclose(a: Any, b: Any, *, rel_tol: float, abs_tol: float) -> bool:
        """Approximate comparison for primitive numerical values."""
        if a > b:
            result = np.isclose([b], [a], rtol=rel_tol, atol=abs_tol)
        else:
            result = np.isclose([a], [b], rtol=rel_tol, atol=abs_tol)
        # Make sure to return compatible bool type.
        return True if result[0] else False
