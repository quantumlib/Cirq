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

import math

from typing import Any

from typing_extensions import Protocol


class SupportsApproxEquality(Protocol):
    """Object which can be approximately compared."""

    def _approx_eq_(self, other: Any, rel_tol: float, abs_tol: float) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for approximate
        comparison with other types.

        Args:
        other: Target object for approximate comparison.
        rel_tol: The relative tolerance. See math.isclose() documentation for
            details.
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
     - For `val` and `other` both iterable, consecutive elements are compared
       recursively. Types of `val` and `other` does not necessarily needs to
       match each other. They just need to be iterable and have the same
       structure.

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        rel_tol: The relative tolerance. See math.isclose() documentation for
            details.
        abs_tol: The minimum absolute tolerance. See math.isclose()
            documentation for details.

    Returns:
        True if objects are approximately equal, False otherwise. Returns
        NotImplemented when approximate equality is not implemented for given
        types.
    """

    # Check if object defines approximate equality via _approx_eq_. This takes
    # precedence over all other overloads.
    approx_eq_getter = getattr(val, '_approx_eq_', None)
    if approx_eq_getter is not None:
        return approx_eq_getter(other, rel_tol, abs_tol)

    val_t = type(val)
    other_t = type(other)

    # Fallback to math.isclose() for primitive types, which is symmetric
    # contrary to numpy.isclose().
    if val_t in [int, float]:
        if other_t not in [int, float]:
            return NotImplemented
        return math.isclose(val, other, rel_tol=rel_tol, abs_tol=abs_tol)

    # For complex types, treat real and imaginary parts independently.
    if val_t == complex:
        if val_t != other_t:
            return NotImplemented
        return math.isclose(
            val.real,
            other.real,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        ) and math.isclose(
            val.imag,
            other.imag,
            rel_tol=rel_tol,
            abs_tol=abs_tol
        )

    # Try to compare source and target recursively, assuming they're iterable.
    return _approx_eq_iterables(val, other, rel_tol=rel_tol, abs_tol=abs_tol)


def _approx_eq_iterables(
        val: Any,
        other: Any,
        rel_tol: float,
        abs_tol: float) -> bool:
    """Iterates over arguments and calls approx_eq recursively.

    Types of `val` and `other` does not necessarily needs to match each other.
    They just need to be iterable and have the same structure, approx_eq() will
    be called on each consecutive element of `val` and `other`.

    Args:
        val: Source for approximate comparison
        other: Target for approximate comparison
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
