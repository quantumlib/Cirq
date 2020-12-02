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

from typing import Any, Union, Iterable
from fractions import Fraction
from decimal import Decimal

import numbers
import numpy as np
import sympy

from typing_extensions import Protocol

from cirq._doc import doc_private


class SupportsApproximateEquality(Protocol):
    """Object which can be compared approximately."""

    @doc_private
    def _approx_eq_(self, other: Any, *, atol: Union[int, float]) -> bool:
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
    if isinstance(val, numbers.Number):
        if not isinstance(other, numbers.Number):
            return False
        result = _isclose(val, other, atol=atol)
        if result is not NotImplemented:
            return result

    if isinstance(val, str):
        return val == other

    if isinstance(val, sympy.Basic) or isinstance(other, sympy.Basic):
        delta = sympy.Abs(other - val).simplify()
        if not delta.is_number:
            raise AttributeError(
                'Insufficient information to decide whether '
                'expressions are approximately equal '
                f'[{val}] vs [{other}]'
            )
        return sympy.LessThan(delta, atol) == sympy.true

    # If the values are iterable, try comparing recursively on items.
    if isinstance(val, Iterable) and isinstance(other, Iterable):
        return _approx_eq_iterables(val, other, atol=atol)

    # Last resort: exact equality.
    return val == other


def _approx_eq_iterables(val: Iterable, other: Iterable, *, atol: Union[int, float]) -> bool:
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

    iter1 = iter(val)
    iter2 = iter(other)
    done = object()
    cur_item1 = None

    while cur_item1 is not done:
        try:
            cur_item1 = next(iter1)
        except StopIteration:
            cur_item1 = done
        try:
            cur_item2 = next(iter2)
        except StopIteration:
            cur_item2 = done

        if not approx_eq(cur_item1, cur_item2, atol=atol):
            return False

    return True


def _isclose(a: Any, b: Any, *, atol: Union[int, float]) -> bool:
    """Convenience wrapper around np.isclose."""

    # support casting some standard numeric types
    x1 = np.asarray([a])
    if isinstance(a, (Fraction, Decimal)):
        x1 = x1.astype(np.float64)
    x2 = np.asarray([b])
    if isinstance(b, (Fraction, Decimal)):
        x2 = x2.astype(np.float64)

    # workaround np.isfinite type limitations. Cast to bool to avoid np.bool_
    try:
        result = bool(np.isclose(x1, x2, atol=atol, rtol=0.0)[0])
    except TypeError:
        return NotImplemented

    return result
