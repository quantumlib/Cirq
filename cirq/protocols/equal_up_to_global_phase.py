# Copyright 2019 The Cirq Developers
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

import cirq


from typing import Any, Union

import numpy as np

from typing_extensions import Protocol


class SupportsEqualUpToGlobalPhase(Protocol):
    """Object which can be compared for equality mod global phase."""

    def _equal_up_to_global_phase_(
            self,
            other: Any,
            *,
            atol: Union[int, float]
        ) -> bool:
        """Approximate comparator.

        Types implementing this protocol define their own logic for comparison
        with other types.

        Args:
            other: Target object for comparison of equality up to global phase.
            atol: The minimum absolute tolerance. See np.isclose() documentation
                  for details.

        Returns:
            True if objects are equal up to a global phase, False otherwise.
            Returns NotImplemented when checking equality up to a global phase
            is not implemented for given types.
        """


def equal_up_to_global_phase(
    val: Any,
    other: Any,
    *,
    atol: Union[int, float] = 1e-8
) -> bool:
    """Determine whether two objects are equal up to global phase.

    If `val` implements SupportsEqualUpToGlobalPhase protocol then it is
    invoked and takes precedence over all other checks:
     - For complex primitive type the magnitudes of the values are compared.
     - For `val` and `other` both iterable of the same length, consecutive
       elements are compared recursively. Types of `val` and `other` does not
       necessarily needs to match each other. They just need to be iterable and
       have the same structure.
     - For all other types, fall back on _approx_eq_

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        True if objects are approximately equal, False otherwise.
    """

    # fall back to _equal_up_to_global_phase_ for val.
    eq_up_to_phase_getter = getattr(val, '_equal_up_to_global_phase_', None)
    if eq_up_to_phase_getter is not None:
        result = eq_up_to_phase_getter(other, atol)
        if result is not NotImplemented:
            return result

    # fall back to _equal_up_to_global_phase_ for other.
    other_eq_up_to_phase_getter = getattr(
        other, '_equal_up_to_global_phase_', None
    )
    if other_eq_up_to_phase_getter is not None:
        result = other_eq_up_to_phase_getter(val, atol)
        if result is not NotImplemented:
            return result

    # Try to compare the magnitude of two numbers.
    if isinstance(val, complex):
        if not isinstance(other, (complex, float, int)):
            return False
        return cirq.approx_eq(np.abs(val), np.abs(other))

    # Try to compare source and target recursively, assuming they're iterable.
    result = _eq_up_to_phase_iterables(val, other, atol=atol)

    # Fallback to cir approx_eq for remaining types
    if result is NotImplemented:
        return cirq.approx_eq(val, other)
    return result


def _eq_up_to_phase_iterables(val: Any, other: Any, *,
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


# def _isclose(a: Any, b: Any, *, atol: Union[int, float]) -> bool:
#     """Convenience wrapper around np.isclose."""
#     return True if np.isclose([a], [b], atol=atol, rtol=0.0)[0] else False

def _phase_difference(a: complex, b: complex):
    """Compute the angle between two complex numbers in the Argand plane."""
    phi1 = np.arctan2(a.imag, a.real)
    phi2 = np.arctan2(b.imag, b.real)
    return cirq.PeriodicValue(phi1 - phi2, 2.0 * np.pi)
