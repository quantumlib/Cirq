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

from collections import Iterable
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
     - For all other types, fall back to _approx_eq_

    Args:
        val: Source object for approximate comparison.
        other: Target object for approximate comparison.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        True if objects are approximately equal up to phase, False otherwise.
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
        return cirq.approx_eq(np.abs(val), np.abs(other), atol=atol)

    # Try to compare source and target recursively, assuming they're iterable.
    result = _eq_up_to_phase_iterables(val, other, atol=atol)

    # Fallback to cir approx_eq for remaining types
    if result is NotImplemented:
        return cirq.approx_eq(val, other, atol=atol)
    return result


def _eq_up_to_phase_iterables(val: Any, other: Any, *,
                         atol: Union[int, float]) -> bool:
    """Iterates over arguments and checks for equality up to global phase.

    For iterables of the same length and comparable structure, check that the
    difference between phases of corresponding elements can be described by a
    single complex value.

    Args:
        val: Source for comparison.
        other: Target for comparison.
        atol: The minimum absolute tolerance.

    Returns:
        True if objects are approximately equal up to phase, False otherwise.
        Returns NotImplemented when approximate equality is not implemented
        for given types.
    """

    if not (isinstance(val, Iterable) and isinstance(other, Iterable)):
        return NotImplemented

    val_it = iter(val)
    other_it = iter(other)
    global_phase = None

    while True:
        try:
            val_next = next(val_it)
        # only allow phase comparison for equal-length containers
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
        # check that a single value describes all phase differences between
        # corresponding values in the containers
        if global_phase is None:
            global_phase = _phase_difference(val_next, other_next)
        current_phase = _phase_difference(val_next, other_next)
        if not cirq.approx_eq(current_phase, global_phase, atol=atol):
            return False

    return NotImplemented


def _phase_difference(a: complex, b: complex):
    """Compute the angle between two complex numbers in the Argand plane."""
    phi1 = np.arctan2(a.imag, a.real)
    phi2 = np.arctan2(b.imag, b.real)
    return cirq.PeriodicValue(phi1 - phi2, 2.0 * np.pi)
