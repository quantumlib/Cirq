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

from typing import Any, Optional

import numpy as np
from typing_extensions import Protocol

from cirq import abc


class SupportsUnitaryEffect(Protocol):
    """An effect that can be described by a unitary matrix."""

    @abc.abstractmethod
    def _maybe_unitary_effect_(self) -> Optional[np.ndarray]:
        """The unitary matrix of the effect, or None if not known.

        The matrix order is implicit for both gates and operations. For a gate,
        the matrix must be in order with respect to the list of qubits that the
        gate is applied to. For an operation, the order must be with respect to
        its qubits attribute. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        For example, when applying a CNOT gate the control qubit goes first and
        so the CNOT gate's matrix is:

            1 _ _ _
            _ 1 _ _
            _ _ _ 1
            _ _ 1 _
        """

    def _has_unitary_effect_(self) -> bool:
        """Determines if the effect has a known unitary matrix or not."""
        return self._maybe_unitary_effect_() is not None

    def _unitary_effect_(self) -> np.ndarray:
        """The unitary matrix of the effect, or else an exception.

        Returns:
            The unitary matrix.

        Raises:
            ValueError:
                The receiving value doesn't have a unitary effect.
        """
        result = self._maybe_unitary_effect_()
        if result is None:
            raise ValueError('No unitary effect.')
        return result


def maybe_unitary_effect(val: Any) -> Optional[np.ndarray]:
    """Determines the matrix of the given object, if any.

    Returns:
        A 2d numpy array if the given object follows the SupportsUnitaryEffect
        protocol and returns a non-None unitary effect. Otherwise None.
    """
    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        return maybe()

    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        try:
            return get()
        except ValueError:
            return None

    has = getattr(val, '_has_unitary_effect_', None)
    if has is not None:
        if has():
            raise AttributeError(
                "'{}' object has attribute '_has_unitary_effect_' and it "
                "returned True, but object has no attribute '_unitary_effect_' "
                "or '_maybe_unitary_effect_'".format(type(val)))
        return None

    # Not following SupportsUnitaryEffect means don't have a unitary effect.
    return None


def unitary_effect(val: Any) -> np.ndarray:
    """Gets the unitary matrix of the given object.

    Returns:
        A 2d numpy array if the given object follows the SupportsUnitaryEffect
        protocol and returns a non-None unitary effect. Otherwise None.

    Raises:
        ValueError:
            The given object doesn't follow the SupportsUnitaryEffect protocol,
            or it does but doesn't have a known unitary effect.
    """
    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        return get()

    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        result = maybe()
        if result is None:
            raise ValueError(
                "Expected a value with a non-None _unitary_effect_, but "
                "{!r}._maybe_unitary_effect_() returned None".format(type(val)))
        return result

    has = getattr(val, '_has_unitary_effect_', None)
    if has is not None:
        if has():
            raise AttributeError(
                "'{}' object has attribute '_has_unitary_effect_' and it "
                "returned True, but object has no attribute '_unitary_effect_' "
                "or '_maybe_unitary_effect_'".format(type(val)))
        else:
            raise ValueError(
                "Expected a value with a non-None _unitary_effect_, but "
                "{!r}._has_unitary_effect_() returned False".format(type(val)))

    raise ValueError(
        "'{}' object doesn't follow the SupportsUnitaryEffect protocol, and so "
        "does not have a unitary effect.".format(type(val)))


def has_unitary_effect(val: Any) -> bool:
    """Determines the matrix of the given object, if any.

    Returns:
        Whether or not the given object follows the SupportsUnitaryEffect and
        also has a unitary effect.
    """
    has = getattr(val, '_has_unitary_effect_', None)
    if has is not None:
        return has()

    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        return maybe() is not None

    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        try:
            _ = get()
            return True
        except ValueError:
            return False

    # Not following SupportsUnitaryEffect means don't have a unitary effect.
    return False
