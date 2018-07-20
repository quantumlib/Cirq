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

import cirq
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

    def _has_matrix_(self) -> bool:
        """Determines if the gate/operation has a known matrix or not."""
        return self._maybe_unitary_effect_() is not None

    def _unitary_effect_(self) -> np.ndarray:
        result = self._maybe_unitary_effect_()
        assert result is not None, 'self._unitary_effect_() returned None'
        return result


def maybe_unitary_effect(val: Any) -> Optional[np.ndarray]:
    """Determines the matrix of the given object, if any.

    Returns:
        A 2d numpy array when the given object is a KnownMatrix and its
        matrix() method returns a non-None result.

        None when the given object isn't a KnownMatrix or else its matrix
        method returns None.
    """
    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        return maybe()

    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        return get()

    # TODO: delete after KnownMatrix is deleted.
    v = cirq.try_cast(cirq.KnownMatrix, val)
    if v is not None:
        return v.matrix()

    # Things that don't specify a unitary effect don't have a unitary effect.
    return None


def unitary_effect(val: Any) -> np.ndarray:
    """Determines the matrix of the given object, if any.

    Returns:
        A 2d numpy array when the given object is a KnownMatrix and its
        matrix() method returns a non-None result.

        None when the given object isn't a KnownMatrix or else its matrix
        method returns None.
    """
    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        return get()

    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        result = maybe()
        if result is None:
            raise TypeError(
                "Expected a value with a non-None _unitary_effect_, but "
                "{!r}._maybe_unitary_effect_() returned None".format(type(val)))
        return result

    # TODO: delete after KnownMatrix is deleted.
    v = cirq.try_cast(cirq.KnownMatrix, val)
    if v is not None:
        return v.matrix()

    raise AttributeError(
        "'{}' object has no attribute '_unitary_effect_' "
        "or '_maybe_unitary_effect_'".format(type(val)))


def has_unitary_effect(val: Any) -> bool:
    """Determines the matrix of the given object, if any.

    Returns:
        A 2d numpy array when the given object is a KnownMatrix and its
        matrix() method returns a non-None result.

        None when the given object isn't a KnownMatrix or else its matrix
        method returns None.
    """
    has = getattr(val, '_has_unitary_effect_', None)
    if has is not None:
        return has()

    maybe = getattr(val, '_maybe_unitary_effect_', None)
    if maybe is not None:
        return maybe() is not None

    get = getattr(val, '_unitary_effect_', None)
    if get is not None:
        assert get() is not None, 'val._unitary_effect_() returned None'
        return True

    # TODO: delete after KnownMatrix is deleted.
    v = cirq.try_cast(cirq.KnownMatrix, val)
    if v is not None:
        return v.has_matrix()

    return False
