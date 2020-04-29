# Copyright 2020 The Cirq Developers
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

from typing import (
    Any,
    TypeVar,
    Optional,
)

from typing_extensions import Protocol

from cirq._doc import document
from cirq.protocols import has_unitary_protocol, mixture_protocol
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits,)

TDefault = TypeVar('TDefault')


class SupportsCanActOnStateVector(Protocol):
    """An object that explicitly specifies whether it has a unitary effect."""

    @document
    def _can_act_on_state_vector_(self) -> Optional[bool]:
        """Determines whether or not the receiver can act on a state vector.

        This method is used preferentially by the global
        `cirq.can_act_on_state_vector` method, because this method is much
        cheaper than the fallback strategies such as checking for a unitary
        matrix.

        Returns:
            True: The receiving object (`self`) can act on a state vector.
            False: The receiving object cannot act on a state vector.
            None or NotImplemented: Inconclusive result. The caller should try
                fallback strategies such as decomposing the receiving object.
        """


def can_act_on_state_vector(val: Any, *, allow_decompose: bool = True) -> bool:
    """Determines whether the value can act on a state vector.

    Values that can act on state vectors can be passed into
    `cirq.act_on_state_vector`. Examples of such values include unitary
    operations, measurements, and mixtures. The action may be probabilistic.

    1. Try to use `val._can_act_on_state_vector_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Inconclusive.
        Case b) Method returns `True`.
            Can act on state vector.
        Case c) Method returns `False`.
            Cannot act on state vectors.

    2. Try to use `cirq.has_mixture_channel(val, allow_decompose=False)`.
        Case a) Method returns `True`.
            Can act on state vector.
        Case b) Method returns `False`.
            Inconclusive.

    3. Try to use `val._decompose_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Inconclusive.
        Case b) Method returns an OP_TREE containing operations that can all act
                on a state vector.
            Can act on state vector.
        Case c) Method returns an OP_TREE containing any operation that cannot
                act on a state vector.
            Cannot act on state vector.

    It is assumed that, when multiple of these strategies give a conclusive
    result, that these results will all be consistent with each other. If all
    strategies are inconclusive, the value is classified as not being able to
    act on a state vector.

    Args:
        val: The value that may or may not be able to act on a state vector.
        allow_decompose: Used by internal methods to stop redundant
            decompositions from being performed (e.g. there's no need to
            decompose an object to check if it is unitary as part of determining
            if the object is a quantum channel, when the quantum channel check
            will already be doing a more general decomposition check). Defaults
            to True. When false, the decomposition strategy for determining
            the result is skipped.

    Returns:
        Whether or not `val` can act on a state vector.
    """
    strats = [
        _strat_can_act_from_can_act,
        _strat_can_act_from_has_mixture,
    ]
    if allow_decompose:
        strats.append(_strat_can_act_from_decompose)
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result

    # If you can't tell that it acts on a state vector, then it can't.
    return False


def _strat_can_act_from_can_act(val: Any) -> Optional[bool]:
    getter = getattr(val, '_can_act_on_state_vector_', None)
    if getter is None:
        return None
    result = getter()
    if result is NotImplemented:
        return None
    return result


def _strat_can_act_from_has_mixture(val: Any) -> Optional[bool]:
    result = mixture_protocol.has_mixture_channel(val, allow_decompose=False)
    if result:
        return True
    return None


def _strat_can_act_from_decompose(val: Any) -> Optional[bool]:
    operations, _, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return None
    return all(can_act_on_state_vector(op) for op in operations)
