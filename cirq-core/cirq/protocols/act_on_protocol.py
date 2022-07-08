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

from typing import Any, Sequence, TYPE_CHECKING, Union

from typing_extensions import Protocol

from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


class SupportsActOn(Protocol):
    """An object that explicitly specifies how to act on simulator states."""

    @doc_private
    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> Union[NotImplementedType, bool]:
        """Applies an action to the given argument, if it is a supported type.

        For example, unitary operations can implement an `_act_on_` method that
        checks if `isinstance(sim_state, cirq.StateVectorSimulationState)` and,
        if so, apply their unitary effect to the state vector.

        The global `cirq.act_on` method looks for whether or not the given
        argument has this value, before attempting any fallback strategies
        specified by the argument being acted on.

        This should only be implemented on `Operation` subclasses. Others such
        as gates should use `SupportsActOnQubits`.

        Args:
            sim_state: An object of unspecified type. The method must check if
                this object is of a recognized type and act on it if so.

        Returns:
            True: The receiving object (`self`) acted on the argument.
            NotImplemented: The receiving object did not act on the argument.

            All other return values are considered to be errors.
        """


class SupportsActOnQubits(Protocol):
    """An object that explicitly specifies how to act on specific qubits."""

    @doc_private
    def _act_on_(
        self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']
    ) -> Union[NotImplementedType, bool]:
        """Applies an action to the given argument, if it is a supported type.

        For example, unitary operations can implement an `_act_on_` method that
        checks if `isinstance(sim_state, cirq.StateVectorSimulationState)` and,
        if so, apply their unitary effect to the state vector.

        The global `cirq.act_on` method looks for whether or not the given
        argument has this value, before attempting any fallback strategies
        specified by the argument being acted on.

        If implementing this on an `Operation`, use `SupportsActOn` instead.

        Args:
            sim_state: An object of unspecified type. The method must check if
                this object is of a recognized type and act on it if so.
            qubits: The sequence of qubits to use when applying the action.

        Returns:
            True: The receiving object (`self`) acted on the argument.
            NotImplemented: The receiving object did not act on the argument.

            All other return values are considered to be errors.
        """


def act_on(
    action: Any,
    sim_state: 'cirq.SimulationStateBase',
    qubits: Sequence['cirq.Qid'] = None,
    *,
    allow_decompose: bool = True,
):
    """Applies an action to a state argument.

    For example, the action may be a `cirq.Operation` and the state argument may
    represent the internal state of a state vector simulator (a
    `cirq.StateVectorSimulationState`).

    For non-operations, the `qubits` argument must be explicitly supplied.

    The action is applied by first checking if `action._act_on_` exists and
    returns `True` (instead of `NotImplemented`) for the given object. Then
    fallback strategies specified by the state argument via `_act_on_fallback_`
    are attempted. If those also fail, the method fails with a `TypeError`.

    Args:
        action: The operation, gate, or other to apply to the state tensor.
        sim_state: A mutable state object that should be modified by the
            action. May specify an `_act_on_fallback_` method to use in case
            the action doesn't recognize it.
        qubits: The sequence of qubits to use when applying the action.
        allow_decompose: Defaults to True. Forwarded into the
            `_act_on_fallback_` method of `sim_state`. Determines if
            decomposition should be used or avoided when attempting to act
            `action` on `sim_state`. Used by internal methods to avoid
            redundant decompositions.

    Returns:
        Nothing. Results are communicated by editing `sim_state`.

    Raises:
        ValueError: If called on an operation and supplied qubits, if not called
            on an operation and no qubits are supplied, or if `_act_on_` or
             `_act_on_fallback_` returned something other than `True` or
             `NotImplemented`.
        TypeError: Failed to act `action` on `sim_state`.
    """
    is_op = isinstance(action, ops.Operation)

    if is_op and qubits is not None:
        raise ValueError('Calls to act_on should not supply qubits if the action is an Operation.')

    if not is_op and qubits is None:
        raise ValueError('Calls to act_on should supply qubits if the action is not an Operation.')

    action_act_on = getattr(action, '_act_on_', None)
    if action_act_on is not None:
        result = action_act_on(sim_state) if is_op else action_act_on(sim_state, qubits)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_ must return True or NotImplemented but got '
                f'{result!r} from {action!r}._act_on_'
            )

    arg_fallback = getattr(sim_state, '_act_on_fallback_', None)
    if arg_fallback is not None:
        qubits = action.qubits if isinstance(action, ops.Operation) else qubits
        result = arg_fallback(action, qubits=qubits, allow_decompose=allow_decompose)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_fallback_ must return True or NotImplemented but got '
                f'{result!r} from {type(sim_state)}._act_on_fallback_'
            )

    raise TypeError(
        "Failed to act action on state argument.\n"
        "Tried both action._act_on_ and sim_state._act_on_fallback_.\n"
        "\n"
        f"State argument type: {type(sim_state)}\n"
        f"Action type: {type(action)}\n"
        f"Action repr: {action!r}\n"
    )
