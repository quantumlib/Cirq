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
"""A protocol that wouldn't exist if python had __rimul__."""

from typing import Any, TYPE_CHECKING, Union, Sequence

from typing_extensions import Protocol

from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
from cirq import ops

if TYPE_CHECKING:
    import cirq


class SupportsActOn(Protocol):
    """An object that explicitly specifies how to act on simulator states."""

    @doc_private
    def _act_on_(self, args: 'cirq.ActOnArgs') -> Union[NotImplementedType, bool]:
        """Applies an action to the given argument, if it is a supported type.

        For example, unitary operations can implement an `_act_on_` method that
        checks if `isinstance(args, cirq.ActOnStateVectorArgs)` and, if so,
        apply their unitary effect to the state vector.

        The global `cirq.act_on` method looks for whether or not the given
        argument has this value, before attempting any fallback strategies
        specified by the argument being acted on.

        This method is analogous to python's `__imul__` in that it is expected
        to perform an inline effect if it recognizes the type of an argument,
        and return NotImplemented otherwise. It is also analogous to python's
        `__rmul__` in that dispatch is being done on the right hand side value
        instead of the left hand side value. If python had an `__rimul__`
        method, then `_act_on_` would not exist because it would be redundant.

        Args:
            args: An object of unspecified type. The method must check if this
                object is of a recognized type and act on it if so.

        Returns:
            True: The receiving object (`self`) acted on the argument.
            NotImplemented: The receiving object did not act on the argument.

            All other return values are considered to be errors.
        """


class SupportsActOnQubits(Protocol):
    """An object that explicitly specifies how to act on simulator states."""

    @doc_private
    def _act_on_qubits_(
        self, qubits: Sequence['cirq.Qid'], args: 'cirq.ActOnArgs'
    ) -> Union[NotImplementedType, bool]:
        """Applies an action to the given argument, if it is a supported type.

        For example, unitary operations can implement an `_act_on_` method that
        checks if `isinstance(args, cirq.ActOnStateVectorArgs)` and, if so,
        apply their unitary effect to the state vector.

        The global `cirq.act_on` method looks for whether or not the given
        argument has this value, before attempting any fallback strategies
        specified by the argument being acted on.

        This method is analogous to python's `__imul__` in that it is expected
        to perform an inline effect if it recognizes the type of an argument,
        and return NotImplemented otherwise. It is also analogous to python's
        `__rmul__` in that dispatch is being done on the right hand side value
        instead of the left hand side value. If python had an `__rimul__`
        method, then `_act_on_` would not exist because it would be redundant.

        Args:
            qubits: The sequence of qubits to use when applying the action.
            args: An object of unspecified type. The method must check if this
                object is of a recognized type and act on it if so.

        Returns:
            True: The receiving object (`self`) acted on the argument.
            NotImplemented: The receiving object did not act on the argument.

            All other return values are considered to be errors.
        """


def act_on(
    action: 'cirq.Operation',
    args: 'cirq.ActOnArgs',
    *,
    allow_decompose: bool = True,
):
    """Applies an action to a state argument.

    For example, the action may be a `cirq.Operation` and the state argument may
    represent the internal state of a state vector simulator (a
    `cirq.ActOnStateVectorArgs`).

    The action is applied by first checking if `action._act_on_` exists and
    returns `True` (instead of `NotImplemented`) for the given object. Then
    fallback strategies specified by the state argument via `_act_on_fallback_`
    are attempted. If those also fail, the method fails with a `TypeError`.

    Args:
        action: The operation to apply to the state tensor.
        args: A mutable state object that should be modified by the action. May
            specify an `_act_on_fallback_` method to use in case the action
            doesn't recognize it.
        allow_decompose: Defaults to True. Forwarded into the
            `_act_on_fallback_` method of `args`. Determines if decomposition
            should be used or avoided when attempting to act `action` on `args`.
            Used by internal methods to avoid redundant decompositions.

    Returns:
        Nothing. Results are communicated by editing `args`.

    Raises:
        TypeError: Failed to act `action` on `args`.
    """
    action_act_on = getattr(action, '_act_on_', None)
    if action_act_on is not None:
        result = action_act_on(args)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_ must return True or NotImplemented but got '
                f'{result!r} from {action!r}._act_on_'
            )

    arg_fallback = getattr(args, '_act_on_fallback_', None)
    if arg_fallback is not None:
        result = arg_fallback(action, allow_decompose=allow_decompose)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_fallback_ must return True or NotImplemented but got '
                f'{result!r} from {type(args)}._act_on_fallback_'
            )

    raise TypeError(
        "Failed to act action on state argument.\n"
        "Tried both action._act_on_ and args._act_on_fallback_.\n"
        "\n"
        f"State argument type: {type(args)}\n"
        f"Action type: {type(action)}\n"
        f"Action repr: {action!r}\n"
    )


def act_on_qubits(
    action: Any,
    qubits: Sequence['cirq.Qid'],
    args: 'cirq.ActOnArgs',
    *,
    allow_decompose: bool = True,
):
    """Applies an action to qubits of a state argument.

    For example, the action may be a `cirq.Gate` and the state argument may
    represent the internal state of a state vector simulator (a
    `cirq.ActOnStateVectorArgs`).

    The action is applied by first checking if `action._act_on_qubits_` exists
    and returns `True` (instead of `NotImplemented`) for the given object. Then
    fallback strategies specified by the state argument via `_act_on_fallback_`
    are attempted. If those also fail, the method fails with a `TypeError`.

    Args:
        action: The action to apply to the state tensor. Typically a
            `cirq.Gate`.
        qubits: The sequence of qubits to use when applying the action.
        args: A mutable state object that should be modified by the action. May
            specify an `_act_on_fallback_` method to use in case the action
            doesn't recognize it.
        allow_decompose: Defaults to True. Forwarded into the
            `_act_on_fallback_` method of `args`. Determines if decomposition
            should be used or avoided when attempting to act `action` on `args`.
            Used by internal methods to avoid redundant decompositions.

    Returns:
        Nothing. Results are communicated by editing `args`.

    Raises:
        TypeError: Failed to act `action` on `args`.
    """
    assert not isinstance(action, ops.Operation), "Use `cirq.act_on` for action type `Operation`"
    action_act_on = getattr(action, '_act_on_qubits_', None)
    if action_act_on is not None:
        result = action_act_on(qubits, args)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_ must return True or NotImplemented but got '
                f'{result!r} from {action!r}._act_on_'
            )

    arg_fallback = getattr(args, '_act_on_qubits_fallback_', None)
    if arg_fallback is not None:
        result = arg_fallback(action, allow_decompose=allow_decompose, qubits=qubits)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(
                f'_act_on_fallback_ must return True or NotImplemented but got '
                f'{result!r} from {type(args)}._act_on_fallback_'
            )

    raise TypeError(
        "Failed to act action on state argument.\n"
        "Tried both action._act_on_ and args._act_on_fallback_.\n"
        "\n"
        f"State argument type: {type(args)}\n"
        f"Action type: {type(action)}\n"
        f"Action repr: {action!r}\n"
    )
