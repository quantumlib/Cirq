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
"""A protocol for implementing high performance unitary left-multiplies."""

from typing import (
    Any,
    cast,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
    DefaultDict, List)

import numpy as np
from typing_extensions import Protocol

from cirq import linalg
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits, )
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

# This is a special indicator value used by the apply_unitary method
# to determine whether or not the caller provided a 'default' argument. It must
# be of type np.ndarray to ensure the method has the correct type signature in
# that case. It is checked for using `is`, so it won't have a false positive if
# the user provides a different np.array([]) value.

RaiseTypeErrorIfNotProvided = np.array([])  # type: np.ndarray

TDefault = TypeVar('TDefault')


class ActOnStateVectorArgs:
    """State and context for an operation acting on a state vector."""

    def __init__(self,
                 target_tensor: np.ndarray,
                 available_buffer: np.ndarray,
                 axes: Iterable[int],
                 prng: np.random.RandomState,
                 log_of_measurement_results: DefaultDict[str, List[List[int]]]):
        """
        Args:
            target_tensor: The state vector to act on, stored as a numpy array
                with one dimension for each qubit in the system. Operations are
                expected to perform inplace edits of this object.
            available_buffer: A workspace with the same shape and dtype as
                `target_tensor`. The result of an operation can be put into this
                buffer, instead of directly editing `target_tensor`, if
                `swap_target_tensor_for` is called afterward.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStateVectorArgs.record_measurement_result`.
        """
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.axes = tuple(axes)
        self.prng = prng
        self.log_of_measurement_results = log_of_measurement_results

    def swap_target_tensor_for(self, new_target_tensor: np.ndarray):
        """Gives a new state vector for the system.

        Typically, the new state vector should be `args.available_buffer` where
        `args` is this `cirq.ActOnStateVectorArgs` instance.

        Args:
            new_target_tensor: The new system state. Must have the same shape
                and dtype as the old system state.
        """
        if new_target_tensor is self.available_buffer:
            self.available_buffer = self.target_tensor
        self.target_tensor = new_target_tensor

    def record_measurement_result(self, key: str, value: Union[int, List[int]]):
        """Adds a measurement result to the log.

        Args:
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_keys_` method.
            value: The value to log for the measurement. This can be an integer
                (e.g. 0 or 1 for a qubit measurement) or a list of integers
                (e.g. for a multi-qubit measurement).
        """
        if isinstance(value, int):
            self.log_of_measurement_results[key].append([value])
        else:
            self.log_of_measurement_results[key].append(value)


def act_on_state_vector(action: Any,
                        args: 'cirq.ActOnStateVectorArgs',
                        *,
                        allow_decompose: bool = True,
                        ):
    """High performance application of an action to a state vector.

    Edits or replaces the `target_tensor` property of `args`, and adds entries
    into `args.log_of_measurement_results`, by using the following strategies:

    A. Try to use `action._act_on_state_vector_(args)`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns `False`.
            Conclude `action` cannot be acted upon a state vector.
        Case c) Method returns `True`.
            Forward the successful result to the caller.

    B. Try to use `action._apply_unitary_(*)`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns `None`.
            Conclude `action` cannot be acted upon a state vector.
        Case c) Method returns a numpy array.
            Translate and forward the successful result to the caller.

    C. Try to use `action._unitary_()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns `None`.
            Conclude `action` cannot be acted upon a state vector.
        Case c) Method returns a numpy array.
            Multiply the matrix onto the target tensor and return to the caller.

    D. Try to use `action._decompose_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Continue to next strategy.
        Case b) Method returns an OP_TREE.
            Delegate to `cirq.act_all_on_state_vector(*)`.

    E. Conclude `action` cannot be acted upon a state vector.

    Args:
        action: The action to apply to the state tensor. Typically a
            `cirq.Operation`.
        args: A mutable `cirq.ActOnStateVectorArgs` object describing the target
            tensor to act upon as well as other context. The result is returned
            by mutating this object.
        allow_decompose: Defaults to True. If set to False, and acting on the
            state vector requires decomposing the action, the method will
            pretend the object cannot act on a state vector.

    Returns:
        Nothing. Results are communicated by editing `args`.

    Raises:
        TypeError: `action` can't act on a state vector.
    """

    # Decide on order to attempt application strategies.
    strats = [
        _strat_act_on_state_vector_from_apply_unitary,
    ]
    if allow_decompose:
        strats.append(_strat_act_on_state_vector_from_apply_decompose)

    # Try each strategy, stopping if one works.
    for strat in strats:
        result = strat(action, args)
        if result is False:
            break
        if result is not NotImplemented:
            return result

    # TODO: MIXTURE
    # # Don't know how to apply. Fallback to specified default behavior.
    # if default is not RaiseTypeErrorIfNotProvided:
    #     return default
    raise TypeError(
        "Value can't act on a state vector.\n"
        "\n"
        f"type: {type(action)}\n"
        f"value: {action!r}\n")


def _strat_act_on_state_vector_from_act_on_state_vector_protocol(
        action: Any,
        args: 'cirq.ActOnStateVectorArgs',
) -> bool:
    func = getattr(action, '_act_on_state_vector_', None)
    if func is None:
        return NotImplemented
    return func(args)
    # op_qid_shape = qid_shape_protocol.qid_shape(unitary_value,
    #                                             (2,) * len(args.axes))
    # sub_args = args._for_operation_with_qid_shape(range(len(op_qid_shape)),
    #                                               op_qid_shape)
    # sub_result = func(sub_args)
    # if sub_result is NotImplemented or sub_result is None:
    #     return sub_result
    # return _incorporate_result_into_target(args, sub_args, sub_result)
    #
    # operations, qubits, _ = _try_decompose_into_operations_and_qubits(val)
    # if operations is None:
    #     return NotImplemented
    # return act_all_on_state_vector(operations, qubits, args)


def _strat_act_on_state_vector_from_apply_unitary(
        unitary_value: Any,
        args: 'cirq.ActOnStateVectorArgs',
) -> bool:
    new_target_tensor = cirq.apply_unitary(
        unitary_value,
        cirq.ApplyUnitaryArgs(
            target_tensor=args.target_tensor,
            available_buffer=args.available_buffer,
            axes=args.axes,
        ),
        allow_decompose=False,
        default=NotImplemented)
    if new_target_tensor is NotImplemented:
        return NotImplemented
    args.swap_target_tensor_for(new_target_tensor)
    return True


def _strat_act_on_state_vector_from_apply_decompose(
        val: Any,
        args: ApplyUnitaryArgs,
) -> bool:
    operations, qubits, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented
    return act_all_on_state_vector(operations, qubits, args)


def act_all_on_state_vector(actions: Iterable[Any],
                            qubits: Sequence['cirq.Qid'],
                            args: 'cirq.ActOnStateVectorArgs'):
    """Apply a series of unitaries onto a state tensor.

    Uses `cirq.apply_unitary` on each of the unitary values, to apply them to
    the state tensor from the `args` argument.

    CAUTION: if one of the given unitary values does not have a unitary effect,
    forcing the method to terminate, the method will not rollback changes
    from previous unitary values.

    Args:
        actions: Values that implement the `_act_on_state_vector_` protocol and
            have a `qubits` property. Each will be acted upon the state vector
            in turn. The `qubits` property is needed in order to determine
            what part of the state is being acted on by each action. Typically
            an action will be a `cirq.Operation`.
        qubits: The qubits that will be targeted by the actions. These qubits
            must match up, index by index, with `args.indices`.
        args: A mutable `cirq.ActOnStateVectorArgs` object describing the target
            tensor to act upon as well as other context. The result is returned
            by mutating this object.

    Returns:
        Nothing. Results are communicated by editing the `args` object.

    Raises:
        TypeError: One of the actions did not implement the
            `_act_on_state_vector_` protocol.
    """
    if len(qubits) != len(args.axes):
        raise ValueError('len(qubits) != len(args.axes)')
    qubit_map = {
        q: args.axes[i] for i, q in enumerate(qubits)
    }

    old_indices = args.indices
    try:
        for action in actions:
            args.indices = [qubit_map[q] for q in action.qubits]
            act_on_state_vector(action=action, args=args)
    finally:
        args.indices = old_indices
