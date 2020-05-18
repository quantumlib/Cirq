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
"""Objects and methods for acting efficiently on a state vector."""

from typing import (Any, Iterable, Sequence, Tuple, TYPE_CHECKING, Union,
                    DefaultDict, List, Dict)

import numpy as np

from cirq import linalg, protocols
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits,)

if TYPE_CHECKING:
    import cirq


class ActOnStateVectorArgs:
    """State and context for an operation acting on a state vector."""

    def __init__(self, target_tensor: np.ndarray, available_buffer: np.ndarray,
                 axes: Iterable[int], prng: np.random.RandomState,
                 log_of_measurement_results: Dict[str, Any]):
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

    def record_measurement_result(self, key: str, value: Any):
        """Adds a measurement result to the log.

        Args:
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_keys_` method.
            value: The value to log for the measurement.
        """
        if key in self.log_of_measurement_results:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self.log_of_measurement_results[key] = value

    def subspace_index(self,
                       little_endian_bits_int: int = 0,
                       *,
                       big_endian_bits_int: int = 0
                      ) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
        """An index for the subspace where the target axes equal a value.

        Args:
            little_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The least significant
                bit of the integer is the desired bit for the first axis, and
                so forth in increasing order. Can't be specified at the same
                time as `big_endian_bits_int`.
            big_endian_bits_int: The desired value of the qubits at the
                targeted `axes`, packed into an integer. The most significant
                bit of the integer is the desired bit for the first axis, and
                so forth in decreasing order. Can't be specified at the same
                time as `little_endian_bits_int`.
            value_tuple: The desired value of the qids at the targeted `axes`,
                packed into a tuple.  Specify either `little_endian_bits_int` or
                `value_tuple`.

        Returns:
            A value that can be used to index into `target_tensor` and
            `available_buffer`, and manipulate only the part of Hilbert space
            corresponding to a given bit assignment.

        Example:
            If `target_tensor` is a 4 qubit tensor and `axes` is `[1, 3]` and
            then this method will return the following when given
            `little_endian_bits=0b01`:

                `(slice(None), 0, slice(None), 1, Ellipsis)`

            Therefore the following two lines would be equivalent:

                args.target_tensor[args.subspace_index(0b01)] += 1

                args.target_tensor[:, 0, :, 1] += 1
        """
        return linalg.slice_for_qubits_equal_to(
            self.axes,
            little_endian_qureg_value=little_endian_bits_int,
            big_endian_qureg_value=big_endian_bits_int,
            qid_shape=self.target_tensor.shape)

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        strats = [
            _strat_act_on_state_vector_from_apply_unitary,
            _strat_act_on_state_vector_from_mixture,
        ]
        if allow_decompose:
            strats.append(_strat_act_on_state_vector_from_apply_decompose)

        # Try each strategy, stopping if one works.
        for strat in strats:
            result = strat(action, self)
            if result is False:
                break
            if result is True:
                return True
            assert result is NotImplemented

        return NotImplemented


def _strat_act_on_state_vector_from_apply_unitary(
        unitary_value: Any,
        args: 'cirq.ActOnStateVectorArgs',
) -> bool:
    new_target_tensor = protocols.apply_unitary(
        unitary_value,
        protocols.ApplyUnitaryArgs(
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
        args: ActOnStateVectorArgs,
) -> bool:
    operations, qubits, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented
    return _act_all_on_state_vector(operations, qubits, args)


def _act_all_on_state_vector(actions: Iterable[Any],
                             qubits: Sequence['cirq.Qid'],
                             args: 'cirq.ActOnStateVectorArgs'):
    if len(qubits) != len(args.axes):
        raise ValueError('len(qubits) != len(args.axes)')
    qubit_map = {q: args.axes[i] for i, q in enumerate(qubits)}

    old_indices = args.indices
    try:
        for action in actions:
            args.indices = [qubit_map[q] for q in action.qubits]
            protocols.act_on(action, args)
    finally:
        args.indices = old_indices


def _strat_act_on_state_vector_from_mixture(action: Any,
                                            args: 'cirq.ActOnStateVectorArgs'
                                           ) -> bool:
    mixture = protocols.mixture(action, default=None)
    if mixture is None:
        return NotImplemented
    probabilities, unitaries = zip(*mixture)

    index = args.prng.choice(range(len(unitaries)), p=probabilities)
    shape = protocols.qid_shape(action) * 2
    unitary = unitaries[index].astype(args.target_tensor.dtype).reshape(shape)
    linalg.targeted_left_multiply(unitary,
                                  args.target_tensor,
                                  args.axes,
                                  out=args.available_buffer)
    args.swap_target_tensor_for(args.available_buffer)
    return True
