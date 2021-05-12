# Copyright 2021 The Cirq Developers
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
"""Objects and methods for acting efficiently on a density matrix."""

from typing import Any, Iterable, Dict, List, Tuple, TYPE_CHECKING, Sequence

import numpy as np

from cirq import protocols, sim
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose
from cirq.linalg import transformations as tf

if TYPE_CHECKING:
    import cirq


class ActOnDensityMatrixArgs(ActOnArgs):
    """State and context for an operation acting on a density matrix.

    To act on this object, directly edit the `target_tensor` property, which is
    storing the density matrix of the quantum system with one axis per qubit.
    """

    def __init__(
        self,
        target_tensor: np.ndarray,
        available_buffer: List[np.ndarray],
        axes: Iterable[int],
        qid_shape: Tuple[int, ...],
        prng: np.random.RandomState,
        log_of_measurement_results: Dict[str, Any],
        qubits: Sequence['cirq.Qid'] = None,
    ):
        """
        Args:
            target_tensor: The state vector to act on, stored as a numpy array
                with one dimension for each qubit in the system. Operations are
                expected to perform inplace edits of this object.
            available_buffer: A workspace with the same shape and dtype as
                `target_tensor`. Used by operations that cannot be applied to
                `target_tensor` inline, in order to avoid unnecessary
                allocations.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            qid_shape: The shape of the target tensor.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStateVectorArgs.record_measurement_result`.
        """
        super().__init__(prng, qubits, axes, log_of_measurement_results)
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.qid_shape = qid_shape

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        strats = [
            _strat_apply_channel_to_state,
        ]
        if allow_decompose:
            strats.append(strat_act_on_from_apply_decompose)  # type: ignore

        # Try each strategy, stopping if one works.
        for strat in strats:
            result = strat(action, self)
            if result is False:
                break  # coverage: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)
        raise TypeError(
            "Can't simulate operations that don't implement "
            "SupportsUnitary, SupportsConsistentApplyUnitary, "
            "SupportsMixture, SupportsChannel or is a measurement: {!r}".format(action)
        )

    def _perform_measurement(self) -> List[int]:
        """Delegates the call to measure the density matrix."""
        bits, _ = sim.measure_density_matrix(
            self.target_tensor,
            self.axes,
            out=self.target_tensor,
            qid_shape=self.qid_shape,
            seed=self.prng,
        )
        return bits

    def copy(self) -> 'cirq.ActOnDensityMatrixArgs':
        return ActOnDensityMatrixArgs(
            target_tensor=self.target_tensor.copy(),
            available_buffer=[b.copy() for b in self.available_buffer],
            qubits=self.qubits,
            axes=self.axes,
            qid_shape=self.qid_shape,
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results,
        )

    def join(self, other: 'cirq.ActOnDensityMatrixArgs') -> 'cirq.ActOnDensityMatrixArgs':
        target_tensor = tf.merge_density_matrices(self.target_tensor, other.target_tensor)
        buffer = [np.empty_like(target_tensor) for _ in self.available_buffer]
        return ActOnDensityMatrixArgs(
            target_tensor=target_tensor,
            available_buffer=buffer,
            qubits=self.qubits + other.qubits,
            axes=(),
            qid_shape=target_tensor.shape[: int(target_tensor.ndim / 2)],
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results,
        )

    def extract(
        self, qubits: Sequence['cirq.Qid']
    ) -> Tuple['cirq.ActOnDensityMatrixArgs', 'cirq.ActOnDensityMatrixArgs']:
        axes = [self.qubit_map[q] for q in qubits]
        extracted_tensor, remainder_tensor = tf.split_density_matrices(self.target_tensor, axes)
        buffer = [np.empty_like(extracted_tensor) for _ in self.available_buffer]
        extracted_args = ActOnDensityMatrixArgs(
            target_tensor=extracted_tensor,
            available_buffer=buffer,
            qubits=qubits,
            qid_shape=extracted_tensor.shape[: int(extracted_tensor.ndim / 2)],
            axes=(),
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results,
        )
        buffer = [np.empty_like(remainder_tensor) for _ in self.available_buffer]
        remainder_args = ActOnDensityMatrixArgs(
            target_tensor=remainder_tensor,
            available_buffer=buffer,
            qubits=tuple(q for q in self.qubits if q not in qubits),
            qid_shape=remainder_tensor.shape[: int(remainder_tensor.ndim / 2)],
            axes=(),
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results,
        )
        return extracted_args, remainder_args

    def reorder(self, qubits: Sequence['cirq.Qid']) -> 'cirq.ActOnDensityMatrixArgs':
        assert len(qubits) == len(self.qubits)
        axes = [self.qubit_map[q] for q in qubits]
        axes = axes + [i + len(qubits) for i in axes]
        new_tensor = np.moveaxis(self.target_tensor, axes, range(len(qubits) * 2))
        buffer = [np.empty_like(new_tensor) for _ in self.available_buffer]
        return ActOnDensityMatrixArgs(
            target_tensor=new_tensor,
            available_buffer=buffer,
            qubits=qubits,
            qid_shape=new_tensor.shape[: int(new_tensor.ndim / 2)],
            axes=(),
            prng=self.prng,
            log_of_measurement_results=self.log_of_measurement_results,
        )


def _strat_apply_channel_to_state(
    action: Any,
    args: ActOnDensityMatrixArgs,
) -> bool:
    """Apply channel to state."""
    result = protocols.apply_channel(
        action,
        args=protocols.ApplyChannelArgs(
            target_tensor=args.target_tensor,
            out_buffer=args.available_buffer[0],
            auxiliary_buffer0=args.available_buffer[1],
            auxiliary_buffer1=args.available_buffer[2],
            left_axes=args.axes,
            right_axes=[e + len(args.qid_shape) for e in args.axes],
        ),
        default=None,
    )
    if result is None:
        return NotImplemented
    for i in range(3):
        if result is args.available_buffer[i]:
            args.available_buffer[i] = args.target_tensor
    args.target_tensor = result
    return True
