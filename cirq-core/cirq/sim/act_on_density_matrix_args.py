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

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Sequence, Union

import numpy as np

from cirq import protocols, sim
from cirq._compat import proper_repr
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose
from cirq.linalg import transformations

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
        available_buffer: Optional[List[np.ndarray]] = None,
        qid_shape: Optional[Tuple[int, ...]] = None,
        prng: Optional[np.random.RandomState] = None,
        log_of_measurement_results: Optional[Dict[str, Any]] = None,
        qubits: Optional[Sequence['cirq.Qid']] = None,
        ignore_measurement_results: bool = False,
    ):
        """Inits ActOnDensityMatrixArgs.

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
            qid_shape: The shape of the target tensor.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            ignore_measurement_results: If True, then the simulation
                will treat measurement as dephasing instead of collapsing
                process. This is only applicable to simulators that can
                model dephasing.

        Raises:
            ValueError: The dimension of `target_tensor` is not divisible by 2
                and `qid_shape` is not provided.
        """
        super().__init__(prng, qubits, log_of_measurement_results, ignore_measurement_results)
        self.target_tensor = target_tensor
        if available_buffer is None:
            self.available_buffer = [np.empty_like(target_tensor) for _ in range(3)]
        else:
            self.available_buffer = available_buffer
        if qid_shape is None:
            target_shape = target_tensor.shape
            if len(target_shape) % 2 != 0:
                raise ValueError(
                    'The dimension of target_tensor is not divisible by 2.'
                    ' Require explicit qid_shape.'
                )
            self.qid_shape = target_shape[: len(target_shape) // 2]
        else:
            self.qid_shape = qid_shape

    def _act_on_fallback_(
        self,
        action: Union['cirq.Operation', 'cirq.Gate'],
        qubits: Sequence['cirq.Qid'],
        allow_decompose: bool = True,
    ) -> bool:
        strats = [
            _strat_apply_channel_to_state,
        ]
        if allow_decompose:
            strats.append(strat_act_on_from_apply_decompose)  # type: ignore

        # Try each strategy, stopping if one works.
        for strat in strats:
            result = strat(action, self, qubits)
            if result is False:
                break  # coverage: ignore
            if result is True:
                return True
            assert result is NotImplemented, str(result)
        raise TypeError(
            "Can't simulate operations that don't implement "
            "SupportsUnitary, SupportsConsistentApplyUnitary, "
            "SupportsMixture or SupportsKraus or is a measurement: {!r}".format(action)
        )

    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Delegates the call to measure the density matrix."""
        bits, _ = sim.measure_density_matrix(
            self.target_tensor,
            self.get_axes(qubits),
            out=self.target_tensor,
            qid_shape=self.qid_shape,
            seed=self.prng,
        )
        return bits

    def _on_copy(self, target: 'cirq.ActOnDensityMatrixArgs', deep_copy_buffers: bool = True):
        target.target_tensor = self.target_tensor.copy()
        if deep_copy_buffers:
            target.available_buffer = [b.copy() for b in self.available_buffer]
        else:
            target.available_buffer = self.available_buffer

    def _on_kronecker_product(
        self, other: 'cirq.ActOnDensityMatrixArgs', target: 'cirq.ActOnDensityMatrixArgs'
    ):
        target_tensor = transformations.density_matrix_kronecker_product(
            self.target_tensor, other.target_tensor
        )
        target.target_tensor = target_tensor
        target.available_buffer = [
            np.empty_like(target_tensor) for _ in range(len(self.available_buffer))
        ]
        target.qid_shape = target_tensor.shape[: int(target_tensor.ndim / 2)]

    def _on_factor(
        self,
        qubits: Sequence['cirq.Qid'],
        extracted: 'cirq.ActOnDensityMatrixArgs',
        remainder: 'cirq.ActOnDensityMatrixArgs',
        validate=True,
        atol=1e-07,
    ):
        axes = self.get_axes(qubits)
        extracted_tensor, remainder_tensor = transformations.factor_density_matrix(
            self.target_tensor, axes, validate=validate, atol=atol
        )
        extracted.target_tensor = extracted_tensor
        extracted.available_buffer = [
            np.empty_like(extracted_tensor) for _ in self.available_buffer
        ]
        extracted.qid_shape = extracted_tensor.shape[: int(extracted_tensor.ndim / 2)]
        remainder.target_tensor = remainder_tensor
        remainder.available_buffer = [
            np.empty_like(remainder_tensor) for _ in self.available_buffer
        ]
        remainder.qid_shape = remainder_tensor.shape[: int(remainder_tensor.ndim / 2)]

    def _on_transpose_to_qubit_order(
        self, qubits: Sequence['cirq.Qid'], target: 'cirq.ActOnDensityMatrixArgs'
    ):
        axes = self.get_axes(qubits)
        new_tensor = transformations.transpose_density_matrix_to_axis_order(
            self.target_tensor, axes
        )
        buffer = [np.empty_like(new_tensor) for _ in self.available_buffer]
        target.target_tensor = new_tensor
        target.available_buffer = buffer
        target.qid_shape = new_tensor.shape[: int(new_tensor.ndim / 2)]

    def sample(
        self,
        qubits: Sequence['cirq.Qid'],
        repetitions: int = 1,
        seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    ) -> np.ndarray:
        indices = [self.qubit_map[q] for q in qubits]
        return sim.sample_density_matrix(
            self.target_tensor,
            indices,
            qid_shape=tuple(q.dimension for q in self.qubits),
            repetitions=repetitions,
            seed=seed,
        )

    @property
    def can_represent_mixed_states(self) -> bool:
        return True

    def __repr__(self) -> str:
        return (
            'cirq.ActOnDensityMatrixArgs('
            f'target_tensor={proper_repr(self.target_tensor)},'
            f' available_buffer={proper_repr(self.available_buffer)},'
            f' qid_shape={self.qid_shape!r},'
            f' qubits={self.qubits!r},'
            f' log_of_measurement_results={proper_repr(self.log_of_measurement_results)})'
        )


def _strat_apply_channel_to_state(
    action: Any, args: 'cirq.ActOnDensityMatrixArgs', qubits: Sequence['cirq.Qid']
) -> bool:
    """Apply channel to state."""
    axes = args.get_axes(qubits)
    result = protocols.apply_channel(
        action,
        args=protocols.ApplyChannelArgs(
            target_tensor=args.target_tensor,
            out_buffer=args.available_buffer[0],
            auxiliary_buffer0=args.available_buffer[1],
            auxiliary_buffer1=args.available_buffer[2],
            left_axes=axes,
            right_axes=[e + len(args.qubits) for e in axes],
        ),
        default=None,
    )
    if result is None:
        return NotImplemented
    for i in range(len(args.available_buffer)):
        if result is args.available_buffer[i]:
            args.available_buffer[i] = args.target_tensor
    args.target_tensor = result
    return True
