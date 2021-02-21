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
"""Objects and methods for acting efficiently on a density matrix."""

from typing import Any, Iterable, Dict, List, Tuple

import numpy as np

from cirq import protocols
from cirq.sim.act_on_args import ActOnArgs, strat_act_on_from_apply_decompose


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
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            qid_shape: The shape of the target tensor.
            prng: The pseudo random number generator to use for probabilistic
                effects.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStateVectorArgs.record_measurement_result`.
        """
        super().__init__(axes, prng, log_of_measurement_results)
        self.target_tensor = target_tensor
        self.available_buffer = available_buffer
        self.qid_shape = qid_shape

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        """Apply channel to state."""
        result = protocols.apply_channel(
            action,
            args=protocols.ApplyChannelArgs(
                target_tensor=self.target_tensor,
                out_buffer=self.available_buffer[0],
                auxiliary_buffer0=self.available_buffer[1],
                auxiliary_buffer1=self.available_buffer[2],
                left_axes=self.axes,
                right_axes=[e + len(self.qid_shape) for e in self.axes],
            ),
            default=None,
        )
        if result is not None:
            for i in range(3):
                if result is self.available_buffer[i]:
                    self.available_buffer[i] = self.target_tensor
            self.target_tensor = result
            return True

        if allow_decompose:
            return strat_act_on_from_apply_decompose(action, self)

        return NotImplemented  # coverage: ignore
