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

from typing import Any, Iterable, Dict, List

import numpy as np

from cirq import protocols
from cirq.protocols.decompose_protocol import (
    _try_decompose_into_operations_and_qubits,
)


class ActOnDensityMatrixArgs:
    """State and context for an operation acting on a density matrix.

    There are three common ways to act on this object:

    1. Directly edit the `target_tensor` property, which is storing the density
        matrix of the quantum system as a numpy array with one axis per qudit.
    2. Overwrite the `available_buffer` property with the new state vector, and
        then pass `available_buffer` into `swap_target_tensor_for`.
    3. Call `record_measurement_result(key, val)` to log a measurement result.
    """

    def __init__(
        self,
        target_tensor: np.ndarray,
        available_buffer: List[np.ndarray],
        axes: List[int],
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
                allocations. Passing `available_buffer` into
                `swap_target_tensor_for` will swap it for `target_tensor`.
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
        self.axes = axes
        self.prng = prng
        self.log_of_measurement_results = log_of_measurement_results

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

    def _act_on_fallback_(self, action: Any, allow_decompose: bool):
        """Apply channel to state."""
        operations, qubits, _ = _try_decompose_into_operations_and_qubits(action)
        assert len(qubits) == len(self.axes)
        qubit_map = {q: self.axes[i] for i, q in enumerate(qubits)}
        indices = [qubit_map[qubit] for qubit in qubits]
        result = protocols.apply_channel(
            action,
            args=protocols.ApplyChannelArgs(
                target_tensor=self.target_tensor,
                out_buffer=self.available_buffer[0],
                auxiliary_buffer0=self.available_buffer[1],
                auxiliary_buffer1=self.available_buffer[2],
                left_axes=indices,
                right_axes=[e + len(qubits) for e in indices],
            ),
        )
        for i in range(3):
            if result is self.available_buffer[i]:
                self.available_buffer[i] = self.target_tensor
        self.tensor = result
        return True
