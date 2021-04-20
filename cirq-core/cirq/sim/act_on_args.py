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
"""Objects and methods for acting efficiently on a state tensor."""
import abc
from typing import Any, Iterable, Dict, List, TypeVar, TYPE_CHECKING, Sequence

import numpy as np

from cirq import protocols
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits

TSelf = TypeVar('TSelf', bound='ActOnArgs')

if TYPE_CHECKING:
    import cirq


class ActOnArgs:
    """State and context for an operation acting on a state tensor."""

    def __init__(
        self,
        prng: np.random.RandomState,
        qubits: Sequence['cirq.Qid'] = None,
        axes: Iterable[int] = None,
        log_of_measurement_results: Dict[str, Any] = None,
    ):
        """
        Args:
            prng: The pseudo random number generator to use for probabilistic
                effects.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            axes: The indices of axes corresponding to the qubits that the
                operation is supposed to act upon.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into. Edit it easily by calling
                `ActOnStateVectorArgs.record_measurement_result`.
        """
        if qubits is None:
            qubits = ()
        if axes is None:
            axes = ()
        if log_of_measurement_results is None:
            log_of_measurement_results = {}
        self.qubits = tuple(qubits)
        self.qubit_map = {q: i for i, q in enumerate(self.qubits)}
        self.axes = tuple(axes)
        self.prng = prng
        self.log_of_measurement_results = log_of_measurement_results

    def measure(self, key, invert_mask):
        """Adds a measurement result to the log.

        Args:
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_keys_` method.
            invert_mask: The invert mask for the measurement.
        """
        bits = self._perform_measurement()
        corrected = [bit ^ (bit < 2 and mask) for bit, mask in zip(bits, invert_mask)]
        if key in self.log_of_measurement_results:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self.log_of_measurement_results[key] = corrected

    @abc.abstractmethod
    def _perform_measurement(self) -> List[int]:
        """Child classes that perform measurements should implement this with
        the implementation."""

    @abc.abstractmethod
    def copy(self: TSelf) -> TSelf:
        """Creates a copy of the object."""


def strat_act_on_from_apply_decompose(
    val: Any,
    args: ActOnArgs,
) -> bool:
    operations, qubits, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return NotImplemented
    assert len(qubits) == len(args.axes)
    qubit_map = {q: args.axes[i] for i, q in enumerate(qubits)}

    old_axes = args.axes
    try:
        for operation in operations:
            args.axes = tuple(qubit_map[q] for q in operation.qubits)
            protocols.act_on(operation, args)
    finally:
        args.axes = old_axes
    return True
