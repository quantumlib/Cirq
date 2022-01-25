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
import copy
import inspect
from typing import (
    Any,
    Dict,
    List,
    TypeVar,
    TYPE_CHECKING,
    Sequence,
    Tuple,
    cast,
    Optional,
    Iterator,
)
import warnings

import numpy as np

from cirq import protocols, ops
from cirq.protocols.decompose_protocol import _try_decompose_into_operations_and_qubits
from cirq.sim.operation_target import OperationTarget

TSelf = TypeVar('TSelf', bound='ActOnArgs')

if TYPE_CHECKING:
    import cirq


class ActOnArgs(OperationTarget[TSelf]):
    """State and context for an operation acting on a state tensor."""

    def __init__(
        self,
        prng: np.random.RandomState = None,
        qubits: Sequence['cirq.Qid'] = None,
        log_of_measurement_results: Dict[str, List[int]] = None,
        ignore_measurement_results: bool = False,
    ):
        """Inits ActOnArgs.

        Args:
            prng: The pseudo random number generator to use for probabilistic
                effects.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            log_of_measurement_results: A mutable object that measurements are
                being recorded into.
            ignore_measurement_results: If True, then the simulation
                will treat measurement as dephasing instead of collapsing
                process, and not log the result. This is only applicable to
                simulators that can represent mixed states.
        """
        if prng is None:
            prng = cast(np.random.RandomState, np.random)
        if qubits is None:
            qubits = ()
        if log_of_measurement_results is None:
            log_of_measurement_results = {}
        self._set_qubits(qubits)
        self.prng = prng
        self._log_of_measurement_results = log_of_measurement_results
        self._ignore_measurement_results = ignore_measurement_results

    def _set_qubits(self, qubits: Sequence['cirq.Qid']):
        self._qubits = tuple(qubits)
        self.qubit_map = {q: i for i, q in enumerate(self.qubits)}

    def measure(self, qubits: Sequence['cirq.Qid'], key: str, invert_mask: Sequence[bool]):
        """Measures the qubits and records to `log_of_measurement_results`.

        Any bitmasks will be applied to the measurement record. If
        `self._ignore_measurement_results` is set, it dephases instead of
        measuring, and no measurement result will be logged.

        Args:
            qubits: The qubits to measure.
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_key_names_` method.
            invert_mask: The invert mask for the measurement.

        Raises:
            ValueError: If a measurement key has already been logged to a key.
        """
        if self.ignore_measurement_results:
            self._act_on_fallback_(ops.phase_damp(1), qubits)
            return
        bits = self._perform_measurement(qubits)
        corrected = [bit ^ (bit < 2 and mask) for bit, mask in zip(bits, invert_mask)]
        if key in self._log_of_measurement_results:
            raise ValueError(f"Measurement already logged to key {key!r}")
        self._log_of_measurement_results[key] = corrected

    def get_axes(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        return [self.qubit_map[q] for q in qubits]

    @abc.abstractmethod
    def _perform_measurement(self, qubits: Sequence['cirq.Qid']) -> List[int]:
        """Child classes that perform measurements should implement this with
        the implementation."""

    def copy(self: TSelf, deep_copy_buffers: bool = True) -> TSelf:
        """Creates a copy of the object.

        Args:
            deep_copy_buffers: If True, buffers will also be deep-copied.
            Otherwise the copy will share a reference to the original object's
            buffers.

        Returns:
            A copied instance.
        """
        args = copy.copy(self)
        if 'deep_copy_buffers' in inspect.signature(self._on_copy).parameters:
            self._on_copy(args, deep_copy_buffers)
        else:
            warnings.warn(
                (
                    'A new parameter deep_copy_buffers has been added to ActOnArgs._on_copy(). '
                    'The classes that inherit from ActOnArgs should support it before Cirq 0.15.'
                ),
                DeprecationWarning,
            )
            self._on_copy(args)
        args._log_of_measurement_results = self.log_of_measurement_results.copy()
        return args

    def _on_copy(self: TSelf, args: TSelf, deep_copy_buffers: bool = True):
        """Subclasses should implement this with any additional state copy
        functionality."""

    def create_merged_state(self: TSelf) -> TSelf:
        """Creates a final merged state."""
        return self

    def kronecker_product(self: TSelf, other: TSelf, *, inplace=False) -> TSelf:
        """Joins two state spaces together."""
        args = self if inplace else copy.copy(self)
        self._on_kronecker_product(other, args)
        args._set_qubits(self.qubits + other.qubits)
        return args

    def _on_kronecker_product(self: TSelf, other: TSelf, target: TSelf):
        """Subclasses should implement this with any additional state product
        functionality, if supported."""

    def factor(
        self: TSelf,
        qubits: Sequence['cirq.Qid'],
        *,
        validate=True,
        atol=1e-07,
        inplace=False,
    ) -> Tuple[TSelf, TSelf]:
        """Splits two state spaces after a measurement or reset."""
        extracted = copy.copy(self)
        remainder = self if inplace else copy.copy(self)
        self._on_factor(qubits, extracted, remainder, validate, atol)
        extracted._set_qubits(qubits)
        remainder._set_qubits([q for q in self.qubits if q not in qubits])
        return extracted, remainder

    def _on_factor(
        self: TSelf,
        qubits: Sequence['cirq.Qid'],
        extracted: TSelf,
        remainder: TSelf,
        validate=True,
        atol=1e-07,
    ):
        """Subclasses should implement this with any additional state factor
        functionality, if supported."""

    def transpose_to_qubit_order(
        self: TSelf, qubits: Sequence['cirq.Qid'], *, inplace=False
    ) -> TSelf:
        """Physically reindexes the state by the new basis.

        Args:
            qubits: The desired qubit order.
            inplace: True to perform this operation inplace.

        Returns:
            The state with qubit order transposed and underlying representation
            updated.

        Raises:
            ValueError: If the provided qubits do not match the existing ones.
        """
        if len(self.qubits) != len(qubits) or set(qubits) != set(self.qubits):
            raise ValueError(f'Qubits do not match. Existing: {self.qubits}, provided: {qubits}')
        args = self if inplace else copy.copy(self)
        self._on_transpose_to_qubit_order(qubits, args)
        args._set_qubits(qubits)
        return args

    def _on_transpose_to_qubit_order(self: TSelf, qubits: Sequence['cirq.Qid'], target: TSelf):
        """Subclasses should implement this with any additional state transpose
        functionality, if supported."""

    @property
    def log_of_measurement_results(self) -> Dict[str, List[int]]:
        return self._log_of_measurement_results

    @property
    def ignore_measurement_results(self) -> bool:
        return self._ignore_measurement_results

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._qubits

    def swap(self, q1: 'cirq.Qid', q2: 'cirq.Qid', *, inplace=False):
        """Swaps two qubits.

        This only affects the index, and does not modify the underlying
        state.

        Args:
            q1: The first qubit to swap.
            q2: The second qubit to swap.
            inplace: True to swap the qubits in the current object, False to
                create a copy with the qubits swapped.

        Returns:
            The original object with the qubits swapped if inplace is
            requested, or a copy of the original object with the qubits swapped
            otherwise.

        Raises:
            ValueError: If the qubits are of different dimensionality.
        """
        if q1.dimension != q2.dimension:
            raise ValueError(f'Cannot swap different dimensions: q1={q1}, q2={q2}')

        args = self if inplace else copy.copy(self)
        i1 = self.qubits.index(q1)
        i2 = self.qubits.index(q2)
        qubits = list(args.qubits)
        qubits[i1], qubits[i2] = qubits[i2], qubits[i1]
        args._qubits = tuple(qubits)
        args.qubit_map = {q: i for i, q in enumerate(qubits)}
        return args

    def rename(self, q1: 'cirq.Qid', q2: 'cirq.Qid', *, inplace=False):
        """Renames `q1` to `q2`.

        Args:
            q1: The qubit to rename.
            q2: The new name.
            inplace: True to rename the qubit in the current object, False to
                create a copy with the qubit renamed.

        Returns:
            The original object with the qubits renamed if inplace is
            requested, or a copy of the original object with the qubits renamed
            otherwise.

        Raises:
            ValueError: If the qubits are of different dimensionality.
        """
        if q1.dimension != q2.dimension:
            raise ValueError(f'Cannot rename to different dimensions: q1={q1}, q2={q2}')

        args = self if inplace else copy.copy(self)
        i1 = self.qubits.index(q1)
        qubits = list(args.qubits)
        qubits[i1] = q2
        args._qubits = tuple(qubits)
        args.qubit_map = {q: i for i, q in enumerate(qubits)}
        return args

    def __getitem__(self: TSelf, item: Optional['cirq.Qid']) -> TSelf:
        if item not in self.qubit_map:
            raise IndexError(f'{item} not in {self.qubits}')
        return self

    def __len__(self) -> int:
        return len(self.qubits)

    def __iter__(self) -> Iterator[Optional['cirq.Qid']]:
        return iter(self.qubits)

    @property
    def can_represent_mixed_states(self) -> bool:
        return False


def strat_act_on_from_apply_decompose(
    val: Any,
    args: 'cirq.ActOnArgs',
    qubits: Sequence['cirq.Qid'],
) -> bool:
    operations, qubits1, _ = _try_decompose_into_operations_and_qubits(val)
    assert len(qubits1) == len(qubits)
    qubit_map = {q: qubits[i] for i, q in enumerate(qubits1)}
    if operations is None:
        return NotImplemented
    for operation in operations:
        operation = operation.with_qubits(*[qubit_map[q] for q in operation.qubits])
        protocols.act_on(operation, args)
    return True
