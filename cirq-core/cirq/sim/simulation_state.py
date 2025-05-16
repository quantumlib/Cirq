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

from __future__ import annotations

import abc
import copy
from typing import (
    Any,
    cast,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import numpy as np
from typing_extensions import Self

from cirq import ops, protocols, value
from cirq.sim.simulation_state_base import SimulationStateBase

TState = TypeVar('TState', bound='cirq.QuantumStateRepresentation')

if TYPE_CHECKING:
    import cirq


class SimulationState(SimulationStateBase, Generic[TState], metaclass=abc.ABCMeta):
    """State and context for an operation acting on a state tensor."""

    def __init__(
        self,
        *,
        state: TState,
        prng: Optional[np.random.RandomState] = None,
        qubits: Optional[Sequence[cirq.Qid]] = None,
        classical_data: Optional[cirq.ClassicalDataStore] = None,
    ):
        """Inits SimulationState.

        Args:
            prng: The pseudo random number generator to use for probabilistic
                effects.
            qubits: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            classical_data: The shared classical data container for this
                simulation.
            state: The underlying quantum state of the simulation.
        """
        if qubits is None:
            qubits = ()
        classical_data = classical_data or value.ClassicalDataDictionaryStore()
        super().__init__(qubits=qubits, classical_data=classical_data)
        if prng is None:
            prng = cast(np.random.RandomState, np.random)
        self._prng = prng
        self._state = state

    @property
    def prng(self) -> np.random.RandomState:
        return self._prng

    def measure(
        self,
        qubits: Sequence[cirq.Qid],
        key: str,
        invert_mask: Sequence[bool],
        confusion_map: Dict[Tuple[int, ...], np.ndarray],
    ):
        """Measures the qubits and records to `log_of_measurement_results`.

        Any bitmasks will be applied to the measurement record.

        Args:
            qubits: The qubits to measure.
            key: The key the measurement result should be logged under. Note
                that operations should only store results under keys they have
                declared in a `_measurement_key_names_` method.
            invert_mask: The invert mask for the measurement.
            confusion_map: The confusion matrices for the measurement.

        Raises:
            ValueError: If a measurement key has already been logged to a key.
        """
        bits = self._perform_measurement(qubits)
        confused = self._confuse_result(bits, qubits, confusion_map)
        corrected = [bit ^ (bit < 2 and mask) for bit, mask in zip(confused, invert_mask)]
        self._classical_data.record_measurement(
            value.MeasurementKey.parse_serialized(key), corrected, qubits
        )

    def get_axes(self, qubits: Sequence[cirq.Qid]) -> List[int]:
        return [self.qubit_map[q] for q in qubits]

    def _perform_measurement(self, qubits: Sequence[cirq.Qid]) -> List[int]:
        """Delegates the call to measure the `QuantumStateRepresentation`."""
        if self._state is not None:
            return self._state.measure(self.get_axes(qubits), self.prng)
        raise NotImplementedError()

    def _confuse_result(
        self,
        bits: List[int],
        qubits: Sequence[cirq.Qid],
        confusion_map: Dict[Tuple[int, ...], np.ndarray],
    ):
        """Applies confusion matrices to measured results.

        Compare with _confuse_results in cirq-core/cirq/sim/simulator.py.
        """
        confused = list(bits)
        dims = [q.dimension for q in qubits]
        for indices, confuser in confusion_map.items():
            mat_dims = [dims[k] for k in indices]
            row = value.big_endian_digits_to_int((bits[k] for k in indices), base=mat_dims)
            new_val = self.prng.choice(len(confuser), p=confuser[row])
            new_bits = value.big_endian_int_to_digits(new_val, base=mat_dims)
            for i, k in enumerate(indices):
                confused[k] = new_bits[i]
        return confused

    def sample(
        self,
        qubits: Sequence[cirq.Qid],
        repetitions: int = 1,
        seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
    ) -> np.ndarray:
        if self._state is not None:
            return self._state.sample(self.get_axes(qubits), repetitions, seed)
        raise NotImplementedError()

    def copy(self, deep_copy_buffers: bool = True) -> Self:
        """Creates a copy of the object.

        Args:
            deep_copy_buffers: If True, buffers will also be deep-copied.
            Otherwise the copy will share a reference to the original object's
            buffers.

        Returns:
            A copied instance.
        """
        args = copy.copy(self)
        args._classical_data = self._classical_data.copy()
        args._state = self._state.copy(deep_copy_buffers=deep_copy_buffers)
        return args

    def create_merged_state(self) -> Self:
        """Creates a final merged state."""
        return self

    def add_qubits(self: Self, qubits: Sequence[cirq.Qid]) -> Self:
        """Add `qubits` in the `|0>` state to a new state space and take the kron product.

        Args:
            qubits: Sequence of qubits to be added.

        Returns:
            NotImplemented: If the subclass does not implement this method.
            Self: A `cirq.SimulationState` with qubits added or `self` if there are no qubits to
                add.
        """
        if not qubits:
            return self
        return NotImplemented

    def remove_qubits(self: Self, qubits: Sequence[cirq.Qid]) -> Self:
        """Remove `qubits` from the state space.

        The qubits to be removed should be untangled from rest of the system and in the |0> state.

        Args:
            qubits: Sequence of qubits to be removed.

        Returns:
            NotImplemented: If the subclass does not implement this method.
            Self: A `cirq.SimulationState` with qubits removed or `self` if there are no qubits to
                remove.
        """
        if not qubits:
            return self
        return NotImplemented

    def kronecker_product(self, other: Self, *, inplace=False) -> Self:
        """Joins two state spaces together."""
        args = self if inplace else copy.copy(self)
        args._state = self._state.kron(other._state)
        args._set_qubits(self.qubits + other.qubits)
        return args

    def factor(
        self, qubits: Sequence[cirq.Qid], *, validate=True, atol=1e-07, inplace=False
    ) -> Tuple[Self, Self]:
        """Splits two state spaces after a measurement or reset."""
        extracted = copy.copy(self)
        remainder = self if inplace else copy.copy(self)
        e, r = self._state.factor(self.get_axes(qubits), validate=validate, atol=atol)
        extracted._state = e
        remainder._state = r
        extracted._set_qubits(qubits)
        remainder._set_qubits([q for q in self.qubits if q not in qubits])
        return extracted, remainder

    @property
    def allows_factoring(self):
        """Subclasses that allow factorization should override this."""
        return self._state.supports_factor if self._state is not None else False

    def transpose_to_qubit_order(self, qubits: Sequence[cirq.Qid], *, inplace=False) -> Self:
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
        args._state = self._state.reindex(self.get_axes(qubits))
        args._set_qubits(qubits)
        return args

    @property
    def qubits(self) -> Tuple[cirq.Qid, ...]:
        return self._qubits

    def swap(self, q1: cirq.Qid, q2: cirq.Qid, *, inplace=False):
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
        args._set_qubits(qubits)
        return args

    def rename(self, q1: cirq.Qid, q2: cirq.Qid, *, inplace=False):
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
        args._set_qubits(qubits)
        return args

    def __getitem__(self, item: Optional[cirq.Qid]) -> Self:
        if item not in self.qubit_map:
            raise IndexError(f'{item} not in {self.qubits}')
        return self

    def __len__(self) -> int:
        return len(self.qubits)

    def __iter__(self) -> Iterator[Optional[cirq.Qid]]:
        return iter(self.qubits)

    @property
    def can_represent_mixed_states(self) -> bool:
        return self._state.can_represent_mixed_states if self._state is not None else False


def strat_act_on_from_apply_decompose(
    val: Any, args: cirq.SimulationState, qubits: Sequence[cirq.Qid]
) -> bool:
    if isinstance(val, ops.Gate):
        decomposed = protocols.decompose_once_with_qubits(val, qubits, flatten=False, default=None)
    else:
        decomposed = protocols.decompose_once(val, flatten=False, default=None)
    if decomposed is None:
        return NotImplemented
    all_ancilla: Set[cirq.Qid] = set()
    for operation in ops.flatten_to_ops(decomposed):
        curr_ancilla = tuple(q for q in operation.qubits if q not in args.qubits)
        args = args.add_qubits(curr_ancilla)
        if args is NotImplemented:
            return NotImplemented  # pragma: no cover
        all_ancilla.update(curr_ancilla)
        protocols.act_on(operation, args)
    args = args.remove_qubits(tuple(all_ancilla))
    if args is NotImplemented:
        raise TypeError(f"{type(args)} implements add_qubits but not remove_qubits.")
    return True


TSimulationState = TypeVar('TSimulationState', bound=SimulationState)
