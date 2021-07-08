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

"""Marker classes for indicating which additional features gates support.

For example: some gates are reversible, some have known matrices, etc.
"""

import abc
from typing import Union, Iterable, Any, List, Sequence

from cirq.ops import raw_types


class InterchangeableQubitsGate(metaclass=abc.ABCMeta):
    """Indicates operations should be equal under some qubit permutations."""

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        return 0


class SupportsOnEachGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that can be applied to exactly one qubit."""

    def on_each(self, *targets: Union[raw_types.Qid, Iterable[Any]]) -> List[raw_types.Operation]:
        """Returns a list of operations applying the gate to all targets.
        Args:
            *targets: The qubits to apply this gate to. For single-qubit gates
            this can be provided as varargs or a combination of nested
            iterables. For multi-qubit gates this must be provided as an
            `Iterable[Sequence[Qid]]`, where each sequence has `num_qubits`
            qubits.
        Returns:
            Operations applying this gate to the target qubits.
        Raises:
            ValueError if targets are not instances of Qid or Iterable[Qid].
            ValueError if the gate qubit number is incompatible.
        """
        operations: List[raw_types.Operation] = []
        if self._num_qubits_() > 1:
            iterator: Iterable = targets
            if len(targets) == 1:
                if not isinstance(targets[0], Iterable):
                    raise TypeError(f'{targets[0]} object is not iterable.')
                t0 = list(targets[0])
                iterator = [t0] if t0 and isinstance(t0[0], raw_types.Qid) else t0
            for target in iterator:
                if not isinstance(target, Sequence):
                    raise ValueError(
                        f'Inputs to multi-qubit gates must be Sequence[Qid].'
                        f' Type: {type(target)}'
                    )
                if not all(isinstance(x, raw_types.Qid) for x in target):
                    raise ValueError(f'All values in sequence should be Qids, but got {target}')
                if len(target) != self._num_qubits_():
                    raise ValueError(f'Expected {self._num_qubits_()} qubits, got {target}')
                operations.append(self.on(*target))
            return operations

        for target in targets:
            if isinstance(target, raw_types.Qid):
                operations.append(self.on(target))
            elif isinstance(target, Iterable) and not isinstance(target, str):
                operations.extend(self.on_each(*target))
            else:
                raise ValueError(
                    f'Gate was called with type different than Qid. Type: {type(target)}'
                )
        return operations


class SingleQubitGate(SupportsOnEachGate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly one qubit."""

    def _num_qubits_(self) -> int:
        return 1


class TwoQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly two qubits."""

    def _num_qubits_(self) -> int:
        return 2


class ThreeQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly three qubits."""

    def _num_qubits_(self) -> int:
        return 3
