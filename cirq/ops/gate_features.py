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

from cirq.ops import op_tree, raw_types


class InterchangeableQubitsGate(metaclass=abc.ABCMeta):
    """Indicates operations should be equal under some qubit permutations."""

    def qubit_index_to_equivalence_group_key(self, index: int) -> int:
        """Returns a key that differs between non-interchangeable qubits."""
        return 0


class SingleQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly one qubit."""
    def num_qubits(self) -> int:
        return 1

    def on_each(self, *targets: raw_types.Qid) -> op_tree.OP_TREE:
        """Returns a list of operations apply this gate to each of the targets.

        Args:
            *targets: The qubits to apply this gate to.

        Returns:
            Operations applying this gate to the target qubits.

        Raises:
            ValueError if targets are not instances of Qid.
        """
        return [self.on(target) for target in targets]


class TwoQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly two qubits."""
    def num_qubits(self) -> int:
        return 2


class ThreeQubitGate(raw_types.Gate, metaclass=abc.ABCMeta):
    """A gate that must be applied to exactly three qubits."""
    def num_qubits(self) -> int:
        return 3
