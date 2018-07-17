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

from typing import Dict, Sequence, Any

from cirq import abc
from cirq.ops import CompositeGate, Gate, QubitId, OP_TREE, SWAP

class PermutationGate(Gate, CompositeGate, metaclass=abc.ABCMeta):
    """Permutation gate."""

    def __init__(self, swap_gate: Gate=SWAP) -> None:
        self.swap_gate = swap_gate

    @abc.abstractmethod
    def permutation(self, qubit_count: int) -> Dict[int, int]:
        """permutation = {i: s[i]} indicates that the i-th qubit is mapped to
        the s[i]-th qubit."""
        pass

    def update_mapping(self,
                       mapping: Dict[Any, Any],
                       elements: Sequence[Any]
                       ) -> None:
        n_elements = len(elements)
        permutation = self.permutation(n_elements)
        permuted_elements = [elements[permutation.get(i, i)]
                             for i in range(n_elements)]
        for i, e in enumerate(elements):
            mapping[e] = permuted_elements[i]

class SwapPermutationGate(PermutationGate):
    """Generic swap gate."""

    def permutation(self, qubit_count: int) -> Dict[int, int]:
        return {0: 1, 1: 0}

class LinearPermutationGate(PermutationGate):
    """A permutation gate that decomposes a given permutation using a linear
        sorting network."""

    def __init__(self,
                 permutation: Dict[int, int],
                 swap_gate: Gate=SWAP
                 ) -> None:
        self._permutation = permutation
        self.swap_gate = swap_gate

    def permutation(self, qubit_count: int) -> Dict[int, int]:
        return self._permutation

    def default_decompose(self, qubits: Sequence[QubitId]) -> OP_TREE:
        swap_gate = SwapPermutationGate(self.swap_gate)
        n_qubits = len(qubits)
        mapping = {i: self._permutation.get(i, i) for i in range(n_qubits)}
        for layer_index in range(n_qubits):
            for i in range(layer_index % 2, n_qubits - 1, 2):
                if mapping[i] > mapping[i + 1]:
                    yield swap_gate(*qubits[i:i+2])
                    mapping[i], mapping[i+1] = mapping[i+1], mapping[i]
