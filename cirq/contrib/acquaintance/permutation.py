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
from cirq.ops import CompositeGate, Gate

class PermutationGate(Gate, CompositeGate, metaclass=abc.ABCMeta):
    """Permutation gate."""

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


class LinearPermutationGate(PermutationGate):
    """A permutation gate that decomposes a given permutation using a linear
        sorting network."""
    pass
