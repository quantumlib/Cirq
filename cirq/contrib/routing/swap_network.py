# Copyright 2019 The Cirq Developers
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

from typing import Dict, Iterable, TYPE_CHECKING

from cirq import ops
import cirq.contrib.acquaintance as cca

if TYPE_CHECKING:
    import cirq


class SwapNetwork:
    """A swap network, i.e. a circuit containing logical operations and swaps
    together with an initial mapping of physical to logical qubits.

    Only instances of PermutationGate are considered as changing the mapping
    between logical and physical qubits. This is, in part, to distinguish
    between such gates and those with the same unitaries but that are part of
    the logical circuit to be routed.

    Args:
        circuit: The circuit.
        initial_mapping: The initial mapping from physical to logical qubits.
    """

    def __init__(self, circuit: 'cirq.Circuit',
                 initial_mapping: Dict['cirq.Qid', 'cirq.Qid']) -> None:
        if not all(
                isinstance(i, ops.Qid)
                for I in initial_mapping.items()
                for i in I):
            raise ValueError('Mapping must be from Qids to Qids.')
        self.circuit = circuit
        self.initial_mapping = initial_mapping

    def final_mapping(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        mapping = dict(self.initial_mapping)
        cca.update_mapping(mapping, self.circuit.all_operations())
        return mapping

    def get_logical_operations(self) -> Iterable['cirq.Operation']:
        return cca.get_logical_operations(self.circuit.all_operations(),
                                          self.initial_mapping)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return (self.circuit == other.circuit and
                self.initial_mapping == other.initial_mapping)

    @property
    def device(self) -> 'cirq.Device':
        return self.circuit.device

    def __str__(self) -> str:
        circuit = self.circuit.copy()
        cca.display_mapping(circuit, self.initial_mapping)
        return str(circuit)
