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

from typing import (Any, Dict, ItemsView, Iterable, Iterator, KeysView, Mapping,
                    Optional, Tuple, ValuesView)

from cirq import ops

from cirq.contrib.paulistring import Pauli, CliffordGate, PauliInteractionGate


class PauliString:
    def __init__(self,
                 qubit_pauli_map: Mapping[ops.QubitId, Pauli],
                 inverted: bool = False) -> None:
        self._qubit_pauli_map = dict(qubit_pauli_map)
        self.inverted = inverted

    @staticmethod
    def from_single(qubit: ops.QubitId, pauli: Pauli) -> 'PauliString':
        """Creates a PauliString with a single qubit."""
        return PauliString({qubit: pauli})

    def _eq_tuple(self) -> Tuple[Any, ...]:
        return (PauliString,
                self._qubit_pauli_map,
                self.inverted)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._eq_tuple() == other._eq_tuple()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((PauliString, self.inverted, frozenset(self.items())))

    def __getitem__(self, key: ops.QubitId) -> Pauli:
        return self._qubit_pauli_map[key]

    def get(self, key: ops.QubitId, default: Optional[Pauli] = None
            ) -> Optional[Pauli]:
        return self._qubit_pauli_map.get(key, default)

    def __contains__(self, key: ops.QubitId) -> bool:
        return key in self._qubit_pauli_map

    def keys(self) -> KeysView[ops.QubitId]:
        return self._qubit_pauli_map.keys()

    def qubits(self) -> KeysView[ops.QubitId]:
        return self.keys()

    def values(self) -> ValuesView[Pauli]:
        return self._qubit_pauli_map.values()

    def items(self) -> ItemsView:
        return self._qubit_pauli_map.items()

    def __iter__(self) -> Iterator[ops.QubitId]:
        return iter(self._qubit_pauli_map.keys())

    def __len__(self) -> int:
        return len(self._qubit_pauli_map)

    def __repr__(self):
        map_str = ', '.join(('{!r}: {!r}'.format(qubit, self[qubit])
                             for qubit in
                                ops.QubitOrder.DEFAULT.order_for(self)))
        return 'PauliString({{{}}}, {})'.format(map_str,
                                                self.inverted)

    def __str__(self):
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(self.qubits())
        return '{{{}, {}}}'.format('+-'[self.inverted],
                                   ', '.join(('{!s}:{!s}'.format(q, self[q])
                                             for q in ordered_qubits)))

    def zip_items(self, other: 'PauliString'
                  ) -> Iterator[Tuple[ops.QubitId, Tuple[Pauli, Pauli]]]:
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield qubit, (pauli0, other[qubit])

    def zip_paulis(self, other: 'PauliString') -> Iterator[Tuple[Pauli, Pauli]]:
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield pauli0, other[qubit]

    def commutes_with_string(self, other: 'PauliString') -> bool:
        return sum((not p0.commutes_with(p1)
                    for p0, p1 in self.zip_paulis(other))
                   ) % 2 == 0

    def inverse(self) -> 'PauliString':
        return PauliString(self._qubit_pauli_map, not self.inverted)

    def map_qubits(self, qubit_map: Dict[ops.QubitId, ops.QubitId]
                   ) -> 'PauliString':
        new_qubit_pauli_map = {qubit_map[qubit]: pauli
                               for qubit, pauli in self.items()}
        return PauliString(new_qubit_pauli_map, self.inverted)

    def pass_operations_over(self,
                             ops: Iterable[ops.Operation],
                             after_to_before: bool = False) -> 'PauliString':
        """Return a new PauliString such that the circuits
            --op--...--op--self-- and --output--op--...--op--
        are equivalent up to global phase.

        Args:
            op: The operation to move
            after_to_before: If true, passes op over the other direction such
                that the circuits
                    --self--op--...--op-- and --op--...--op--output--
                are equivalent up to global phase.
        """
        pauli_map = dict(self._qubit_pauli_map)
        inv = self.inverted
        for op in ops:
            inv ^= self._pass_operation_over(pauli_map, op, after_to_before)
        return PauliString(pauli_map, inv)

    @classmethod
    def _pass_operation_over(cls,
                             pauli_map: Dict[ops.QubitId, Pauli],
                             op: ops.Operation,
                             after_to_before: bool = False) -> bool:
        if isinstance(op.gate, CliffordGate):
            return cls._pass_single_clifford_gate_over(
                        pauli_map, op.gate, op.qubits[0],
                        after_to_before=after_to_before)
        elif isinstance(op.gate, PauliInteractionGate):
            return cls._pass_pauli_interaction_gate_over(
                        pauli_map, op.gate, op.qubits[0], op.qubits[1],
                        after_to_before=after_to_before)
        else:
            raise TypeError('Unsupported gate type: {}'.format(
                            type(op.gate).__name__))

    @staticmethod
    def _pass_single_clifford_gate_over(pauli_map: Dict[ops.QubitId, Pauli],
                                        gate: CliffordGate,
                                        qubit: ops.QubitId,
                                        after_to_before: bool = False) -> bool:
        if qubit not in pauli_map:
            return False
        if not after_to_before:
            gate = gate.inverse()
        pauli, inv = gate.transform(pauli_map[qubit])
        pauli_map[qubit] = pauli
        return inv

    @staticmethod
    def _pass_pauli_interaction_gate_over(pauli_map: Dict[ops.QubitId, Pauli],
                                          gate: PauliInteractionGate,
                                          qubit0: ops.QubitId,
                                          qubit1: ops.QubitId,
                                          after_to_before: bool = False
                                          ) -> bool:
        def merge_and_kickback(qubit, pauli_left, pauli_right, inv):
            if pauli_left is None or pauli_right is None:
                pauli_map[qubit] = pauli_left or pauli_right
                return 0
            elif pauli_left == pauli_right:
                del pauli_map[qubit]
                return 0
            else:
                pauli_map[qubit] = pauli_left.third(pauli_right)
                if (pauli_left < pauli_right) ^ after_to_before:
                    return inv * 2 + 1
                else:
                    return inv * 2 - 1
        quarter_kickback = 0
        if (qubit0 in pauli_map
            and not pauli_map[qubit0].commutes_with(gate.pauli0)):
            quarter_kickback += merge_and_kickback(qubit1,
                                                   gate.pauli1,
                                                   pauli_map.get(qubit1),
                                                   gate.invert1)
        if (qubit1 in pauli_map
            and not pauli_map[qubit1].commutes_with(gate.pauli1)):
            quarter_kickback += merge_and_kickback(qubit0,
                                                   pauli_map.get(qubit0),
                                                   gate.pauli0,
                                                   gate.invert0)
        assert quarter_kickback % 2 == 0, ('Impossible condition.  '
            'quarter_kickback is either incremented twice or never.')
        return (quarter_kickback % 4 == 2)
