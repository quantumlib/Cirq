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

from typing import (Dict, ItemsView, Iterable, Iterator, KeysView, Mapping,
                    Tuple, TypeVar, Union, ValuesView, overload)

from cirq import value
from cirq.ops import (
    raw_types, gate_operation, common_gates, op_tree
)
from cirq.ops.pauli import Pauli
from cirq.ops.clifford_gate import SingleQubitCliffordGate
from cirq.ops.pauli_interaction_gate import PauliInteractionGate

TDefault = TypeVar('TDefault')


@value.value_equality
class PauliString(raw_types.Operation):
    def __init__(self,
                 qubit_pauli_map: Mapping[raw_types.QubitId, Pauli],
                 negated: bool = False) -> None:
        self._qubit_pauli_map = dict(qubit_pauli_map)
        self.negated = negated

    @staticmethod
    def from_single(qubit: raw_types.QubitId, pauli: Pauli) -> 'PauliString':
        """Creates a PauliString with a single qubit."""
        return PauliString({qubit: pauli})

    def _value_equality_values_(self):
        return (frozenset(self._qubit_pauli_map.items()),
                self.negated)

    def equal_up_to_sign(self, other: 'PauliString') -> bool:
        return self._qubit_pauli_map == other._qubit_pauli_map

    def __getitem__(self, key: raw_types.QubitId) -> Pauli:
        return self._qubit_pauli_map[key]

    # pylint: disable=function-redefined
    @overload
    def get(self, key: raw_types.QubitId) -> Pauli:
        pass

    @overload
    def get(self, key: raw_types.QubitId, default: TDefault
            ) -> Union[Pauli, TDefault]:
        pass

    def get(self, key: raw_types.QubitId, default=None):
        return self._qubit_pauli_map.get(key, default)
    # pylint: enable=function-redefined

    def __contains__(self, key: raw_types.QubitId) -> bool:
        return key in self._qubit_pauli_map

    def keys(self) -> KeysView[raw_types.QubitId]:
        return self._qubit_pauli_map.keys()

    @property
    def qubits(self) -> Tuple[raw_types.QubitId, ...]:
        return tuple(sorted(self.keys()))

    def with_qubits(self, *new_qubits: raw_types.QubitId) -> 'PauliString':
        return PauliString(dict(zip(new_qubits,
                                    (self[q] for q in self.qubits))),
                           self.negated)

    def values(self) -> ValuesView[Pauli]:
        return self._qubit_pauli_map.values()

    def items(self) -> ItemsView:
        return self._qubit_pauli_map.items()

    def __iter__(self) -> Iterator[raw_types.QubitId]:
        return iter(self._qubit_pauli_map.keys())

    def __len__(self) -> int:
        return len(self._qubit_pauli_map)

    def __repr__(self):
        map_str = ', '.join(('{!r}: {!r}'.format(qubit, self[qubit])
                             for qubit in sorted(self.qubits)))
        return 'cirq.PauliString({{{}}}, {})'.format(map_str, self.negated)

    def __str__(self):
        ordered_qubits = sorted(self.qubits)
        return '{{{}, {}}}'.format('+-'[self.negated],
                                   ', '.join(('{!s}:{!s}'.format(q, self[q])
                                             for q in ordered_qubits)))

    def zip_items(self, other: 'PauliString'
                  ) -> Iterator[Tuple[raw_types.QubitId, Tuple[Pauli, Pauli]]]:
        for qubit, pauli0 in self.items():
            if qubit in other:
                yield qubit, (pauli0, other[qubit])

    def zip_paulis(self, other: 'PauliString') -> Iterator[Tuple[Pauli, Pauli]]:
        return (paulis for qubit, paulis in self.zip_items(other))

    def commutes_with(self, other: 'PauliString') -> bool:
        return sum(not p0.commutes_with(p1)
                   for p0, p1 in self.zip_paulis(other)
                   ) % 2 == 0

    def negate(self) -> 'PauliString':
        return PauliString(self._qubit_pauli_map, not self.negated)

    def __neg__(self) -> 'PauliString':
        return self.negate()

    def __pos__(self) -> 'PauliString':
        return self

    def map_qubits(self, qubit_map: Dict[raw_types.QubitId, raw_types.QubitId]
                   ) -> 'PauliString':
        new_qubit_pauli_map = {qubit_map[qubit]: pauli
                               for qubit, pauli in self.items()}
        return PauliString(new_qubit_pauli_map, self.negated)

    def to_z_basis_ops(self) -> op_tree.OP_TREE:
        """Returns operations to convert the qubits to the computational basis.
        """
        for qubit, pauli in self.items():
            yield SingleQubitCliffordGate.from_single_map(
                {pauli: (Pauli.Z, False)})(qubit)

    def pass_operations_over(self,
                             ops: Iterable[raw_types.Operation],
                             after_to_before: bool = False) -> 'PauliString':
        """Determines how the Pauli string changes when conjugated by Cliffords.

        The output and input pauli strings are related by a circuit equivalence.
        In particular, this circuit:

            ───ops───INPUT_PAULI_STRING───

        will be equivalent to this circuit:

            ───OUTPUT_PAULI_STRING───ops───

        up to global phase (assuming `after_to_before` is not set).

        If ops together have matrix C, the Pauli string has matrix P, and the
        output Pauli string has matrix P', then P' == C^-1 P C up to
        global phase.

        Setting `after_to_before` inverts the relationship, so that the output
        is the input and the input is the output. Equivalently, it inverts C.

        Args:
            ops: The operations to move over the string.
            after_to_before: Determines whether the operations start after the
                pauli string, instead of before (and so are moving in the
                opposite direction).
        """
        pauli_map = dict(self._qubit_pauli_map)
        inv = self.negated
        for op in ops:
            if not set(op.qubits) & set(pauli_map.keys()):
                # op operates on an independent set of qubits from the Pauli
                # string.  The order can be switched with no change no matter
                # what op is.
                continue
            inv ^= PauliString._pass_operation_over(pauli_map,
                                                    op,
                                                    after_to_before)
        return PauliString(pauli_map, inv)

    @staticmethod
    def _pass_operation_over(pauli_map: Dict[raw_types.QubitId, Pauli],
                             op: raw_types.Operation,
                             after_to_before: bool = False) -> bool:
        if isinstance(op, gate_operation.GateOperation):
            gate = op.gate
            if isinstance(gate, SingleQubitCliffordGate):
                return PauliString._pass_single_clifford_gate_over(
                    pauli_map, gate, op.qubits[0],
                    after_to_before=after_to_before)
            if isinstance(gate, common_gates.CZPowGate):
                gate = PauliInteractionGate.CZ
            if isinstance(gate, PauliInteractionGate):
                return PauliString._pass_pauli_interaction_gate_over(
                    pauli_map, gate, op.qubits[0], op.qubits[1],
                    after_to_before=after_to_before)
        raise TypeError('Unsupported operation: {!r}'.format(op))

    @staticmethod
    def _pass_single_clifford_gate_over(pauli_map: Dict[raw_types.QubitId,
                                                        Pauli],
                                        gate: SingleQubitCliffordGate,
                                        qubit: raw_types.QubitId,
                                        after_to_before: bool = False) -> bool:
        if qubit not in pauli_map:
            return False
        if not after_to_before:
            gate **= -1
        pauli, inv = gate.transform(pauli_map[qubit])
        pauli_map[qubit] = pauli
        return inv

    @staticmethod
    def _pass_pauli_interaction_gate_over(pauli_map: Dict[raw_types.QubitId,
                                                          Pauli],
                                          gate: PauliInteractionGate,
                                          qubit0: raw_types.QubitId,
                                          qubit1: raw_types.QubitId,
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
