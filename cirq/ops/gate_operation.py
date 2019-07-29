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

"""Basic types defining qubits, gates, and operations."""

from typing import (
    Any, FrozenSet, Optional, Sequence, Tuple, Type, TypeVar, TYPE_CHECKING,
    Union
)

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types, gate_features, op_tree
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    # pylint: disable=unused-import
    from typing import Dict, List


@value.value_equality(approximate=True)
class GateOperation(raw_types.Operation):
    """An application of a gate to a sequence of qubits."""

    def __init__(self,
                 gate: raw_types.Gate,
                 qubits: Sequence[raw_types.Qid]) -> None:
        """
        Args:
            gate: The gate to apply.
            qubits: The qubits to operate on.
        """
        gate.validate_args(qubits)
        self._gate = gate
        self._qubits = tuple(qubits)

    @property
    def gate(self) -> raw_types.Gate:
        """The gate applied by the operation."""
        return self._gate

    @property
    def qubits(self) -> Tuple[raw_types.Qid, ...]:
        """The qubits targeted by the operation."""
        return self._qubits

    def with_qubits(self, *new_qubits: raw_types.Qid) -> 'raw_types.Operation':
        return self.gate.on(*new_qubits)

    def with_gate(self, new_gate: raw_types.Gate) -> 'raw_types.Operation':
        return new_gate.on(*self.qubits)

    def __repr__(self):
        # Abbreviate when possible.
        if self == self.gate.on(*self.qubits):
            return '{!r}.on({})'.format(
                self.gate,
                ', '.join(repr(q) for q in self.qubits))

        return 'cirq.GateOperation(gate={!r}, qubits={!r})'.format(
            self.gate,
            list(self.qubits))

    def __str__(self):
        return '{}({})'.format(self.gate,
                               ', '.join(str(e) for e in self.qubits))

    def _group_interchangeable_qubits(self) -> Tuple[
            Union[raw_types.Qid,
                  Tuple[int, FrozenSet[raw_types.Qid]]],
            ...]:

        if not isinstance(self.gate, gate_features.InterchangeableQubitsGate):
            return self.qubits

        groups = {}  # type: Dict[int, List[raw_types.Qid]]
        for i, q in enumerate(self.qubits):
            k = self.gate.qubit_index_to_equivalence_group_key(i)
            if k not in groups:
                groups[k] = []
            groups[k].append(q)
        return tuple(sorted((k, frozenset(v)) for k, v in groups.items()))

    def _value_equality_values_(self):
        return self.gate, self._group_interchangeable_qubits()

    def _qid_shape_(self):
        return protocols.qid_shape(self.gate)

    def _num_qubits_(self):
        return len(self._qubits)

    def _decompose_(self) -> op_tree.OP_TREE:
        return protocols.decompose_once_with_qubits(self.gate,
                                                    self.qubits,
                                                    NotImplemented)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return protocols.pauli_expansion(self.gate)

    def _apply_unitary_(self, args: protocols.ApplyUnitaryArgs
                        ) -> Union[np.ndarray, None, NotImplementedType]:
        return protocols.apply_unitary(self.gate, args, default=None)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        return protocols.unitary(self.gate, default=None)

    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.gate)

    def _mixture_(self) -> Sequence[Tuple[float, Any]]:
        return protocols.mixture(self.gate, NotImplemented)

    def _has_channel_(self) -> bool:
        return protocols.has_channel(self.gate)

    def _channel_(self) -> Union[Tuple[np.ndarray], NotImplementedType]:
        return protocols.channel(self.gate, NotImplemented)

    def _measurement_key_(self) -> str:
        return protocols.measurement_key(self.gate, NotImplemented)

    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.gate)

    def _resolve_parameters_(self, resolver):
        resolved_gate = protocols.resolve_parameters(self.gate, resolver)
        return GateOperation(resolved_gate, self._qubits)

    def _circuit_diagram_info_(self,
                               args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.circuit_diagram_info(self.gate,
                                              args,
                                              NotImplemented)

    def _trace_distance_bound_(self) -> float:
        return protocols.trace_distance_bound(self.gate)

    def _phase_by_(self, phase_turns: float,
                   qubit_index: int) -> 'GateOperation':
        phased_gate = protocols.phase_by(self.gate,
                                         phase_turns,
                                         qubit_index,
                                         default=None)
        if phased_gate is None:
            return NotImplemented
        return GateOperation(phased_gate, self._qubits)

    def __pow__(self, exponent: Any) -> 'raw_types.Operation':
        """Raise gate to a power, then reapply to the same qubits.

        Only works if the gate implements cirq.ExtrapolatableEffect.
        For extrapolatable gate G this means the following two are equivalent:

            (G ** 1.5)(qubit)  or  G(qubit) ** 1.5

        Args:
            exponent: The amount to scale the gate's effect by.

        Returns:
            A new operation on the same qubits with the scaled gate.
        """
        new_gate = protocols.pow(self.gate,
                                 exponent,
                                 NotImplemented)
        if new_gate is NotImplemented:
            return NotImplemented
        return self.with_gate(new_gate)

    def _qasm_(self, args: protocols.QasmArgs) -> Optional[str]:
        return protocols.qasm(self.gate,
                              args=args,
                              qubits=self.qubits,
                              default=None)


TV = TypeVar('TV', bound=raw_types.Gate)


def op_gate_of_type(op: raw_types.Operation,
                    gate_type: Type[TV]) -> Optional[TV]:
    """Returns gate of given type, if op has that gate otherwise None."""
    if isinstance(op, GateOperation) and isinstance(op.gate, gate_type):
        return op.gate
    return None


def op_gate_isinstance(op: raw_types.Operation, gate_type: Type[TV]) -> bool:
    """Returns True if op has that gate type otherwise False."""
    if isinstance(op, GateOperation):
        return isinstance(op.gate, gate_type)
    return False
