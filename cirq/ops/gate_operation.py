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

from typing import (Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Type,
                    TypeVar, Union, TYPE_CHECKING)

import numpy as np

from cirq import protocols, value
from cirq._compat import deprecated
from cirq.ops import raw_types, gate_features
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq


@value.value_equality(approximate=True)
class GateOperation(raw_types.Operation):
    """An application of a gate to a sequence of qubits."""

    def __init__(self, gate: 'cirq.Gate', qubits: Sequence['cirq.Qid']) -> None:
        """
        Args:
            gate: The gate to apply.
            qubits: The qubits to operate on.
        """
        gate.validate_args(qubits)
        self._gate = gate
        self._qubits = tuple(qubits)

    @property
    def gate(self) -> 'cirq.Gate':
        """The gate applied by the operation."""
        return self._gate

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """The qubits targeted by the operation."""
        return self._qubits

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'cirq.Operation':
        return self.gate.on(*new_qubits)

    def with_gate(self, new_gate: 'cirq.Gate') -> 'cirq.Operation':
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

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['gate', 'qubits'])

    def _group_interchangeable_qubits(
            self
    ) -> Tuple[Union['cirq.Qid', Tuple[int, FrozenSet['cirq.Qid']]], ...]:

        if not isinstance(self.gate, gate_features.InterchangeableQubitsGate):
            return self.qubits

        groups: Dict[int, List['cirq.Qid']] = {}
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

    def _decompose_(self) -> 'cirq.OP_TREE':
        return protocols.decompose_once_with_qubits(self.gate,
                                                    self.qubits,
                                                    NotImplemented)

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return protocols.pauli_expansion(self.gate)

    def _apply_unitary_(self, args: 'protocols.ApplyUnitaryArgs'
                       ) -> Union[np.ndarray, None, NotImplementedType]:
        return protocols.apply_unitary(self.gate, args, default=None)

    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.gate)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        return protocols.unitary(self.gate, default=None)

    def _commutes_(self, other: Any,
                   atol: float) -> Union[bool, NotImplementedType, None]:
        return self.gate._commutes_on_qids_(self.qubits, other, atol=atol)

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

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'
                              ) -> 'cirq.CircuitDiagramInfo':
        return protocols.circuit_diagram_info(self.gate,
                                              args,
                                              NotImplemented)

    def _decompose_into_clifford_(self):
        sub = getattr(self.gate, '_decompose_into_clifford_with_qubits_', None)
        if sub is None:
            return NotImplemented
        return sub(self.qubits)

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

    def __pow__(self, exponent: Any) -> 'cirq.Operation':
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

    def __mul__(self, other: Any) -> Any:
        result = self.gate._mul_with_qubits(self._qubits, other)

        # python will not auto-attempt the reverse order for same type.
        if result is NotImplemented and isinstance(other, GateOperation):
            return other.__rmul__(self)

        return result

    def __rmul__(self, other: Any) -> Any:
        return self.gate._rmul_with_qubits(self._qubits, other)

    def _qasm_(self, args: 'protocols.QasmArgs') -> Optional[str]:
        return protocols.qasm(self.gate,
                              args=args,
                              qubits=self.qubits,
                              default=None)

    def _equal_up_to_global_phase_(self,
                                   other: Any,
                                   atol: Union[int, float] = 1e-8
                                  ) -> Union[NotImplementedType, bool]:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.qubits != other.qubits:
            return False
        return protocols.equal_up_to_global_phase(self.gate,
                                                  other.gate,
                                                  atol=atol)


TV = TypeVar('TV', bound=raw_types.Gate)


@deprecated(deadline='v0.8.0',
            fix='use: `op.gate if isinstance(op.gate, gate_type) else None`')
def op_gate_of_type(op: Any, gate_type: Type[TV]) -> Optional[TV]:
    """Returns gate of given type, if op has that gate otherwise None."""
    gate = getattr(op, 'gate', None)
    return gate if isinstance(gate, gate_type) else None
