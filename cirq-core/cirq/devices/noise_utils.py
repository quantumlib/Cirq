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

from typing import TYPE_CHECKING, Any, Dict, Tuple, Type, Union

from cirq import ops, protocols, value
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


# Tag for gates to which noise must be applied.
PHYSICAL_GATE_TAG = 'physical_gate'


@value.value_equality(distinct_child_types=True)
class OpIdentifier:
    """Identifies an operation by gate and (optionally) target qubits."""

    def __init__(self, gate_type: Type['cirq.Gate'], *qubits: 'cirq.Qid'):
        self._gate_type = gate_type
        self._gate_family = ops.GateFamily(gate_type)
        self._qubits: Tuple['cirq.Qid', ...] = tuple(qubits)

    @property
    def gate_type(self) -> Type['cirq.Gate']:
        # set to a type during initialization, never modified
        return self._gate_type

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self._qubits

    def _predicate(self, *args, **kwargs):
        return self._gate_family._predicate(*args, **kwargs)

    def is_proper_subtype_of(self, op_id: 'OpIdentifier'):
        """Returns true if this is contained within op_id, but not equal to it.

        If this returns true, (x in self) implies (x in op_id), but the reverse
        implication does not hold. op_id must be more general than self (either
        by accepting any qubits or having a more general gate type) for this
        to return true.
        """
        more_specific_qubits = self.qubits and not op_id.qubits
        more_specific_gate = self.gate_type != op_id.gate_type and issubclass(
            self.gate_type, op_id.gate_type
        )
        if more_specific_qubits:
            return more_specific_gate or self.gate_type == op_id.gate_type
        elif more_specific_gate:
            return more_specific_qubits or self.qubits == op_id.qubits
        else:
            return False

    def __contains__(self, item: Union[ops.Gate, ops.Operation]) -> bool:
        if isinstance(item, ops.Gate):
            return (not self._qubits) and self._predicate(item)
        return (
            (not self.qubits or (item.qubits == self._qubits))
            and item.gate is not None
            and self._predicate(item.gate)
        )

    def __str__(self):
        return f'{self.gate_type}{self.qubits}'

    def __repr__(self) -> str:
        qubits = ', '.join(map(repr, self.qubits))
        return f'cirq.devices.noise_utils.OpIdentifier({proper_repr(self.gate_type)}, {qubits})'

    def _value_equality_values_(self) -> Any:
        return (self.gate_type, self.qubits)

    def _json_dict_(self) -> Dict[str, Any]:
        if hasattr(self.gate_type, '__name__'):
            return {'gate_type': protocols.json_cirq_type(self._gate_type), 'qubits': self._qubits}
        return {'gate_type': self._gate_type, 'qubits': self._qubits}

    @classmethod
    def _from_json_dict_(cls, gate_type, qubits, **kwargs) -> 'OpIdentifier':
        if isinstance(gate_type, str):
            gate_type = protocols.cirq_type_from_json(gate_type)
        return cls(gate_type, *qubits)
