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

from cirq import ops

from cirq.contrib.rearrange.axis import IdentAxis, I_AXIS
from cirq.contrib.rearrange.clifford_pauli_gate import CliffordPauliGate
from cirq.contrib.rearrange.interaction_gate import InteractionGate


class PauliString:
    def __init__(self, qubit_axis_map):
        self.qubit_axis_map = qubit_axis_map

    @classmethod
    def from_tuples(cls, qubits, axes):
        return cls(dict(zip(qubits, axes)))

    def copy(self):
        return PauliString(dict(self.qubit_axis_map))

    def get_axis(self, qubit, default=I_AXIS):
        return self.qubit_axis_map.get(qubit, default)

    def set_axis(self, qubit, axis):
        self.qubit_axis_map[qubit] = axis

    def add_axis_left(self, qubit, axis):
        old_axis = self.get_axis(qubit)
        new_axis = axis.merge_rotations(old_axis)
        self.set_axis(qubit, new_axis)

    def add_axis_right(self, qubit, axis):
        old_axis = self.get_axis(qubit)
        new_axis = old_axis.merge_rotations(axis)
        self.set_axis(qubit, new_axis)

    def clean_axes(self):
        extra = [qubit for qubit, axis in self.qubit_axis_map.items() if isinstance(axis, IdentAxis)]
        for qubit in extra:
            del self.qubit_axis_map[qubit]

    def commutes_with(self, other):
        anti_total = 0
        for qubit, axis0 in self.qubit_axis_map.items():
            axis1 = other.get_axis(qubit)
            anti_total += not axis0.commutes_with(axis1)
        return anti_total % 2 == 0

    def qubits(self):
        return self.qubit_axis_map.keys()

    def qubit_axis_pairs(self):
        return self.qubit_axis_map.items()

    def pass_op_over(self, op: ops.Operation):
        self.pass_gate_over(op.gate, *op.qubits)

    def pass_gate_over(self, gate, *qubits):
        if len(qubits) == 1:
            self.pass_single_gate_over(gate, qubits[0])
        else:
            self.pass_interaction_gate_over(gate, *qubits)

    def pass_single_gate_over(self, gate: CliffordPauliGate, qubit: ops.QubitId):
        """Modifies the instance."""
        if qubit not in self.qubit_axis_map:
            return
        new_axis = gate.axis_after_passing_over(self.qubit_axis_map[qubit])
        self.qubit_axis_map[qubit] = new_axis

    def pass_interaction_gate_over(self, gate: InteractionGate, qubit0: ops.QubitId, qubit1: ops.QubitId):
        """Modifies the instance."""
        if not self.get_axis(qubit0).commutes_with(gate.axis0):
            self.add_axis_left(qubit1, gate.axis1)
        if not self.get_axis(qubit1).commutes_with(gate.axis1):
            self.add_axis_left(qubit0, gate.axis0)
