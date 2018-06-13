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

from cirq.contrib.rearrange.axis import IdentAxis, Z_AXIS
from cirq.contrib.rearrange.clifford_pauli_gate import CliffordPauliGate
from cirq.contrib.rearrange.interaction_gate import InteractionGate
from cirq.contrib.rearrange.pauli_string import PauliString


class NonCliffordGate(ops.CompositeGate, ops.ExtrapolatableGate, ops.TextDiagrammableGate
                      #ops.InterchangeableQubitsGate,  # Sometimes
                      ):
    def __init__(self, pauli_string, half_turns: float):
        self.pauli_string = pauli_string
        self.half_turns = half_turns

    @classmethod
    def op_from_single(cls, axis, half_turns: float, qubit: ops.QubitId):
        if axis.is_negative():
            axis = axis.negative()
            half_turns = -half_turns
        pauli_string = PauliString({qubit: axis})
        gate = NonCliffordGate(pauli_string, half_turns)
        return gate(qubit)

    def updated_op(self):
        return self(*self.pauli_string.qubits())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.pauli_string == self.pauli_string and self.half_turns == other.half_turns

    def __hash__(self):
        return hash((NonCliffordGate, self.pauli_string, self.half_turns))

    def _with_half_turns(self, half_turns: float):
        return NonCliffordGate(self.pauli_string.copy(), half_turns=half_turns)

    def extrapolate_effect(self, factor: float):
        return self._with_half_turns(self.half_turns * factor)

    def default_decompose(self, qubits_ignore):
        qubits_axes = tuple(((qubit, axis) for qubit, axis in self.pauli_string.qubit_axis_pairs()
                             if not isinstance(axis, IdentAxis)))
        neg = False
        left_ops = []
        right_ops = []
        cnot_ops = []
        rotation_op = ops.Z(qubits_axes[-1][0]) ** self.half_turns
        for qubit, axis in qubits_axes:
            neg ^= axis.is_negative()
            axis = axis.abs()
            if axis == Z_AXIS:
                pass
            else:
                gate_left, gate_right = CliffordPauliGate.axis_change_gates(axis, Z_AXIS)
                left_ops.append(gate_left(qubit))
                right_ops.append(gate_right(qubit))
        if neg:
            rotation_op = rotation_op ** -1
        for i in range(len(qubits_axes) - 1):
            qubit0, qubit1 = qubits_axes[i][0], qubits_axes[i+1][0]
            cnot_ops.append(ops.CNOT(qubit0, qubit1))
        return left_ops + cnot_ops + [rotation_op] + cnot_ops[::-1] + right_ops[::-1]

    def text_diagram_wire_symbols(self, qubit_count=None, use_unicode_characters=True, precision=3):
        # TODO: qubit order
        return tuple(('[{!s}??]'.format(axis) for qubit, axis in self.pauli_string.qubit_axis_pairs()))

    def text_diagram_exponent(self):
        return self.half_turns
