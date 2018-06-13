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

from cirq.contrib.rearrange.axis import AxisAbc, IdentAxis, Axis


class CliffordPauliGate(ops.CompositeGate, ops.ReversibleGate, ops.TextDiagrammableGate):
    def __init__(self, axis: AxisAbc, is_sqrt):
        self.axis = axis
        self._is_sqrt = is_sqrt

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.axis == other.axis and self._is_sqrt == other._is_sqrt

    def __ne__(self, other):
        return not self == other

    def __hash__(self, other):
        return hash((CliffordPauliGate, self.axis, self._is_sqrt))

    def __repr__(self):
        return 'CliffordPauliGate({}, {})'.format(self.axis, self._is_sqrt)

    def __str__(self):
        return str(self.axis) + ('^0.5' if self._is_sqrt else '')

    def is_sqrt_pauli(self):
        return self._is_sqrt

    def commutes_with(self, other):
        return self.axis.commutes_with(other.axis)

    def _with_axis(self, new_axis):
        return CliffordPauliGate(new_axis, self._is_sqrt)

    def default_decompose(self, qubits):
        q, = qubits
        if isinstance(self.axis, IdentAxis):
            return ()
        else:
            sign = 1 if self.axis.axis_i < 3 else -1
            factor = 0.5 if self._is_sqrt else 1
            gate = (ops.X, ops.Y, ops.Z)[self.axis.axis_i % 3]
            return ((gate ** (sign * factor))(q),)

    def inverse(self):
        return CliffordPauliGate(self.axis.negative(), self._is_sqrt)

    def text_diagram_wire_symbols(self, qubit_count=None, use_unicode_characters=True, precision=3):
        return str(self.axis)

    def text_diagram_exponent(self):
        return 0.5 if self._is_sqrt else 1.0

    def gate_after_passing_over(self, gate: 'CliffordPauliGate'):
        return gate._with_axis(self.axis_after_passing_over(gate.axis))

    def axis_after_passing_over(self, axis: AxisAbc):
        if self.axis.commutes_with(axis):
            return axis
        elif self.is_sqrt_pauli():
            return self.axis.complement(axis)
        else:
            return axis.negative()

    @staticmethod
    def axis_change_gates(old_axis: Axis, new_axis: Axis):
        if old_axis == new_axis:
            return (None, None)
        if old_axis == new_axis.negative():
            gate_axis = old_axis.next()  # or .next(2), or .next().abs()
            is_sqrt = False
        else:
            gate_axis = new_axis.complement(old_axis)
            is_sqrt = True
        gate_left = CliffordPauliGate(gate_axis, is_sqrt)
        gate_right = CliffordPauliGate(gate_axis.negative(), is_sqrt)
        return (gate_left, gate_right)
