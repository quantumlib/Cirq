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
from cirq import CompositeGate, InterchangeableQubitsGate, TextDiagrammableGate

from cirq.contrib.rearrange.axis import Axis, Z_AXIS


def to_axis_ops(old_axis: Axis, new_axis: Axis, qubit: ops.QubitId):
    return (gate.on(qubit) for gate in to_axis_gates(old_axis, new_axis))

def to_axis_gates(old_axis: Axis, new_axis: Axis):
    gates = (ops.X, ops.Y, ops.Z)
    axis = old_axis.axis_i
    axis2 = new_axis.axis_i
    flip = (axis < 3) != (axis2 < 3)
    cycle = (axis + axis2) % 3
    gate_i = (-cycle) % 3
    quarter_turns = (axis2 - axis + 1) % 3 - 1  # -1, 0, or +1
    if flip:
        quarter_turns = -quarter_turns
    if quarter_turns == 0:
        if flip:
            gate = gates[(axis + 1) % 3]  # Alternatively +2
            return (gate,)
        else:
            return ()
    else:
        gate = gates[gate_i] ** (quarter_turns * 0.5)
        return (gate,)


class InteractionGate(ops.CompositeGate,
                      ops.ExtrapolatableGate,
                      #####ops.InterchangeableQubitsGate,  # Sometimes
                      ops.TextDiagrammableGate):
    def __init__(self, axis0: Axis, axis1: Axis,
                 *positional_args,
                 half_turns: float = 1.0):
        assert not positional_args
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1
        self.half_turns = half_turns

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.axis0 == other.axis0 and self.axis1 == other.axis1 and self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((InteractionGate, self.axis0, self.axis1, self.half_turns))

    def _with_half_turns(self, half_turns: float) -> 'InteractionGate':
        return InteractionGate(self.axis0, self.axis1,
                               half_turns=half_turns)

    def extrapolate_effect(self, factor: float) -> 'InteractionGate':
        return self._with_half_turns(self.half_turns * factor)

    def conversion_to(self,
                      other_interaction: 'InteractionGate',
                      q0: ops.QubitId,
                      q1: ops.QubitId):
        yield to_axis_ops(self.axis0, other_interaction.axis0, q0)
        yield to_axis_ops(self.axis1, other_interaction.axis1, q1)
        yield other_interaction._with_half_turns(self.half_turns).on(q0, q1)
        yield to_axis_ops(other_interaction.axis0, self.axis0, q0)
        yield to_axis_ops(other_interaction.axis1, self.axis1, q1)

    def default_decompose(self, qubits):
        q0, q1 = qubits
        yield from to_axis_ops(self.axis0, Z_AXIS, q0)
        yield from to_axis_ops(self.axis1, Z_AXIS, q1)
        yield ops.CZ(q0, q1) ** self.half_turns
        yield from to_axis_ops(Z_AXIS, self.axis0, q0)
        yield from to_axis_ops(Z_AXIS, self.axis1, q1)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return (str(self.axis0), str(self.axis1))

    def text_diagram_exponent(self):
        return self.half_turns
