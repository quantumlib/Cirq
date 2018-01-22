# Copyright 2017 Google LLC
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

"""Quantum gates that are commonly used in the literature."""

import math
import numpy as np

from cirq import linalg
from cirq.ops import gate_features
from cirq.ops import native_gates


class CZGate(native_gates.ParameterizedCZGate,
             gate_features.ConstantAdjacentTwoQubitGate,
             gate_features.ExtrapolatableGate,
             gate_features.BoundedEffectGate,
             gate_features.AsciiDiagrammableGate):
    """Phases the |11> state of two adjacent qubits by a fixed amount.

  A ParameterizedCZGate guaranteed to not be using the parameter key field.
  """

    def __init__(self, *positional_args,
                 half_turns: float=1.0):
        assert not positional_args
        super(CZGate, self).__init__(turns_offset=half_turns / 2.0)

    def extrapolate_effect(self, factor) -> 'CZGate':
        """See base class."""
        return CZGate(half_turns=self.half_turns * factor)

    def ascii_exponent(self):
        return self.half_turns

    def ascii_wire_symbols(self):
        return 'Z', 'Z'

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.half_turns) * 3.5

    def matrix(self):
        """See base class."""
        return np.diag([1, 1, 1, np.exp(1j * np.pi * self.half_turns)])

    def __repr__(self):
        if self.half_turns == 1:
            return 'CZ'
        return 'CZ**{}'.format(repr(self.half_turns))


class XYGate(native_gates.ParameterizedXYGate,
             gate_features.ConstantSingleQubitGate,
             gate_features.ExtrapolatableGate,
             gate_features.BoundedEffectGate,
             gate_features.AsciiDiagrammableGate,
             gate_features.PhaseableGate):
    """Fixed rotation around an axis in the XY plane of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: float = 1,
                 axis_half_turns: float = 0):
        assert not positional_args
        super(XYGate, self).__init__(
            axis_phase_turns_offset=axis_half_turns / 2.0,
            turns_offset=half_turns / 2.0)

    def phase_by(self, phase_turns, qubit_index):
        return XYGate(axis_half_turns=self.axis_half_turns + 2*phase_turns,
                      half_turns=self.half_turns)

    def ascii_exponent(self):
        return self.half_turns

    def ascii_wire_symbols(self):
        if self.axis_half_turns == 0:
            return 'X',
        if self.axis_half_turns == 0.5:
            return 'Y',
        return 'XY({})'.format(repr(self.axis_half_turns)),

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.half_turns) * 3.5

    def extrapolate_effect(self, factor) -> 'XYGate':
        """See base class."""
        return XYGate(axis_half_turns=self.axis_half_turns,
                      half_turns=self.half_turns * factor)

    def matrix(self):
        """See base class."""
        phase = ZGate(half_turns=self.axis_half_turns).matrix()
        c = np.exp(1j * np.pi * self.half_turns)
        rot = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
        return phase.dot(rot).dot(np.conj(phase))

    def __repr__(self):
        if self.axis_half_turns == 0:
            if self.half_turns == 1:
                return 'X'
            return 'X**{}'.format(repr(self.half_turns))

        if self.axis_half_turns == 0.5:
            if self.half_turns == 1:
                return 'Y'
            return 'Y**{}'.format(repr(self.half_turns))

        return 'XYGate(axis_half_turns={}, half_turns={})'.format(
            repr(self.axis_half_turns), repr(self.half_turns))


class ZGate(native_gates.ParameterizedZGate,
            gate_features.ConstantSingleQubitGate,
            gate_features.ExtrapolatableGate,
            gate_features.BoundedEffectGate,
            gate_features.AsciiDiagrammableGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self, *positional_args,
                 half_turns: float = 1.0):
        assert not positional_args
        super(ZGate, self).__init__(turns_offset=half_turns / 2.0)

    def ascii_exponent(self):
        return self.half_turns

    def ascii_wire_symbols(self):
        return 'Z',

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.half_turns) * 3.5

    def extrapolate_effect(self, factor) -> 'ZGate':
        """See base class."""
        return ZGate(half_turns=self.half_turns * factor)

    def matrix(self):
        """See base class."""
        return np.diag([1, np.exp(1j * np.pi * self.half_turns)])

    def __repr__(self):
        if self.half_turns == 1:
            return 'Z'
        return 'Z**{}'.format(repr(self.half_turns))


X = XYGate()  # Pauli X gate.
Y = XYGate(axis_half_turns=0.5)  # Pauli Y gate.
Z = ZGate()  # Pauli Z gate.
CZ = CZGate()  # Negates the amplitude of the |11> state.


class HGate(gate_features.ConstantSingleQubitGate,
            gate_features.AsciiDiagrammableGate):
    """180 degree rotation around the X+Z axis of the Bloch sphere."""

    def ascii_wire_symbols(self):
        return 'H',

    def matrix(self):
        """See base class."""
        s = math.sqrt(0.5)
        return np.array([[s, s], [s, -s]])

    def __repr__(self):
        return 'H'


H = HGate()  # Hadamard gate.


class CNotGate(gate_features.ConstantAdjacentTwoQubitGate,
               gate_features.CompositeGate,
               gate_features.SelfInverseGate,
               gate_features.AsciiDiagrammableGate):
    """A controlled-NOT. Toggle the second qubit when the first qubit is on."""

    def matrix(self):
        """See base class."""
        return np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]])

    def ascii_wire_symbols(self):
        return '@', 'X'

    def default_decompose(self, qubits):
        """See base class."""
        c, t = qubits
        yield (Y**-0.5)(t)
        yield CZ(c, t)
        yield (Y**0.5)(t)

    def __repr__(self):
        return 'CNOT'


CNOT = CNotGate()  # Controlled Not Gate.


class SwapGate(gate_features.CompositeGate,
               gate_features.ConstantAdjacentTwoQubitGate):
    """Swaps two qubits."""

    def matrix(self):
        """See base class."""
        return np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])

    def default_decompose(self, qubits):
        """See base class."""
        a, b = qubits
        yield CNOT(a, b)
        yield CNOT(b, a)
        yield CNOT(a, b)

    def __repr__(self):
        return 'SWAP'


SWAP = SwapGate()  # Exchanges two qubits' states.
