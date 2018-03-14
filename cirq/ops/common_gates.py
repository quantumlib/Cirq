# Copyright 2018 Google LLC
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
import abc
import math
import numpy as np

from cirq.ops import gate_features
from cirq.ops.raw_types import InterchangeableQubitsGate


def _canonicalize_half_turns(half_turns: float) -> float:
    v = half_turns
    v %= 2
    if v > 1:
        v -= 2
    return v


class _TurnGate(gate_features.ExtrapolatableGate,
                gate_features.BoundedEffectGate,
                gate_features.AsciiDiagrammableGate):

    def __init__(self, *positional_args, half_turns: float = 1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    @abc.abstractmethod
    def ascii_wire_symbols(self):
        pass

    def ascii_exponent(self):
        return self.half_turns

    def __repr__(self):
        base = ''.join(self.ascii_wire_symbols())
        if self.half_turns == 1:
            return base
        return '{}**{}'.format(base, repr(self.half_turns))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.half_turns == other.half_turns

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((type(self), self.half_turns))

    def trace_distance_bound(self):
        return abs(self.half_turns) * 3.5

    def extrapolate_effect(self, factor) -> '_TurnGate':
        return type(self)(half_turns=self.half_turns * factor)


class Rot11Gate(_TurnGate,
                gate_features.TwoQubitGate,
                InterchangeableQubitsGate):
    """Phases the |11> state of two adjacent qubits by a fixed amount.

  A ParameterizedCZGate guaranteed to not be using the parameter key field.
  """

    def __init__(self, *positional_args, half_turns: float=1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def ascii_wire_symbols(self):
        return 'Z', 'Z'

    def matrix(self):
        """See base class."""
        return np.diag([1, 1, 1, np.exp(1j * np.pi * self.half_turns)])


class RotXGate(_TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the X axis of the Bloch sphere."""

    def __init__(self, *positional_args, half_turns: float = 1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def ascii_wire_symbols(self):
        return 'X',

    def phase_by(self, phase_turns, qubit_index):
        return self

    def matrix(self):
        c = np.exp(1j * np.pi * self.half_turns)
        return np.array([[1 + c, 1 - c],
                         [1 - c, 1 + c]]) / 2


class RotYGate(_TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the Y axis of the Bloch sphere."""

    def __init__(self, *positional_args, half_turns: float = 1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def ascii_wire_symbols(self):
        return 'Y',

    def phase_by(self, phase_turns, qubit_index):
        return self

    def matrix(self):
        s = np.sin(np.pi * self.half_turns)
        c = np.cos(np.pi * self.half_turns)
        return np.array([[1 + s*1j + c, c*1j - s - 1j],
                         [1j + s - c*1j, 1 + s*1j + c]]) / 2


class RotZGate(_TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self, *positional_args, half_turns: float = 1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def ascii_wire_symbols(self):
        return 'Z',

    def phase_by(self, phase_turns, qubit_index):
        return self

    def matrix(self):
        """See base class."""
        return np.diag([1, np.exp(1j * np.pi * self.half_turns)])


class MeasurementGate(gate_features.AsciiDiagrammableGate):
    """Indicates that a qubit should be measured, and where the result goes."""

    def __init__(self, key: str = '', invert_result=False) -> None:
        self.key = key
        self.invert_result = invert_result

    def ascii_wire_symbols(self):
        return 'M',

    def __repr__(self):
        return 'MeasurementGate({})'.format(repr(self.key))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((MeasurementGate, self.key))


X = RotXGate()  # Pauli X gate.
Y = RotYGate()  # Pauli Y gate.
Z = RotZGate()  # Pauli Z gate.
CZ = Rot11Gate()  # Negates the amplitude of the |11> state.

S = Z**0.5
T = Z**0.25


class HGate(gate_features.SingleQubitGate,
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


class CNotGate(gate_features.TwoQubitGate,
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
        yield Y(t)**-0.5
        yield CZ(c, t)
        yield Y(t)**0.5

    def __repr__(self):
        return 'CNOT'


CNOT = CNotGate()  # Controlled Not Gate.


class SwapGate(gate_features.CompositeGate,
               gate_features.TwoQubitGate):
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
