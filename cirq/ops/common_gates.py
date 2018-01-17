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

    def __init__(self, turns=0.5):
        super(CZGate, self).__init__(turns_offset=turns)

    def extrapolate_effect(self, factor) -> 'CZGate':
        """See base class."""
        return CZGate(self.turns * factor)

    def ascii_exponent(self):
        return self.turns * 2

    def ascii_wire_symbols(self):
        return 'Z', 'Z'

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.turns) * 7

    def matrix(self):
        """See base class."""
        return np.diag([1, 1, 1, np.exp(2j * np.pi * self.turns)])

    def __repr__(self):
        if self.turns == 0.5:
            return 'CZ'
        return 'CZ**{}'.format(repr(self.turns * 2))


class XYGate(native_gates.ParameterizedXYGate,
             gate_features.ConstantSingleQubitGate,
             gate_features.ExtrapolatableGate,
             gate_features.BoundedEffectGate,
             gate_features.AsciiDiagrammableGate,
             gate_features.PhaseableGate):
    """Fixed rotation around an axis in the XY plane of the Bloch sphere."""

    def __init__(self, axis_phase_turns=0.0, turns=0.5):
        super(XYGate, self).__init__(
            axis_phase_turns_offset=axis_phase_turns, turns_offset=turns)

    def phase_by(self, phase_turns, qubit_index):
        return XYGate(self.axis_phase_turns + phase_turns, self.turns)

    def ascii_exponent(self):
        return self.turns * 2

    def ascii_wire_symbols(self):
        if self.axis_phase_turns == 0:
            return 'X',
        if self.axis_phase_turns == 0.25:
            return 'Y',
        return 'XY({})'.format(repr(self.axis_phase_turns)),

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.turns) * 7

    def extrapolate_effect(self, factor) -> 'XYGate':
        """See base class."""
        return XYGate(self.axis_phase_turns, self.turns * factor)

    def matrix(self):
        """See base class."""
        phase = ZGate(self.axis_phase_turns).matrix()
        c = np.exp(2j * np.pi * self.turns)
        rot = np.array([[1 + c, 1 - c], [1 - c, 1 + c]]) / 2
        return phase.dot(rot).dot(np.conj(phase))

    def __repr__(self):
        if self.axis_phase_turns == 0:
            if self.turns == 0.5:
                return 'X'
            return 'X**{}'.format(repr(self.turns * 2))

        if self.axis_phase_turns == 0.25:
            if self.turns == 0.5:
                return 'Y'
            return 'Y**{}'.format(repr(self.turns * 2))

        return 'XYGate({}, {})'.format(
            repr(self.axis_phase_turns), repr(self.turns))


class ZGate(native_gates.ParameterizedZGate,
            gate_features.ConstantSingleQubitGate,
            gate_features.ExtrapolatableGate,
            gate_features.BoundedEffectGate,
            gate_features.AsciiDiagrammableGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self, turns=0.5):
        super(ZGate, self).__init__(turns_offset=turns)

    def ascii_exponent(self):
        return self.turns * 2

    def ascii_wire_symbols(self):
        return 'Z',

    def trace_distance_bound(self):
        """See base class."""
        return abs(self.turns) * 7

    def extrapolate_effect(self, factor) -> 'ZGate':
        """See base class."""
        return ZGate(self.turns * factor)

    def matrix(self):
        """See base class."""
        return np.diag([1, np.exp(2j * np.pi * self.turns)])

    def __repr__(self):
        if self.turns == 0.5:
            return 'Z'
        return 'Z**{}'.format(repr(self.turns * 2))


X = XYGate()  # Pauli X gate.
Y = XYGate(axis_phase_turns=0.25)  # Pauli Y gate.
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


class SingleQubitMatrixGate(gate_features.ConstantSingleQubitGate,
                            gate_features.PhaseableGate,
                            gate_features.ExtrapolatableGate,
                            gate_features.BoundedEffectGate):
    """A 1-qubit gate defined only by its matrix.

    More general than specialized classes like ZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray):
        if matrix.shape != (2, 2):
            raise ValueError('Not a 2x2 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def extrapolate_effect(self, factor: float):
        new_mat = linalg.map_eigenvalues(self._matrix,
                                         lambda e: complex(e)**factor)
        return SingleQubitMatrixGate(new_mat)

    def trace_distance_bound(self):
        vals = np.linalg.eigvals(self._matrix)
        rotation_angle = abs(np.angle(vals[0] / vals[1]))
        return rotation_angle * 1.2

    def phase_by(self, phase_turns: float, qubit_index: int):
        phaser = ZGate(phase_turns).matrix()
        phased_matrix = phaser.dot(self._matrix).dot(np.conj(phaser))
        return SingleQubitMatrixGate(phased_matrix)

    def matrix(self):
        return self._matrix

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def approx_eq(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other.matrix())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other.matrix())

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'SingleQubitMatrixGate({})'.format(repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class TwoQubitMatrixGate(gate_features.ConstantAdjacentTwoQubitGate,
                         gate_features.PhaseableGate,
                         gate_features.ExtrapolatableGate):
    """A 2-qubit gate defined only by its matrix.

    More general than specialized classes like CZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray):
        if matrix.shape != (4, 4):
            raise ValueError('Not a 4x4 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def extrapolate_effect(self, factor: float):
        new_mat = linalg.map_eigenvalues(self._matrix,
                                         lambda e: complex(e)**factor)
        return TwoQubitMatrixGate(new_mat)

    def phase_by(self, phase_turns: float, qubit_index: int):
        z = ZGate(phase_turns).matrix()
        i = np.eye(2)
        z2 = np.kron(z, i) if qubit_index else np.kron(i, z)
        phased_matrix = z2.dot(self._matrix).dot(np.conj(z2))
        return TwoQubitMatrixGate(phased_matrix)

    def approx_eq(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other.matrix())

    def matrix(self):
        return self._matrix

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other.matrix())

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'TwoQubitMatrixGate({})'.format(repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))
