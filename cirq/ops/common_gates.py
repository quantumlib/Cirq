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

"""Quantum gates that are commonly used in the literature."""
import math
from typing import Union, Tuple, Optional, List, Callable

import numpy as np

from cirq import value
from cirq.ops import gate_features, eigen_gate, raw_types


class Rot11Gate(eigen_gate.EigenGate,
                gate_features.PhaseableGate,
                gate_features.TwoQubitGate,
                gate_features.TextDiagrammable,
                raw_types.InterchangeableQubitsGate):
    """Phases the |11> state of two adjacent qubits by a fixed amount.

    A ParameterizedCZGate guaranteed to not be using the parameter key field.
    """

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: Relative phasing of CZ's eigenstates, in half_turns.
            rads: Relative phasing of CZ's eigenstates, in radians.
            degs: Relative phasing of CZ's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def _eigen_components(self):
        return [
            (0, np.diag([1, 1, 1, 0])),
            (1, np.diag([0, 0, 0, 1])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'Rot11Gate':
        return Rot11Gate(half_turns=exponent)

    def phase_by(self, phase_turns, qubit_index):
        return self

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=('@', '@'),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CZ'
        return 'CZ**{!r}'.format(self.half_turns)


class RotXGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate):
    """Fixed rotation around the X axis of the Bloch sphere."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The relative phasing of X's eigenstates, in half_turns.
            rads: The relative phasing of X's eigenstates, in radians.
            degs: The relative phasing of X's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def _eigen_components(self):
        return [
            (0, np.array([[0.5, 0.5], [0.5, 0.5]])),
            (1, np.array([[0.5, -0.5], [-0.5, 0.5]])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'RotXGate':
        return RotXGate(half_turns=exponent)

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=('X',),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'X'
        return 'X**{!r}'.format(self.half_turns)


class RotYGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Y axis of the Bloch sphere."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The relative phasing of Y's eigenstates, in half_turns.
            rads: The relative phasing of Y's eigenstates, in radians.
            degs: The relative phasing of Y's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def _eigen_components(self):
        return [
            (0, np.array([[0.5, -0.5j], [0.5j, 0.5]])),
            (1, np.array([[0.5, 0.5j], [-0.5j, 0.5]])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'RotYGate':
        return RotYGate(half_turns=exponent)

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=('Y',),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'Y'
        return 'Y**{!r}'.format(self.half_turns)


class RotZGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The relative phasing of Z's eigenstates, in half_turns.
            rads: The relative phasing of Z's eigenstates, in radians.
            degs: The relative phasing of Z's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def _eigen_components(self):
        return [
            (0, np.diag([1, 0])),
            (1, np.diag([0, 1])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'RotZGate':
        return RotZGate(half_turns=exponent)

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        if self.half_turns in [-0.25, 0.25]:
            return gate_features.TextDiagramInfo(
                wire_symbols=('T',),
                exponent=self._exponent * 4)

        if self.half_turns in [-0.5, 0.5]:
            return gate_features.TextDiagramInfo(
                wire_symbols=('S',),
                exponent=self._exponent * 2)

        return gate_features.TextDiagramInfo(
            wire_symbols=('Z',),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 0.25:
            return 'T'
        if self.half_turns == -0.25:
            return 'T**-1'
        if self.half_turns == 0.5:
            return 'S'
        if self.half_turns == -0.5:
            return 'S**-1'
        if self.half_turns == 1:
            return 'Z'
        return 'Z**{!r}'.format(self.half_turns)


class MeasurementGate(raw_types.Gate,
                      gate_features.TextDiagrammable):
    """Indicates that qubits should be measured plus a key to identify results.

    Params:
        key: The string key of the measurement.
        invert_mask: A list of Truthy or Falsey values indicating whether
            the corresponding qubits should be flipped. None indicates no
            inverting should be done.
    """

    def __init__(self,
                 key: str = '',
                 invert_mask: Optional[Tuple[bool, ...]] = None) -> None:
        self.key = key
        self.invert_mask = invert_mask

    def validate_args(self, qubits):
        if (self.invert_mask is not None and
                len(self.invert_mask) > len(qubits)):
            raise ValueError('len(invert_mask) > len(qubits)')

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        n = 1 if args.known_qubit_count is None else args.known_qubit_count
        return gate_features.TextDiagramInfo(('M',) * n)

    def __repr__(self):
        return 'MeasurementGate({}, {})'.format(repr(self.key),
                                                repr(self.invert_mask))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key and self.invert_mask == other.invert_mask

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((MeasurementGate, self.key, self.invert_mask))


def measure(*qubits: raw_types.QubitId,
            key: Optional[str] = None,
            invert_mask: Optional[Tuple[bool, ...]] = None
            ) -> raw_types.Operation:
    """Returns a single MeasurementGate applied to all the given qubits.

    The qubits are measured in the computational basis.

    Args:
        *qubits: The qubits that the measurement gate should measure.
        key: The string key of the measurement. If this is None, it defaults
            to a comma-separated list of the target qubits' str values.
        invert_mask: A list of Truthy or Falsey values indicating whether
            the corresponding qubits should be flipped. None indicates no
            inverting should be done.

    Returns:
        An operation targeting the given qubits with a measurement.
    """
    if key is None:
        key = ','.join(str(q) for q in qubits)
    return MeasurementGate(key, invert_mask).on(*qubits)


def measure_each(*qubits: raw_types.QubitId,
                 key_func: Callable[[raw_types.QubitId], str] = str
                 ) -> List[raw_types.Operation]:
    """Returns a list of operations individually measuring the given qubits.

    The qubits are measured in the computational basis.

    Args:
        *qubits: The qubits to measure.
        key_func: Determines the key of the measurements of each qubit. Takes
            the qubit and returns the key for that qubit. Defaults to str.

    Returns:
        A list of operations individually measuring the given qubits.
    """
    return [MeasurementGate(key_func(q)).on(q) for q in qubits]


X = RotXGate()  # Pauli X gate.
Y = RotYGate()  # Pauli Y gate.
Z = RotZGate()  # Pauli Z gate.
CZ = Rot11Gate()  # Negates the amplitude of the |11> state.

S = Z**0.5
T = Z**0.25


class HGate(gate_features.CompositeGate,
            gate_features.TextDiagrammable,
            gate_features.ReversibleEffect,
            gate_features.KnownMatrixGate,
            gate_features.SingleQubitGate):
    """180 degree rotation around the X+Z axis of the Bloch sphere."""

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(('H',))

    def default_decompose(self, qubits):
        q = qubits[0]
        yield Y(q)**0.5
        yield X(q)

    def inverse(self):
        return self

    def matrix(self):
        """See base class."""
        s = math.sqrt(0.5)
        return np.array([[s, s], [s, -s]])

    def __repr__(self):
        return 'H'


H = HGate()  # Hadamard gate.


class CNotGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.TwoQubitGate):
    """A controlled-NOT. Toggle the second qubit when the first qubit is on."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: Relative phasing of CNOT's eigenstates, in half_turns.
            rads: Relative phasing of CNOT's eigenstates, in radians.
            degs: Relative phasing of CNOT's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def default_decompose(self, qubits):
        c, t = qubits
        yield Y(t)**-0.5
        yield Rot11Gate(half_turns=self.half_turns).on(c, t)
        yield Y(t)**0.5

    def _eigen_components(self):
        return [
            (0, np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0.5, 0.5],
                          [0, 0, 0.5, 0.5]])),
            (1, np.array([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0.5, -0.5],
                          [0, 0, -0.5, 0.5]])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'CNotGate':
        return CNotGate(half_turns=exponent)

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=('@', 'X'),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CNOT'
        return 'CNOT**{!r}'.format(self.half_turns)


CNOT = CNotGate()  # Controlled Not Gate.


class SwapGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.TwoQubitGate,
               gate_features.CompositeGate,
               raw_types.InterchangeableQubitsGate):
    """Swaps two qubits."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Union[value.Symbol, float] = 1.0) -> None:
        super().__init__(exponent=half_turns)

    def default_decompose(self, qubits):
        """See base class."""
        a, b = qubits
        yield CNOT(a, b)
        yield CNOT(b, a) ** self.half_turns
        yield CNOT(a, b)

    def _eigen_components(self):
        return [
            (0, np.array([[1, 0,   0,   0],
                          [0, 0.5, 0.5, 0],
                          [0, 0.5, 0.5, 0],
                          [0, 0,   0,   1]])),
            (1, np.array([[0,  0,    0,   0],
                          [0,  0.5, -0.5, 0],
                          [0, -0.5,  0.5, 0],
                          [0,  0,    0,   0]])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'SwapGate':
        return SwapGate(half_turns=exponent)

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        if not args.use_unicode_characters:
            return gate_features.TextDiagramInfo(
                wire_symbols=('swap', 'swap'),
                exponent=self._exponent)
        return gate_features.TextDiagramInfo(
            wire_symbols=('×', '×'),
            exponent=self._exponent)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'SWAP'
        return 'SWAP**{!r}'.format(self.half_turns)


SWAP = SwapGate()  # Exchanges two qubits' states.


class ISwapGate(eigen_gate.EigenGate,
                gate_features.CompositeGate,
                raw_types.InterchangeableQubitsGate,
                gate_features.TextDiagrammable,
                gate_features.TwoQubitGate):
    """Rotates the |01⟩-vs-|10⟩ subspace of two qubits around its Bloch X-axis.

    When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by i. More
    generally, this gate's matrix is defined as follows:

        ISWAP**t ≡ exp(+i π t (X⊗X + Y⊗Y))
                 ≡ [1 0         0         0]
                   [0 cos(πt)   i·sin(πt) 0]
                   [0 i·sin(πt) cos(πt)   0]
                   [0 0         0         1]
    """

    @property
    def exponent(self) -> Union[value.Symbol, float]:
        return self._exponent

    def _eigen_components(self):
        return [
            (0, np.diag([1, 0, 0, 1])),
            (+0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0.5, 0.5, 0],
                             [0, 0, 0, 0]])),
            (-0.5, np.array([[0, 0, 0, 0],
                             [0, 0.5, -0.5, 0],
                             [0, -0.5, 0.5, 0],
                             [0, 0, 0, 0]])),
        ]

    def _canonical_exponent_period(self) -> Optional[float]:
        return 4

    def _with_exponent(self, exponent: Union[value.Symbol, float]
                       ) -> 'ISwapGate':
        return ISwapGate(exponent=exponent)

    def default_decompose(self, qubits):
        a, b = qubits

        yield CNOT(a, b)
        yield H(a)
        yield CNOT(b, a)
        yield S(a)**self.exponent
        yield CNOT(b, a)
        yield S(a)**-self.exponent
        yield H(a)
        yield CNOT(a, b)

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(
            wire_symbols=('iSwap', 'iSwap'),
            exponent=self._exponent)

    def __repr__(self):
        if self.exponent == 1:
            return 'ISWAP'
        return 'ISWAP**{!r}'.format(self.exponent)


ISWAP = ISwapGate()  # Swaps two qubits while phasing the swapped subspace by i.
