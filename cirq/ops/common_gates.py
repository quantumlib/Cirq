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
from typing import Union, Tuple, Optional

import numpy as np

from cirq import value
from cirq.ops import gate_features, eigen_gate
from cirq.ops.raw_types import InterchangeableQubitsGate


class Rot11Gate(eigen_gate.EigenGate,
                gate_features.TwoQubitGate,
                gate_features.TextDiagrammableGate,
                InterchangeableQubitsGate):
    """Phases the |11> state of two adjacent qubits by a fixed amount.

    A ParameterizedCZGate guaranteed to not be using the parameter key field.
    """

    def __init__(self,
                 *positional_args,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            *positional_args: Not an actual argument. Forces all arguments to
                be keyword arguments. Prevents angle unit confusion by forcing
                "rads=", "degs=", or "half_turns=".
            half_turns: Relative phasing of CZ's eigenstates, in half_turns.
            rads: Relative phasing of CZ's eigenstates, in radians.
            degs: Relative phasing of CZ's eigenstates, in degrees.
        """
        assert not positional_args
        super().__init__(exponent=value.chosen_angle_to_canonical_half_turns(
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

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'Z'

    def text_diagram_exponent(self):
        return self._exponent

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CZ'
        return 'CZ**{!r}'.format(self.half_turns)


class RotXGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammableGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the X axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            *positional_args: Not an actual argument. Forces all arguments to
                be keyword arguments. Prevents angle unit confusion by forcing
                "rads=", "degs=", or "half_turns=".
            half_turns: The relative phasing of X's eigenstates, in half_turns.
            rads: The relative phasing of X's eigenstates, in radians.
            degs: The relative phasing of X's eigenstates, in degrees.
        """
        assert not positional_args
        super().__init__(exponent=value.chosen_angle_to_canonical_half_turns(
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'X',

    def text_diagram_exponent(self):
        return self._exponent

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'X'
        return 'X**{!r}'.format(self.half_turns)


class RotYGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammableGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Y axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            *positional_args: Not an actual argument. Forces all arguments to
                be keyword arguments. Prevents angle unit confusion by forcing
                "rads=", "degs=", or "half_turns=".
            half_turns: The relative phasing of Y's eigenstates, in half_turns.
            rads: The relative phasing of Y's eigenstates, in radians.
            degs: The relative phasing of Y's eigenstates, in degrees.
        """
        assert not positional_args
        super().__init__(exponent=value.chosen_angle_to_canonical_half_turns(
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'Y',

    def text_diagram_exponent(self):
        return self._exponent

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'Y'
        return 'Y**{!r}'.format(self.half_turns)


class RotZGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammableGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            *positional_args: Not an actual argument. Forces all arguments to
                be keyword arguments. Prevents angle unit confusion by forcing
                "rads=", "degs=", or "half_turns=".
            half_turns: The relative phasing of Z's eigenstates, in half_turns.
            rads: The relative phasing of Z's eigenstates, in radians.
            degs: The relative phasing of Z's eigenstates, in degrees.
        """
        assert not positional_args
        super().__init__(exponent=value.chosen_angle_to_canonical_half_turns(
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'Z',

    def text_diagram_exponent(self):
        return self._exponent

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'Z'
        return 'Z**{!r}'.format(self.half_turns)


class MeasurementGate(gate_features.TextDiagrammableGate):
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'M',

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


X = RotXGate()  # Pauli X gate.
Y = RotYGate()  # Pauli Y gate.
Z = RotZGate()  # Pauli Z gate.
CZ = Rot11Gate()  # Negates the amplitude of the |11> state.

S = Z**0.5
T = Z**0.25


class HGate(gate_features.TextDiagrammableGate,
            gate_features.CompositeGate,
            gate_features.SelfInverseGate,
            gate_features.KnownMatrixGate,
            gate_features.SingleQubitGate):
    """180 degree rotation around the X+Z axis of the Bloch sphere."""

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return 'H',

    def default_decompose(self, qubits):
        q = qubits[0]
        yield Y(q)**0.5
        yield X(q)

    def matrix(self):
        """See base class."""
        s = math.sqrt(0.5)
        return np.array([[s, s], [s, -s]])

    def __repr__(self):
        return 'H'


H = HGate()  # Hadamard gate.


class CNotGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammableGate,
               gate_features.CompositeGate,
               gate_features.TwoQubitGate):
    """A controlled-NOT. Toggle the second qubit when the first qubit is on."""

    def __init__(self,
                 *positional_args,
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            *positional_args: Not an actual argument. Forces all arguments to
                be keyword arguments. Prevents angle unit confusion by forcing
                "rads=", "degs=", or "half_turns=".
            half_turns: Relative phasing of CNOT's eigenstates, in half_turns.
            rads: Relative phasing of CNOT's eigenstates, in radians.
            degs: Relative phasing of CNOT's eigenstates, in degrees.
        """
        assert not positional_args
        super().__init__(exponent=value.chosen_angle_to_canonical_half_turns(
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        return '@', 'X'

    def text_diagram_exponent(self):
        return self._exponent

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'CNOT'
        return 'CNOT**{!r}'.format(self.half_turns)


CNOT = CNotGate()  # Controlled Not Gate.


class SwapGate(gate_features.TextDiagrammableGate,
               gate_features.CompositeGate,
               gate_features.SelfInverseGate,
               gate_features.KnownMatrixGate,
               gate_features.TwoQubitGate,
               InterchangeableQubitsGate):
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

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True,
                                  precision=3):
        if not use_unicode_characters:
            return 'swap', 'swap'
        return '×', '×'


SWAP = SwapGate()  # Exchanges two qubits' states.
