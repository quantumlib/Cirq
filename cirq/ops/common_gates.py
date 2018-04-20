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
import math
from typing import Union, Tuple

import numpy as np

from cirq import abc
from cirq.extension import PotentialImplementation
from cirq.ops import gate_features
from cirq.ops.raw_types import InterchangeableQubitsGate
from cirq.value import Symbol


def _canonicalize_half_turns(
        half_turns: Union[Symbol, float]
) -> Union[Symbol, float]:
    if isinstance(half_turns, Symbol):
        return half_turns
    half_turns %= 2
    if half_turns > 1:
        half_turns -= 2
    return half_turns


class TurnGate(gate_features.BoundedEffectGate,
               gate_features.TextDiagrammableGate,
               PotentialImplementation):
    """A gate with two active eigenvalues.

    TurnGates have a direct sum decomposition I \oplus U, where I is the
    identity and U has exactly two eigenvalues. Extrapolating the gate phases
    one eigenspace of U relative to the other, with half_turns=1 corresponding
    to the point where the relative phase factor is exactly -1.
    """

    def __init__(self,
                 *positional_args,
                 half_turns: Union[Symbol, float] = 1.0) -> None:
        assert not positional_args
        self.half_turns = _canonicalize_half_turns(half_turns)

    @abc.abstractmethod
    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True
                                  ) -> Tuple[str, ...]:
        pass

    def text_diagram_exponent(self):
        return self.half_turns

    def __pow__(self, power: float) -> 'TurnGate':
        return self.extrapolate_effect(power)

    def inverse(self) -> 'TurnGate':
        return self.extrapolate_effect(-1)

    def __repr__(self):
        base = ''.join(self.text_diagram_wire_symbols())
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
        if isinstance(self.half_turns, Symbol):
            return 1
        return abs(self.half_turns) * 3.5

    def try_cast_to(self, desired_type):
        if (desired_type in [gate_features.ExtrapolatableGate,
                             gate_features.ReversibleGate] and
                self.can_extrapolate_effect()):
            return self
        if desired_type is gate_features.KnownMatrixGate and self.has_matrix():
            return self
        return super().try_cast_to(desired_type)

    @abc.abstractmethod
    def _matrix_impl_assuming_unparameterized(self) -> np.ndarray:
        pass

    def has_matrix(self) -> bool:
        return not isinstance(self.half_turns, Symbol)

    def matrix(self) -> np.ndarray:
        if not self.has_matrix():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return self._matrix_impl_assuming_unparameterized()

    def can_extrapolate_effect(self) -> bool:
        return not isinstance(self.half_turns, Symbol)

    def extrapolate_effect(self, factor) -> 'TurnGate':
        if not self.can_extrapolate_effect():
            raise ValueError("Parameterized. Don't have a known matrix.")
        return type(self)(half_turns=self.half_turns * factor)


class Rot11Gate(TurnGate,
                gate_features.TwoQubitGate,
                InterchangeableQubitsGate):
    """Phases the |11> state of two adjacent qubits by a fixed amount.

    A ParameterizedCZGate guaranteed to not be using the parameter key field.
    """

    def __init__(self,
                 *positional_args,
                 half_turns: Union[float, Symbol]=1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
        return 'Z', 'Z'

    def _matrix_impl_assuming_unparameterized(self):
        """See base class."""
        return np.diag([1, 1, 1, np.exp(1j * np.pi * self.half_turns)])


class RotXGate(TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the X axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Union[float, Symbol]=1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
        return 'X',

    def _matrix_impl_assuming_unparameterized(self):
        c = np.exp(1j * np.pi * self.half_turns)
        return np.array([[1 + c, 1 - c],
                         [1 - c, 1 + c]]) / 2


class RotYGate(TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the Y axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Union[float, Symbol]=1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
        return 'Y',

    def _matrix_impl_assuming_unparameterized(self):
        s = np.sin(np.pi * self.half_turns)
        c = np.cos(np.pi * self.half_turns)
        return np.array([[1 + s*1j + c, c*1j - s - 1j],
                         [1j + s - c*1j, 1 + s*1j + c]]) / 2


class RotZGate(TurnGate, gate_features.SingleQubitGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def __init__(self,
                 *positional_args,
                 half_turns: Union[float, Symbol]=1.0) -> None:
        assert not positional_args
        super().__init__(half_turns=half_turns)

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
        return 'Z',

    def phase_by(self, phase_turns, qubit_index):
        return self

    def _matrix_impl_assuming_unparameterized(self):
        """See base class."""
        return np.diag([1, np.exp(1j * np.pi * self.half_turns)])


class MeasurementGate(gate_features.TextDiagrammableGate):
    """Indicates that qubits should be measured plus a key to identify results.

    Params:
        key: The string key of the measurement.
        invert_mask: A list of Truthy or Falsey values indicating whether
        the corresponding qubits should be flipped. None indicates no
        inverting should be done.
    """

    def __init__(self, key: str = '', invert_mask: Tuple[bool] = None) -> None:
        self.key = key
        self.invert_mask = invert_mask

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
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
                                  use_unicode_characters=True):
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


class CNotGate(gate_features.TextDiagrammableGate,
               gate_features.CompositeGate,
               gate_features.KnownMatrixGate,
               gate_features.SelfInverseGate,
               gate_features.TwoQubitGate):
    """A controlled-NOT. Toggle the second qubit when the first qubit is on."""

    def matrix(self):
        """See base class."""
        return np.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]])

    def text_diagram_wire_symbols(self,
                                  qubit_count=None,
                                  use_unicode_characters=True):
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
                                  use_unicode_characters=True):
        if not use_unicode_characters:
            return 'swap', 'swap'
        return '×', '×'


SWAP = SwapGate()  # Exchanges two qubits' states.
