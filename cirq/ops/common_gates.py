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
from typing import Union, Tuple, Optional, List, Callable, cast, Iterable

import numpy as np

from cirq import value
from cirq.ops import gate_features, eigen_gate, raw_types, gate_operation


class Rot11Gate(eigen_gate.EigenGate,
                gate_features.PhaseableEffect,
                gate_features.TwoQubitGate,
                gate_features.TextDiagrammable,
                gate_features.InterchangeableQubitsGate,
                gate_features.QasmConvertibleGate):
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

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        if self.half_turns != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cz {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self.half_turns == 1:
            return 'CZ'
        return 'CZ**{!r}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'cirq.CZ'
        return '(cirq.CZ**{!r})'.format(self.half_turns)


class RotXGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate,
               gate_features.QasmConvertibleGate):
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

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        if self.half_turns == 1:
            return args.format('x {0};\n', qubits[0])
        else:
            return args.format('rx({0:half_turns}) {1};\n',
                               self.half_turns, qubits[0])

    def __str__(self) -> str:
        if self.half_turns == 1:
            return 'X'
        return 'X**{!r}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'cirq.X'
        return '(cirq.X**{!r})'.format(self.half_turns)


class RotYGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate,
               gate_features.QasmConvertibleGate):
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

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        if self.half_turns == 1:
            return args.format('y {0};\n', qubits[0])
        else:
            return args.format('ry({0:half_turns}) {1};\n',
                               self.half_turns, qubits[0])

    def __str__(self) -> str:
        if self.half_turns == 1:
            return 'Y'
        return 'Y**{!r}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'cirq.Y'
        return '(cirq.Y**{!r})'.format(self.half_turns)


class RotZGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.SingleQubitGate,
               gate_features.PhaseableEffect,
               gate_features.QasmConvertibleGate):
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

    def phase_by(self,
                 phase_turns: float,
                 qubit_index: int):
        return self

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        if self.half_turns in [-0.25, 0.25]:
            return gate_features.TextDiagramInfo(
                wire_symbols=('T',),
                exponent=cast(float, self._exponent) * 4)

        if self.half_turns in [-0.5, 0.5]:
            return gate_features.TextDiagramInfo(
                wire_symbols=('S',),
                exponent=cast(float, self._exponent) * 2)

        return gate_features.TextDiagramInfo(
            wire_symbols=('Z',),
            exponent=self._exponent)

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        if self.half_turns == 1:
            return args.format('z {0};\n', qubits[0])
        else:
            return args.format('rz({0:half_turns}) {1};\n',
                               self.half_turns, qubits[0])

    def __str__(self) -> str:
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
        return 'Z**{}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 0.25:
            return 'cirq.T'
        if self.half_turns == -0.25:
            return '(cirq.T**-1)'
        if self.half_turns == 0.5:
            return 'cirq.S'
        if self.half_turns == -0.5:
            return '(cirq.S**-1)'
        if self.half_turns == 1:
            return 'cirq.Z'
        return '(cirq.Z**{!r})'.format(self.half_turns)


class MeasurementGate(raw_types.Gate,
                      gate_features.TextDiagrammable,
                      gate_features.QasmConvertibleGate):
    """Indicates that qubits should be measured plus a key to identify results.

    Attributes:
        key: The string key of the measurement.
        invert_mask: A list of values indicating whether the corresponding
            qubits should be flipped. The list's length must not be longer than
            the number of qubits, but it is permitted to be shorted.
    Qubits with indices past the end of the mask are not flipped.
    """

    def __init__(self,
                 key: str = '',
                 invert_mask: Tuple[bool, ...] = ()) -> None:
        self.key = key
        self.invert_mask = invert_mask or ()

    @staticmethod
    def is_measurement(op: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if isinstance(op, MeasurementGate):
            return True
        if (isinstance(op, gate_operation.GateOperation) and
                isinstance(op.gate, MeasurementGate)):
            return True
        return False

    def with_bits_flipped(self, *bit_positions: int) -> 'MeasurementGate':
        """Toggles whether or not the measurement inverts various outputs."""
        old_mask = self.invert_mask or ()
        n = max(len(old_mask) - 1, *bit_positions) + 1
        new_mask = [k < len(old_mask) and old_mask[k] for k in range(n)]
        for b in bit_positions:
            new_mask[b] = not new_mask[b]
        return MeasurementGate(key=self.key, invert_mask=tuple(new_mask))

    def validate_args(self, qubits):
        if (self.invert_mask is not None and
                len(self.invert_mask) > len(qubits)):
            raise ValueError('len(invert_mask) > len(qubits)')

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        n = (max(1, len(self.invert_mask))
             if args.known_qubit_count is None
             else args.known_qubit_count)
        symbols = ['M'] * n

        # Show which output bits are negated.
        if self.invert_mask:
            for i, b in enumerate(self.invert_mask):
                if b:
                    symbols[i] = '!M'

        # Mention the measurement key.
        if (not args.known_qubits or
                self.key != _default_measurement_key(args.known_qubits)):
            symbols[0] += "('{}')".format(self.key)

        return gate_features.TextDiagramInfo(tuple(symbols))

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        invert_mask = self.invert_mask
        if len(invert_mask) < len(qubits):
            invert_mask = (invert_mask
                           + (False,) * (len(qubits) - len(invert_mask)))
        lines = []
        for i, (qubit, inv) in enumerate(zip(qubits, invert_mask)):
            if inv:
                lines.append(args.format(
                    'x {0};  // Invert the following measurement\n', qubit))
            lines.append(args.format('measure {0} -> {1:meas}[{2}];\n',
                                     qubit, self.key, i))
        return ''.join(lines)

    def __repr__(self):
        return 'cirq.MeasurementGate({}, {})'.format(repr(self.key),
                                                     repr(self.invert_mask))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.key == other.key and self.invert_mask == other.invert_mask

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((MeasurementGate, self.key, self.invert_mask))


def _default_measurement_key(qubits: Iterable[raw_types.QubitId]) -> str:
    return ','.join(str(q) for q in qubits)


def measure(*qubits: raw_types.QubitId,
            key: Optional[str] = None,
            invert_mask: Tuple[bool, ...] = ()
            ) -> gate_operation.GateOperation:
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
        key = _default_measurement_key(qubits)
    return MeasurementGate(key, invert_mask).on(*qubits)


def measure_each(*qubits: raw_types.QubitId,
                 key_func: Callable[[raw_types.QubitId], str] = str
                 ) -> List[gate_operation.GateOperation]:
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


class HGate(eigen_gate.EigenGate,
            gate_features.TextDiagrammable,
            gate_features.CompositeGate,
            gate_features.SingleQubitGate,
            gate_features.QasmConvertibleGate):
    """180 degree rotation around the X+Z axis of the Bloch sphere."""

    def __init__(self, *,  # Forces keyword args.
                 half_turns: Optional[Union[value.Symbol, float]] = None,
                 rads: Optional[float] = None,
                 degs: Optional[float] = None) -> None:
        """Initializes the gate.

        At most one angle argument may be specified. If more are specified,
        the result is considered ambiguous and an error is thrown. If no angle
        argument is given, the default value of one half turn is used.

        Args:
            half_turns: The relative phasing of H's eigenstates, in half_turns.
            rads: The relative phasing of H's eigenstates, in radians.
            degs: The relative phasing of H's eigenstates, in degrees.
        """
        super().__init__(exponent=value.chosen_angle_to_half_turns(
            half_turns=half_turns,
            rads=rads,
            degs=degs))

    def _canonical_exponent_period(self) -> Optional[float]:
        return 2

    def _with_exponent(self,
                       exponent: Union[value.Symbol, float]) -> 'HGate':
        return HGate(half_turns=exponent)

    def _eigen_components(self):
        component0 = np.divide(np.array([[(3 + 2 * np.sqrt(2)), (1 + np.sqrt(2))],
                                         [(1 + np.sqrt(2)), (1)]]), 2 * (2 + np.sqrt(2)))

        component1 = np.divide(np.array([[(3 - 2 * np.sqrt(2)), (1 - np.sqrt(2))],
                                         [(1 - np.sqrt(2)), (1)]]), 2 * (2 - np.sqrt(2)))

        return [(0, component0), (1, component1), ]

    @property
    def half_turns(self) -> Union[value.Symbol, float]:
        return self._exponent

    def default_decompose(self, qubits):
        q = qubits[0]

        if self._exponent == 1:
            yield Y(q)**0.5, X(q)
            return

        yield Y(q)**0.25
        yield X(q)**self.half_turns
        yield Y(q)**-0.25

    def inverse(self):
        return self

    def text_diagram_info(self, args: gate_features.TextDiagramInfoArgs
                          ) -> gate_features.TextDiagramInfo:
        return gate_features.TextDiagramInfo(('H',))

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        args.validate_version('2.0')
        if self.half_turns == 1:
            return args.format('h {0};\n', qubits[0])
        else:
            return args.format('rh({0:half_turns}) {1};\n',
                               self.half_turns, qubits[0])

    def __str__(self):
        return 'H'

    def __repr__(self):
        return 'cirq.H'


H = HGate()  # Hadamard gate.


class CNotGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.CompositeGate,
               gate_features.TwoQubitGate,
               gate_features.QasmConvertibleGate):
    """When applying CNOT (controlled-not) to QuBits, you can either use
    positional arguments CNOT(q1, q2), where q2 is toggled when q1 is on,
    or named arguments CNOT(control=q1, target=q2).
    (Mixing the two is not permitted.)"""

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

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        if self.half_turns != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cx {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self.half_turns == 1:
            return 'CNOT'
        return 'CNOT**{!r}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'cirq.CNOT'
        return '(cirq.CNOT**{!r})'.format(self.half_turns)

    def on(self, *args: raw_types.QubitId,
           **kwargs: raw_types.QubitId) -> gate_operation.GateOperation:
        if not kwargs:
            return super().on(*args)
        if not args and set(kwargs.keys()) == {'control', 'target'}:
            return super().on(kwargs['control'], kwargs['target'])
        raise ValueError(
            "Expected two positional argument or else 'target' AND 'control' "
            "keyword arguments. But got args={!r}, kwargs={!r}.".format(
                args, kwargs))


CNOT = CNotGate()  # Controlled Not Gate.


class SwapGate(eigen_gate.EigenGate,
               gate_features.TextDiagrammable,
               gate_features.TwoQubitGate,
               gate_features.CompositeGate,
               gate_features.InterchangeableQubitsGate,
               gate_features.QasmConvertibleGate):
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

    def known_qasm_output(self,
                          qubits: Tuple[raw_types.QubitId, ...],
                          args: gate_features.QasmOutputArgs) -> Optional[str]:
        if self.half_turns != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('swap {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self.half_turns == 1:
            return 'SWAP'
        return 'SWAP**{!r}'.format(self.half_turns)

    def __repr__(self) -> str:
        if self.half_turns == 1:
            return 'cirq.SWAP'
        return '(cirq.SWAP**{!r})'.format(self.half_turns)


SWAP = SwapGate()  # Exchanges two qubits' states.


class ISwapGate(eigen_gate.EigenGate,
                gate_features.CompositeGate,
                gate_features.InterchangeableQubitsGate,
                gate_features.TextDiagrammable,
                gate_features.TwoQubitGate):
    """Rotates the |01⟩-vs-|10⟩ subspace of two qubits around its Bloch X-axis.

    When exponent=1, swaps the two qubits and phases |01⟩ and |10⟩ by i. More
    generally, this gate's matrix is defined as follows:

        ISWAP**t ≡ exp(+i π t (X⊗X + Y⊗Y) / 4)
                 ≡ [1 0            0            0]
                   [0 cos(π·t/2)   i·sin(π·t/2) 0]
                   [0 i·sin(π·t/2) cos(π·t/2)   0]
                   [0 0            0            1]
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

    def __str__(self) -> str:
        if self.exponent == 1:
            return 'ISWAP'
        return 'ISWAP**{!r}'.format(self.exponent)

    def __repr__(self):
        if self.exponent == 1:
            return 'cirq.ISWAP'
        return '(cirq.ISWAP**{!r})'.format(self.exponent)


# Swaps two qubits while phasing the swapped subspace by i.
ISWAP = ISwapGate()
