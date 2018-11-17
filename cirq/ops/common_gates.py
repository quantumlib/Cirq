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
from typing import (
    Union, Tuple, Optional, List, Callable, cast, Iterable, Sequence,
    Any)

import numpy as np

from cirq import linalg, protocols, value
from cirq.ops import (
    gate_features,
    eigen_gate,
    raw_types,
    gate_operation,
)
from cirq.type_workarounds import NotImplementedType

# Note: avoiding 'from/as' because it creates a circular dependency in python 2.
import cirq.ops.phased_x_gate


class CZPowGate(eigen_gate.EigenGate,
                gate_features.TwoQubitGate,
                gate_features.InterchangeableQubitsGate):
    """Phases the |11⟩ state of two adjacent qubits by a fixed amount.

    A ParameterizedCZGate guaranteed to not be using the parameter key field.
    """

    def _eigen_components(self):
        return [
            (0, np.diag([1, 1, 1, 0])),
            (1, np.diag([0, 0, 0, 1])),
        ]

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if protocols.is_parameterized(self):
            return NotImplemented

        c = 1j**(2 * self._exponent)
        one_one = linalg.slice_for_qubits_equal_to(axes, 0b11)
        target_tensor[one_one] *= c
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            target_tensor *= p
        return target_tensor

    def _phase_by_(self, phase_turns, qubit_index):
        return self

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', '@'),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cz {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CZ'
        return 'CZ**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._exponent == 1:
            return 'cirq.CZ'
        return '(cirq.CZ**{!r})'.format(self._exponent)


def _rads_func_symbol(func_name: str,
                      args: protocols.CircuitDiagramInfoArgs,
                      half_turns: Any) -> str:
    unit = 'π' if args.use_unicode_characters else 'pi'
    if half_turns == 1:
        return '{}({})'.format(func_name, unit)
    if half_turns == -1:
        return '{}(-{})'.format(func_name, unit)
    return '{}({}{})'.format(func_name, half_turns, unit)


class XPowGate(eigen_gate.EigenGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the X axis of the Bloch sphere."""

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self._exponent != 1:
            return NotImplemented
        zero = linalg.slice_for_qubits_equal_to(axes, 0)
        one = linalg.slice_for_qubits_equal_to(axes, 1)
        available_buffer[zero] = target_tensor[one]
        available_buffer[one] = target_tensor[zero]
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            available_buffer *= p
        return available_buffer

    def _eigen_components(self):
        return [
            (0, np.array([[0.5, 0.5], [0.5, 0.5]])),
            (1, np.array([[0.5, -0.5], [-0.5, 0.5]])),
        ]

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> Union[str, protocols.CircuitDiagramInfo]:
        if self._global_shift == -0.5:
            return _rads_func_symbol(
                'Rx',
                args,
                self._diagram_exponent(args, ignore_global_phase=False))

        return protocols.CircuitDiagramInfo(
            wire_symbols=('X',),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1:
            return args.format('x {0};\n', qubits[0])
        else:
            return args.format('rx({0:half_turns}) {1};\n',
                               self._exponent, qubits[0])

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(
            exponent=self._exponent,
            phase_exponent=phase_turns * 2)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'X'
        return 'X**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._global_shift == -0.5:
            return 'cirq.Rx(np.pi*{!r})'.format(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.X'
            return '(cirq.X**{!r})'.format(self._exponent)
        return (
            'cirq.XPowGate(exponent={!r}, '
            'global_shift={!r})'
        ).format(self._exponent, self._global_shift)


class YPowGate(eigen_gate.EigenGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Y axis of the Bloch sphere."""

    def _eigen_components(self):
        return [
            (0, np.array([[0.5, -0.5j], [0.5j, 0.5]])),
            (1, np.array([[0.5, 0.5j], [-0.5j, 0.5]])),
        ]

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> Union[str, protocols.CircuitDiagramInfo]:
        if self._global_shift == -0.5:
            return _rads_func_symbol(
                'Ry',
                args,
                self._diagram_exponent(args, ignore_global_phase=False))

        return protocols.CircuitDiagramInfo(
            wire_symbols=('Y',),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1:
            return args.format('y {0};\n', qubits[0])
        else:
            return args.format('ry({0:half_turns}) {1};\n',
                               self._exponent, qubits[0])

    def _phase_by_(self, phase_turns, qubit_index):
        """See `cirq.SupportsPhase`."""
        return cirq.ops.phased_x_gate.PhasedXPowGate(
            exponent=self._exponent,
            phase_exponent=0.5 + phase_turns * 2)

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'Y'
        return 'Y**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._global_shift == -0.5:
            return 'cirq.Ry(np.pi*{!r})'.format(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.Y'
            return '(cirq.Y**{!r})'.format(self._exponent)
        return (
            'cirq.YPowGate(exponent={!r}, '
            'global_shift={!r})'
        ).format(self._exponent, self._global_shift)


class ZPowGate(eigen_gate.EigenGate,
               gate_features.SingleQubitGate):
    """Fixed rotation around the Z axis of the Bloch sphere."""

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if protocols.is_parameterized(self):
            return NotImplemented

        one = linalg.slice_for_qubits_equal_to(axes, 1)
        c = 1j**(self._exponent * 2)
        target_tensor[one] *= c
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            target_tensor *= p
        return target_tensor

    def _eigen_components(self):
        return [
            (0, np.diag([1, 0])),
            (1, np.diag([0, 1])),
        ]

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return self

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> Union[str, protocols.CircuitDiagramInfo]:
        if self._global_shift == -0.5:
            return _rads_func_symbol(
                'Rz',
                args,
                self._diagram_exponent(args, ignore_global_phase=False))

        e = self._diagram_exponent(args)
        if e in [-0.25, 0.25]:
            return protocols.CircuitDiagramInfo(
                wire_symbols=('T',),
                exponent=cast(float, e) * 4)

        if e in [-0.5, 0.5]:
            return protocols.CircuitDiagramInfo(
                wire_symbols=('S',),
                exponent=cast(float, e) * 2)

        return protocols.CircuitDiagramInfo(
            wire_symbols=('Z',),
            exponent=e)

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1:
            return args.format('z {0};\n', qubits[0])
        else:
            return args.format('rz({0:half_turns}) {1};\n',
                               self._exponent, qubits[0])

    def __str__(self) -> str:
        if self._exponent == 0.25:
            return 'T'
        if self._exponent == -0.25:
            return 'T**-1'
        if self._exponent == 0.5:
            return 'S'
        if self._exponent == -0.5:
            return 'S**-1'
        if self._exponent == 1:
            return 'Z'
        return 'Z**{}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._global_shift == -0.5:
            return 'cirq.Rz(np.pi*{!r})'.format(self._exponent)
        if self._global_shift == 0:
            if self._exponent == 0.25:
                return 'cirq.T'
            if self._exponent == -0.25:
                return '(cirq.T**-1)'
            if self._exponent == 0.5:
                return 'cirq.S'
            if self._exponent == -0.5:
                return '(cirq.S**-1)'
            if self._exponent == 1:
                return 'cirq.Z'
            return '(cirq.Z**{!r})'.format(self._exponent)
        return (
            'cirq.ZPowGate(exponent={!r}, '
            'global_shift={!r})'
        ).format(self._exponent, self._global_shift)


@value.value_equality
class MeasurementGate(raw_types.Gate):
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

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
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

        return protocols.CircuitDiagramInfo(tuple(symbols))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
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

    def _value_equality_values_(self):
        return self.key, self.invert_mask


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

    Raises:
        ValueError if the qubits are not instances of QubitId.
    """
    for qubit in qubits:
        if isinstance(qubit, np.ndarray):
            raise ValueError(
                    'measure() was called a numpy ndarray. Perhaps you meant '
                    'to call measure_state_vector on numpy array?'
            )
        elif not isinstance(qubit, raw_types.QubitId):
            raise ValueError(
                    'measure() was called with type different than QubitId.')

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


X = XPowGate()  # Pauli X gate.
Y = YPowGate()  # Pauli Y gate.
Z = ZPowGate()  # Pauli Z gate.
CZ = CZPowGate()  # Negates the amplitude of the |11⟩ state.

S = Z**0.5
T = Z**0.25


class HPowGate(eigen_gate.EigenGate, gate_features.SingleQubitGate):
    """Rotation around the X+Z axis of the Bloch sphere."""

    def _eigen_components(self):
        s = np.sqrt(2)

        component0 = np.array([
            [3 + 2 * s, 1 + s],
            [1 + s, 1]
        ]) / (4 + 2 * s)

        component1 = np.array([
            [3 - 2 * s, 1 - s],
            [1 - s, 1]
        ]) / (4 - 2 * s)

        return [(0, component0), (1, component1)]

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self._exponent != 1:
            return NotImplemented

        zero = linalg.slice_for_qubits_equal_to(axes, 0)
        one = linalg.slice_for_qubits_equal_to(axes, 1)
        target_tensor[one] -= target_tensor[zero]
        target_tensor[one] *= -0.5
        target_tensor[zero] -= target_tensor[one]
        p = 1j**(2 * self._exponent * self._global_shift)
        target_tensor *= np.sqrt(2) * p
        return target_tensor

    def _decompose_(self, qubits):
        q = qubits[0]

        if self._exponent == 1:
            yield cirq.Y(q)**0.5
            yield cirq.XPowGate(global_shift=-0.25).on(q)
            return

        yield Y(q)**0.25
        yield X(q)**self._exponent
        yield Y(q)**-0.25

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(('H',))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        args.validate_version('2.0')
        if self._exponent == 1:
            return args.format('h {0};\n', qubits[0])
        else:
            return args.format('ry({0:half_turns}) {3};\n'
                               'rx({1:half_turns}) {3};\n'
                               'ry({2:half_turns}) {3};\n',
                               0.25,  self._exponent, -0.25, qubits[0])

    def __str__(self):
        if self._exponent == 1:
            return 'H'
        return 'H^{}'.format(self._exponent)

    def __repr__(self):
        if self._exponent == 1:
            return 'cirq.H'
        return '(cirq.H**{!r})'.format(self._exponent)


H = HPowGate()  # Hadamard gate.


class CNotPowGate(eigen_gate.EigenGate, gate_features.TwoQubitGate):
    """The controlled-not gate, possibly raised to a power.

    When applying CNOT (controlled-not) to QuBits, you can either use
    positional arguments CNOT(q1, q2), where q2 is toggled when q1 is on,
    or named arguments CNOT(control=q1, target=q2).
    (Mixing the two is not permitted.)
    """

    def _decompose_(self, qubits):
        c, t = qubits
        yield Y(t)**-0.5
        yield CZ(c, t)**self._exponent
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

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('@', 'X'),
            exponent=self._diagram_exponent(args))

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self._exponent != 1:
            return NotImplemented

        oo = linalg.slice_for_qubits_equal_to(axes, 0b11)
        zo = linalg.slice_for_qubits_equal_to(axes, 0b01)
        available_buffer[oo] = target_tensor[oo]
        target_tensor[oo] = target_tensor[zo]
        target_tensor[zo] = available_buffer[oo]
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            target_tensor *= p
        return target_tensor

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('cx {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'CNOT'
        return 'CNOT**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._exponent == 1:
            return 'cirq.CNOT'
        return '(cirq.CNOT**{!r})'.format(self._exponent)

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


CNOT = CNotPowGate()  # Controlled Not Gate.


class SwapPowGate(eigen_gate.EigenGate,
                  gate_features.TwoQubitGate,
                  gate_features.InterchangeableQubitsGate):
    """The SWAP gate, possibly raised to a power. Exchanges qubits."""

    def _decompose_(self, qubits):
        """See base class."""
        a, b = qubits
        yield CNOT(a, b)
        yield CNOT(b, a) ** self._exponent
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

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self._exponent != 1:
            return NotImplemented

        zo = linalg.slice_for_qubits_equal_to(axes, 0b01)
        oz = linalg.slice_for_qubits_equal_to(axes, 0b10)
        available_buffer[zo] = target_tensor[zo]
        target_tensor[zo] = target_tensor[oz]
        target_tensor[oz] = available_buffer[zo]
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            target_tensor *= p
        return target_tensor

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return protocols.CircuitDiagramInfo(
                wire_symbols=('swap', 'swap'),
                exponent=self._diagram_exponent(args))
        return protocols.CircuitDiagramInfo(
            wire_symbols=('×', '×'),
            exponent=self._diagram_exponent(args))

    def _qasm_(self,
               args: protocols.QasmArgs,
               qubits: Tuple[raw_types.QubitId, ...]) -> Optional[str]:
        if self._exponent != 1:
            return None  # Don't have an equivalent gate in QASM
        args.validate_version('2.0')
        return args.format('swap {0},{1};\n', qubits[0], qubits[1])

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'SWAP'
        return 'SWAP**{!r}'.format(self._exponent)

    def __repr__(self) -> str:
        if self._exponent == 1:
            return 'cirq.SWAP'
        return '(cirq.SWAP**{!r})'.format(self._exponent)


SWAP = SwapPowGate()  # Exchanges two qubits' states.


class ISwapPowGate(eigen_gate.EigenGate,
                   gate_features.InterchangeableQubitsGate,
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

    def _decompose_(self, qubits):
        a, b = qubits

        yield CNOT(a, b)
        yield H(a)
        yield CNOT(b, a)
        yield S(a)**self._exponent
        yield CNOT(b, a)
        yield S(a)**-self._exponent
        yield H(a)
        yield CNOT(a, b)

    def _apply_unitary_to_tensor_(self,
                                  target_tensor: np.ndarray,
                                  available_buffer: np.ndarray,
                                  axes: Sequence[int],
                                  ) -> Union[np.ndarray, NotImplementedType]:
        if self._exponent != 1:
            return NotImplemented

        zo = linalg.slice_for_qubits_equal_to(axes, 0b01)
        oz = linalg.slice_for_qubits_equal_to(axes, 0b10)
        available_buffer[zo] = target_tensor[zo]
        target_tensor[zo] = target_tensor[oz]
        target_tensor[oz] = available_buffer[zo]
        target_tensor[zo] *= 1j
        target_tensor[oz] *= 1j
        p = 1j**(2 * self._exponent * self._global_shift)
        if p != 1:
            target_tensor *= p
        return target_tensor

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=('iSwap', 'iSwap'),
            exponent=self._diagram_exponent(args))

    def __str__(self) -> str:
        if self._exponent == 1:
            return 'ISWAP'
        return 'ISWAP**{!r}'.format(self._exponent)

    def __repr__(self):
        if self._exponent == 1:
            return 'cirq.ISWAP'
        return '(cirq.ISWAP**{!r})'.format(self._exponent)


# Swaps two qubits while phasing the swapped subspace by i.
ISWAP = ISwapPowGate()


def Rx(rads: float) -> XPowGate:
    """Returns a gate with the matrix e^{-i X rads / 2}."""
    return XPowGate(exponent=rads / np.pi, global_shift=-0.5)


def Ry(rads: float) -> YPowGate:
    """Returns a gate with the matrix e^{-i Y rads / 2}."""
    return YPowGate(exponent=rads / np.pi, global_shift=-0.5)


def Rz(rads: float) -> ZPowGate:
    """Returns a gate with the matrix e^{-i Z rads / 2}."""
    return ZPowGate(exponent=rads / np.pi, global_shift=-0.5)
