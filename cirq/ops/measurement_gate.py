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
"""Quantum gates that are commonly used in the literature.

This module creates Gate instances for the following gates:
    X,Y,Z: Pauli gates.
    H,S: Clifford gates.
    T: A non-Clifford gate.
    CZ: Controlled phase gate.
    CNOT: Controlled not gate.

Each of these are implemented as EigenGates, which means that they can be
raised to a power (i.e. cirq.H**0.5). See the definition in EigenGate.

In addition MeasurementGate is defined and convenience methods for
measurements are provided
    measure
    measure_each
"""
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types


@value.value_equality
class MeasurementGate(raw_types.Gate):
    """A gate that measures qubits in the computational basis.

    The measurement gate contains a key that is used to identify results
    of measurements.
    """

    def __init__(self,
                 num_qubits: Optional[int] = None,
                 key: str = '',
                 invert_mask: Tuple[bool, ...] = (),
                 qid_shape: Tuple[int, ...] = None) -> None:
        """
        Args:
            num_qubits: The number of qubits to act upon.
            key: The string key of the measurement.
            invert_mask: A list of values indicating whether the corresponding
                qubits should be flipped. The list's length must not be longer
                than the number of qubits, but it is permitted to be shorter.
                Qubits with indices past the end of the mask are not flipped.
            qid_shape: Specifies the dimension of each qid the measurement
                applies to.  The default is 2 for every qubit.

        Raises:
            ValueError: If the length of invert_mask is greater than num_qubits.
                or if the length of qid_shape doesn't equal num_qubits.
        """
        if qid_shape is None:
            if num_qubits is None:
                raise ValueError(
                    'Specify either the num_qubits or qid_shape argument.')
            qid_shape = (2,) * num_qubits
        elif num_qubits is None:
            num_qubits = len(qid_shape)
        if num_qubits == 0:
            raise ValueError('Measuring an empty set of qubits.')
        self._qid_shape = qid_shape
        if len(self._qid_shape) != num_qubits:
            raise ValueError('len(qid_shape) != num_qubits')
        self.key = key
        self.invert_mask = invert_mask or ()
        if (self.invert_mask is not None and
                len(self.invert_mask) > self.num_qubits()):
            raise ValueError('len(invert_mask) > num_qubits')

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def with_bits_flipped(self, *bit_positions: int) -> 'MeasurementGate':
        """Toggles whether or not the measurement inverts various outputs."""
        old_mask = self.invert_mask or ()
        n = max(len(old_mask) - 1, *bit_positions) + 1
        new_mask = [k < len(old_mask) and old_mask[k] for k in range(n)]
        for b in bit_positions:
            new_mask[b] = not new_mask[b]
        return MeasurementGate(self.num_qubits(),
                               key=self.key,
                               invert_mask=tuple(new_mask))

    def full_invert_mask(self):
        """Returns the invert mask for all qubits.

        If the user supplies a partial invert_mask, this returns that mask
        padded by False.

        Similarly if no invert_mask is supplies this returns a tuple
        of size equal to the number of qubits with all entries False.
        """
        mask = self.invert_mask or self.num_qubits() * (False,)
        deficit = self.num_qubits() - len(mask)
        mask += (False,) * deficit
        return mask

    def _measurement_key_(self):
        return self.key

    def _channel_(self):
        size = np.prod(self._qid_shape, dtype=int)

        def delta(i):
            result = np.zeros((size, size))
            result[i][i] = 1
            return result

        return tuple(delta(i) for i in range(size))

    def _has_channel_(self):
        return True

    def _circuit_diagram_info_(self, args: 'protocols.CircuitDiagramInfoArgs'
                              ) -> 'protocols.CircuitDiagramInfo':
        symbols = ['M'] * self.num_qubits()

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

    def _qasm_(self, args: 'protocols.QasmArgs',
               qubits: Tuple[raw_types.Qid, ...]) -> Optional[str]:
        if not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        args.validate_version('2.0')
        invert_mask = self.invert_mask
        if len(invert_mask) < len(qubits):
            invert_mask = (invert_mask + (False,) *
                           (len(qubits) - len(invert_mask)))
        lines = []
        for i, (qubit, inv) in enumerate(zip(qubits, invert_mask)):
            if inv:
                lines.append(
                    args.format('x {0};  // Invert the following measurement\n',
                                qubit))
            lines.append(
                args.format('measure {0} -> {1:meas}[{2}];\n', qubit, self.key,
                            i))
        return ''.join(lines)

    def __repr__(self):
        other = ''
        if not all(d == 2 for d in self._qid_shape):
            other = ', {!r}'.format(self._qid_shape)
        return 'cirq.MeasurementGate({!r}, {!r}, {!r}{})'.format(
            self.num_qubits(), self.key, self.invert_mask, other)

    def _value_equality_values_(self):
        return self.key, self.invert_mask, self._qid_shape

    def _json_dict_(self):
        other = {}
        if not all(d == 2 for d in self._qid_shape):
            other['qid_shape'] = self._qid_shape
        return {
            'cirq_type': self.__class__.__name__,
            'num_qubits': len(self._qid_shape),
            'key': self.key,
            'invert_mask': self.invert_mask,
            **other,
        }

    @classmethod
    def _from_json_dict_(cls,
                         num_qubits,
                         key,
                         invert_mask,
                         qid_shape=None,
                         **kwargs):
        return cls(num_qubits=num_qubits,
                   key=key,
                   invert_mask=tuple(invert_mask),
                   qid_shape=None if qid_shape is None else tuple(qid_shape))


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)


def measure(*qubits: raw_types.Qid,
            key: Optional[str] = None,
            invert_mask: Tuple[bool, ...] = ()) -> raw_types.Operation:
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
        ValueError if the qubits are not instances of Qid.
    """
    for qubit in qubits:
        if isinstance(qubit, np.ndarray):
            raise ValueError(
                'measure() was called a numpy ndarray. Perhaps you meant '
                'to call measure_state_vector on numpy array?')
        elif not isinstance(qubit, raw_types.Qid):
            raise ValueError(
                'measure() was called with type different than Qid.')

    if key is None:
        key = _default_measurement_key(qubits)
    qid_shape = protocols.qid_shape(qubits)
    return MeasurementGate(len(qubits), key, invert_mask, qid_shape).on(*qubits)


def measure_each(*qubits: raw_types.Qid,
                 key_func: Callable[[raw_types.Qid], str] = str
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
    return [
        MeasurementGate(1, key_func(q), qid_shape=(q.dimension,)).on(q)
        for q in qubits
    ]
