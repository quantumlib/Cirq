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

from typing import Any, Dict, Iterable, Optional, Tuple, Sequence, TYPE_CHECKING, Union

import numpy as np

from cirq import protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class MeasurementGate(raw_types.Gate):
    """A gate that measures qubits in the computational basis.

    The measurement gate contains a key that is used to identify results
    of measurements.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        key: Union[str, value.MeasurementKey] = '',
        invert_mask: Tuple[bool, ...] = (),
        qid_shape: Tuple[int, ...] = None,
    ) -> None:
        """Inits MeasurementGate.

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
                raise ValueError('Specify either the num_qubits or qid_shape argument.')
            qid_shape = (2,) * num_qubits
        elif num_qubits is None:
            num_qubits = len(qid_shape)
        if num_qubits == 0:
            raise ValueError('Measuring an empty set of qubits.')
        self._qid_shape = qid_shape
        if len(self._qid_shape) != num_qubits:
            raise ValueError('len(qid_shape) != num_qubits')
        self.key = key  # type: ignore
        self.invert_mask = invert_mask or ()
        if self.invert_mask is not None and len(self.invert_mask) > self.num_qubits():
            raise ValueError('len(invert_mask) > num_qubits')

    @property
    def key(self) -> str:
        return str(self.mkey)

    @key.setter
    def key(self, key: Union[str, value.MeasurementKey]):
        if isinstance(key, value.MeasurementKey):
            self.mkey = key
        else:
            self.mkey = value.MeasurementKey(name=key)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def with_key(self, key: Union[str, value.MeasurementKey]) -> 'MeasurementGate':
        """Creates a measurement gate with a new key but otherwise identical."""
        if key == self.key:
            return self
        return MeasurementGate(
            self.num_qubits(), key=key, invert_mask=self.invert_mask, qid_shape=self._qid_shape
        )

    def _with_key_path_(self, path: Tuple[str, ...]):
        return self.with_key(self.mkey._with_key_path_(path))

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        return self.with_key(protocols.with_measurement_key_mapping(self.mkey, key_map))

    def with_bits_flipped(self, *bit_positions: int) -> 'MeasurementGate':
        """Toggles whether or not the measurement inverts various outputs."""
        old_mask = self.invert_mask or ()
        n = max(len(old_mask) - 1, *bit_positions) + 1
        new_mask = [k < len(old_mask) and old_mask[k] for k in range(n)]
        for b in bit_positions:
            new_mask[b] = not new_mask[b]
        return MeasurementGate(
            self.num_qubits(), key=self.key, invert_mask=tuple(new_mask), qid_shape=self._qid_shape
        )

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

    def _is_measurement_(self) -> bool:
        return True

    def _measurement_key_(self):
        return self.key

    def _kraus_(self):
        size = np.prod(self._qid_shape, dtype=np.int64)

        def delta(i):
            result = np.zeros((size, size))
            result[i][i] = 1
            return result

        return tuple(delta(i) for i in range(size))

    def _has_kraus_(self):
        return True

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        symbols = ['M'] * self.num_qubits()

        # Show which output bits are negated.
        if self.invert_mask:
            for i, b in enumerate(self.invert_mask):
                if b:
                    symbols[i] = '!M'

        # Mention the measurement key.
        if not args.known_qubits or self.key != _default_measurement_key(args.known_qubits):
            symbols[0] += f"('{self.key}')"

        return protocols.CircuitDiagramInfo(tuple(symbols))

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        args.validate_version('2.0')
        invert_mask = self.invert_mask
        if len(invert_mask) < len(qubits):
            invert_mask = invert_mask + (False,) * (len(qubits) - len(invert_mask))
        lines = []
        for i, (qubit, inv) in enumerate(zip(qubits, invert_mask)):
            if inv:
                lines.append(args.format('x {0};  // Invert the following measurement\n', qubit))
            lines.append(args.format('measure {0} -> {1:meas}[{2}];\n', qubit, self.key, i))
        return ''.join(lines)

    def _quil_(
        self, qubits: Tuple['cirq.Qid', ...], formatter: 'cirq.QuilFormatter'
    ) -> Optional[str]:
        if not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        invert_mask = self.invert_mask
        if len(invert_mask) < len(qubits):
            invert_mask = invert_mask + (False,) * (len(qubits) - len(invert_mask))
        lines = []
        for i, (qubit, inv) in enumerate(zip(qubits, invert_mask)):
            if inv:
                lines.append(
                    formatter.format('X {0} # Inverting for following measurement\n', qubit)
                )
            lines.append(formatter.format('MEASURE {0} {1:meas}[{2}]\n', qubit, self.key, i))
        return ''.join(lines)

    def _op_repr_(self, qubits: Sequence['cirq.Qid']) -> str:
        args = list(repr(q) for q in qubits)
        if self.key != _default_measurement_key(qubits):
            args.append(f'key={self.key!r}')
        if self.invert_mask:
            args.append(f'invert_mask={self.invert_mask!r}')
        arg_list = ', '.join(args)
        return f'cirq.measure({arg_list})'

    def __repr__(self):
        qid_shape_arg = ''
        if any(d != 2 for d in self._qid_shape):
            qid_shape_arg = f', {self._qid_shape!r}'
        return (
            f'cirq.MeasurementGate('
            f'{self.num_qubits()!r}, '
            f'{self.key!r}, '
            f'{self.invert_mask}'
            f'{qid_shape_arg})'
        )

    def _value_equality_values_(self) -> Any:
        return self.key, self.invert_mask, self._qid_shape

    def _json_dict_(self) -> Dict[str, Any]:
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
    def _from_json_dict_(cls, num_qubits, key, invert_mask, qid_shape=None, **kwargs):
        return cls(
            num_qubits=num_qubits,
            key=value.MeasurementKey.parse_serialized(key),
            invert_mask=tuple(invert_mask),
            qid_shape=None if qid_shape is None else tuple(qid_shape),
        )

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']) -> bool:
        args.measure(qubits, self.key, self.full_invert_mask())
        return True


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)
