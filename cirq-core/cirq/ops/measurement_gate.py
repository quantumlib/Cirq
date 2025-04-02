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

from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np

from cirq import _compat, protocols, value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class MeasurementGate(raw_types.Gate):
    """A gate that measures qubits in the computational basis.

    The measurement gate contains a key that is used to identify results
    of measurements.

    Instead of constructing this directly, consider using the `cirq.measure`
    helper method.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        key: Union[str, 'cirq.MeasurementKey'] = '',
        invert_mask: Tuple[bool, ...] = (),
        qid_shape: Optional[Tuple[int, ...]] = None,
        confusion_map: Optional[Dict[Tuple[int, ...], np.ndarray]] = None,
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
            confusion_map: A map of qubit index sets (using indices in the
                operation generated from this gate) to the 2D confusion matrix
                for those qubits. Indices not included use the identity.
                Applied before invert_mask if both are provided.

        Raises:
            ValueError: If invert_mask or confusion_map have indices
                greater than the available qubit indices, or if the length of
                qid_shape doesn't equal num_qubits.
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
        self._mkey = (
            key if isinstance(key, value.MeasurementKey) else value.MeasurementKey(name=key)
        )
        self._invert_mask = invert_mask or ()
        if self.invert_mask is not None and len(self.invert_mask) > self.num_qubits():
            raise ValueError('len(invert_mask) > num_qubits')
        self._confusion_map = confusion_map or {}
        if any(x >= self.num_qubits() for idx in self._confusion_map for x in idx):
            raise ValueError('Confusion matrices have index out of bounds.')

    @property
    def key(self) -> str:
        return str(self.mkey)

    @property
    def mkey(self) -> 'cirq.MeasurementKey':
        return self._mkey

    @property
    def invert_mask(self) -> Tuple[bool, ...]:
        return self._invert_mask

    @property
    def confusion_map(self) -> Dict[Tuple[int, ...], np.ndarray]:
        return self._confusion_map

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return False

    def with_key(self, key: Union[str, 'cirq.MeasurementKey']) -> 'MeasurementGate':
        """Creates a measurement gate with a new key but otherwise identical."""
        if key == self.key:
            return self
        return MeasurementGate(
            self.num_qubits(),
            key=key,
            invert_mask=self.invert_mask,
            qid_shape=self._qid_shape,
            confusion_map=self.confusion_map,
        )

    def _with_key_path_(self, path: Tuple[str, ...]):
        return self.with_key(self.mkey._with_key_path_(path))

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return self.with_key(self.mkey._with_key_path_prefix_(prefix))

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ):
        return self.with_key(protocols.with_rescoped_keys(self.mkey, path, bindable_keys))

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        return self.with_key(protocols.with_measurement_key_mapping(self.mkey, key_map))

    def with_bits_flipped(self, *bit_positions: int) -> 'MeasurementGate':
        """Toggles whether or not the measurement inverts various outputs.

        This only affects the invert_mask, which is applied after confusion
        matrices if any are defined.
        """
        old_mask = self.invert_mask or ()
        n = max(len(old_mask) - 1, *bit_positions) + 1
        new_mask = [k < len(old_mask) and old_mask[k] for k in range(n)]
        for b in bit_positions:
            new_mask[b] = not new_mask[b]
        return MeasurementGate(
            self.num_qubits(),
            key=self.key,
            invert_mask=tuple(new_mask),
            qid_shape=self._qid_shape,
            confusion_map=self.confusion_map,
        )

    def full_invert_mask(self) -> Tuple[bool, ...]:
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

    def _measurement_key_name_(self) -> str:
        return self.key

    def _measurement_key_obj_(self) -> 'cirq.MeasurementKey':
        return self.mkey

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
        flipped_indices = {i for i, x in enumerate(self.full_invert_mask()) if x}
        confused_indices = {x for idxs in self.confusion_map for x in idxs}

        # Show which output bits are negated and/or confused.
        for i in range(self.num_qubits()):
            prefix = ''
            if i in flipped_indices:
                prefix += '!'
            if i in confused_indices:
                prefix += '?'
            symbols[i] = prefix + symbols[i]

        # Mention the measurement key.
        label_map = args.label_map or {}
        if not args.known_qubits or self.key != _default_measurement_key(args.known_qubits):
            if self.key not in label_map:
                symbols[0] += f"('{self.key}')"
        if self.key in label_map:
            symbols += '@'

        return protocols.CircuitDiagramInfo(symbols)

    def _qasm_(self, args: 'cirq.QasmArgs', qubits: Tuple['cirq.Qid', ...]) -> Optional[str]:
        if self.confusion_map or not all(d == 2 for d in self._qid_shape):
            return NotImplemented
        args.validate_version('2.0', '3.0')
        invert_mask = self.invert_mask
        if len(invert_mask) < len(qubits):
            invert_mask = invert_mask + (False,) * (len(qubits) - len(invert_mask))
        lines = []
        for i, (qubit, inv) in enumerate(zip(qubits, invert_mask)):
            if inv:
                lines.append(args.format('x {0};  // Invert the following measurement\n', qubit))
            if args.version == '2.0':
                lines.append(args.format('measure {0} -> {1:meas}[{2}];\n', qubit, self.key, i))
            else:
                lines.append(args.format('{1:meas}[{2}] = measure {0};\n', qubit, self.key, i))
            if inv:
                lines.append(args.format('x {0};  // Undo the inversion\n', qubit))
        return ''.join(lines)

    def _op_repr_(self, qubits: Sequence['cirq.Qid']) -> str:
        args = list(repr(q) for q in qubits)
        if self.key != _default_measurement_key(qubits):
            args.append(f'key={self.mkey!r}')
        if self.invert_mask:
            args.append(f'invert_mask={self.invert_mask!r}')
        if self.confusion_map:
            proper_map_str = ', '.join(
                f"{k!r}: {_compat.proper_repr(v)}" for k, v in self.confusion_map.items()
            )
            args.append(f'confusion_map={{{proper_map_str}}}')
        arg_list = ', '.join(args)
        return f'cirq.measure({arg_list})'

    def __repr__(self):
        args = [f'{self.num_qubits()!r}', f'{self.mkey!r}', f'{self.invert_mask}']
        if any(d != 2 for d in self._qid_shape):
            args.append(f'qid_shape={self._qid_shape!r}')
        if self.confusion_map:
            proper_map_str = ', '.join(
                f"{k!r}: {_compat.proper_repr(v)}" for k, v in self.confusion_map.items()
            )
            args.append(f'confusion_map={{{proper_map_str}}}')
        return f'cirq.MeasurementGate({", ".join(args)})'

    def _value_equality_values_(self) -> Any:
        hashable_cmap = frozenset(
            (idxs, tuple(v for _, v in np.ndenumerate(cmap)))
            for idxs, cmap in self._confusion_map.items()
        )
        return self.key, self.full_invert_mask(), self._qid_shape, hashable_cmap

    def _json_dict_(self) -> Dict[str, Any]:
        other: Dict[str, Any] = {}
        if not all(d == 2 for d in self._qid_shape):
            other['qid_shape'] = self._qid_shape
        if self.confusion_map:
            json_cmap = [(k, v.tolist()) for k, v in self.confusion_map.items()]
            other['confusion_map'] = json_cmap
        return {
            'num_qubits': len(self._qid_shape),
            'key': self.key,
            'invert_mask': self.invert_mask,
            **other,
        }

    @classmethod
    def _from_json_dict_(
        cls, num_qubits, key, invert_mask, qid_shape=None, confusion_map=None, **kwargs
    ):
        return cls(
            num_qubits=num_qubits,
            key=value.MeasurementKey.parse_serialized(key),
            invert_mask=tuple(invert_mask),
            qid_shape=None if qid_shape is None else tuple(qid_shape),
            confusion_map={tuple(k): np.array(v) for k, v in confusion_map or []},
        )

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return True

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase', qubits: Sequence['cirq.Qid']) -> bool:
        from cirq.sim import SimulationState

        if not isinstance(sim_state, SimulationState):
            return NotImplemented
        sim_state.measure(qubits, self.key, self.full_invert_mask(), self.confusion_map)
        return True


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)
